import warnings
import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model
from transformers.modeling_outputs import CausalLMOutput
from dat_utils import GradientReversalFunction
from dat_utils import GradientReversalLayer

#
# ─── Wav2Vec2ForCTC with Domain-Adversarial Branch ───────────────────────────────
#
class Wav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
    def __init__(
        self,
        config,
        target_lang: Optional[str] = None,
        domain_classes: int = 2,
        lambda_grl: float = 1.0,
    ):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)

        # Domain-Adversarial Components
        self.grl = GradientReversalLayer(lambda_grl=lambda_grl)

        feat_dim = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.final_dropout),
            nn.Linear(feat_dim // 2, domain_classes),
        )

        self.target_lang = target_lang

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        output_hidden_size = feat_dim
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # weight init
        self.post_init()

    def tie_weights(self):
        # unchanged from parent, used for adapter loading if needed
        super().tie_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        for p in self.wav2vec2.parameters():
            p.requires_grad = False

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.LongTensor] = None,
        lambda_grl: Optional[float] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1) Feature extraction + CTC head
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=return_dict,
            return_dict=return_dict,
        )
        hidden_states = outputs.last_hidden_state if return_dict else outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)

        # 2) Compute CTC loss if labels provided
        ctc_loss = None
        if labels is not None:
            # derive input_lengths
            attn = attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            input_lengths = self._get_feat_extract_output_lengths(attn.sum(-1)).to(torch.long)
            mask = labels >= 0
            target_lengths = mask.sum(-1)
            flat_targets = labels.masked_select(mask)

            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss = nn.functional.ctc_loss(
                    log_probs,
                    flat_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        # 3) Domain‐adversarial branch
        domain_loss = None
        if domain_labels is not None:
            # optionally override λ coefficient
            if lambda_grl is not None:
                self.grl.lambda_grl = lambda_grl

            # aggregate over time dim: (B, T, D) → (B, D)
            feat = hidden_states.mean(dim=1)
            rev_feat = self.grl(feat)
            domain_logits = self.domain_classifier(rev_feat)
            domain_loss = nn.functional.cross_entropy(domain_logits, domain_labels)

        # 4) Combine losses
        if ctc_loss is not None and domain_loss is not None:
            total_loss = ctc_loss + domain_loss
        else:
            total_loss = ctc_loss if domain_loss is None else domain_loss

        # 5) Return
        if not return_dict:
            output = (logits,) + (outputs.hidden_states if return_dict else outputs[1:])
            return ((total_loss,) + output) if total_loss is not None else output

        return CausalLMOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
