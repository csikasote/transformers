import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForAudioClassification
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    set_seed,
)


import warnings
warnings.filterwarnings('ignore')

"""
    Dual model that jointly tackles automatic speech recognition (ASR) and adversarial tasks
"""
class DualModel(torch.nn.Module):
    def __init__(
        self, 
        model_name_or_path,
        cache_dir,
        token, 
        num_subgroups=2, 
        hidden_size=768,
        trust_remote_code=False,
        ignore_mismatched_sizes=True, 
        local_files_only=False
        ):
        super(DualModel, self).__init__()
        # 1) Load Wav2Vec2 config to pull out vocab_size and hidden_size
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=trust_remote_code
        )
        self.hidden_size = config.hidden_size     # e.g. 768 for base, 1024 for large
        self.vocab_size = config.vocab_size

        # 2) Backbone: Wav2Vec2Model (no LM head). Returns hidden states: [batch, time, hidden_size]
        # create model
        self.base_model = AutoModelForCTC.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            config=config,
            token=token,
            trust_remote_code=trust_remote_code,
            ignore_mismatched_sizes=True,
        )

        # 3) CTC head: map hidden_size → vocab_size (applied at each time step)
        self.ctc_head = nn.Linear(self.hidden_size, self.vocab_size)

        # 4) Subgroup/adversarial head: takes a pooled hidden-vector, outputs single logit
        self.cls_subgroups = nn.Sequential(
            nn.Linear(self.hidden_size, 100),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(100, 1),
        )
        
    def forward(self, x):
        
        output = self.base_model(**x)

        output_intents = self.cls_intents(output.last_hidden_state[:,-1])
        output_subgroups = self.cls_subgroups(output.last_hidden_state[:,-1])

        return output_intents, output_subgroups