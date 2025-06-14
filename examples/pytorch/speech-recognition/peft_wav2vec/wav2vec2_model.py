
"""Fine-tuning a 🤗 Transformers CTC model for automatic speech recognition"""

import functools
import json
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union
from pprint import pprint
import pandas as pd
import numpy as np
import wandb
sys.path.append(os.pardir)
import argparse
from distutils.util import strtobool
import torch
import bitsandbytes as bnb

# dataset utils
import datasets
import evaluate
from evaluate import load
from datasets import DatasetDict
from datasets import load_dataset

import transformers
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
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from utils import (
  read_data,
  get_num_trainable_parameters,
  Dataset, 
  DataCollatorCTCWithPadding,
  train_model,
  create_vocabulary_from_data,
  fix_seed
)

import logging
logger = logging.getLogger(__name__)

fix_seed(42)

def main():
    parser = argparse.ArgumentParser()

    # General Configs
    parser.add_argument('--run_name', type=str, default='sample_run')
    parser.add_argument('--use_skip', type=strtobool, default=False)
    parser.add_argument('--save_model', type=strtobool, default=False)
    parser.add_argument('--use_steplr', type=strtobool, default=True) 
    parser.add_argument('--classifier_lr', type=float, default=1e-3)
    parser.add_argument('--train_encoder', type=strtobool, default=False)
    parser.add_argument('--train_adapter', type=strtobool, default=False)
    parser.add_argument('--adapter_lr', type=float, default=1e-3)
    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    parser.add_argument('--wandb_log', type=strtobool, default=False)
    parser.add_argument('--verbose', type=strtobool, default=True)
    parser.add_argument('--max_duration', type=int, default=20)

    # Model Traning Arguments
    parser.add_argument('--model_name_or_path', type=str, default='facebook/wav2vec2-large-xlsr-53')
    parser.add_argument('--tokenizer_name_or_path', type=str, default=None)
    parser.add_argument('--freeze_feature_encoder', type=strtobool, default=True)
    parser.add_argument('--attention_dropout', type=float, default=0.0)
    parser.add_argument('--feat_proj_dropout', type=float, default=0.0)
    parser.add_argument('--activation_dropout', type=float, default=0.0)
    parser.add_argument('--hidden_dropout', type=float, default=0.0)
    parser.add_argument('--final_dropout', type=float, default=0.0)
    parser.add_argument('--mask_time_prob', type=float, default=0.05)
    parser.add_argument('--mask_time_length', type=int, default=10)
    parser.add_argument('--mask_feature_prob', type=float, default=0.0)
    parser.add_argument('--mask_feature_length', type=int, default=10)
    parser.add_argument('--layerdrop', type=float, default=0.0)
    parser.add_argument('--ctc_loss_reduction', type=str, default="mean")
    parser.add_argument('--ctc_zero_infinity', type=strtobool, default=True)
    parser.add_argument('--add_adapter', type=strtobool, default=True)

    # Data Training Arguments
    parser.add_argument('--audio_column_name', type=str, default="wav")
    parser.add_argument('--text_column_name', type=str, default="wrd")
    parser.add_argument('--train_split_path', type=str, default="./")
    parser.add_argument('--eval_split_path', type=str, default="./")
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--unk_token', type=str, default="[UNK]")
    parser.add_argument('--pad_token', type=str, default="[PAD]")
    parser.add_argument('--word_delimiter_token', type=str, default="|")
    parser.add_argument('--trust_remote_code', type=strtobool, default=True)
    parser.add_argument('--min_duration_in_seconds', type=int, default=2)
    parser.add_argument('--max_duration_in_seconds', type=int, default=20)
    parser.add_argument('--preprocessing_num_workers', type=int, default=12)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--phoneme_language', type=None, default=None)
    parser.add_argument('--chars_to_ignore', type=str, default=None)
    parser.add_argument('--preprocessing_only', type=strtobool, default=False)

    # Trainer Configurations
    parser.add_argument('--output_dir', type=str, default='./output_dir')
    parser.add_argument('--cache_dir', type=str, default='./cache_dir')
    parser.add_argument('--group_by_length', type=strtobool, default=True)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--gradient_checkpointing', type=strtobool, default=True)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--overwrite_output_dir', type=strtobool, default=True)
    parser.add_argument('--do_train', type=strtobool, default=False)
    parser.add_argument('--do_eval', type=strtobool, default=False)
    parser.add_argument('--seed', type=int, default=42)

    # Huggingface Configs
    parser.add_argument('--push_to_hub', type=strtobool, default=False)

    args = parser.parse_args()

    training_args = TrainingArguments(
      output_dir=args.output_dir,
      overwrite_output_dir=args.overwrite_output_dir,
      group_by_length=args.group_by_length,
      per_device_train_batch_size=args.per_device_train_batch_size,
      per_device_eval_batch_size=args.per_device_eval_batch_size,
      gradient_accumulation_steps=2,
      num_train_epochs=args.num_train_epochs,
      gradient_checkpointing=args.gradient_checkpointing,
      fp16=True,
      save_steps=400,
      eval_steps=400,
      logging_steps=400,
      learning_rate=args.learning_rate,
      warmup_steps=400,
      greater_is_better=False,
      do_train=args.do_train,
      do_eval=args.do_eval,
      push_to_hub=False,
      seed=42,
      )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)


    # 1. Tokenization
    raw_datasets = load_dataset("csv",
                                data_files={
                                  "train": args.train_split_path,  
                                  "eval": args.eval_split_path, 
                                  },
                                delimiter=",",
                                streaming=False,
                                cache_dir="./cache",
                                )
    # 2. We remove some special characters from the datasets
    # that make training complicated and do not help in transcribing the speech
    # E.g. characters, such as `,` and `.` do not really have an acoustic characteristic
    # that could be easily picked up by the model
    chars_to_ignore_regex = (
        f"[{''.join(args.chars_to_ignore)}]" if args.chars_to_ignore is not None else None
    )

    text_column_name = args.text_column_name

    def remove_special_characters(batch):
        if chars_to_ignore_regex is not None:
            batch["target_text"] = re.sub(
              chars_to_ignore_regex, 
              "", 
              batch[args.text_column_name]).lower() + " "
        else:
            batch["target_text"] = batch[args.text_column_name].lower() + " "
        return batch

    with training_args.main_process_first(desc="dataset map special characters removal"):
        raw_datasets = raw_datasets.map(
            remove_special_characters,
            remove_columns=[args.text_column_name],
            desc="remove special characters from datasets",
        )

    # save special tokens for tokenizer
    word_delimiter_token = args.word_delimiter_token
    unk_token = args.unk_token
    pad_token = args.pad_token


    # 3. Next, let's load the config as we might need it to create
    # the tokenizer
    # Tokenizer config
    config = AutoConfig.from_pretrained(
      args.model_name_or_path,
      cache_dir=args.cache_dir,
      token=args.token,
       trust_remote_code=args.trust_remote_code,
      )

    # 4. Next, if no tokenizer file is defined,
    # we create the vocabulary of the model by extracting all unique characters from
    # the training and evaluation datasets
    # We need to make sure that only first rank saves vocabulary
    # make sure all processes wait until vocab is created
    tokenizer_name_or_path = args.tokenizer_name_or_path
    tokenizer_kwargs = {}
    if tokenizer_name_or_path is None:
      tokenizer_name_or_path = training_args.output_dir
      vocab_file = os.path.join(tokenizer_name_or_path, "vocab.json")

      with training_args.main_process_first():
            if training_args.overwrite_output_dir and os.path.isfile(vocab_file):
                try:
                    os.remove(vocab_file)
                except OSError:
                    # in shared file-systems it might be the case that
                    # two processes try to delete the vocab file at the some time
                    pass
      with training_args.main_process_first(desc="dataset map vocabulary creation"):
            if not os.path.isfile(vocab_file):
                os.makedirs(tokenizer_name_or_path, exist_ok=True)
                vocab_dict = create_vocabulary_from_data(
                    raw_datasets,
                    word_delimiter_token=args.word_delimiter_token,
                    unk_token=args.unk_token,
                    pad_token=args.pad_token,
                )

                # save vocab dict to be loaded into tokenizer
                with open(vocab_file, "w") as file:
                    json.dump(vocab_dict, file)
       # if tokenizer has just been created
      # it is defined by `tokenizer_class` if present in config else by `model_type`
      tokenizer_kwargs = {
        "config": config if config.tokenizer_class is not None else None,
        "tokenizer_type": config.model_type if config.tokenizer_class is None else None,
        "unk_token": args.unk_token,
        "pad_token": args.pad_token,
        "word_delimiter_token": args.word_delimiter_token,}
      
    # 5. Now we can instantiate the feature extractor, tokenizer and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.
    # load feature_extractor and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
      tokenizer_name_or_path,
      token=args.token,
      trust_remote_code=args.trust_remote_code,
      **tokenizer_kwargs,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )

    # adapt config
    config.update(
        {
          "feat_proj_dropout": args.feat_proj_dropout,
          "attention_dropout": args.attention_dropout,
          "hidden_dropout": args.hidden_dropout,
          "final_dropout": args.final_dropout,
          "mask_time_prob": args.mask_time_prob,
          "mask_time_length": args.mask_time_length,
          "mask_feature_prob": args.mask_feature_prob,
          "mask_feature_length": args.mask_feature_length,
          "gradient_checkpointing": training_args.gradient_checkpointing,
          "layerdrop": args.layerdrop,
          "ctc_loss_reduction": args.ctc_loss_reduction,
          "ctc_zero_infinity": args.ctc_zero_infinity,
          "pad_token_id": tokenizer.pad_token_id,
          "vocab_size": len(tokenizer),
          "activation_dropout": args.activation_dropout,
          "add_adapter": args.add_adapter,
        }
    )

    # Create Wav2Vec model
    model = AutoModelForCTC.from_pretrained(
      args.model_name_or_path, 
      config=config,
      token=args.token,
      trust_remote_code=args.trust_remote_code
      )
    # freeze encoder
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # 6. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`
     # make sure that dataset decodes audio with correct sampling rate
    raw_datasets = raw_datasets.cast_column(
      args.audio_column_name,
      datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
      )


    # `phoneme_language` is only relevant if the model is fine-tuned on phoneme classification
    phoneme_language = args.phoneme_language

    # derive max & min input length for sample rate & max duration
    max_input_length = args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = args.min_duration_in_seconds * feature_extractor.sampling_rate
    audio_column_name = args.audio_column_name
    num_workers = args.preprocessing_num_workers
    feature_extractor_input_name = feature_extractor.model_input_names[0]

    # We need to read the audio files as arrays and tokenize the targets.
    def prepare_dataset(batch):
        # load audio
        sample = batch[audio_column_name]

        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch[feature_extractor_input_name] = getattr(inputs, feature_extractor_input_name)[0]
        # take length of raw audio waveform
        batch["input_length"] = len(sample["array"].squeeze())

        # encode targets
        additional_kwargs = {}
        if phoneme_language is not None:
            additional_kwargs["phonemizer_lang"] = phoneme_language

        batch["labels"] = tokenizer(batch["target_text"], **additional_kwargs).input_ids
        return batch

    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=num_workers,
            desc="preprocess datasets",
        )

        def is_audio_in_length_range(length):
            return length > min_input_length and length < max_input_length

        # filter data that is shorter than min_input_length
        vectorized_datasets = vectorized_datasets.filter(
            is_audio_in_length_range,
            num_proc=num_workers,
            input_columns=["input_length"],
        )

    
    # 7. Next, we can prepare the training.
    # Let's use word error rate (WER) as our evaluation metric,
    # instantiate a data collator and the trainer
    metric = load('wer')

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with ``args.preprocessing_only`` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step ``args.preprocessing_only`` can then be set to `False` to load the
    # cached dataset
    if args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}")
        return

    # For languages like Chinese with large vocabulary size, we need to discard logits
    # and only keep the argmax, otherwise we run out of memory during evaluation.
    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels

    def compute_metrics(pred):
        pred_ids = pred.predictions[0]
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

        metrics = {k: v.compute(predictions=pred_str, references=label_str) for k, v in metrics.items()}

        return metrics

    # Now save everything to be able to create a single processor later
    # make sure all processes wait until data is saved
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    try:
        processor = AutoProcessor.from_pretrained(training_args.output_dir)
    except (OSError, KeyError):
        warnings.warn(
            "Loading a processor from a feature extractor config that does not"
            " include a `processor_class` attribute is deprecated and will be removed in v5. Please add the following "
            " attribute to your `preprocessor_config.json` file to suppress this warning: "
            " `'processor_class': 'Wav2Vec2Processor'`",
            FutureWarning,
        )
        processor = Wav2Vec2Processor.from_pretrained(training_args.output_dir)

    # Instantiate custom data collator
    collator = DataCollatorCTCWithPadding(
      processor=processor, 
      feature_extractor_input_name=feature_extractor_input_name)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        processing_class=processor,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # 8. Finally, we can start training the model
    if training_args.do_train:
        # use last checkpoint if exist
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(args.model_name_or_path):
            checkpoint = args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            args.max_train_samples
            if args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            args.max_eval_samples if args.max_eval_samples is not None else len(vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["eval"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    
    #########
    configs={
      "scheduler": {'type':'StepLR', 'step':25, 'gamma':0.3} if args.use_steplr else {'type':'LambdaLR', 'param':{'alpha':0.20, 'beta':0.03, 'start':10, 'end':1.0, 'scale':10}},
    }

    num_epochs = training_args.num_train_epochs
    learning_rate = training_args.learning_rate
    sc_setting = configs['scheduler']

    # Counting model parameters
    down_param = []
    adapter_param = []
    encoder_param = []
    pcount = 0
    adapcount = 0
    flag = True

    if args.train_encoder:
      layer_names = [str(i) for i in range(0, 12)]

    for name, param in model.named_parameters():
      for layer in layer_names:
          if layer in name:
              flag=True
              break
          else:
              flag=False
      if 'lm_head' in name:
        print('down_param: ', name)
        pcount += param.numel()
        down_param.append(param)
      
      elif 'encoder.layers' in name and flag and args.train_encoder:
          encoder_param.append(param)
          pcount += param.numel();print('encoder: ', name)

      elif 'adapter.layers' in name:
          adapter_param.append(param)
          print('adapter: ', name)
          pcount += param.numel(); adapcount += param.numel()

    print('\ncount of parameters: ', round(pcount/1e6, 2), 'M')
    print('count of adapter parameters: ', round(adapcount/1e6, 2), 'M')
    trainable_params = get_num_trainable_parameters(model)
    print("Number of trainable paramters",round(trainable_params/1e6, 2), 'M\n')

    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    # defining the optimizer
    optimizer = bnb.optim.Adam8bit(
        params=optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )
    
    # Scheduler for model optimization
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sc_setting['step'], gamma=sc_setting['gamma'])

    if args.wandb_log:
        wandb.init(
            project="ASR",
            config=config,
            id=args.run_name
        )

    train_dataset=vectorized_datasets["train"] #if training_args.do_train else None,
    val_dataset=vectorized_datasets["eval"] #if training_args.do_eval else None

    print(train_dataset)

    train_loader = torch.utils.data.DataLoader(
      train_dataset, 
      batch_size=training_args.per_device_train_batch_size, 
      collate_fn=collator, 
      shuffle=True, 
      num_workers=num_workers, 
      pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
      val_dataset, 
      batch_size=training_args.per_device_eval_batch_size, 
      collate_fn=collator, 
      shuffle=False, 
      num_workers=num_workers, 
      pin_memory=True)
    dataloaders_dict = {
      'train':train_loader, 
      'val':val_loader
      }

    # Model Initialization
    model = train_model(
      model, 
      processor,
      tokenizer, 
      dataloaders_dict, 
      optimizer, 
      scheduler, 
      metric, 
      num_epochs, 
      report_wandb=False, 
      val_interval=100)

    if args.save_model:
        # Save model properly whether DataParallel is used or not
        if hasattr(model, "module"):
            torch.save(model.module.state_dict(), args.run_name + ".pth")  # If wrapped in DataParallel
        else:
            torch.save(model.state_dict(), args.run_name + ".pth")  # If not wrapped



    # Write model card and (optionally) push to hub
    #config_name = args.dataset_config_name if args.dataset_config_name is not None else "na"
    #kwargs = {
    #    "finetuned_from": args.model_name_or_path,
    #    "tasks": "automatic-speech-recognition",
    #    "tags": ["automatic-speech-recognition", args.dataset_name],
    #    "dataset_args": (
    #        f"Config: {config_name}, Training split: {args.train_split_name}, Eval split:"
    #        f" {args.eval_split_name}"
    #    ),
    #    "dataset": f"{args.dataset_name.upper()} - {config_name.upper()}",
    #}
    #if "common_voice" in args.dataset_name:
    #    kwargs["language"] = config_name

    #if training_args.push_to_hub:
    #    trainer.push_to_hub(**kwargs)
    #else:
    #    trainer.create_model_card(**kwargs)

    #return results


if __name__ == '__main__':
    main()

