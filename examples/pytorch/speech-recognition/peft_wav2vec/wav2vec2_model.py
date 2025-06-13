
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
import datasets
import evaluate
import torch
from datasets import DatasetDict, load_dataset

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
  Dataset, 
  DataCollatorCTCWithPadding,
  train_model,
  create_vocabulary_from_data,
  fix_seed
)

fix_seed(42)

def main():
    parser = argparse.ArgumentParser()

    # Genderal Configs
    parser.add_argument('--run_name', type=str, default='sample_run')
    parser.add_argument('--use_skip', type=strtobool, default=False)
    parser.add_argument('--save_model', type=strtobool, default=False)
    parser.add_argument('--use_steplr', type=strtobool, default=True) 
    parser.add_argument('--classifier_lr', type=float, default=1e-3)
    parser.add_argument('--train_encoder', type=strtobool, default=False)
    parser.add_argument('--encoder_adapter_lr', type=float, default=1e-3)
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

    # Trainer Configurations
    parser.add_argument('--output_dir', type=str, default='./wav2vec')
    parser.add_argument('--group_by_length', type=strtobool, default=True)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--gradient_checkpointing', type=strtobool, default=True)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--overwrite_output_dir', type=strtobool, default=True)

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
      push_to_hub=False,
      )


    # Model & Feature Extractor
    model_checkpoint = args.model_name_or_path
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)
    feature_extractor_input_name = feature_extractor.model_input_names[0]

    # 2. Tokenization
    raw_datasets = load_dataset("csv",
                                data_files={
                                  "train": args.train_split_path,  
                                  "eval": args.eval_split_path, 
                                  },
                                delimiter=",",
                                streaming=False,
                                cache_dir="./cache",
                                )
    chars_to_ignore_regex = None

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

    # Tokenizer config
    config = AutoConfig.from_pretrained(
      args.model_name_or_path,
      token=args.token
      )

    # Create tokenizor
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

    tokenizer_kwargs = {
      "config": config if config.tokenizer_class is not None else None,
      "tokenizer_type": config.model_type if config.tokenizer_class is None else None,
      "unk_token": args.unk_token,
      "pad_token": args.pad_token,
      "word_delimiter_token": args.word_delimiter_token,}
      
    # load feature_extractor and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
      tokenizer_name_or_path,
      token=args.token,
      trust_remote_code=args.trust_remote_code,
      **tokenizer_kwargs,
    )

    # loading dataset
    df_train, df_valid = read_data(os.getcwd(), verbose=args.verbose)

    train_dataset = Dataset(
      examples = df_train, 
      feature_extractor=feature_extractor,
      tokenizer = tokenizer,
      max_duration = args.max_duration, 
      )
      
    val_dataset = Dataset(
      examples=df_valid,  
      feature_extractor = feature_extractor,
      tokenizer = tokenizer,
      max_duration = args.max_duration,
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
    configs={
        "pretrained_model": args.model_name_or_path,
        "dataset": 'LL10h',
        "epochs": 2,
        "batch_size": {'train':8, 'val':4},
        "model_config": config,
        "learning_rate": training_args.learning_rate,
        'optimizer':'Adam',
        "scheduler": {'type':'StepLR', 'step':25, 'gamma':0.3} if args.use_steplr else {'type':'LambdaLR', 'param':{'alpha':0.20, 'beta':0.03, 'start':10, 'end':1.0, 'scale':10}},
    }

    if args.wandb_log:
        wandb.init(
            project="ASR",
            config=config,
            id=args.run_name
        )

    num_epochs = training_args.num_train_epochs
    train_batch_size = training_args.per_device_train_batch_size
    eval_batch_size = training_args.per_device_eval_batch_size
    learning_rate = training_args.learning_rate
    sc_setting = configs['scheduler']
    pretrained_model = args.model_name_or_path

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

    # Create Wav2Vec model
    model = AutoModelForCTC.from_pretrained(
      args.model_name_or_path, 
      config=config,
      token=args.token,
      trust_remote_code=args.trust_remote_code
      )

    # Counting model parameters
    down_param = []
    layernorm_param = []
    encoder_param = []
    encoder_adapter_param = []
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
      
        elif 'encoder.layers' in name and 'layer_norm' in name and flag and not args.train_encoder:
            layernorm_param.append(param)
            print('layer_norm: ', name);pcount += param.numel()
        
        elif 'encoder.layers' in name and flag and args.train_encoder:
            encoder_param.append(param)
            pcount += param.numel();print('encoder: ', name)

        elif 'wav2vec2.adapter.layers' in name:
            encoder_adapter_param.append(param)
            print('adapter: ', name)
            pcount += param.numel(); adapcount += param.numel()
        else:
            print('frozen: ', name)
            param.requires_grad = False

    print('\ncount of parameters: ', round(pcount/1e6, 2), 'M')
    print('\ncount of adapter_parameters: ', round(adapcount/1e6, 2), 'M\n')
    # freeze encoder
    #if args.freeze_feature_encoder:
    #    model.freeze_feature_encoder()


    if args.train_encoder:
        optimizer = torch.optim.Adam([
            {'params': down_param, 'lr': args.classifier_lr},
            {'params': encoder_param, 'lr': args.encoder_lr},
        ])
     
    elif args.train_encoder_adapter:
        optimizer = torch.optim.Adam([
          {'params': down_param, 'lr': args.classifier_lr},
          {'params':encoder_adapter_param, 'lr':args.encoder_adapter_lr},
          {'params': layernorm_param, 'lr': args.encoder_adapter_lr},
          ])
    else:
        optimizer = torch.optim.Adam([
            {'params': down_param, 'lr': args.classifier_lr},
            {'params': layernorm_param, 'lr': args.encoder_lr},
            ])

    
    if args.add_adapter and args.use_steplr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sc_setting['step'], gamma=sc_setting['gamma'])
    else:
        hyparam = sc_setting['param']
        def func(epoch):
            alpha = hyparam['alpha']; beta = hyparam['beta']
            start = hyparam['start']; end = hyparam['end']; scale=hyparam['scale']
            warmup = np.linspace(start,num_epochs, int(num_epochs*alpha)) / num_epochs
            stag = np.ones(int(num_epochs*(beta)))
            decay = np.linspace(num_epochs, end, int(num_epochs*(1-alpha-beta)+1)) / np.linspace(num_epochs,num_epochs*scale,int(num_epochs*(1-alpha-beta)+1))
            steps = np.concatenate([warmup, stag, decay], axis=-1)
            return steps[epoch]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)

    metric = evaluate.load('wer')
    train_loader = torch.utils.data.DataLoader(
      train_dataset, 
      batch_size=train_batch_size, 
      collate_fn=collator, 
      shuffle=True, 
      num_workers=12, 
      pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
      val_dataset, 
      batch_size=eval_batch_size, 
      collate_fn=collator, 
      shuffle=False, 
      num_workers=12, 
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

if __name__ == '__main__':
    main()

