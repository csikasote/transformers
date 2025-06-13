import functools
import json
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import datasets
import evaluate
import torch
import torch.nn as nn
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
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Initiations
metric = evaluate.load('wer')
# Defing the model
def train_model(
        model, 
        processor, 
        dataloaders_dict, 
        optimizer, 
        scheduler, 
        metric, 
        num_epochs, 
        log_interval=10, 
        report_wandb=False, 
        val_interval=5):
    
    #setting device configs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)

    torch.backends.cudnn.benchmark = True

    pbar_update = 1 / sum([len(v) for v in dataloaders_dict.values()])

    opt_flag = (type(optimizer) == list)
    sc_flag = (type(scheduler) == list)

    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            for phase in ['train', 'val']:

                if phase == 'train':
                    # Training stage
                    model.train()
                else:
                    # Evaluation stage
                    if (epoch+1) % val_interval:
                        for _ in range(len(dataloaders_dict[phase])):
                            pbar.update(pbar_update)
                        continue
                    model.eval()
                
                epoch_loss = 0.0
                epoch_wer = 0.0

                epoch_preds_str=[]; epoch_labels_str=[]

                for step, inputs in enumerate(dataloaders_dict[phase]):
                    minibatch_size = inputs['input_values'].size(0)
                    labels_ids = inputs['labels']
                    inputs = inputs.to(device)

                    if opt_flag:
                        for optim in optimizer:
                            optim.zero_grad()
                    else:
                        optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(**inputs) # get outputs from the model
                        del inputs
                        loss = outputs.loss.mean(dim=-1)

                        preds_logits = outputs.logits
                        preds_ids = torch.argmax(preds_logits, dim=-1)
                        preds_str = processor.batch_decode(preds_ids)
                        labels_ids[labels_ids==-100] = processor.tokenizer.pad_token_id
                        labels_str = processor.batch_decode(labels_ids, group_tokens=False)
                        wer = metric.compute(predictions=preds_str, references=labels_str)
                        epoch_preds_str += preds_str
                        epoch_labels_str += labels_str
                    if phase == 'train':
                        loss.backward()

                        if opt_flag:
                            for opt in optimizer:
                                opt.step()
                        else:
                            optimizer.step()
                        loss_log = loss.item()

                        del loss

                        if report_wandb:
                            wandb.log({'train/loss':loss_log})

                    epoch_loss += loss_log * minibatch_size







            
                

