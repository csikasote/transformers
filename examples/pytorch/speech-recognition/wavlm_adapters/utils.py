import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pandas as pd
import random
import wandb
import librosa
import os

from tqdm.notebook import tqdm
from transformers import Wav2Vec2Processor
from transformers import AutoProcessor
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

""" Dataset Class """
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        examples, 
        feature_extractor,
        tokenizer, 
        max_duration, 
        ):
        self.examples = examples['wav']
        self.labels = examples['wrd']
        self.max_duration = max_duration
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __getitem__(self, idx):

      try:
          inputs = self.feature_extractor(
              librosa.load(self.examples[idx], sr=16_000)[0].squeeze(),
              sampling_rate=self.feature_extractor.sampling_rate, 
              return_tensors="pt",
              max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
              truncation=True,
              padding='max_length'
              )
      except:
          print("Audio not available")

      text = self.labels[idx]
      labels = self.tokenizer(
        text=text, 
        return_tensors="pt", 
        padding=False).input_ids[0]
      
      try:
        item = {
          'input_values': inputs['input_values'].squeeze(0),
          'labels': labels
        }
          #item = {'input_values': inputs['input_values'].squeeze(0)}
          #item["labels"] = torch.tensor(self.labels[idx])
      except:
          item = {'input_values': [], 'labels': []}
          
      return item

    def __len__(self):
        return len(self.examples)


""" Read and Process Data"""
def read_data(df_folder, verbose=False):

    df_train = pd.read_csv(
        os.path.join(
            df_folder, 
            'train.csv'
            ), 
            index_col=None)
    df_valid = pd.read_csv(
        os.path.join(
            df_folder, 
            'dev.csv'
            ), 
            index_col=None)
    df_train = df_train.dropna()
    df_valid = df_valid.dropna()

    if verbose:
        print("Train size: ", len(df_train))
        print("Valid size: ", len(df_valid))

    return df_train, df_valid


class LibriSpeechDataset(object):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")    
    def __init__(self, csv_file, lthresh=None):
        self.df = pd.read_csv(csv_file)
        self.lthresh = lthresh
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self.df.loc[idx]
        array, sampling_rate = torchaudio.load(data['wav'])
        
        if self.lthresh:
            array = array.numpy().flatten()[:self.lthresh]
        else:
            array = array.numpy().flatten()
        array = self.processor(array,sampling_rate=sampling_rate).input_values[0]

        text = data['wrd']
        file = data['wav']
        
        with self.processor.as_target_processor():
            labels = self.processor.tokenizer(text, return_tensors="pt").input_ids[0]
        sample = {'input_values': array,  
                  'labels': labels,
                  'files': file}
        return sample

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for processing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    feature_extractor_input_name: Optional[str] = "input_values"

    def __call__(self, features: list[dict[str, Union[list[int], torch.Tensor]]]) -> dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            {self.feature_extractor_input_name: feature[self.feature_extractor_input_name]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)

        return batch

@dataclass
class DataCollatorCTCWithPadding_:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    preprocess: Optional[bool] = False
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if self.preprocess:
            input_features = []; label_features = []
            # chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
            for data in features:
            #     text = re.sub(chars_to_ignore_regex, '', data[2]).lower() + " "
                inputs = self.processor(data[0].numpy().flatten(), sampling_rate=data[1]).input_values[0]
                with self.processor.as_target_processor():
                    labels = self.processor(data[2]).input_ids
                input_features.append({'input_values': inputs})
                label_features.append({'input_ids': labels})
        else:
            # split inputs and labels since they have to be of different lenghts and need
            # different padding methods
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

# train
def train_model(model, processor, tokenizer, dataloaders_dict, optimizer, scheduler, metric, num_epochs, log_interval=10, report_wandb=False, val_interval=5):
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
                    model.train()  
                else:
                    if (epoch+1) % val_interval:
                        for _ in range(len(dataloaders_dict[phase])):
                            pbar.update(pbar_update)
                        continue
                    model.eval()
                epoch_loss = 0.0  
                epoch_wer = 0.
                epoch_preds_str=[]; epoch_labels_str=[]

                for step, inputs in enumerate(dataloaders_dict[phase]):
                   
                    minibatch_size = inputs['input_values'].size(0)
                    labels_ids = inputs['labels']
                    inputs = inputs.to(device)

                    if opt_flag:
                        for opt in optimizer:
                            opt.zero_grad()
                    else:
                        optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(**inputs)
                        print(outputs)
                        del inputs
                        loss = outputs.loss.mean(dim=-1)
                        preds_logits = outputs.logits
                        #print(preds_logits)
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
                    
                    
                    pbar.update(pbar_update)
                
                epoch_wer = metric.compute(predictions=epoch_preds_str, references=epoch_labels_str)
                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                
                if phase=='train':
                    if scheduler:
                        if sc_flag:
                            for sc in scheduler:
                                sc.step()
                        else:        
                            scheduler.step()
                    print('Epoch {}/{} | {:^5} |  Loss: {:.4f} WER: {:.4f}'.format(epoch+1,\
                                                               num_epochs, phase, epoch_loss, epoch_wer))
                    if report_wandb:
                        wandb.log({'train/epoch':epoch+1,
                                'train/epoch_loss':epoch_loss,
                                'train/epoch_WER':epoch_wer})
                else:
                    print('Epoch {}/{} | {:^5} |  Loss: {:.4f} WER: {:.4f}'.format(epoch+1,\
                                                                 num_epochs, phase, epoch_loss, epoch_wer))
                    if report_wandb:
                        wandb.log({'val/epoch':epoch+1,
                                'val/epoch_loss':epoch_loss,
                                'val/epoch_WER':epoch_wer})
    
    
    return model

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
