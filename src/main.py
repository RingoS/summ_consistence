# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

'''
This script is modified based on transformers/examples/run_language_modeling.py
'''

import torch
import torch.nn as nn
import random, os, numpy as np
import argparse
import csv
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import copy, time, pickle
import numpy as np

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, NewType, Tuple, Callable, NamedTuple
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoModelForTokenClassification,
)

from transformers import DataCollator, PreTrainedModel, EvalPrediction, DefaultDataCollator, EncoderDecoderConfig
from tqdm.auto import tqdm, trange

from models import BertForTextEditing, EncoderDecoderInsertionModel, BertForTokenClassification_modified
from trainer_for_factual_editing import TrainerForFactualEditing, torch_distributed_zero_first

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# label ids for sequence labeling (to label tokens with factual inconsistencies)
label_map = {0: 'O', 1: 'B', 2: 'I'}
sequence_labeling_label_ids = [0, 1, 2]  # [O, B, I]
pad_token_label_id = -100
sequence_a_segment_id = 0
START_OF_ENTITY_index = 30522
END_OF_ENTITY_index = 30523



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    test_data_file: Optional[str] = field(
        default=None,
        metadata={"help": ""},
    )

@dataclass
class AdditionalArguments:
    '''
    arguments added by Sen Yang.
    '''
    share_bert_param: bool = field(
        default=False, metadata={"help": "Whether the two BERTs (for classification/generation, respectively) share parameters"}
    )
    classify_or_insertion: str = field(
        default='classify', metadata={"help": "to do sequence labeling or seq2seq insertion"}
    )
    max_inference_len: int = field(
        default=12, metadata={"help": "the max length of a seq2seq-based decoded entity (noted that this is bpe-based)"}
    )
    do_inference: bool = field(
        default=False, metadata={"help": "whether do seq2seq-based decoding inference"}
    )

def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, description='train', local_rank=-1):
    if description is 'train':
        file_path = args.train_data_file
    elif description is 'eval':
        file_path = args.eval_data_file
    elif description is 'detect':
        # for sequence labeling classification
        file_path = args.test_data_file
    elif description is 'edit':
        # for seq2seq insertion decoding
        file_path = args.test_data_file
    return CsvDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, local_rank=local_rank, description=description
        )

class CsvDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, local_rank=-1, overwrite_cache=False, description=None):
        assert description in ['train', 'eval', 'detect', 'edit']
        if description in ['train', 'eval', 'edit']:
            assert os.path.isfile(file_path)
        else:
            assert os.path.isdir(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        self.tokenizer = tokenizer
        self.description = description

        if description in ['train', 'eval']:
            if 'only_mask' in file_path: 
                block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

            # pad_to_max_length = True
            # pad_to_max_length_file_prefix = '_PaddedToMaxLen'
            # if 'train' in file_path:
            #     pad_to_max_length = False
            #     pad_to_max_length_file_prefix = ''
            pad_to_max_length = False
            pad_to_max_length_file_prefix = ''


            directory, filename = os.path.split(file_path)
            cached_features_file = os.path.join(
                directory, "cached_input_feature_{}_{}{}_{}".format(tokenizer.__class__.__name__, str(block_size), pad_to_max_length_file_prefix, filename.replace('csv', 'pkl'),),
            )

            with torch_distributed_zero_first(local_rank):
                # Make sure only the first process in distributed training processes the dataset,
                # and the others will use the cache.

                # load only_mask data file
                if 'only_mask' in filename:
                    if os.path.exists(cached_features_file) and not overwrite_cache:
                        start = time.time()
                        with open(cached_features_file, "rb") as handle:
                            self.examples, self.labels, self.gold_ids, self.entity_positions = pickle.load(handle)
                        logger.info(
                            f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                        )

                    else:
                        logger.info("Creating features from dataset file at %s", file_path)

                        csv_data = self.read_csv(file_path)
                        
                        logger.info('Finishing reading csv file.')

                        # lines = [' '.join(_['text']) for _ in csv_data]  # for un-separated sentences
                        lines = [(' '.join(_['text1']), ' '.join(_['text2'])) for _ in csv_data]
                        original_masked_ids = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]
                        logger.info('Finishing encoding original masked tokens.')

                        # replace [MASK] tokens with original gold tokens
                        gold_lines = []
                        entity_ids = []
                        consecutive_entity_positions = []
                        for _ in csv_data:
                            tmp_line = _['text2']
                            # sep_position = _['text'].index('[SEP]') + 1
                            # sep_position = _['text'].index('[SEP]')
                            _['entity'].sort(key=lambda x:x['position'][0])
                            entity_ids.append([tokenizer.encode(' '.join(__['text']), add_special_tokens=False) for __ in _['entity']])
                            consecutive_entity_positions.append([])
                            for j, __ in enumerate(_['entity']):
                                # for later processing
                                if (j != len(_['entity']) - 1) and (_['entity'][j]['position'][1] == _['entity'][j+1]['position'][0]):
                                    consecutive_entity_positions[-1].append(j+1)

                                del tmp_line[__['position'][0]: __['position'][1]]
                                for i in range(__['position'][0], __['position'][1]): 
                                    tmp_line.insert(i, __['text'][i-__['position'][0]])
                            gold_lines.append((' '.join(_['text1']), ' '.join(tmp_line)))
                        logger.info('Finishing preprocessing gold_lines.')

                        batch_encoding = tokenizer.batch_encode_plus(gold_lines, add_special_tokens=True, max_length=block_size)
                        logger.info('Finishing encoding gold_lines.')

                        self.gold_ids = batch_encoding["input_ids"]
                        self.examples = copy.deepcopy(self.gold_ids)
                        self.labels = copy.deepcopy(self.examples)
                        for i in range(len(self.labels)):
                            for j in range(len(self.labels[i])):
                                self.labels[i][j] = -100
                        logger.info('Finishing preprocessing labels.')

                        # func = lambda x, y, z : self.final_processing(x, y, z)
                        entity_positions = []
                        for i, _ in enumerate(self.examples):
                            tmp_count = 0
                            tmp_intervals = [0]
                            sep_position = _.index(self.tokenizer.sep_token_id)
                            
                            # try: 
                            #     sep_position = _.index(tokenizer.sep_token_id)
                            # except ValueError:
                            #     print(i)
                            #     print(tokenizer.convert_ids_to_tokens(_))
                            #     raise KeyboardInterrupt
                            j = sep_position + 1

                            mask_signal = False
                            while j < len(original_masked_ids[i]): 
                                if not mask_signal:
                                    if original_masked_ids[i][j] != tokenizer.mask_token_id:
                                        tmp_intervals[-1] += 1
                                    else:
                                        mask_signal = True
                                        # tmp_intervals.append(0)
                                        # logger.info('zzzz{}'.format(j))
                                else:
                                    if original_masked_ids[i][j] != tokenizer.mask_token_id:
                                        mask_signal = False
                                        tmp_intervals.append(1)
                                    # else:
                                    #     continue
                                # if j == len(original_masked_ids[i]) - 1 and original_masked_ids[i][j] != tokenizer.mask_token_id::
                                #     del tmp_intervals[-1]
                                j+=1
                            del tmp_intervals[-1]

                            for __ in consecutive_entity_positions[i]:
                                tmp_intervals.insert(__, 0)

                            assert len(entity_ids[i]) == len(tmp_intervals)

                            j = sep_position + 1
                            curr_position = j
                            entity_positions.append([])
                            for k in range(len(tmp_intervals)):
                                # tmp_entity_position: [start_index, end_index]
                                tmp_entity_position = [curr_position +tmp_intervals[k], curr_position +tmp_intervals[k] + len(entity_ids[i][k])]
                                entity_positions[-1].append(tmp_entity_position)
                                curr_position = tmp_entity_position[-1]
                                

                            # try:
                            #     assert len(entity_ids[i]) == len(tmp_intervals)
                            # except AssertionError:
                            #     logger.info(len(entity_ids[i]))
                            #     logger.info(len(tmp_intervals))
                            #     logger.info(_)
                            #     logger.info(original_masked_ids[i])
                            #     logger.info(entity_ids[i])
                            #     logger.info(tmp_intervals)
                            #     logger.info(csv_data[i]['entity'])
                            #     logger.info(gold_lines[i])
                            #     logger.info(lines[i])
                            #     raise KeyboardInterrupt
                            j = sep_position + 1
                            for tmp_count in range(len(entity_ids[i])):
                                j = j + tmp_intervals[tmp_count]
                                for k in range(len(entity_ids[i][tmp_count])): 
                                    _[j+k] = self.tokenizer.mask_token_id
                                    # try: 
                                    #     _[j+k] = self.tokenizer.mask_token_id
                                    # except IndexError:
                                    #     logger.info(k)
                                    #     logger.info(j+k)
                                    #     logger.info(entity_ids[i][tmp_count])
                                    #     logger.info(_)
                                    #     logger.info(entity_ids[i])
                                    #     logger.info(csv_data[i]['entity'])
                                    #     logger.info(gold_lines[i])
                                    #     raise KeyboardInterrupt
                                    self.labels[i][j+k] = entity_ids[i][tmp_count][k]
                                j = j + len(entity_ids[i][tmp_count])
                                # tmp_count += 1
                            assert len(self.examples[i]) == len(self.labels[i])
                        
                        logger.info('#'*100)
                        logger.info(tokenizer.convert_ids_to_tokens(self.examples[10]))
                        logger.info('#'*40)
                        logger.info(tokenizer.convert_ids_to_tokens(self.labels[10]))

                        self.entity_positions = entity_positions

                        # input_data include: self.examples, self.labels
                        start = time.time()
                        with open(cached_features_file, "wb") as handle:
                            pickle.dump((self.examples, self.labels, self.gold_ids, self.entity_positions), handle, protocol=pickle.HIGHEST_PROTOCOL)
                        logger.info(
                            f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                        )
                
                # for pseudo_data file
                if os.path.exists(cached_features_file) and not overwrite_cache:
                    start = time.time()
                    with open(cached_features_file, "rb") as handle:
                        self.classify_data, self.insertion_data = pickle.load(handle)
                    logger.info(
                        f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                    )
                    
                    ###############   debug only   ############################
                    if description == 'eval':
                        threshold = 100
                        for _ in self.classify_data.keys():
                            self.classify_data[_] = self.classify_data[_][:threshold]
                        for _ in self.insertion_data.keys():
                            self.insertion_data[_] = self.insertion_data[_][:threshold]
                    ############################################################
                
                else:
                    logger.info("Creating features from dataset file at %s", file_path)
                    csv_data = self.read_csv(file_path)
                    # header: text, gold_entity, statistic
                    logger.info('Finishing reading csv file.')


                    # 1. preprocessing for sequence labeling (to label tokens with factual errors)

                    logger.info('Start to process sequence labeling data. ')
                    self.classify_data = {
                        'input_ids': [],
                        'label_ids': [],
                        'attention_mask': [],
                        'token_type_ids': [],
                    }
                    classify_tokens = []
                    for i, _ in enumerate(csv_data): 
                        temp_words, temp_word_label_ids = [], []
                        temp_words.append(self.tokenizer.cls_token)
                        temp_word_label_ids.append(0)
                        for j, __ in enumerate(_['text']):
                            if j == 0:
                                temp_words += __
                                temp_words.append(self.tokenizer.sep_token)
                                temp_word_label_ids += [0 for k in range(len(__))]
                                temp_word_label_ids.append(0)
                            else:
                                for word in __: 
                                    if type(word) == str:
                                        temp_words.append(word)
                                        temp_word_label_ids.append(0)
                                    elif type(word) == list:
                                        for k, sub_word in enumerate(word):
                                            temp_words.append(sub_word)
                                            if k == 0:
                                                # append label "B"
                                                temp_word_label_ids.append(1)
                                            else:
                                                # append label "I"
                                                temp_word_label_ids.append(2)
                                    else:
                                        raise TypeError("The word {} of type {} is neither str nor list. ".format(word, type(word)))
                        temp_words.append(self.tokenizer.sep_token)
                        temp_word_label_ids.append(0)
                        assert len(temp_words) == len(temp_word_label_ids)

                        # bpe tokenize & processing labels
                        temp_tokens, temp_token_label_ids = [], []
                        for word, label in zip(temp_words, temp_word_label_ids):
                            word_tokens = self.tokenizer.tokenize(word)
                            if len(word_tokens) > 0:
                                temp_tokens.extend(word_tokens)
                                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                                temp_token_label_ids.extend([label] + [pad_token_label_id] * (len(word_tokens) - 1))
                        assert len(temp_tokens) == len(temp_token_label_ids)

                        # convert tokens to vocab-ids
                        # self.classify_data['input_ids'].append(
                        #     self.tokenizer.convert_tokens_to_ids(temp_tokens)
                        # )
                        classify_tokens.append(temp_tokens)
                        if pad_to_max_length:
                            padding_length = block_size - len(temp_tokens)
                            temp_token_label_ids += [pad_token_label_id] * padding_length
                            assert len(temp_token_label_ids) == block_size, 'Current len(temp_tokens)={}, len(temp_token_label_ids)={}'.format(len(temp_tokens), len(temp_token_label_ids))
                        self.classify_data['label_ids'].append(torch.tensor(temp_token_label_ids).type_as(torch.LongTensor()))
                    self.classify_data['input_ids'] = self.tokenizer.batch_encode_plus(
                        classify_tokens,
                        add_special_tokens=False,
                        max_length=block_size,
                        is_pretokenized=True,
                        pad_to_max_length=pad_to_max_length,
                    )['input_ids']
                    logger.info('Finished processing sequence labeling data. ')


                    
                    # 2. preprocessing for seq2seq insertion deocding
                    
                    logger.info('Start to process seq2seq insertion data. ')
                    self.insertion_data = {
                        'input_ids': [],
                        'tgt_ids': [],
                        'tgt_label_ids': [],
                        'entity_positions': []
                    }
                    '''
                        'input_ids': one entity is replaced by exactly one '[soe]' token.
                        'tgt_ids': gold target sequence, with '[soe]' token inserted to the start position of each entity
                        'tgt_mask': seq2seq decoder mask for insertion transformer

                        examples:
                            tokens of input_ids: 
                                ['[CLS]', '-LRB-', 'CNN', '-RRB-', 'Former', 'Vice', 'President', 'Walter', 'Mondale', 'was', 'released', 
                                'from', 'the', 'Mayo', 'Clinic', 'on', 'Saturday', 'after', 'being', 'admitted', 'with', 'influenza', ',', 
                                'hospital', 'spokeswoman', 'Kelley', 'Luckstein', 'said', '.', '[SEP]', '[soe]', 'was', 
                                'released', 'from', '[soe]', 'on', 'Saturday', ',', 'hospital', 'spokeswoman', 'said', '.', '[SEP]']
                            tokens of tgt_ids: 
                                ['[CLS]', '-LRB-', 'CNN', '-RRB-', 'Former', 'Vice', 'President', 'Walter', 'Mondale', 'was', 'released', 
                                'from', 'the', 'Mayo', 'Clinic', 'on', 'Saturday', 'after', 'being', 'admitted', 'with', 'influenza', ',', 
                                'hospital', 'spokeswoman', 'Kelley', 'Luckstein', 'said', '.', '[SEP]', 'Walter', 'Mondale',, '[eoe]' 'was', 
                                'released', 'from', 'the', 'Mayo', 'Clinic', '[eoe]', 'on', 'Saturday', ',', 'hospital', 'spokeswoman', 'said', '.', '[SEP]']
                    '''
                    insertion_tokens, tgt_tokens, tgt_label_tokens_ids = [], [], []
                    for i, _ in enumerate(csv_data): 
                        temp_token_input, temp_token_tgt, temp_token_tgt_label_ids, temp_entity_position = [], [], [], []
                        temp_token_input.append(self.tokenizer.cls_token)
                        temp_token_tgt.append(self.tokenizer.cls_token)
                        count_entity = 0
                        for j, __ in enumerate(_['text']):
                            assert j in [0, 1]
                            if j == 0:
                                for word in __:
                                    temp_tokens = self.tokenizer.tokenize(word)
                                    temp_token_input += temp_tokens
                                    temp_token_tgt += temp_tokens
                                temp_token_input.append(self.tokenizer.sep_token)
                                temp_token_tgt.append(self.tokenizer.sep_token)
                                temp_token_tgt_label_ids += [-100 for kk in range(len(temp_token_tgt))]
                            else:
                                for word in __: 
                                    if type(word) == str:
                                        temp_tokens = self.tokenizer.tokenize(word)
                                        temp_token_input += temp_tokens
                                        temp_token_tgt += temp_tokens
                                        temp_token_tgt_label_ids += [-100 for kk in range(len(temp_tokens))]
                                    elif type(word) == list:
                                        temp_token_input.append('[soe]')
                                        temp_entity_position.append([len(temp_token_tgt)])
                                        temp_token_tgt.append('[soe]')
                                        for entity_word in _['gold_entity'][count_entity]:
                                            temp_tokens = self.tokenizer.tokenize(entity_word)
                                            temp_token_tgt += temp_tokens
                                            temp_token_tgt_label_ids += self.tokenizer.convert_tokens_to_ids(temp_tokens)
                                        temp_token_tgt_label_ids += self.tokenizer.convert_tokens_to_ids(['[eoe]'])
                                        count_entity += 1
                                        temp_entity_position[-1].append(len(temp_token_tgt))
                                    else:
                                        raise TypeError("The word {} of type {} is neither str nor list. ".format(word, type(word)))
                        temp_token_input.append(self.tokenizer.sep_token)
                        temp_token_tgt.append(self.tokenizer.sep_token)
                        temp_token_tgt_label_ids.append(-100)
                        assert len(temp_token_tgt) == len(temp_token_tgt_label_ids)
                        if pad_to_max_length:
                            padding_length = block_size - len(temp_token_tgt)
                            temp_token_tgt_label_ids += [-100] * padding_length
                            assert len(temp_token_tgt_label_ids) == block_size
                        # convert tokens to ids
                        # self.insertion_data['input_ids'].append(self.tokenizer.convert_tokens_to_ids(temp_token_input))
                        # self.insertion_data['tgt_ids'].append(self.tokenizer.convert_tokens_to_ids(temp_token_tgt))
                        # self.insertion_data['tgt_label_ids'].append(self.tokenizer.convert_tokens_to_ids(temp_token_tgt_label_ids))
                        insertion_tokens.append(temp_token_input)
                        tgt_tokens.append(temp_token_tgt)
                        tgt_label_tokens_ids.append(temp_token_tgt_label_ids)
                        self.insertion_data['entity_positions'].append(temp_entity_position)
                    self.insertion_data['input_ids'] = self.tokenizer.batch_encode_plus(
                        insertion_tokens,
                        add_special_tokens=False,
                        max_length=block_size,
                        is_pretokenized=True,
                        pad_to_max_length=pad_to_max_length,
                    )['input_ids']
                    self.insertion_data['tgt_ids'] = self.tokenizer.batch_encode_plus(
                        tgt_tokens,
                        add_special_tokens=False,
                        max_length=block_size,
                        is_pretokenized=True,
                        pad_to_max_length=pad_to_max_length,
                    )['input_ids']
                    self.insertion_data['tgt_label_ids'] = tgt_label_tokens_ids
                    # self.insertion_data['tgt_label_ids'] = self.tokenizer.batch_encode_plus(
                    #     tgt_label_tokens,
                    #     add_special_tokens=False,
                    #     max_length=block_size,
                    #     is_pretokenized=True,
                    #     pad_to_max_length=pad_to_max_length,
                    # )['input_ids']
                    logger.info('Finished processing seq2seq insertion data. ')

                    # input_data include: self.classify_data, self.insertion_data
                    start = time.time()
                    with open(cached_features_file, "wb") as handle:
                        pickle.dump([self.classify_data, self.insertion_data], handle, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info(
                        f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                    )
        elif description is 'detect': 
            # block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

            # pad_to_max_length = True
            # pad_to_max_length_file_prefix = '_PaddedToMaxLen'
            # if 'train' in file_path:
            #     pad_to_max_length = False
            #     pad_to_max_length_file_prefix = ''
            pad_to_max_length = False
            pad_to_max_length_file_prefix = ''

            dir_name = file_path.split('/')[-1] if file_path.split('/')[-1] != '' else file_path.split('/')[-2]
            cached_features_file = os.path.join(
                file_path, "cached_input_feature_for_detection_{}_{}{}_{}".format(tokenizer.__class__.__name__, str(block_size), pad_to_max_length_file_prefix, dir_name+'.pkl',),
            )

            with torch_distributed_zero_first(local_rank):
                # Make sure only the first process in distributed training processes the dataset,
                # and the others will use the cache.

                # load data file
                if os.path.exists(cached_features_file) and not overwrite_cache:
                    start = time.time()
                    with open(cached_features_file, "rb") as handle:
                        self.examples = pickle.load(handle)
                    logger.info(
                        f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                    )

                else:
                    logger.info("Creating features from dataset file at %s", file_path)
                    articles = self.read_tsv(os.path.join(file_path, 'articles.tsv'))
                    summaries = self.read_tsv(os.path.join(file_path, 'summaries.tsv'))
                    highlight_positions = self.read_tsv(os.path.join(file_path, 'highlight.tsv'))
                    logger.info('Finishing reading tsv files.')

                    self.examples = dict()

                    inputs = []

                    for i, _ in enumerate(highlight_positions):
                        for j, __ in enumerate(_):
                            __ = __.strip()
                            if __ == '':
                                continue
                            temp_sent_1 = ''
                            temp_sent_2 = summaries[i][j]
                            for temp_position in __.split(','):
                                temp_sent_1 += articles[i][int(temp_position)]
                                # temp_position = temp_position.strip()
                                # if temp_position != '':
                                #     temp_sent_1 += articles[i][int(temp_position)]
                            temp_input = self.tokenizer.cls_token + temp_sent_1 + self.tokenizer.sep_token + temp_sent_2 +self.tokenizer.sep_token
                            inputs.append(temp_input)
                    self.examples['input_ids'] = self.tokenizer.batch_encode_plus(
                        inputs, 
                        add_special_tokens=False,
                        max_length=block_size,
                    )["input_ids"]

                    # input_data include: self.examples
                    start = time.time()
                    with open(cached_features_file, "wb") as handle:
                        pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info(
                        f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                    )
        elif description == 'edit':

            pad_to_max_length = False
            pad_to_max_length_file_prefix = ''

            directory, filename = os.path.split(file_path)
            filename = filename.replace('for_detection_', '')
            cached_features_file = os.path.join(
                directory, "cached_input_feature_for_editing_{}_{}{}_{}".format(tokenizer.__class__.__name__, str(block_size), pad_to_max_length_file_prefix, filename+'.pkl',),
            )

            with torch_distributed_zero_first(local_rank):
                # Make sure only the first process in distributed training processes the dataset,
                # and the others will use the cache.

                # load data file
                if os.path.exists(cached_features_file) and not overwrite_cache:
                    start = time.time()
                    with open(cached_features_file, "rb") as handle:
                        self.examples = pickle.load(handle)
                    logger.info(
                        f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                    )

                else:
                    logger.info("Creating features from dataset file at %s", file_path)
                    csv_data = self.read_csv(file_path, with_header=False)
                    logger.info('Finishing reading csv files.')

                    self.examples = dict()
                    inputs = []

                    for _ in csv_data:
                        tmp_input = []
                        for __ in _[1].split():
                            if __[:2] == '##':
                                continue
                            tmp_input.append(__)
                        inputs.append(' '.join(tmp_input))

                    self.examples['input_ids'] = self.tokenizer.batch_encode_plus(
                        inputs, 
                        add_special_tokens=False,
                        max_length=block_size,
                    )["input_ids"]

                    # input_data include: self.examples
                    start = time.time()
                    with open(cached_features_file, "wb") as handle:
                        pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info(
                        f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                    )


    def __len__(self):
        # return len(self.examples)
        if self.description in ['train', 'eval']:
            return len(self.classify_data['input_ids'])
        elif self.description == 'detect':
            return len(self.examples["input_ids"])
        elif self.description == 'edit':
            return len(self.examples["input_ids"])

    def __getitem__(self, i) -> torch.Tensor:
        # return {
        #         'input_ids': torch.tensor(self.examples[i], dtype=torch.long), 
        #         'mlm_label_ids': torch.tensor(self.labels[i], dtype=torch.long),
        #         'tgt_ids': torch.tensor(self.gold_ids[i], dtype=torch.long),
        #         'entity_positions': self.gold_ids[i],
        #         }
        if self.description in ['train', 'eval']:
            return {
                    'input_ids_classify': torch.tensor(self.classify_data['input_ids'][i], dtype=torch.long), 
                    'input_ids_insertion': torch.tensor(self.insertion_data['input_ids'][i], dtype=torch.long),
                    'label_ids_classify': torch.tensor(self.classify_data['label_ids'][i], dtype=torch.long),
                    'tgt_ids': torch.tensor(self.insertion_data['tgt_ids'][i], dtype=torch.long),
                    'tgt_label_ids': torch.tensor(self.insertion_data['tgt_label_ids'][i], dtype=torch.long),
                    'entity_positions': self.insertion_data['entity_positions'][i]
                    }
        elif self.description == 'detect':
            return {
                    "input_ids": torch.tensor(self.examples['input_ids'][i], dtype=torch.long)
                    }
        elif self.description is 'edit':
            return {
                    "input_ids": torch.tensor(self.examples['input_ids'][i], dtype=torch.long)
                    }
    
    def read_csv(self, file_path, with_header=True): 
        if with_header:
            '''
            return:
                a list of dict, the keys of dict are headers of the csv file
            '''
            header, rst = [], []
            with open(file_path, 'r') as f: 
                f_csv = csv.reader(f)
                for i, _ in enumerate(f_csv): 
                    continue_signal = False
                    if i == 0:
                        header = _
                        continue
                    tmp_dict = dict()
                    for j, __ in enumerate(header):
                        tmp_dict[__] = eval(_[j])
                        # ['entity' in header] means only_mask.csv
                        if 'entity' in header:
                            # classifying samples that have multi-hop entities
                            if __ == 'entity':
                                intervals = []
                                for entity in tmp_dict[__]: 
                                    entity['text'] = ' '.join(entity['text']).replace('\xa0', '[UNK]').split()
                                    for interval in intervals:
                                        if (entity['position'][0] >= interval[0] and entity['position'][0] <= interval[1]) \
                                            or (entity['position'][1] >= interval[0] and entity['position'][1] <= interval[1]):
                                            continue_signal = True
                                            break
                                    if continue_signal:
                                        break
                                    intervals.append([entity['position'][0], entity['position'][1]])
                            if continue_signal:
                                break
                            if __ == 'text': 
                                # tmp_dict[__].append('[SEP]')
                                tmp_dict[__] = ' '.join(tmp_dict[__]).replace('\xa0', '[UNK]').split()
                                sep_position = tmp_dict[__].index('[SEP]')
                                tmp_dict['text1'] = tmp_dict[__][:sep_position]
                                tmp_dict['text1'].remove('[CLS]')
                                tmp_dict['text2'] = tmp_dict[__][sep_position:]
                                tmp_dict['text2'].remove('[SEP]')
                            if continue_signal:
                                continue
                    rst.append(tmp_dict)
            return rst
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                result = list(csv_reader)
            return result

    def read_tsv(self, file_path):
        result = []
        with open(file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            result = list(csv_reader)
        return result

@dataclass
class DataCollatorForFactualEditing(DataCollator):
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15
    description: str = 'classify'  # 'classify' or 'insertion'

    def collate_batch(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        # if key "input_ids" exists, it refers to non-training mode, which means inference mode
        if examples[0].get('input_ids') != None:
            input_ids = [_['input_ids'] for _ in examples]
            inputs = self._tensorize_batch(input_list=[input_ids])[0]
            if self.description == 'classify':
                return {
                    "input_ids": inputs
                    }
            elif self.description == 'insertion':
                return {
                    "input_ids": inputs,
                    "decoder_input_ids": inputs
                    }
        # input_ids = [_['input_ids'] for _ in examples]
        # mlm_label_ids = [_['mlm_label_ids'] for _ in examples]
        # tgt_ids = [_['tgt_ids'] for _ in examples]
        # entity_positions = [_['entity_positions'] for _ in examples]

        else:
            input_ids_classify = [_['input_ids_classify'] for _ in examples]
            input_ids_insertion = [_['input_ids_insertion'] for _ in examples]
            label_ids_classify = [_['label_ids_classify'] for _ in examples]
            tgt_ids = [_['tgt_ids'] for _ in examples]
            tgt_label_ids = [_['tgt_label_ids'] for _ in examples]
            entity_positions = [_['entity_positions'] for _ in examples]
            
            # inputs, labels, tgt = self._tensorize_batch(input_ids, mlm_label_ids, tgt_ids)
            inputs_classify, inputs_insertion, labels_classify, tgt, tgt_labels = \
                self._tensorize_batch(input_list=[input_ids_classify, input_ids_insertion, label_ids_classify, tgt_ids, tgt_label_ids])
            tgt_mask = self.make_decoder_attention_mask(tgt, entity_positions)
            # batch = self._tensorize_batch(examples)
            
            # # for masked-lm
            # return {"input_ids": inputs, "masked_lm_labels": labels}
            
            # for classify
            if self.description == 'classify':
                return {
                        'input_ids': inputs_classify, 
                        'labels': labels_classify,
                        }
            # for seq2seq insertion
            elif self.description == 'insertion':
                return {
                        'input_ids': inputs_insertion, 
                        'decoder_input_ids': tgt,
                        'masked_lm_labels': tgt_labels, 
                        'decoder_attention_mask': tgt_mask
                        }
            else:
                raise ValueError('The description for DataCollator, {}, is neither \'classify\' nor \'insertion\''.format(self.description))

    def TakeSpanLength(self, x):
        return x[1]-x[0]
    def make_decoder_attention_mask(self, tgt, entity_positions):
        '''
        to make tgt masks for seq2seq-based insertion decoding
        '''
        # tgt = [batch_size, tgt_len]
        # print('#'*80, type(tgt), '\n', type(tgt[0]), '#'*80)
        tgt_pad_mask = (tgt != self.tokenizer.pad_token_id).type_as(torch.LongTensor()).unsqueeze(1)
        # tgt_pad_mask = [batch size, 1, tgt_len]
        batch_size = tgt.shape[0]
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.ones((batch_size, tgt_len, tgt_len))
        for i in range(batch_size):
            entity_positions[i].sort(key=self.TakeSpanLength)
        for i in range(batch_size):
            for _ in entity_positions[i]:
                tgt_sub_mask[i, :, _[0]:_[1]] = 0
            for j, _ in enumerate(entity_positions[i]):
                tmp_entity_len = _[1] - _[0]
                tgt_sub_mask[i, _[0]:_[1], _[0]:_[1]] = torch.tril(torch.ones((tmp_entity_len, tmp_entity_len)))
                k = j + 1
                while k < len(entity_positions[i]):
                    tgt_sub_mask[i, _[0]:_[1], entity_positions[i][k][0]:entity_positions[i][k][0]+tmp_entity_len] \
                        = torch.tril(torch.ones((tmp_entity_len, tmp_entity_len)))
                    tgt_sub_mask[i, entity_positions[i][k][0]:entity_positions[i][k][0]+tmp_entity_len, _[0]:_[1]] \
                        = torch.tril(torch.ones((tmp_entity_len, tmp_entity_len)))
                    tgt_sub_mask[i, entity_positions[i][k][0]+tmp_entity_len: entity_positions[i][k][1], _[0]:_[1]] = 1
                    k += 1
        # tgt_sub_mask: [batch_size, tgt_len, tgt_len]
        tgt_sub_mask = tgt_sub_mask.type_as(torch.LongTensor())
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

    # input_ids_classify, input_ids_insertion, label_ids_classify, tgt_ids, tgt_label_ids
    def _tensorize_batch(
                    self, 
                    input_list = None
                    ) -> torch.Tensor:
        are_tensors_same_length = True
        for i, _ in enumerate(input_list):
            length_of_first = input_list[i][0].size(0)
            are_tensors_same_length = all(x.size(0) == length_of_first for x in input_list[0])
            if not are_tensors_same_length:
                break
        if are_tensors_same_length:
            return tuple([torch.stack(_, dim=0) for _ in input_list])
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return tuple([pad_sequence(_, batch_first=True, padding_value=self.tokenizer.pad_token_id) for _ in input_list])

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
                # out_label_list[i].append(label_ids[i][j])
                # preds_list[i].append(preds[i][j])

    return preds_list, out_label_list

def compute_metrics(args: TrainingArguments, p: EvalPrediction) -> Dict:
    if args.classify_or_insertion == 'classify':
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        return {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
            'accuracy': accuracy_score(out_label_list, preds_list),
        }
    elif args.classify_or_insertion == 'insertion':
        # here, p.predictions have been processed by argmax.index  
        # preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        return {
            'accuracy': mlm_acc_score(p.predictions, p.label_ids),
        }

def mlm_acc_score(y_pred: np.ndarray, y_true: np.ndarray):
    '''
    y_pred: shape(batch_size, seq_len)
    y_true: the same as above
    '''
    count, correct = 0, 0
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    assert y_pred.shape == y_true.shape, 'y_pred.shape={}, y_true.shape={}'.format(y_pred.shape, y_true.shape)
    for i, _ in enumerate(y_pred):
        if y_true[i] != -100:
            count += 1
            if y_true[i] == _:
                correct += 1
    return correct/count

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalArguments))
    model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()

    training_args.max_inference_len = additional_args.max_inference_len
    training_args.do_inference = additional_args.do_inference

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if (not additional_args.do_inference) or (additional_args.classify_or_insertion == 'classify'):
        if model_args.config_name:
            config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
        elif model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
        else:
            config = CONFIG_MAPPING[model_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
        config_encoder = config
        config_decoder = config

    # do_inference and is_insertion
    elif additional_args.classify_or_insertion == 'insertion':
        if model_args.config_name:
            config = EncoderDecoderConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
        elif model_args.model_name_or_path:
            logger.info("###########################\nLoading predefined EncoderDecoderConfig\n##############################")
            config = EncoderDecoderConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
        else:
            config = CONFIG_MAPPING[model_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
        config_encoder = config.encoder
        config_decoder = config.decoder

    config.share_bert_param = additional_args.share_bert_param
    config.classify_or_insertion = additional_args.classify_or_insertion
    training_args.classify_or_insertion = additional_args.classify_or_insertion

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    # if model_args.model_name_or_path:
    #     model = BertForTextEditing.from_pretrained(
    #         model_args.model_name_or_path,
    #         from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #         config=config,
    #         cache_dir=model_args.cache_dir,
    #     )
    # else:
    #     logger.info("Training new model from scratch")
    #     model = BertForTextEditing.from_config(config)

    assert additional_args.classify_or_insertion in ['classify', 'insertion'], \
        'additional_args.classify_or_insertion should not be {}. It should be either \'classify\' or \'insertion\'. '.format(additional_args.classify_or_insertion)

    if model_args.model_name_or_path:
        if additional_args.classify_or_insertion == 'classify':
            config.num_labels = 3
            model = BertForTokenClassification_modified.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                )
        elif additional_args.classify_or_insertion == 'insertion':
            if not additional_args.do_inference:
                # load original bert-base-uncased
                model = EncoderDecoderInsertionModel.from_encoder_decoder_pretrained(
                    model_args.model_name_or_path, 
                    model_args.model_name_or_path, 
                    encoder_from_tf=bool(".ckpt" in model_args.model_name_or_path), 
                    decoder_from_tf=bool(".ckpt" in model_args.model_name_or_path), 
                )
            else:
                encoder_decoder_state_dict = torch.load(os.path.join(model_args.model_name_or_path, "pytorch_model.bin"))
                encoder_state_dict, decoder_state_dict = dict(), dict()
                for _ in encoder_decoder_state_dict.keys():
                    if _[:7] == 'encoder':
                        tmp_key = _[8:]
                        encoder_state_dict[tmp_key] = encoder_decoder_state_dict[_]
                    elif _[:7] == 'decoder':
                        tmp_key = _[8:]
                        decoder_state_dict[tmp_key] = encoder_decoder_state_dict[_]
                    else:
                        raise KeyError("key \"{}\" doesn't match any model config".format(_))
                    
                # load state_dict of my pretrained model for inference
                model = EncoderDecoderInsertionModel.from_encoder_decoder_pretrained(
                    model_args.model_name_or_path, 
                    model_args.model_name_or_path, 
                    encoder_from_tf=bool(".ckpt" in model_args.model_name_or_path), 
                    decoder_from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    encoder_config=config_encoder,
                    decoder_config=config_decoder, 
                    encoder_state_dict=encoder_state_dict,
                    decoder_state_dict=decoder_state_dict,
                    encoder_cache_dir=model_args.cache_dir,
                    decoder_cache_dir=model_args.cache_dir,
                )
    else:
        logger.info("Training new model from scratch")
        if additional_args.classify_or_insertion == 'classify':
            model = AutoModelForTokenClassification.from_config(config)
        elif additional_args.classify_or_insertion == 'insertion':
            model = EncoderDecoderInsertionModel.from_config(config)

    # add new tokens for seq2seq insertion decoding (similar to <s> and <\s> in common left-to-right seq2seq decoding)
    new_tokens = ['[soe]', '[eoe]']  # StartOfEntity, EndOfEntity
    num_added_toks = tokenizer.add_tokens(new_tokens)
    logger.info('We have added {} tokens'.format(num_added_toks))
    if additional_args.classify_or_insertion == 'classify':
        model.resize_token_embeddings(len(tokenizer))
    elif additional_args.classify_or_insertion == 'insertion':
        model.encoder.resize_token_embeddings(len(tokenizer))
        model.decoder.resize_token_embeddings(len(tokenizer))


    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets
    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, description='train')
        if training_args.do_train
        else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, description='eval')
        if training_args.do_eval or training_args.evaluate_during_training
        else None
    )
    data_collator = DataCollatorForFactualEditing(
        tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability, description=additional_args.classify_or_insertion
    )

    # Initialize our Trainer
    trainer = TrainerForFactualEditing(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=False,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    
    # Inference
    if training_args.do_inference and training_args.local_rank in [-1, 0]:
        logger.info("*** Inference ***")
        # sequence labeling detection
        if additional_args.classify_or_insertion == 'classify':
            test_dataset = get_dataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, description='detect')
            inference_output, inference_input = trainer.predict(test_dataset=test_dataset, mode=additional_args.classify_or_insertion)
            # for classify, test_file_name is a dir 
            #   (e.g. "test_chen18_org" for 172.16.71.6: /data/senyang/summ_consistence/previous_work/ACL-19_Ranking/summary-correctness-v1.0/test_chen18_org/)
            test_file_name = data_args.test_data_file.split('/')[-1] if data_args.test_data_file.split('/')[-1] != '' else data_args.test_data_file.split('/')[-2]
            inference_output_file = os.path.join(training_args.output_dir, 'detection_output_'+test_file_name+'.csv')

            with open(inference_output_file, 'w', encoding='utf-8') as f:
                f_csv = csv.writer(f)
                for i, _ in enumerate(inference_output):
                    for j, __ in enumerate(_):
                        __ = __.tolist()
                        temp_output = inference_input[i][j].tolist()
                        temp_del_position = []
                        for k in range(len(inference_input[i][j])):
                            if __[k] == 1:
                                temp_output[k] = START_OF_ENTITY_index
                            elif __[k] == 2:
                                temp_del_position.append(k)
                        temp_del_position.sort(reverse=True)
                        for k in temp_del_position:
                            del temp_output[k]

                        line = [tokenizer.decode(inference_input[i][j]), tokenizer.decode(temp_output)]
                        f_csv.writerow(line)

        # seq2seq insertion editing
        elif additional_args.classify_or_insertion == 'insertion':
            test_dataset = get_dataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, description='edit')
            inference_output, inference_input = trainer.predict(test_dataset=test_dataset, mode=additional_args.classify_or_insertion)
            test_file_name = os.path.split(data_args.test_data_file)[1]
            inference_output_file = os.path.join(training_args.output_dir, 'editing_output_'+test_file_name.split('.')[0]+'.csv')
            with open(inference_output_file, 'w', encoding='utf-8') as f:
                f_csv = csv.writer(f)
                for i, _ in enumerate(inference_output):
                    line = [tokenizer.decode(inference_input[i]), tokenizer.decode(inference_output[i])]
                    f_csv.writerow(line)


    return results


if __name__ == "__main__":
    main()
