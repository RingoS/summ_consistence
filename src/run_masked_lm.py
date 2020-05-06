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
import models
import random, os, numpy as np
import argparse
import csv
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import copy

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, NewType, Tuple

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
)

from transformers import DataCollator

# #*****************************************************************
# '''
# set random seed
# '''
# seed = 42
# random.seed(seed)
# os.environ['PYTHONHASHSEED']  =  str(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
# torch.backends.cudnn.benchmark  =  False
# torch.backends.cudnn.deterministic  =  True
# def _init_fn(worker_id):
#     ''' for dataloader workers init,  freeze dataloader's randomness '''
#     np.random.seed(seed + worker_id)
# #******************************************************************

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


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


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False, local_rank=-1):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    # if args.line_by_line:
    #     return LineByLineTextDataset(
    #         tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, local_rank=local_rank
    #     )
    # else:
    #     return TextDataset(
    #         tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, local_rank=local_rank,
    #     )
    return CsvDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, local_rank=local_rank
        )

class CsvDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, local_rank=-1):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        self.tokenizer = tokenizer
        logger.info("Creating features from dataset file at %s", file_path)

        csv_data = self.read_csv(file_path)

        # with open(file_path, encoding="utf-8") as f:
        #     lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        logger.info('Finishing reading csv file.')

        lines = [' '.join(_['text1']) for _ in csv_data]
        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]

        # self.examples
        # self.labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
    
    def read_csv(self, file_path): 
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

    def collate_batch(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [_['input_ids'] for _ in examples]
        mlm_label_ids = [_['mlm_label_ids'] for _ in examples]
        inputs, labels = self._tensorize_batch(input_ids, mlm_label_ids)
        # batch = self._tensorize_batch(examples)
        # if self.mlm:
        #     inputs, labels = self.mask_tokens(batch)
        #     return {"input_ids": inputs, "masked_lm_labels": labels}
        # else:
        #     return {"input_ids": batch, "labels": batch}
        return {"input_ids": inputs, "masked_lm_labels": labels}

    def _tensorize_batch(self, input_ids: List[torch.Tensor], mlm_label_ids: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = input_ids[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in input_ids)
        if are_tensors_same_length:
            return torch.stack(input_ids, dim=0), torch.stack(mlm_label_ids, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id), \
                    pad_sequence(mlm_label_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

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
        get_dataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank)
        if training_args.do_train
        else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, evaluate=True)
        if training_args.do_eval or training_args.evaluate_during_training
        else None
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
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

    return results


if __name__ == "__main__":
    main()

'''
    python run_language_modeling.py --help
    I0504 16:16:48.721203 139835076368128 file_utils.py:38] PyTorch version 1.4.0 available.
    usage: run_language_modeling.py [-h] [--model_name_or_path MODEL_NAME_OR_PATH]
                                    [--model_type MODEL_TYPE]
                                    [--config_name CONFIG_NAME]
                                    [--tokenizer_name TOKENIZER_NAME]
                                    [--cache_dir CACHE_DIR]
                                    [--train_data_file TRAIN_DATA_FILE]
                                    [--eval_data_file EVAL_DATA_FILE]
                                    [--line_by_line] [--mlm]
                                    [--mlm_probability MLM_PROBABILITY]
                                    [--block_size BLOCK_SIZE] [--overwrite_cache]
                                    --output_dir OUTPUT_DIR
                                    [--overwrite_output_dir] [--do_train]
                                    [--do_eval] [--do_predict]
                                    [--evaluate_during_training]
                                    [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
                                    [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
                                    [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                                    [--learning_rate LEARNING_RATE]
                                    [--weight_decay WEIGHT_DECAY]
                                    [--adam_epsilon ADAM_EPSILON]
                                    [--max_grad_norm MAX_GRAD_NORM]
                                    [--num_train_epochs NUM_TRAIN_EPOCHS]
                                    [--max_steps MAX_STEPS]
                                    [--warmup_steps WARMUP_STEPS]
                                    [--logging_dir LOGGING_DIR]
                                    [--logging_first_step]
                                    [--logging_steps LOGGING_STEPS]
                                    [--save_steps SAVE_STEPS]
                                    [--save_total_limit SAVE_TOTAL_LIMIT]
                                    [--no_cuda] [--seed SEED] [--fp16]
                                    [--fp16_opt_level FP16_OPT_LEVEL]
                                    [--local_rank LOCAL_RANK]

    optional arguments:
    -h, --help            show this help message and exit
    --model_name_or_path MODEL_NAME_OR_PATH
                            The model checkpoint for weights initialization. Leave
                            None if you want to train a model from scratch.
    --model_type MODEL_TYPE
                            If training from scratch, pass a model type from the
                            list: t5, distilbert, albert, camembert, xlm-roberta,
                            bart, roberta, bert, openai-gpt, gpt2, transfo-xl,
                            xlnet, flaubert, xlm, ctrl, electra, encoder_decoder
    --config_name CONFIG_NAME
                            Pretrained config name or path if not the same as
                            model_name
    --tokenizer_name TOKENIZER_NAME
                            Pretrained tokenizer name or path if not the same as
                            model_name
    --cache_dir CACHE_DIR
                            Where do you want to store the pretrained models
                            downloaded from s3
    --train_data_file TRAIN_DATA_FILE
                            The input training data file (a text file).
    --eval_data_file EVAL_DATA_FILE
                            An optional input evaluation data file to evaluate the
                            perplexity on (a text file).
    --line_by_line        Whether distinct lines of text in the dataset are to
                            be handled as distinct sequences.
    --mlm                 Train with masked-language modeling loss instead of
                            language modeling.
    --mlm_probability MLM_PROBABILITY
                            Ratio of tokens to mask for masked language modeling
                            loss
    --block_size BLOCK_SIZE
                            Optional input sequence length after tokenization.The
                            training dataset will be truncated in block of this
                            size for training.Default to the model max input
                            length for single sentence inputs (take into account
                            special tokens).
    --overwrite_cache     Overwrite the cached training and evaluation sets
    --output_dir OUTPUT_DIR
                            The output directory where the model predictions and
                            checkpoints will be written.
    --overwrite_output_dir
                            Overwrite the content of the output directory.Use this
                            to continue training if output_dir points to a
                            checkpoint directory.
    --do_train            Whether to run training.
    --do_eval             Whether to run eval on the dev set.
    --do_predict          Whether to run predictions on the test set.
    --evaluate_during_training
                            Run evaluation during training at each logging step.
    --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                            Batch size per GPU/CPU for training.
    --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
                            Batch size per GPU/CPU for evaluation.
    --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                            Number of updates steps to accumulate before
                            performing a backward/update pass.
    --learning_rate LEARNING_RATE
                            The initial learning rate for Adam.
    --weight_decay WEIGHT_DECAY
                            Weight decay if we apply some.
    --adam_epsilon ADAM_EPSILON
                            Epsilon for Adam optimizer.
    --max_grad_norm MAX_GRAD_NORM
                            Max gradient norm.
    --num_train_epochs NUM_TRAIN_EPOCHS
                            Total number of training epochs to perform.
    --max_steps MAX_STEPS
                            If > 0: set total number of training steps to perform.
                            Override num_train_epochs.
    --warmup_steps WARMUP_STEPS
                            Linear warmup over warmup_steps.
    --logging_dir LOGGING_DIR
                            Tensorboard log dir.
    --logging_first_step  Log and eval the first global_step
    --logging_steps LOGGING_STEPS
                            Log every X updates steps.
    --save_steps SAVE_STEPS
                            Save checkpoint every X updates steps.
    --save_total_limit SAVE_TOTAL_LIMIT
                            Limit the total amount of checkpoints.Deletes the
                            older checkpoints in the output_dir. Default is
                            unlimited checkpoints
    --no_cuda             Avoid using CUDA even if it is available
    --seed SEED           random seed for initialization
    --fp16                Whether to use 16-bit (mixed) precision (through
                            NVIDIA apex) instead of 32-bit
    --fp16_opt_level FP16_OPT_LEVEL
                            For fp16: Apex AMP optimization level selected in
                            ['O0', 'O1', 'O2', and 'O3'].See details at
                            https://nvidia.github.io/apex/amp.html
    --local_rank LOCAL_RANK
                            For distributed training: local_rank
'''