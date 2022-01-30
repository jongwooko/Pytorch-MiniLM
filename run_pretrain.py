# coding: utf-8
# 2021.12.30-Changed for pretraining
#      Huawei Technologies Co., Ltd. <yinyichun@huawei.com>
# Copyright 2021 Huawei Technologies Co., Ltd.
# Copyright 2019 Sinovation Ventures AI Institute
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

# from __future__ import absolute_import, division, print_function

import os

from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

import math
import json
import random
import numpy as np
from collections import namedtuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import pickle
import collections

from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling import MiniLMForPreTraining, BertModel
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import logging
from apex.parallel import DistributedDataParallel as DDP

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

InputFeatures = namedtuple(
    "InputFeatures",
    "input_ids input_mask segment_ids lm_label_ids is_next")


def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]
    
    if len(tokens) > max_seq_length:
        logger.info('len(tokens): {}'.format(len(tokens)))
        logger.info('tokens: {}'.format(tokens))
        tokens = tokens[:max_seq_length]
        
    if len(tokens) != len(segment_ids):
        logger.info('tokens: {}\nsegment_ids: {}'.format(tokens, segment_ids))
        segment_ids = [0] * len(tokens)
        
    assert len(tokens) == len(segment_ids) <= max_seq_length
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
    
    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids
    
    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids
    
    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids
    
    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next)
    return features

class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False, gpuid=0):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = int(epoch % num_data_epochs)
        logger.info('training_path: {}'.format(training_path))
        data_file = Path(training_path) / "epoch_{}.json".format(self.data_epoch)
        metrics_file = Path(training_path) / "epoch_{}_metrics.json".format(self.data_epoch)
        
        logger.info('data_file: {}'.format(data_file))
        logger.info('metrics_file: {}'.format(metrics_file))
        
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path('/tmp'+str(gpuid))
            os.makedirs(str(self.working_dir), exist_ok=True)
            
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir/'is_nexts.memmap',
                                 shape=(num_samples,), mode='w+', dtype=np.bool)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
            
        logging.info("Loading training examples for epoch {}".format(epoch))
        
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next
                
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.is_nexts[item].astype(np.int64)))    

def main():
    parser = ArgumentParser()
    
    parser.add_argument('--pregenerated_data', type=str, required=True)
    parser.add_argument('--student_model', type=str, required=True, default='./models/3layer_bert')
    parser.add_argument('--teacher_model', type=str, default='./models/bert-base-uncased')
    parser.add_argument('--cache_dir', type=str, default='/input/osilab-nlp/cache', help='')
    
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_seq_length", type=int, default=512)
    
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--scratch',
                        action='store_true',
                        help="Whether to train from scratch")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument('--debug',
                        action='store_true',
                        help="Whether to debug")
    
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--already_trained_epoch",
                        default=0,
                        type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--warmup_steps", default=10000, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--num_workers", type=int, default=4, help="num_workers.")
    parser.add_argument("--continue_index", type=int, default=0, help="")
    parser.add_argument("--threads", type=int, default=27,
                        help="Number of threads to preprocess input data")
    
    parser.add_argument('--further_train', action='store_true')
    parser.add_argument('--mlm_loss', action='store_true')
    parser.add_argument('--mode', type=str, choices=["MiniLM_v1", "MiniLM_v2"])
    
    args = parser.parse_args()
    assert (torch.cuda.is_available())
    device_count = torch.cuda.device_count()
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    init_method = "env://"
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    print('device_id: %s' % args.local_rank)
    print('device_count: %s, rank: %s, world_size: %s' % (device_count, args.rank, args.world_size))
    print(init_method)
    
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size,
                                         rank=args.rank, init_method=init_method)
    
    samples_per_epoch = []
    for i in range(int(args.epochs)):
        epoch_file = Path(args.pregenerated_data) / "epoch_{}.json".format(i)
        metrics_file = Path(args.pregenerated_data) / "epoch_{}_metrics.json".format(i)
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            print("Warning! There are fewer epochs of pregenerated data ({}) than training epochs ({}).".format(i, args.epochs))
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break    
    else:
        num_data_epochs = args.epochs
    
    LOCAL_DIR = args.cache_dir

    local_save_dir = os.path.join(LOCAL_DIR, 'output', 'checkpoints')
    local_tsbd_dir = os.path.join(LOCAL_DIR, 'output', 'tensorboard')

    save_name = '_'.join([
    'bert',
    'epoch', str(args.epochs),
    'lr', str(args.learning_rate),
    'bsz', str(args.train_batch_size),
    'grad_accu', str(args.gradient_accumulation_steps),
    str(args.max_seq_length),
    'gpu', str(args.world_size),
    ])

    bash_save_dir = os.path.join(local_save_dir, save_name)
    bash_tsbd_dir = os.path.join(local_tsbd_dir, save_name)
    if args.local_rank == 0:
        if not os.path.exists(bash_save_dir):
            os.makedirs(bash_save_dir)
            logger.info(bash_save_dir + ' created!')
        if not os.path.exists(bash_tsbd_dir):
            os.makedirs(bash_tsbd_dir)
            logger.info(bash_tsbd_dir + ' created!')

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps paramter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
        
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    args.tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=args.do_lower_case)
    args.vocab_list = list(args.tokenizer.vocab.keys())
    
    if args.further_train:
        student_model = MiniLMForPreTraining.from_pretrained(args.student_model)
    else:
        student_model = MiniLMForPreTraining.from_scratch(args.student_model)
        
    student_model.to(device)
    args.student_config = student_model.config
    
    if not args.mlm_loss:
        teacher_model = MiniLMForPreTraining.from_pretrained(args.teacher_model)
        teacher_model.to(device)
        args.teacher_config = teacher_model.config
        
    if args.local_rank == 0:
        tb_writer = SummaryWriter(bash_tsbd_dir)
        
    global_step = 0
    step = 0
    tr_loss, tr_at_loss, tr_vr_loss = 0.0, 0.0, 0.0
    tr_qr_loss, tr_kr_loss = 0.0, 0.0
    tr_mlm_loss, tr_nsp_loss = 0.0, 0.0
    logging_loss, at_logging_loss, vr_logging_loss = 0.0, 0.0, 0.0
    qr_logging_loss, kr_logging_loss = 0.0, 0.0
    mlm_logging_loss, nsp_logging_loss = 0.0, 0.0
    end_time, start_time = 0, 0
    
    for epoch in range(args.epochs):
        if epoch < args.continue_index:
            args.warmup_steps = 0
            continue
            
        epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=args.tokenizer,
                                    num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory, gpuid=args.local_rank)
        train_sampler = DistributedSampler(epoch_dataset)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=args.num_workers)
        
        step_in_each_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_train_optimization_steps = step_in_each_epoch * args.epochs
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(epoch_dataset) * args.world_size)
        logger.info("  Num Epochs = %d", args.epochs)
        logging.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                     args.train_batch_size * args.gradient_accumulation_steps * args.world_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logging.info("  Num steps = %d", num_train_optimization_steps)
        
        if epoch == args.continue_index:
            # Prepare optimizer
            param_optimizer = list(student_model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            
            print("warm_up_ratio: {}".format(warm_up_ratio))
            optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate,
                                 e=args.adam_epsilon, schedule="warmup_linear",
                                 t_total=num_train_optimization_steps,
                                 warmup=args.warmup_proportion)
            
            if args.fp16:
                try:
                    from apex import amp
                except:
                    raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                                      " to use fp16 training.")
                student_model, optimizer = amp.initialize(student_model, optimizer,
                                                          opt_level=args.fp16_opt_level,
                                                          min_loss_scale=1)
                
            # apex
            student_model = DDP(student_model, message_size=10000000,
                                gradient_predivide_factor=torch.distributed.get_world_size(),
                                delay_allreduce=True)
            
            if not args.mlm_loss:
                teacher_model = DDP(teacher_model, message_size=10000000,
                                    gradient_predivide_factor=torch.distributed.get_world_size(),
                                    delay_allreduce=True)
                teacher_model.eval()
                
            logger.info("apex data paralleled!")
            
        from torch.nn import CrossEntropyLoss, KLDivLoss, MSELoss
#         loss_fct = KLDivLoss(reduction="batchmean", log_target=True) if not args.mlm_loss \
        loss_fct = KLDivLoss(log_target=True) if not args.mlm_loss \
                    else CrossEntropyLoss(ignore_index=-1)
        
        student_model.train()
        for step_, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
            step += 1
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks, segment_ids, lm_label_ids, is_next = batch
            
            if not args.mlm_loss and args.mode == "MiniLM_v1":
                at_loss, vr_loss = 0.0, 0.0
            elif not args.mlm_loss and args.mode == "MiniLM_v2":
                qr_loss, kr_loss, vr_loss = 0.0, 0.0, 0.0
            else:
                mlm_loss, nsp_loss = 0.0, 0.0
            
            if not args.mlm_loss and args.mode == "MiniLM_v1":
                with torch.no_grad():
                    _, _, teacher_query_layers, teacher_key_layers, teacher_value_layers = teacher_model(input_ids, segment_ids, input_masks,
                                                                                                         is_student=False)
                    teacher_attention_dist = torch.matmul(teacher_query_layers[0],
                                             teacher_key_layers[0].transpose(-1,-2))
                    teacher_value_relation = torch.matmul(teacher_value_layers[0],
                                             teacher_value_layers[0].transpose(-1,-2))
                    attention_head_size = int(args.teacher_config.hidden_size / args.teacher_config.num_attention_heads)
                    
                    teacher_attention_dist = teacher_attention_dist / math.sqrt(attention_head_size)
                    teacher_attention_dist = F.log_softmax(teacher_attention_dist, dim=-1)
                    
                    teacher_value_relation = teacher_value_relation / math.sqrt(attention_head_size)
                    teacher_value_relation = F.log_softmax(teacher_value_relation, dim=-1)
            
            elif not args.mlm_loss and args.mode == "MiniLM_v2":
                with torch.no_grad():
                    _, _, teacher_query_layers, teacher_key_layers, teacher_value_layers = teacher_model(input_ids, segment_ids, input_masks,
                                                                                                         is_student=False)
                    teacher_query_relation = torch.matmul(teacher_query_layers[0],
                                             teacher_query_layers[0].transpose(-1,-2))
                    teacher_key_relation = torch.matmul(teacher_key_layers[0],
                                           teacher_key_layers[0].transpose(-1,-2))
                    teacher_value_relation = torch.matmul(teacher_value_layers[0],
                                             teacher_value_layers[0].transpose(-1,-2))
                    attention_head_size = int(args.teacher_config.hidden_size / args.teacher_config.num_attention_heads)
                    
                    teacher_query_relation = teacher_query_relation / math.sqrt(attention_head_size)
                    teacher_query_relation = F.log_softmax(teacher_query_relation, dim=-1)
                    
                    teacher_key_relation = teacher_key_relation / math.sqrt(attention_head_size)
                    teacher_key_relation = F.log_softmax(teacher_key_relation, dim=-1)
                    
                    teacher_value_relation = teacher_value_relation / math.sqrt(attention_head_size)
                    teacher_value_relation = F.log_softmax(teacher_value_relation, dim=-1)
                    
            if not args.mlm_loss and args.mode == "MiniLM_v1":
                # knowledge distillation
                _, _, student_query_layers, student_key_layers, student_value_layers = student_model(input_ids, segment_ids, input_masks,
                                                                                                     is_student=True)
                    
                student_attention_dist = torch.matmul(student_query_layers[0],
                                          student_key_layers[0].transpose(-1,-2))
                student_value_relation = torch.matmul(student_value_layers[0],
                                         student_value_layers[0].transpose(-1,-2))
                attention_head_size = int(args.student_config.hidden_size / args.student_config.num_attention_heads)

                student_attention_dist = student_attention_dist / math.sqrt(attention_head_size)
                student_attention_dist = F.log_softmax(student_attention_dist, dim=-1)

                student_value_relation = student_value_relation / math.sqrt(attention_head_size)
                student_value_relation = F.log_softmax(student_value_relation, dim=-1)
                
                scaler = 1.
                at_loss += loss_fct(teacher_attention_dist, student_attention_dist) / scaler
                vr_loss += loss_fct(teacher_value_relation, student_value_relation) / scaler
                loss = at_loss + vr_loss

                if args.gradient_accumulation_steps > 1:
                    at_loss = at_loss / args.gradient_accumulation_steps
                    vr_loss = vr_loss / args.gradient_accumulation_steps
                    loss = loss / args.gradient_accumulation_steps

                tr_at_loss += at_loss.item()
                tr_vr_loss += vr_loss.item()
                
            elif not args.mlm_loss and args.mode == "MiniLM_v2":
                _, _, student_query_layers, student_key_layers, student_value_layers = student_model(input_ids, segment_ids, input_masks,
                                                                                                     is_student=True)
                    
                student_query_relation = torch.matmul(student_query_layers[0],
                                         student_query_layers[0].transpose(-1,-2))
                student_key_relation = torch.matmul(student_key_layers[0],
                                       student_key_layers[0].transpose(-1,-2))
                student_value_relation = torch.matmul(student_value_layers[0],
                                         student_value_layers[0].transpose(-1,-2))
                attention_head_size = int(args.student_config.hidden_size / args.student_config.num_attention_heads)

                student_query_relation = student_query_relation / math.sqrt(attention_head_size)
                student_query_relation = F.log_softmax(student_query_relation, dim=-1)

                student_key_relation = student_key_relation / math.sqrt(attention_head_size)
                student_key_relation = F.log_softmax(student_key_relation, dim=-1)

                student_value_relation = student_value_relation / math.sqrt(attention_head_size)
                student_value_relation = F.log_softmax(student_value_relation, dim=-1)
                
#                 scaler = student_value_relation.size(0) * student_value_relation.size(1)
                scaler = 1.0
                qr_loss += loss_fct(teacher_query_relation, student_query_relation) / scaler
                kr_loss += loss_fct(teacher_key_relation, student_key_relation) / scaler
                vr_loss += loss_fct(teacher_value_relation, student_value_relation) / scaler
                loss = qr_loss + kr_loss + vr_loss

                if args.gradient_accumulation_steps > 1:
                    qr_loss = qr_loss / args.gradient_accumulation_steps
                    kr_loss = kr_loss / args.gradient_accumulation_steps
                    vr_loss = vr_loss / args.gradient_accumulation_steps
                    loss = loss / args.gradient_accumulation_steps

                tr_qr_loss += qr_loss.item()
                tr_kr_loss += kr_loss.item()
                tr_vr_loss += vr_loss.item()
                    
            else:
                # Do not use distillation
                prediction_scores, seq_relationship_score, _, _, _ = student_model(input_ids, segment_ids, input_masks,
                                                                                   is_student=False)
                mlm_loss = loss_fct(
                    prediction_scores.view(-1, len(args.vocab_list)), lm_label_ids.view(-1))
                nsp_loss = loss_fct(
                    seq_relationship_score.view(-1, 2), is_next.view(-1))
                loss = mlm_loss + nsp_loss

                if args.gradient_accumulation_steps > 1:
                    mlm_loss = mlm_loss / args.gradient_accumulation_steps
                    nsp_loss = nsp_loss / args.gradient_accumulation_steps
                    loss = loss / args.gradient_accumulation_steps
                    
                tr_mlm_loss += mlm_loss.item()
                tr_nsp_loss += nsp_loss.item()
                    
            tr_loss += loss.item()
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                    
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
                    
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                if (step + 1) % (args.gradient_accumulation_steps * args.logging_steps) == 0 \
                    or args.local_rank < 2 and global_step % step_in_each_epoch < 100:
                    end_time = time.time()
                    
                    if not args.mlm_loss and args.mode == "MiniLM_v1":
                        logger.info(
                                'Epoch: %s, global_step: %s/%s, lr: %s, loss is %s; '
                                'at_loss is %s; vr_loss is %s; (%.2f sec)' %
                                (epoch, (global_step + 1) % step_in_each_epoch, 
                                 step_in_each_epoch, optimizer.get_lr()[0],
                                 loss.item() * args.gradient_accumulation_steps,
                                 at_loss.item() * args.gradient_accumulation_steps,
                                 vr_loss.item() * args.gradient_accumulation_steps,
                                 end_time - start_time))
                        
                    elif not args.mlm_loss and args.mode in == "MiniLM_v2":
                        logger.info(
                                'Epoch: %s, global_step: %s/%s, lr: %s, loss is %s; '
                                'qr_loss is %s; kr_loss is %s; vr_loss is %s; (%.2f sec)' %
                                (epoch, (global_step + 1) % step_in_each_epoch, 
                                 step_in_each_epoch, optimizer.get_lr()[0],
                                 loss.item() * args.gradient_accumulation_steps,
                                 qr_loss.item() * args.gradient_accumulation_steps,
                                 kr_loss.item() * args.gradient_accumulation_steps,
                                 vr_loss.item() * args.gradient_accumulation_steps,
                                 end_time - start_time))
                    else:
                        logger.info(
                                'Epoch: %s, global_step: %s/%s, lr: %s, loss is %s; '
                                'mlm_loss is %s; nsp_loss is %s; (%.2f sec)' %
                                (epoch, (global_step + 1) % step_in_each_epoch, 
                                 step_in_each_epoch, optimizer.get_lr()[0],
                                 loss.item() * args.gradient_accumulation_steps,
                                 mlm_loss.item() * args.gradient_accumulation_steps,
                                 nsp_loss.item() * args.gradient_accumulation_steps,
                                 end_time - start_time))
                        
                    start_time = time.time()
                    
                if args.logging_steps > 0 and global_step % args.logging_steps == 0 and args.local_rank == 0:
                    tb_writer.add_scalar("lr", optimizer.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    
                    if not args.mlm_loss and args.mode == "MiniLM_v1":
                        tb_writer.add_scalar("at_loss", (tr_at_loss - at_logging_loss) / args.logging_steps, global_step)
                        tb_writer.add_scalar("vr_loss", (tr_vr_loss - vr_logging_loss) / args.logging_steps, global_step)
                        at_logging_loss = tr_at_loss
                        vr_logging_loss = tr_vr_loss
                        
                    elif not args.mlm_loss and args.mode == "MiniLM_v2":
                        tb_writer.add_scalar("qr_loss", (tr_qr_loss - qr_logging_loss) / args.logging_steps, global_step)
                        tb_writer.add_scalar("kr_loss", (tr_kr_loss - kr_logging_loss) / args.logging_steps, global_step)
                        tb_writer.add_scalar("vr_loss", (tr_vr_loss - vr_logging_loss) / args.logging_steps, global_step)
                        qr_logging_loss = tr_qr_loss
                        kr_logging_loss = tr_kr_loss
                        vr_logging_loss = tr_vr_loss
                        
                    else:
                        tb_writer.add_scalar("mlm_loss", (tr_mlm_loss - mlm_logging_loss) / args.logging_steps, global_step)
                        tb_writer.add_scalar("nsp_loss", (tr_nsp_loss - nsp_logging_loss) / args.logging_steps, global_step)
                        mlm_logging_loss = tr_mlm_loss
                        nsp_logging_loss = tr_nsp_loss
                        
                    logging_loss = tr_loss
                    
        # Save a trained model
        if args.rank == 0:
            saving_path = bash_save_dir
            saving_path = Path(os.path.join(saving_path, "epoch_" + str(epoch)))
            
            if saving_path.is_dir() and list(saving_path.iterdir()):
                logging.warning(f"Output directory ({ saving_path }) already exists and is not empty!")
            saving_path.mkdir(parents=True, exist_ok=True)
            
            logging.info("** ** * Saving fine-tuned model ** ** * ")
            model_to_save = student_model.module if hasattr(student_model, 'module')\
                else student_model  # Only save the model it-self
            
            output_model_file = os.path.join(saving_path, WEIGHTS_NAME)
            output_config_file = os.path.join(saving_path, CONFIG_NAME)
            
            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            args.tokenizer.save_vocabulary(saving_path)
            
            torch.save(optimizer.state_dict(), os.path.join(saving_path, "optimizer.pt"))
            logger.info("Saving optimizer and scheduler states to %s", saving_path)
            
    if args.local_rank == 0:
        tb_writer.close()
        
if __name__ == "__main__":
    main()