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
Based on (but heavily modified from):
https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_xnli.py
"""


import argparse
import glob
import logging
import os
import random
import math
import time
import json
import pickle
import copy
import multiprocessing as mp
import functools

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from typing import List, Dict, Callable, Tuple, Generator, Set, Sequence
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from dataset_processor import PuzzleProcessor, InputExample, InputFeatures, convert_examples_to_features, get_QAs, get_Q_trees, traverse_ans_tree
import tython
import top_down
from neural_classifier import TransfomerSolver, TransfomerTreeGenerator
from tython import _RULES_BY_KIND, RULES, Program, nt
from models.model import reachable_rules_by_kind
from challenges import extract_constants, contains_node
import utils

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def compute_influence(args, train_datasets, model, tokenizer, target_puzzle: str, target_sol: Program, update_steps=1):
    '''
    Computes the -log(prob) of target_sol on target_puzzle
    when updating the model with each train_dataset in train_datasets.

    For simplicity: uses single gpu and no fp16.
    '''
    model.do_solve()

    init_state_dict = copy.deepcopy(model.state_dict())

    log_probs = []

    args.train_batch_size = args.per_gpu_train_batch_size
    for train_dataset in tqdm(train_datasets, desc="Influence"):
        
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * update_steps

        # Prepare optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        #optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=0.1)

        total_bs = args.train_batch_size \
            * args.gradient_accumulation_steps \
            * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        set_seed(args)
        for _ in range(int(update_steps)):
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "parent_inds": batch[-3],
                    "child_nums": batch[-2],
                    "softmax_masks": batch[-4].to_dense(),
                    "labels": batch[-1]
                }
                outputs = model(**inputs)
                loss, logits = outputs[:2]

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    model.zero_grad()
                    global_step += 1

        log_proba = get_solution_prob(args, model, tokenizer, target_puzzle, target_sol)
        log_probs.append(log_proba)

    return log_probs


def train(args, train_dataset, model, tokenizer, do_regen=False):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if do_regen:
        model.do_regen()
    else:
        model.do_solve()

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!

    total_bs = args.train_batch_size \
        * args.gradient_accumulation_steps \
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)

    logger.info(f"Running training on {len(train_dataset)} examples. Epochs={args.num_train_epochs}.")
    logger.info(f"per GPU batch size={args.per_gpu_train_batch_size}. Total batch size={total_bs}")
    logger.info(f"Gradient accumulation steps={args.gradient_accumulation_steps}. Total optimization steps={t_total}")

    #logger.info("***** Running training *****")
    #logger.info("  Num examples = %d", len(train_dataset))
    #logger.info("  Num Epochs = %d", args.num_train_epochs)
    #logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    #logger.info(
    #    "  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #    args.train_batch_size
    #    * args.gradient_accumulation_steps
    #    * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    #)
    #logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    #logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    eval_QAs = None
    for _ in train_iterator:
        #epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator = train_dataloader
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "parent_inds": batch[-3],
                "child_nums": batch[-2],
                "softmax_masks": batch[-4].to_dense(),
                "labels": batch[-1]
            }
            outputs = model(**inputs)
            loss, logits = outputs[:2]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    rolling_loss = (tr_loss - logging_loss) / args.logging_steps
                    tb_writer.add_scalar("loss", rolling_loss, global_step)
                    logging_loss = tr_loss
                    logger.info(f"step {global_step}. lr: {scheduler.get_lr()[0]}, train loss: {rolling_loss}")

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training

                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

                #if args.local_rank in [-1, 0] and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                #    if eval_QAs is None:
                #        if do_regen:
                #            eval_QAs, sol_kinds = get_Q_trees(args.eval_challenges_path, max_ticks=args.max_ticks)
                #        else:
                #            eval_QAs, _ = get_QAs(args.eval_challenges_path, max_ticks=args.max_ticks)
                #    result = evaluate(args, model, eval_QAs, tokenizer, prefix=str(global_step), write_res=False)
                #    for key in sorted(result.keys()):
                #        tb_writer.add_scalar("eval_{}".format(key), result[key], global_step)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def get_puzzle_embeddings(args, model, puzzles, tokenizer):
    batch_encoding = tokenizer.batch_encode_plus(
        puzzles, max_length=args.max_seq_length, pad_to_max_length=True,
    )
    features = []
    for i in range(len(puzzles)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs)
        features.append(feature)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.int)
    dataset = TensorDataset(all_input_ids, all_attention_mask)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    encoder = model.transformer_model
    hidden_size = encoder.config.hidden_size
    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(encoder, torch.nn.DataParallel):
        encoder = torch.nn.DataParallel(encoder)

    logger.debug("***** Embeddings puzzles *****")
    logger.debug("  Num examples = %d", len(dataset))
    logger.debug("  Batch size = %d", args.eval_batch_size)
    puzzle_embeddings = torch.zeros((len(dataset), hidden_size))
    #for b, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating", total=math.ceil(len(dataset)/args.eval_batch_size)):
    for b, batch in enumerate(eval_dataloader):
        encoder.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            outputs = encoder(**inputs)
            for i, emb in enumerate(outputs[1]):
                puzzle_embeddings[b * args.eval_batch_size + i, :] = emb

    return puzzle_embeddings


def regen_puzzles(args, model, tokenizer, puzzles, sol_kinds, n_progs=100, lamb=0., loud=True, generate_here=True):
    answers = {} # Dict puzzle str to rule probs.
    processor = PuzzleProcessor()

    model.do_regen()

    all_ans_rule_probs = []
    BOOL = tython.nonterminals.type2nt(bool)

    puzzle_embeddings = get_puzzle_embeddings(args, model, puzzles, tokenizer)
    for i, puzzle in tqdm(enumerate(puzzles), desc="Puzzels", total=len(puzzles)):
        sol_kind = sol_kinds[i]
        #var_name = utils.get_lambda_arg_name(puzzle)
        #assert tython.VAR_KINDS[var_name] == sol_kind, f"VAR_KINDS['{var_name}'] != {sol_kind} for '{puzzle}'"

        rules_by_kind = _RULES_BY_KIND
        examples = []
        count = 0

        # Collect all possible parents.
        logger.debug("Gathering rules for puzzle {}.".format(i))
        for parent_kind in rules_by_kind:
            # None for root.
            for r in rules_by_kind[parent_kind] + ([None] if parent_kind == sol_kind else []):
                for child_num, kind in enumerate([BOOL] if r is None else r.kids):
            #for r in rules_by_kind[parent_kind]:
            #    for child_num, kind in enumerate(r.kids):
                    if r is None:
                        parent_ind = len(RULES)
                    else:
                        parent_ind = r.index
                    guid = "%s" % (count)
                    examples.append(
                        InputExample(guid=guid,
                                     puzzle_str=puzzle,
                                     parent_ind=parent_ind,
                                     child_num=child_num,
                                     target_kind=kind,
                                     ))
                    count += 1

        features = convert_examples_to_features(
            examples, tokenizer, max_length=args.max_seq_length, output_mode='classification',
            loud=False, rules_by_kind=rules_by_kind
        )
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.int)
        all_softmax_mask = torch.stack([f.softmax_mask for f in features])
        all_parent_inds = torch.tensor([f.parent_ind for f in features], dtype=torch.long)
        all_child_nums = torch.tensor([f.child_num for f in features], dtype=torch.long)

        eval_dataset = TensorDataset(all_input_ids, all_attention_mask, all_softmax_mask, all_parent_inds, all_child_nums)
        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logger.debug("***** Computing rule probabilities *****")
        logger.debug("  Num examples = %d", len(eval_dataset))
        logger.debug("  Batch size = %d", args.eval_batch_size)
        preds = None
        cached_question_emb = None
        #for batch in tqdm(eval_dataloader, desc="Rule probs", total=math.ceil(len(eval_dataset)/args.eval_batch_size)):
        for batch in eval_dataloader:
            model.eval()
            if cached_question_emb is None or cached_question_emb.shape[0] != batch[0].shape[0]:
                cached_question_emb = puzzle_embeddings[i]
                cached_question_emb = cached_question_emb.to(args.device)
                cached_question_emb = cached_question_emb.repeat(batch[0].shape[0], 1)
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "softmax_masks": batch[2].to_dense(),
                    "parent_inds": batch[3],
                    "child_nums": batch[4],
                    "question_emb": cached_question_emb,
                }
                logits = model(**inputs)[0]
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        #TODO min_prob arg
        #min_prob = 1e-100
        min_prob = -1
        
        logger.debug("Creating rule probabilities dictionary")
        ans_rule_probs = {}
        for pred, out_mask, example in zip(preds, all_softmax_mask, examples):
            if example.parent_ind < len(RULES) and RULES[example.parent_ind].name == "COPY":
                # no consts?
                # sholdn't see COPY in puzzle.
                cops = []
                #cops = consts.get(example.target_kind, [])
                ans_rule_probs[example.parent_ind] = [([(1. / len(cops), c) for c in cops],
                           [])]
                continue

            rule_inds =  out_mask.coalesce().indices().cpu().numpy()
            masked_logits = pred[rule_inds]
            rule_probs = np.exp(masked_logits) / np.sum(np.exp(masked_logits))
            
            # Smoothing.
            if len(rule_probs[0]) > 0:
                rule_probs = rule_probs * (1 - lamb) + lamb / len(rule_probs[0])
                assert abs(rule_probs.sum() - 1) < 0.05


            rule_probs = [(p, rule_ind) for p, rule_ind in zip(rule_probs[0], rule_inds[0]) if p >= min_prob]

            if example.parent_ind == len(RULES):
                r = None
                if len(RULES) not in ans_rule_probs:
                    # ROOT has one child
                    ans_rule_probs[len(RULES)] = [()]
            else:
                r = RULES[example.parent_ind]
                if r.index not in ans_rule_probs:
                    # Create entries for kids
                    ans_rule_probs[r.index] = [() for _ in  range(len(r.kids))]

            r_index = r.index if r is not None else len(RULES)
            ans_rule_probs[r_index][example.child_num] = (([], rule_probs))

        all_ans_rule_probs.append(ans_rule_probs)
        #answers[puzzle] = ans_rule_probs
        #ans_file = os.path.join(args.output_dir, 'regen_{}.pkl'.format(i))
        #pickle.dump(ans_rule_probs, open(ans_file, 'wb'))

        if generate_here:
            # Generate puzzles.
            candidates = top_down.map_candidates_rules(ans_rule_probs)

            if loud:
                logger.info(puzzle)

            generations = []
            for _ in tqdm(range(20)):
                node = top_down.rand_node(candidates, str(sol_kind))
                gen_node = tython.Program(node)
                generations.append(str(gen_node))

            if loud:
                logger.info(generations)
                logger.info('-'*30)

    return all_ans_rule_probs


def solve_challenge(params, timeout_secs=20, max_n_progs=5000, max_ticks=10000, from_pkl=False):
    if from_pkl:
        name, f_str, sol_kind, file_name = params
        with open(file_name, 'rb') as f:
            candidates = pickle.load(f)
    else:
        name, f_str, sol_kind, candidates = params
    try:
        prog = tython.Program(f_str)
                              #lambda_kinds={utils.get_lambda_arg_name(f_str): sol_kind})
        st_time = time.time()
        ans, count = top_down.solve(
            prog,
            sol_kind,
            candidates,
            timeout_secs=timeout_secs,
            n_progs=max_n_progs,
            max_ticks=max_ticks,
            rule_inds=True,
        )
        if ans is not None:
            a_py = ans.src(safe=False, simplify=False)
        else:
            a_py = None
        return name, a_py, time.time() - st_time, count
    except Exception as e:
        logger.error("Exception while solving '{}' ('{}'): {}", name, f_str, e)
        return name, None, -1, 0


def infer(args, model, tokenizer, puzzles, sol_kinds, lamb=0., file_names=None):

    answers = {} # Dict puzzle str to rule probs.
    if file_names is not None:
        assert len(file_names) == len(puzzles)

    processor = PuzzleProcessor()

    puzzle_embeddings = get_puzzle_embeddings(args, model, puzzles, tokenizer)

    for i, puzzle in tqdm(enumerate(puzzles), desc="Puzzels", total=len(puzzles)):
        if file_names is not None and os.path.exists(file_names[i]):
            try:
                with open(file_names[i], 'rb') as f:
                    ans_rule_probs = pickle.load(f)
                continue
            except:
                logger.warning(f"Failed to load rule probs from '{file_names[i]}'. Recomputing.")
                pass

        sol_kind = sol_kinds[i]

        # lambda x: len(x) == 43 and x.count(x[26]) == 28
        consts = {}
        try:
            p_prog = tython.Program(
                puzzle)
                #lambda_kinds={utils.get_lambda_arg_name(puzzle): sol_kind})
            consts = extract_constants(p_prog)
        except Exception as e:
            logger.warning("Failed to extract consts from program '{}': {}", puzzle, e)

        #rules_by_kind = reachable_rules_by_kind(sol_kind, consts) # pruned
        rules_by_kind = _RULES_BY_KIND # zzz
        examples = []
        count = 0
        # Collect all possible parents.
        logger.debug("Gathering rules for puzzle {}.".format(i))
        for parent_kind in rules_by_kind:
            # None for root.
            for r in rules_by_kind[parent_kind] + ([None] if parent_kind == sol_kind else []):
                for child_num, kind in enumerate([sol_kind] if r is None else r.kids):
                    if r is None:
                        parent_ind = len(RULES)
                    else:
                        parent_ind = r.index
                    guid = "%s" % (count)
                    examples.append(
                        InputExample(guid=guid,
                                     puzzle_str=puzzle,
                                     parent_ind=parent_ind,
                                     child_num=child_num,
                                     target_kind=kind,
                                     ))
                    count += 1

        assert len(examples) > 0
        features = convert_examples_to_features(
            examples, tokenizer, max_length=args.max_seq_length, output_mode='classification',
            loud=False, rules_by_kind=rules_by_kind
        )
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.int)
        all_softmax_mask = torch.stack([f.softmax_mask for f in features])
        all_parent_inds = torch.tensor([f.parent_ind for f in features], dtype=torch.long)
        all_child_nums = torch.tensor([f.child_num for f in features], dtype=torch.long)

        eval_dataset = TensorDataset(all_input_ids, all_attention_mask, all_softmax_mask, all_parent_inds, all_child_nums)
        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logger.debug("***** Computing rule probabilities *****")
        logger.debug("  Num examples = %d", len(eval_dataset))
        logger.debug("  Batch size = %d", args.eval_batch_size)
        preds = None
        cached_question_emb = None
        #for batch in tqdm(eval_dataloader, desc="Evaluating", total=math.ceil(len(eval_dataset)/args.eval_batch_size)):
        for batch in eval_dataloader:
            model.eval()
            if cached_question_emb is None or cached_question_emb.shape[0] != batch[0].shape[0]:
                cached_question_emb = puzzle_embeddings[i]
                cached_question_emb = cached_question_emb.to(args.device)
                cached_question_emb = cached_question_emb.repeat(batch[0].shape[0], 1)
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "softmax_masks": batch[2].to_dense(),
                    "parent_inds": batch[3],
                    "child_nums": batch[4],
                    "question_emb": cached_question_emb,
                }
                logits = model(**inputs)[0]
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        #TODO min_prob arg
        min_prob = 1e-100
        
        logger.debug("Creating rule probabilities dictionary")
        ans_rule_probs = {}
        for pred, out_mask, example in zip(preds, all_softmax_mask, examples):
            if example.parent_ind < len(RULES) and RULES[example.parent_ind].name == "COPY":
                cops = consts.get(example.target_kind, [])
                # Just give probability 1 to each const because they are all uniform
                # and we don't want to reduce the program's probability if using the (two step) copy vs. the const rule.
                #ans_rule_probs[example.parent_ind] = [([(1., c) for c in cops],
                ans_rule_probs[example.parent_ind] = [([(1. / len(cops), c) for c in cops],
                           [])]
                continue

            #rule_inds =  np.where(out_mask.to_dense() == 1)
            rule_inds =  out_mask.coalesce().indices().cpu().numpy()
            masked_logits = pred[rule_inds]
            rule_probs = np.exp(masked_logits) / np.sum(np.exp(masked_logits))

            # Smoothing.
            rule_probs = rule_probs * (1 - lamb) + lamb / len(rule_probs[0])
            assert abs(rule_probs.sum() - 1) < 0.05

            rule_probs = [(p, rule_ind) for p, rule_ind in zip(rule_probs[0], rule_inds[0]) if p >= min_prob]

            if example.parent_ind == len(RULES):
                r = None
                if len(RULES) not in ans_rule_probs:
                    # ROOT has one child
                    ans_rule_probs[len(RULES)] = [()]
            else:
                r = RULES[example.parent_ind]
                if r.index not in ans_rule_probs:
                    # Create entries for kids
                    ans_rule_probs[r.index] = [() for _ in  range(len(r.kids))]

            r_index = r.index if r is not None else len(RULES)
            ans_rule_probs[r_index][example.child_num] = (([], rule_probs))

        if file_names is not None:
            with open(file_names[i], 'wb') as wf:
                pickle.dump(ans_rule_probs, wf)
        else:
            answers[puzzle] = ans_rule_probs

        if False:
            # Regenerate
            logger.info(f"Trying to solve '{puzzle}'...")
            time0 = time.time()
            timeout_secs=180
            n_progs=20
            stop_time = (time0 + timeout_secs) if timeout_secs else None
            count = 0
            candidates = top_down.map_candidates_rules(ans_rule_probs)
            for p, a in top_down.generate_progs(candidates, stop_time):
                count += 1
                a_py = a.src(safe=False, simplify=False)
                print(a_py)
                if count >= n_progs:
                    break

        if False:
            logger.info(f"Trying to solve '{puzzle}'...")
            st_time = time.time()
            ans, count = top_down.solve(
                p_prog,
                sol_kind,
                ans_rule_probs,
                timeout_secs=180, #TODO
                n_progs=100*5000, #TODO
                rule_inds=True,
                max_ticks=args.max_ticks,
            )
            duration = time.time() - st_time
            if ans is None:
                logger.info(f"Failed to solve, generated {count:,} programs in {duration:.2f}s.")
            else:
                logger.info("Got answer in {:.2f}s and {:,} programs: {}".format(duration, count, ans))

    #ans_file = os.path.join(args.output_dir, 'infer.pkl')
    #pickle.dump(answers, open(ans_file, 'wb'))

    return answers


def get_solution_prob(args, model, tokenizer, puzzle: str, solution: Program):
    answers = {} # Dict puzzle str to rule probs.

    puzzle_embedding = get_puzzle_embeddings(args, model, [puzzle], tokenizer)[0]

    model.do_solve()
    model.loss_reduction = 'sum'

    examples = []
    for i, rule in enumerate(traverse_ans_tree(solution)):
        target_rule, parent_ind, child_num = rule
        guid = "%s" % (i)
        examples.append(
            InputExample(guid=guid,
                         puzzle_str=puzzle,
                         parent_ind=parent_ind,
                         child_num=child_num,
                         label=target_rule,
                         target_kind=RULES[target_rule].kind,
                         ))

    assert len(examples) > 0
    features = convert_examples_to_features(
        examples, tokenizer, max_length=args.max_seq_length, output_mode='classification',
        loud=False)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.int)
    all_softmax_mask = torch.stack([f.softmax_mask for f in features])
    all_parent_inds = torch.tensor([f.parent_ind for f in features], dtype=torch.long)
    all_child_nums = torch.tensor([f.child_num for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    eval_dataset = TensorDataset(all_input_ids, all_attention_mask, all_softmax_mask, all_parent_inds, all_child_nums, all_labels)
    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.debug("***** Computing rule probabilities *****")
    logger.debug("  Num examples = %d", len(eval_dataset))
    logger.debug("  Batch size = %d", args.eval_batch_size)
    preds = None
    out_label_ids = None
    cached_question_emb = None
    eval_loss = 0.0
    #for batch in tqdm(eval_dataloader, desc="Evaluating", total=math.ceil(len(eval_dataset)/args.eval_batch_size)):
    for batch in eval_dataloader:
        model.eval()
        if cached_question_emb is None or cached_question_emb.shape[0] != batch[0].shape[0]:
            cached_question_emb = puzzle_embedding
            cached_question_emb = cached_question_emb.to(args.device)
            cached_question_emb = cached_question_emb.repeat(batch[0].shape[0], 1)
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "softmax_masks": batch[2].to_dense(),
                "parent_inds": batch[3],
                "child_nums": batch[4],
                "labels": batch[5],
                "question_emb": cached_question_emb,
            }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.sum().item()

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    return eval_loss


def evaluate(args, model, QAs, tokenizer, prefix="", write_res=True):
    results = {}
    eval_dataset = load_and_cache_examples(args, QAs, tokenizer, evaluate=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.debug("***** Running evaluation {} *****".format(prefix))
    logger.debug("  Num examples = %d", len(eval_dataset))
    logger.debug("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "parent_inds": batch[-3],
                "child_nums": batch[-2],
                "softmax_masks": batch[-4].to_dense(),
                "labels": batch[-1]
            }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    top_preds = np.argmax(preds, axis=1)
    preds_sort = preds.argsort(-1)
    top_ten = preds_sort[:,-10:]
    top_ten_acc = np.mean([np.isin(out_label_ids[i], top_ten[i]).item() for i in range(top_ten.shape[0])])
    top_five = preds_sort[:,-5:]
    top_five_acc = np.mean([np.isin(out_label_ids[i], top_five[i]).item() for i in range(top_five.shape[0])])
    result = {'acc': (top_preds == out_label_ids).mean(), 'top_5': top_five_acc, 'top_10': top_ten_acc}
    results.update(result)

    logger.debug("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.debug("  %s = %s", key, str(result[key]))
    if write_res:
        output_eval_file = os.path.join(args.output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, QAs, tokenizer, evaluate=False, dont_cache=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = PuzzleProcessor()
    # Load data features from cache or dataset file
    if not dont_cache:
        if evaluate:
            data_dir = os.path.dirname(args.eval_challenges_path)
            pattern = os.path.basename(args.eval_challenges_path).replace("*", "^")
        else:
            data_dir = os.path.dirname(args.challenges_path)
            pattern = os.path.basename(args.challenges_path).replace("*", "^")
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(
                pattern,
                "test" if evaluate else "train",
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
            ),
        )
    if not dont_cache and os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features for %s QA pairs", len(QAs))
        label_list = processor.get_labels()
        examples = (
            processor.get_test_examples(QAs) if evaluate else processor.get_train_examples(QAs)
        )
        features = convert_examples_to_features(
            examples, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode='classification',
        )
        if args.local_rank in [-1, 0] and not dont_cache:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    logger.info("Convert to Tensors and build dataset")
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.int)
    all_softmax_mask = torch.stack([f.softmax_mask for f in features])
    all_parent_inds = torch.tensor([f.parent_ind for f in features], dtype=torch.long)
    all_child_nums = torch.tensor([f.child_num for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_softmax_mask, all_parent_inds, all_child_nums, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--challenges_path",
        default=None,
        type=str,
        help="Path pattern for jsons with training challenges.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument("--min_tries", type=int, default=2, help="Train the solver only on augmentation examples that were solved after at least min_tries (to avoid overfitting trivial puzzles).")
    parser.add_argument("--solve_smoothing_lamb", type=float, default=0.01, help="The smoothing lambda to apply for the target solver [0,1].")
    parser.add_argument("--max_ticks", type=int, default=10000, help="max opertaions for solution program.")
    parser.add_argument("--threads", type=int, default=10, help="Number of cpu threads for solver.")
    parser.add_argument("--timeout_secs", type=int, default=120, help="max seconds to try each target puzzle.")
    parser.add_argument("--max_n_progs", type=int, default=10000, help="max programs to try per target puzzle each cycle.")
    parser.add_argument("--transfer", action="store_true", help="Load a model that was already trained to solve/augment.")
    parser.add_argument(
        "--eval_challenges_path",
        default='',
        type=str,
        help="Path pattern for jsons with eval challenges.",
    )
    parser.add_argument(
        "--rule_emb_dir", default="", type=str, help="Path to dir with precomputed embeddings for rules."
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_regen", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_infer", action="store_true", help="Whether to run inference on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=16, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=2000, help="Evaluate every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    if args.do_eval or args.do_infer or args.do_regen:
        assert args.eval_challenges_path

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    os.makedirs(args.output_dir, exist_ok=True)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        #filename = os.path.join(args.output_dir, 'log.txt'),
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    logger.addHandler(fh)

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare task
    processor = PuzzleProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # TODO: for additional finetuninng use torch.load
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir,
    )
    args.model_type = config.model_type
    if args.transfer:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = torch.load(os.path.join(args.model_name_or_path, WEIGHTS_NAME), map_location=args.device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir,
        )
        transformer_model = AutoModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
        #model = TransfomerSolver(transformer_model)
        model = TransfomerTreeGenerator(transformer_model)

        if args.rule_emb_dir:
            model.set_rule_embeddings(args.rule_emb_dir)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        if args.do_regen:
            QAs, _ = get_Q_trees(args.challenges_path, max_ticks=args.max_ticks)
        else:
            QAs, _ = get_QAs(args.challenges_path, max_ticks=args.max_ticks, min_tries=args.min_tries)
        train_dataset = load_and_cache_examples(args, QAs, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, do_regen=args.do_regen)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Verify that we can load back the model and tokenizer.
        model = torch.load(os.path.join(args.output_dir, WEIGHTS_NAME), map_location=args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if (args.do_eval or args.do_infer or args.do_regen) and args.local_rank in [-1, 0]:
        if args.do_regen:
            eval_QAs, sol_kinds = get_Q_trees(args.eval_challenges_path, max_ticks=args.max_ticks)
        elif args.do_eval:
            eval_QAs, sol_kinds  = get_QAs(args.eval_challenges_path, max_ticks=args.max_ticks)
        else:
            pass
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            if args.eval_all_checkpoints:
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            else:
                prefix = ""

            model = torch.load(os.path.join(args.output_dir, prefix, WEIGHTS_NAME), map_location=args.device)
            model.to(args.device)

            if args.do_infer:
                target_puzzles = []
                target_names = []
                target_sol_kinds = []
                target_puzzles_configs = []
                challenges_files = glob.glob(args.eval_challenges_path)
                for f_name in challenges_files:
                    with open(f_name, 'r') as f:
                        chs = json.load(open(f_name, 'r'))
                        for ch in chs:
                            puz = ch['sat']
                            try:
                                q = tython.Program(puz)
                                f = q.run(max_ticks=100000000)
                            except Exception as e:
                                logger.warning(f"Exception parsing {ch['name']} '{puz}': {e}")
                                continue
                            target_puzzles_configs.append(ch)
                            target_puzzles.append(ch['sat'])
                            target_names.append(ch['name'])
                            sol_type_annotation = q.tree.children[0].children[1].children[1]
                            sol_kind = nt.type2nt(eval(sol_type_annotation.src()))
                            target_sol_kinds.append(sol_kind)

                all_ans_rule_probs = infer(args, model, tokenizer, target_puzzles, target_sol_kinds, lamb=args.solve_smoothing_lamb)

                workers = mp.Pool(processes=args.threads, maxtasksperchild=4)
                map_fn = workers.imap_unordered
                worker_fn = functools.partial(solve_challenge, timeout_secs=args.timeout_secs, max_n_progs=args.max_n_progs, max_ticks=args.max_ticks)
                solved, unsolved = [], []
                outputs = {}
                out_file = os.path.join(args.output_dir, 'solutions.json')
                out_file_jsonl = os.path.join(args.output_dir, 'solutions.jsonl')
                #with open(out_file_jsonl, 'w'):
                #    continue
                params = [(name, x, sol_kind, all_ans_rule_probs[x])
                      for name, x, sol_kind in zip(target_names, target_puzzles, target_sol_kinds)]
                with tqdm(total=len(target_puzzles), desc="Solving") as pbar:
                    for name, ans, duration, count in map_fn(worker_fn, params):
                        if ans is not None:
                            solved.append((name, ans, duration, count))
                        else:
                            unsolved.append((name, ans, duration, count))
                        outputs[name] = (ans, duration, count)
                        pbar.update()

                        with open(out_file_jsonl, 'a+') as wf:
                            out_dict = target_puzzles_configs[target_names.index(name)]
                            if ans is not None:
                                out_dict['sols'].append(ans)
                            else:
                                out_dict['sols'].append('')
                            if 'sol_time' not in out_dict:
                                out_dict['sol_time'] = [duration]
                                out_dict['sol_tries'] = [count]
                            else:
                                out_dict['sol_time'].append(duration)
                                out_dict['sol_tries'].append(count)

                            wf.write(json.dumps(out_dict) + '\n')

                workers.close()
                workers.join()
            
                for (success, li) in [("Solved", solved), ("Unsolved", unsolved)]:
                    logger.info(f"--- {success}: name | solution | (time | count)")
                    for ch in li:
                        logger.info("'{}' | '{}' | ({}, {})".format(
                            ch[0], ch[1], ch[2], ch[3]))

                new_target_puzzles_configs = []
                for i, puzzle_config in enumerate(target_puzzles_configs):
                    out_dict = puzzle_config
                    #ans = outputs[puzzle_config['name']][0]
                    #if ans is not None:
                    #    out_dict['sols'].append(ans)
                    #else:
                    #    out_dict['sols'].append('')
                    #if 'sol_time' not in out_dict:
                    #    out_dict['sol_time'] = [outputs[puzzle_config['name']][1]]
                    #    out_dict['sol_tries'] = [outputs[puzzle_config['name']][2]]
                    #else:
                    #    out_dict['sol_time'].append(outputs[puzzle_config['name']][1])
                    #    out_dict['sol_tries'].append(outputs[puzzle_config['name']][2])
                    new_target_puzzles_configs.append(out_dict)

                logger.info(f"Solutions saved to '{out_file}'")
                with open(out_file, 'w') as wf:
                    json.dump(new_target_puzzles_configs, wf, indent=4)


            if args.do_regen:
                puzzles = [q for q, a in eval_QAs]
                regen_puzzles(args, model, tokenizer, puzzles, sol_kinds)
                model.do_solve()

            if args.do_eval:
                result = evaluate(args, model, eval_QAs, tokenizer, prefix=prefix)
                result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
                results.update(result)

    return results


if __name__ == "__main__":
    main()
