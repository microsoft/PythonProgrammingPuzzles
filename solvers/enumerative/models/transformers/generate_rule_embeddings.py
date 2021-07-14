import argparse
import glob
import logging
import csv
import os
import random
import json
import string

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)

from dataset_processor import PuzzleProcessor, InputExample, InputFeatures, convert_examples_to_features

from tython import Program, TastNode, _RULES_BY_KIND, Rule, RULES

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def save_embeds(file_path, embeds, vocab_file):
    n_tokens, emb_dim = embeds.shape
    with open(file_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for i in range(n_tokens):
            writer.writerow(list(embeds[i,:]))

    with open(vocab_file, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for i in range(n_tokens):
            desc = generate_rule_string(RULES[i])
            writer.writerow([desc])

    return


# First attempt to get random values for rules.

#thing_list = ['x', 'y', 'z', 'a', 'b', 'i', 'j', 'k']
#bool_list = ['True', 'False']
#
#
#kind_gens = {
#    'int': lambda: random.choice([str(random.randint(-1000, 1000)), 'i', 'j']) ,
#    'thing': lambda: random.choice(thing_list),
#    'float' : lambda: round(random.random() * random.randint(-100,100), 3),
#    'bool': lambda: random.choice(bool_list),
#    'range': lambda: random.choice(bool_list),
#    'str': lambda: '"{}"'.format(''.join([random.choice(string.ascii_uppercase + string.ascii_lowercase +string.digits) for _ in range(random.randint(1,10))])),
#    'range': lambda: random.choice(['range({})'.format(random.randint(-100,100)),
#                                      'range({},{})'.format(random.randint(-100,100), random.randint(-100,100)),
#                                      'range({},{},{})'.format(random.randint(-100,100), random.randint(-100,100), random.randint(-10,10))]),
#    'none': lambda: 'None',
#}
#
#
#special_gens = {
#    'Callable': lambda: 'func({})',
#    'Tuple': lambda: '({})',
#    'Set': lambda: '{{}}',
#}
#
#
#def gen_random_kind(kind):
#    '''
#    Given a string or list of kind, returns a string with an instance of that kind
#    '''
#    if isinstance(kind, (tuple, list)):
#        var_instances = []
#        for k in kind[1:]:
#            var_instances.append(gen_random_kind(k))
#
#        base_str = special_gens[kind[0]]()
#        combine_str = base_str.format(', '.join([str(i) for i in var_instances]))
#        return combine_str
#    else:
#        return kind_gens[kind]()
#
#
#def open_kind(kind):
#    '''
#    return list of the names of kinds that create it.
#    '''
#    if isinstance(kind, (tuple, list)):
#        return [open_kind(k) for k in kind]
#    else:
#        return kind.__name__ if hasattr(kind, "__name__") else kind._name if hasattr(kind, "_name") else str(kind)
#
#
#def generate_rule_strings(rule, max_num = 100):
#    children_kinds = []
#    for kind in rule.children:
#        children_kinds.append(open_kind(kind))
#
#    rule_strings = []
#    for _ in range(max_num):
#        children_vals = []
#        for k in children_kinds:
#            children_vals.append(gen_random_kind(k))
#
#        desc = rule.to_python(*children_vals)
#        if desc not in rule_strings:
#            rule_strings.append(desc)
#
#    if 'Callable' in str(rule):
#    return rule_strings

def generate_rule_string(rule):
    return rule.var_repr()

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
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
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Process device: %s, n_gpu: %s",
        device,
        args.n_gpu,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    args.model_type = config.model_type
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir,
    )
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )

    model.to(args.device)
    logger.info("Generating embeddings for %s rules", len(RULES))

    examples = []
    for i, rule in enumerate(RULES):
        desc = generate_rule_string(rule)
        examples.append(
            InputExample(guid=i,
                         puzzle_str=desc,
                         parent_ind=None,
                         child_num=None,
                         label=None))

    features = convert_examples_to_features(examples, tokenizer, max_length=args.max_seq_length)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.int)
    dataset = TensorDataset(all_input_ids, all_attention_mask)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    hidden_size = model.config.hidden_size
    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    rule_embeddings = np.zeros((len(RULES), hidden_size))
    for b, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating", total=len(dataset)/args.eval_batch_size):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            mask = batch[1]
            inputs = {"input_ids": batch[0], "attention_mask": mask}
            outputs, _ = model(**inputs)

            # Get Average of token representations
            mask_ex = mask.unsqueeze(-1).expand_as(outputs)
            y = (outputs * mask_ex).sum(1)
            y = y / mask_ex.sum(1)

            y = y.detach().cpu().numpy()
            for i, emb in enumerate(y):
                rule_embeddings[b * args.eval_batch_size + i, :] = emb

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "rule_embeds.tsv")
    vocab_file = os.path.join(args.output_dir, "rule_embeds_vocab.tsv")
    save_embeds(out_file, rule_embeddings, vocab_file)
    logger.info("Saved embeddings to %s", out_file)

if __name__ == "__main__":
    main()
