import logging
import torch
import os
import torch
import json
import csv
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from tython import RULES

logger = logging.getLogger(__name__)

RULE_EMBEDS_NAME = 'rule_embeds.tsv'
RULE_EMBEDS_VOCAB_NAME = 'rule_embeds_vocab.tsv'


class TransfomerSolver(nn.Module):
    def __init__(self, transformer_model, rules=RULES, hidden_size=300, max_num_children=10, dropout_prob=0.3, child_num_emb_size=10, tie_embeddings=True, fix_embeddings=True):
        super().__init__()
        self.transformer_model = transformer_model
        self.config = transformer_model.config
        self.tie_embeddings = tie_embeddings
        self.fix_embeddings = fix_embeddings
        self.num_rules = len(rules) + 1  # +1 for ROOT.
        self.config.solver_config = dict(hidden_size=hidden_size,
                                         num_rules=self.num_rules,
                                         max_num_children=max_num_children,
                                         dropout_prob=dropout_prob,
                                         child_num_emb_size=child_num_emb_size,
                                         tie_embeddings=tie_embeddings,
                                         fix_embeddings=fix_embeddings,
                                         )

        # Store rule vocab.
        self.rule_vocab = [rule.var_repr() for rule in rules]

        transformer_dim = transformer_model.config.hidden_size
        self.transformer_dim = transformer_dim

        self.rule_embeddings = nn.Embedding(self.num_rules, transformer_dim)
        self.rule_embeddings.weight.requires_grad = not self.fix_embeddings
        self.child_num_embeddings = nn.Embedding(max_num_children, child_num_emb_size)
        if not self.tie_embeddings:
            self.output_embeddings = nn.Embedding(self.num_rules, transformer_dim)
            self.output_embeddings.weight.requires_grad = not self.fix_embeddings

        self.linear1 = nn.Linear(2*transformer_dim + child_num_emb_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.output_proj = nn.Linear(transformer_dim, hidden_size)

        self.loss_reduction = 'mean'

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        model_to_save.config.save_pretrained(save_directory)
        weights_path = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save, weights_path)
        vocab_path = os.path.join(save_directory, 'rule_vocab.json')
        json.dump(self.rule_vocab, open(vocab_path, 'w'), indent=4)

    def set_rule_embeddings(self, emb_dir):
        file_path = os.path.join(emb_dir, RULE_EMBEDS_VOCAB_NAME)
        new_rule_vocab = []
        with open(file_path) as f:
            reader = csv.reader(f, delimiter='\t')
            new_rule_vocab = [line[0] for line in reader]
        file_path = os.path.join(emb_dir, RULE_EMBEDS_NAME)
        logger.info("Loaded embeddings from  %s", file_path)
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, line in tqdm(enumerate(reader), total=len(new_rule_vocab)):
                emb = [float(v) for v in line]
                assert len(emb) == self.transformer_dim
                assert i < self.num_rules, f'rule index out of bounds {i}'
                assert RULES[i].var_repr() == new_rule_vocab[i], f'mismatched rule {i}'
                if self.fix_embeddings:
                    self.rule_embeddings.weight[i][:] = torch.Tensor(emb)[:]
                    if not self.tie_embeddings:
                        self.output_embeddings.weight[i][:] = torch.Tensor(emb)[:]
                else:
                    for j, v in enumerate(emb):
                        self.rule_embeddings.weight[i,j] = v
                        if not self.tie_embeddings:
                            self.output_embeddings.weight[i,j] = v

        logger.info("Loaded %s embeddings", i)

    def forward(self, input_ids, parent_inds, child_nums, softmax_masks=None, labels=None, question_emb=None, **kwargs):
        if question_emb is not None:
            q_emb = question_emb
        else:
            outputs = self.transformer_model(input_ids, **kwargs)
            q_emb = outputs[1]
        parent_embeddings = self.rule_embeddings(parent_inds)
        child_num_embeddings = self.child_num_embeddings(child_nums)
        x = torch.cat((q_emb, parent_embeddings, child_num_embeddings), dim=-1)
        x = self.dropout(x)

        # [batch_size] x [2*trasformer_hidden_size + child_num_emb_size]
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.layer_norm(x)

        # [batch_size] x [hidden_size]
        x = self.linear2(x)
        x = F.gelu(x)
        x = self.layer_norm(x)

        # [num_rules] x [hidden_size]
        if self.tie_embeddings:
            y = self.output_proj(self.rule_embeddings.weight)
        else:
            y = self.output_proj(self.output_embeddings.weight)

        # [batch_size] x [num_rules]
        predictions = torch.matmul(x, y.T)

        # Apply mask.
        #predictions = predictions * softmax_masks.to_dense() - 1e18 * (1 - softmax_masks.to_dense())
        predictions = predictions * softmax_masks - 1e18 * (1 - softmax_masks)

        outputs = (predictions,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(predictions.view(-1, self.num_rules), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

    def get_output_embeddings(self):
        if self.tie_embeddings:
            return self.rule_embeddings
        else:
            return self.output_embeddings


class TreeGenHead(nn.Module):
    def __init__(self, transformer_dim=768, parent_proj_size=300, hidden_size=300, child_num_emb_size=10,  dropout_prob=0.3):
        super().__init__()
        self.parent_proj = nn.Linear(transformer_dim, parent_proj_size)
        self.linear1 = nn.Linear(transformer_dim + parent_proj_size +  child_num_emb_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.output_proj = nn.Linear(transformer_dim, hidden_size)

    def forward(self, q_emb, parent_embeddings, child_num_embeddings, output_embeddings):
        parent_embeddings = self.parent_proj(parent_embeddings)
        x = torch.cat((q_emb, parent_embeddings, child_num_embeddings), dim=-1)
        x = self.dropout(x)

        # [batch_size] x [trasformer_hidden_size + parent_proj_size + child_num_emb_size]
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.layer_norm(x)

        # [batch_size] x [hidden_size]
        x = self.linear2(x)
        x = F.gelu(x)
        x = self.layer_norm(x)

        # [num_rules] x [hidden_size]
        y = self.output_proj(output_embeddings)

        # [batch_size] x [num_rules]
        predictions = torch.matmul(x, y.T)
        return predictions


class TransfomerTreeGenerator(nn.Module):
    def __init__(self, transformer_model, rules=RULES, parent_proj_size=300, hidden_size=300, max_num_children=10, dropout_prob=0.3, child_num_emb_size=10, tie_embeddings=True, fix_embeddings=True):
        super().__init__()
        self.transformer_model = transformer_model
        self.config = transformer_model.config
        self.tie_embeddings = tie_embeddings
        self.fix_embeddings = fix_embeddings
        self.num_rules = len(rules) + 1  # +1 for ROOT.
        self.config.solver_config = dict(hidden_size=hidden_size,
                                         num_rules=self.num_rules,
                                         max_num_children=max_num_children,
                                         dropout_prob=dropout_prob,
                                         child_num_emb_size=child_num_emb_size,
                                         tie_embeddings=tie_embeddings,
                                         fix_embeddings=fix_embeddings,
                                         parent_proj_size=parent_proj_size,
                                         )

        # Store rule vocab.
        self.rule_vocab = [rule.var_repr() for rule in rules]

        transformer_dim = transformer_model.config.hidden_size
        self.transformer_dim = transformer_dim

        self.rule_embeddings = nn.Embedding(self.num_rules, transformer_dim)
        self.rule_embeddings.weight.requires_grad = not self.fix_embeddings
        self.child_num_embeddings = nn.Embedding(max_num_children, child_num_emb_size)
        if not self.tie_embeddings:
            self.output_embeddings = nn.Embedding(self.num_rules, transformer_dim)
            self.output_embeddings.weight.requires_grad = not self.fix_embeddings

        self.puzzle_regenerator = TreeGenHead(transformer_dim=transformer_dim,
                                              parent_proj_size=parent_proj_size,
                                              hidden_size=hidden_size,
                                              child_num_emb_size=child_num_emb_size,
                                              dropout_prob=dropout_prob
                                            )

        self.puzzle_solver = TreeGenHead(transformer_dim=transformer_dim,
                                         parent_proj_size=parent_proj_size,
                                         hidden_size=hidden_size,
                                         child_num_emb_size=child_num_emb_size,
                                         dropout_prob=dropout_prob
                                        )
        self.regen_mode = False
        self.loss_reduction = 'mean'

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        model_to_save.config.save_pretrained(save_directory)
        weights_path = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save, weights_path)
        vocab_path = os.path.join(save_directory, 'rule_vocab.json')
        json.dump(self.rule_vocab, open(vocab_path, 'w'), indent=4)

    def set_rule_embeddings(self, emb_dir):
        file_path = os.path.join(emb_dir, RULE_EMBEDS_VOCAB_NAME)
        new_rule_vocab = []
        with open(file_path) as f:
            reader = csv.reader(f, delimiter='\t')
            new_rule_vocab = [line[0] for line in reader]
        file_path = os.path.join(emb_dir, RULE_EMBEDS_NAME)
        logger.info("Loaded embeddings from  %s", file_path)
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, line in tqdm(enumerate(reader), total=len(new_rule_vocab)):
                emb = [float(v) for v in line]
                assert len(emb) == self.transformer_dim
                assert i < self.num_rules, f'rule index out of bounds {i}'
                assert RULES[i].var_repr() == new_rule_vocab[i], f'mismatched rule {i}'
                if self.fix_embeddings:
                    self.rule_embeddings.weight[i][:] = torch.Tensor(emb)[:]
                    if not self.tie_embeddings:
                        self.output_embeddings.weight[i][:] = torch.Tensor(emb)[:]
                else:
                    for j, v in enumerate(emb):
                        self.rule_embeddings.weight[i,j] = v
                        if not self.tie_embeddings:
                            self.output_embeddings.weight[i,j] = v

        logger.info("Loaded %s embeddings", i)

    def forward(self, input_ids, parent_inds, child_nums, softmax_masks=None, labels=None, question_emb=None, **kwargs):
        if question_emb is not None:
            q_emb = question_emb
        else:
            outputs = self.transformer_model(input_ids, **kwargs)
            q_emb = outputs[1]
        parent_embeddings = self.rule_embeddings(parent_inds)
        child_num_embeddings = self.child_num_embeddings(child_nums)

        if self.tie_embeddings:
            output_embeddings = self.rule_embeddings.weight
        else:
            output_embeddings = self.output_embeddings.weight

        if self.regen_mode:
            predictions = self.puzzle_regenerator(q_emb, parent_embeddings, child_num_embeddings, output_embeddings)
        else:
            predictions = self.puzzle_solver(q_emb, parent_embeddings, child_num_embeddings, output_embeddings)

        outputs = (predictions,)
        if labels is not None:
            if hasattr(self, 'loss_reduction'):
                loss_reduction = self.loss_reduction
            else:
                loss_reduction = 'mean'

            loss_fct = CrossEntropyLoss(reduction=loss_reduction)
            loss = loss_fct(predictions.view(-1, self.num_rules), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

    def get_output_embeddings(self):
        if self.tie_embeddings:
            return self.rule_embeddings
        else:
            return self.output_embeddings

    def do_regen(self):
        self.regen_mode = True

    def do_solve(self):
        self.regen_mode = False

    def get_mode(self):
        if self.regen_mode:
            return 'regen'
        else:
            return 'solve'
