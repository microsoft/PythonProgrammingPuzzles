import logging
import os
import numpy as np
import glob
import json
from dataclasses import dataclass
from typing import List, Dict, Callable, Tuple, Generator, Set, Sequence, Optional, Union
import torch
from tqdm import tqdm

from transformers import DataProcessor, PreTrainedTokenizer

from tython import Program, TastNode, _RULES_BY_KIND, Rule, RULES
from challenges import Challenge, Solution, SolverSolution, verify_solutions


logger = logging.getLogger(__name__)


def get_QAs(challenges_path, max_ticks=1000, min_tries=0, verify_sols=True, all_instances=False):
    # Create examples
    challenges_files = glob.glob(challenges_path)
    logger.info(f"Loading puzzles and solutions from {challenges_files}")
    challenge_configs = []
    seen_challenges = set()
    for f_name in challenges_files:
        chs = json.load(open(f_name, 'r'))
        for ch in chs:
            if not all_instances and not ch['name'].endswith('_0'):
                continue
            if ch['name'] not in seen_challenges:
                challenge_configs.append(ch)
                seen_challenges.add(ch['name'])
            seen_challenges.add(ch['name'])

    # Parse challenges.
    challenges = {}
    for config in challenge_configs:
        #ch = Challenge(config, max_ticks=max_ticks)
        ch = Challenge(config)
        if ch.prog is not None:
            challenges[ch.name] = ch

    if len(challenge_configs) == 0:
        logger.warning("Couldn't parse any puzzles.")
        return [], []

    logger.info("Successfully parsed {} ({:.2f}%) out of {} puzzles.".format(
        len(challenges.keys()),
        100 * len(challenges.keys()) / len(challenge_configs),
        len(challenge_configs)))

    if verify_sols:
        verify_solutions(challenges.values())
    # zzz TODO: only last solutions
    #QAs = [(ch.f_str, s.prog) for ch in challenges.values()
    #            for s in ch.gold_solutions if s.prog is not None and (s.count is None or s.count >= min_tries)]
    #QAs = [(ch.f_str, s.prog) for ch in challenges.values() if len(ch.gold_solutions) > 0
    #            for s in [ch.gold_solutions[-1]] if s.prog is not None]
    QAs = []
    for ch in challenges.values():
        added_one_sol = False
        if len(ch.gold_solutions) == 0:
            continue
        for s in reversed(ch.gold_solutions):
            if not added_one_sol and s.prog is not None:
                QAs.append((ch.f_str, s.prog))
                added_one_sol = True


    logger.info(f"collected {len(QAs)} QAs")

    sol_kinds = [ch.sol_kind for ch in challenges.values()]

    return QAs, sol_kinds


def get_Q_trees(challenges_path, max_ticks=1000):
    # Create examples
    challenges_files = glob.glob(challenges_path)
    logger.info(f"Loading puzzles from {challenges_files}")
    challenge_configs = []
    seen_challenges = set()
    for f_name in challenges_files:
        chs = json.load(open(f_name, 'r'))
        for ch in chs:
            if ch['name'] not in seen_challenges:
                challenge_configs.append(ch)
                seen_challenges.add(ch['name'])
            seen_challenges.add(ch['name'])

    # Parse challenges.
    challenges = {}
    for config in challenge_configs:
        ch = Challenge(config, max_ticks=max_ticks)
        if ch.prog is not None:
            challenges[ch.name] = ch

    if len(challenge_configs) == 0:
        logger.warning("Couldn't parse any puzzles.")
        return [], []

    logger.info("Successfully parsed {} ({:.2f}%) out of {} challenges.".format(
        len(challenges.keys()),
        100 * len(challenges.keys()) / len(challenge_configs),
        len(challenge_configs)))

    QAs = [(ch.f_str, ch.prog) for ch in challenges.values()]
    sol_kinds = [ch.sol_kind for ch in challenges.values()]

    return QAs, sol_kinds


def get_Q_augs(challenges_path, max_ticks=1000):
    challenges_files = glob.glob(challenges_path)
    logger.info(f"Loading puzzles and augmentations from {challenges_files}")
    if len(challenges_files) == 0:
        return []

    challenge_configs = []
    seen_challenges = set()
    for f_name in challenges_files:
        chs = json.load(open(f_name, 'r'))
        for ch in chs:
            if ch['name'] not in seen_challenges:
                challenge_configs.append(ch)
                seen_challenges.add(ch['name'])
            seen_challenges.add(ch['name'])

    # Parse augmentations.
    QAs = []
    for config in challenge_configs:
        try:
            aug = Program(config['aug'])
            QAs.append((config["sat"], aug))
        except Exception as e:
            logger.error(f"Exception parsing augmentation '{config['name']}' ('{config['aug']}'): {e}")

    logger.info("Successfully parsed {} ({:.2f}%) out of {} puzzles and augmentations.".format(
        len(QAs),
        100 * len(QAs) / len(challenge_configs),
        len(challenge_configs)))

    return QAs


@dataclass
class InputExample:
    """
    A single training/test example.

    Args:
        guid: Unique id for the example.
        puzzle_str: string. The untokenized text of the puzzle.
        parent_ind: int. The index of the parent rule from the RULES list.
        child_num: int. The index of the child to predict for.
        label: (Optional) int. The rule index (in RULES list) for the child_num of the parent rule.
        target_kind: (Optional) kind. The kind of the rule we are looking for.
            parent in the solution of the puzzle.
    """

    guid: str
    puzzle_str: str
    parent_ind: int
    child_num: int
    label: Optional[int] = None
    target_kind: Optional = -1 # -1 is the default since None is a valid kind.

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        softmax_mask: Mask for softmax. Only include children of the current rule.
        parent_ind: The index of the parent rule from the RULES list.
        child_num: The index of the child to predict for.
        label: (Optional) Label corresponding to the input.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    softmax_mask: torch.int = None
    parent_ind: int = int
    child_num: int = None
    label: Optional[int] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def traverse_ans_tree(prog: Program):
    '''
    Collect all rules with their features.
    '''
    # Initiate with root (parent_ind=len(RULES)
    rules = [[prog.tree.rule.index, len(RULES), 0]]
    queue = [prog.tree]
    while queue:
        node = queue.pop()
        if not isinstance(node, TastNode) or not _RULES_BY_KIND.get(node.nt):
            continue
        parent_ind = node.rule.index
        for num_child, child in enumerate(node.children):
            if not hasattr(child, 'nt') or child.nt == 'LIT':
                continue
            rules.append([child.rule.index, parent_ind, num_child])
            queue.append(child)

    return rules


class PuzzleProcessor(DataProcessor):
    def __init__(self):
        super().__init__()

    def get_train_examples(self, QAs: List[Tuple[str, Program]]):
        examples = []
        targets = []
        i = 0
        for q_str, a in tqdm(QAs):
            rules = traverse_ans_tree(a)
            for r in rules:
                target_rule = r[0]
                parent_ind, child_num = r[1:]

                targets.append(target_rule)

                guid = "%s-%s" % ("train", i)
                i += 1

                examples.append(
                    InputExample(guid=guid,
                                 puzzle_str=q_str,
                                 parent_ind=parent_ind,
                                 child_num=child_num,
                                 label=target_rule,
                                 target_kind=RULES[target_rule].nt
                                 ))
        return examples

    def get_test_examples(self, QAs: List[Tuple[str, Program]]):
        '''
        For now, assume there is gold answer.
        '''
        return self.get_train_examples(QAs)

    def get_questions(self, QAs: List[Tuple[str, Program]]):
        '''
        Get an InputExample per question (ignoring the solution tree)
        '''
        examples = []
        i = 0
        for q_str, a in tqdm(QAs):
            guid = "%s-%s" % ("question", i)
            i += 1
            examples.append(
                InputExample(guid=guid,
                             puzzle_str=q_str,
                             parent_ind=-1,
                             child_num=-1,
                             label=None))
        return examples

    def get_labels(self):
        """See base class."""
        return list(range(len(RULES)))


def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    label_list=None,
    output_mode='classification',
    loud=False,
    rules_by_kind=_RULES_BY_KIND
):
    """
    Convert ``InputExamples`` to ``InputFeatures``

    Args:
        examples: List of ``InputExamples``
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        A list of ``InputFeatures`` which can be fed to the model.

    """
    if max_length is None:
        max_length = tokenizer.max_len

    processor = PuzzleProcessor()
    if label_list is None:
        label_list = processor.get_labels()

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    parent_inds = [example.parent_ind for example in examples]
    child_nums = [example.child_num for example in examples]
    labels = [label_from_example(example) for example in examples]
    target_kinds = [example.target_kind for example in examples]
    output_masks = []
    for rule_ind, target_kind in zip(labels, target_kinds):
        mask = np.zeros(len(RULES) + 1)  # +1 for root (which is always masked)
        if target_kind is None or target_kind != -1:
            mask_inds = [r.index for r in rules_by_kind[target_kind]]
            for ind in mask_inds:
                mask[ind] = 1
            mask = torch.IntTensor(mask).to_sparse()
            output_masks.append(mask)
        elif rule_ind is not None:
            mask_inds = [r.index for r in rules_by_kind[RULES[rule_ind].nt]]
            for ind in mask_inds:
                mask[ind] = 1
            assert rule_ind in mask_inds
            mask = torch.IntTensor(mask).to_sparse()
            output_masks.append(mask)
        else:
            output_masks.append(None)

    batch_encoding = tokenizer.batch_encode_plus(
        [example.puzzle_str for example in examples], max_length=max_length, pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs,
                                softmax_mask=output_masks[i],
                                parent_ind=parent_inds[i],
                                child_num=child_nums[i],
                                label=labels[i])
        features.append(feature)

    if loud:
        for i, example in enumerate(examples[:5]):
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("features: %s" % features[i])

    return features
