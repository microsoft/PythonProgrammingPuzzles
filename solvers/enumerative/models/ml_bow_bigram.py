from typing import List, Dict, Callable, Tuple, Generator, Set, Sequence
from collections import defaultdict
import numpy as np
from copy import deepcopy
import sklearn
import sklearn.linear_model
import sklearn.ensemble
import sklearn.neighbors
import sklearn.tree
from scipy import sparse
from tqdm import tqdm
import time
import logging

from tython import Program, TastNode, _RULES_BY_KIND, Rule, RULES, nt
from models.model import CandidateGenerator, reachable_rules_by_kind
from models import RegisterModel
from challenges import extract_constants

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@RegisterModel("ml_bow_bigram")
class BOWCondParentLRModel(CandidateGenerator):
    '''
    learn rule weights with logistic regression.
    Features are: BOW for question and parent rule + child ind
    '''

    def __init__(self, ml_model='lr') -> None:
        '''
        Learn model per kind.
        featurization: feature extraction method.
        '''
        super().__init__()
        self.vocabs = _RULES_BY_KIND
        self.ml_model = ml_model

        if ml_model == 'lr':
            self.sk_model = sklearn.linear_model.LogisticRegression(penalty='l2')
        elif ml_model == 'rf':
            self.sk_model = sklearn.ensemble.RandomForestClassifier()
        elif ml_model == 'knn':
            self.sk_model = sklearn.neighbors.KNeighborsClassifier()
        elif ml_model == 'dt':
            self.sk_model = sklearn.tree.DecisionTreeClassifier()
        else:
            raise Exception("Unknown ml model %s", ml_model)

    def featurize_question(self, prog: Program):
        '''
        Convert program to features with bow (bag of words/ rules)
        '''

        qf = np.zeros(len(RULES), dtype=int)

        # Get all rules in prog.

        queue = [prog.tree]
        while queue:
            node = queue.pop()
            qf[node.rule.index] += 1
            for child in node.children:
                queue.append(child)

        return sparse.csr_matrix(qf)

    def add_parent_features(self, qf: np.array, parent_rule: int, child_num: int):
        # Assume max 10 children
        assert child_num < 10
        parent_feature = np.zeros(len(RULES) + 1 + 10, dtype=int)
        parent_feature[parent_rule] = 1
        parent_feature[child_num + len(RULES) + 1] = 1

        # features = np.concatenate((qf, parent_feature))
        parent_feature = sparse.csr_matrix(parent_feature)
        features = sparse.hstack([qf, parent_feature])
        return features

    def traverse_ans_tree(self, prog: Program):
        '''
        Collect all rules with their features.
        '''
        # Initiate with root, parent_ind=len(RULES)
        rules = [[prog.tree.rule.index, len(RULES), 0]]
        queue = [prog.tree]
        while queue:
            node = queue.pop()
            parent_ind = node.rule.index
            for num_child, child in enumerate(node.children):
                rules.append([child.rule.index, parent_ind, num_child])
                queue.append(child)

        return rules

    def learn(self, QAs: List[Tuple[Program, Program]]) -> None:
        '''
        Optimize the ML models with the given examples.
        QAs: list of QAs for building vocab.
        '''
        # xs = None
        xs = []
        targets = []
        logger.info("Creating features from puzzle solution pairs")
        for q, a in tqdm(QAs):
            qf = self.featurize_question(q)
            for (target_rule, parent_rule, child_num) in self.traverse_ans_tree(a):
                input_features = self.add_parent_features(qf, parent_rule, child_num)

                # if xs is None:
                #     xs = sparse.csr_matrix(input_features)
                # else:
                #     xs = sparse.vstack([xs, input_features])
                xs.append(sparse.csr_matrix(input_features))
                targets.append(target_rule)

        logger.info(f"Collected {len(targets)} rules")
        xs = sparse.vstack(xs)
        self.sk_model.fit(xs, targets)

    def get_candidates(self, q: Program) -> Dict:
        '''
        Predict candidates for each question
        '''
        st_time = time.time()
        qf = self.featurize_question(q)

        consts = extract_constants(q)

        # Our question is always a function that returns a bool.
        sol_type_annotation = q.tree.children[0].children[1].children[1]
        sol_kind = nt.type2nt(eval(sol_type_annotation.src()))

        rules_by_kind = _RULES_BY_KIND # zzz reachable_rules_by_kind(sol_kind, consts)
        rules_by_kind_sets = {k: {r.index for r in rules} for k, rules in rules_by_kind.items()}

        ans = {}

        times1 = []
        times2 = []
        for parent_kind in rules_by_kind:
            # None for root.
            for r in rules_by_kind[parent_kind] + ([None] if parent_kind == sol_kind else []):
                # Don't include parents that we won't reach anyway (a.k.a massive pruning).
                if r is not None and r.index not in self.sk_model.classes_:
                    continue
                if r and r.name == "COPY":
                    assert len(r.kids) == 1
                    if r.nt in consts:
                        p = 1/len(consts[r.nt])
                        relevant_consts = [(p, n) for n in consts[r.nt]]
                    else:
                        relevant_consts = []
                    ans[r] = [(relevant_consts, [])]
                    continue

                ans[r] = []
                for child_num, kind in enumerate([sol_kind] if r is None else r.kids):
                    tik2 = time.time()
                    class_mask = np.array([c in rules_by_kind_sets[kind] for c in self.sk_model.classes_])
                    assert child_num < 9
                    # child_num = min(child_num, 9)
                    if r is None:
                        parent_rule = len(RULES)
                    else:
                        parent_rule = r.index

                    input_features = self.add_parent_features(qf, parent_rule, child_num)
                    tik1 = time.time()
                    # TODO: create batches to reduce run time.
                    rule_probs = self.sk_model.predict_proba(input_features.reshape(1, -1))[0]
                    tok1 = time.time() - tik1
                    times1.append(tok1)

                    # Mask out predictions to rules of different kind
                    sum_probs = np.sum(rule_probs * class_mask)
                    if sum_probs > 0:
                        rule_probs = rule_probs * class_mask / sum_probs
                        rule_probs = [(rule_probs[i], RULES[self.sk_model.classes_[i]]) for i in
                                      rule_probs.nonzero()[0]]
                        assert all(r.nt == kind for _, r in rule_probs)
                    else:
                        rule_probs = []



                    ans[r].append(([], rule_probs))
                    tok2 = time.time() - tik2
                    times2.append(tok2)

        # print(np.mean(times1))
        # print(np.mean(times2))
        end_time = time.time() - st_time
        logger.debug("Get candidates took {:.2f}s".format(end_time))

        return ans

    def get_likelihood(self, q: TastNode, ans: Program) -> float:
        '''
        Get the probaility of the given answer program to the question.
        '''
        ans_rules = self.traverse_ans_tree(ans)
        qf = self.featurize_question(q)
        rule_probs = []
        for r in ans_rules:
            target_rule_ind = r[0]
            parent_rule, child_num = r[1:]
            input_features = self.add_parent_features(qf, parent_rule, child_num)
            out_probs = self.sk_model.predict_proba(input_features.reshape(1, -1))[0]
            target_prob = out_probs[np.where(self.sk_model.classes_ == target_rule_ind)].item() \
                if target_rule_ind in self.sk_model.classes_ else 0
            rule_probs.append(target_prob)

        return np.prod(rule_probs)
