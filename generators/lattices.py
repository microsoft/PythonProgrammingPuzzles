"""Lattice problems with and without noise"""

from puzzle_generator import PuzzleGenerator
from typing import List


# See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle to learn about adding puzzles


class LearnParity(PuzzleGenerator):
    """Parity learning (Gaussian elimination)

    The canonical solution to this 
    [Parity learning problem](https://en.wikipedia.org/w/index.php?title=Parity_learning)
    is to use 
    [Gaussian Elimination](https://en.wikipedia.org/w/index.php?title=Gaussian_elimination).

    The vectors are encoded as binary integers for succinctness.
    """

    @staticmethod
    def sat(inds: List[int], vecs=[169, 203, 409, 50, 37, 479, 370, 133, 53, 159, 161, 367, 474, 107, 82, 447, 385]):
        """
        Parity learning: Given binary vectors in a subspace, find the secret set S of indices such that:
        $\\sum_{i \in S} x_i = 1 (mod 2)$
        """
        return all(sum((v >> i) & 1 for i in inds) % 2 == 1 for v in vecs)

    @staticmethod
    def sol(vecs):
        # Gaussian elimination
        d = 0  # decode vectors into arrays
        m = max(vecs)
        while m:
            m >>= 1
            d += 1
        vecs = [[(n >> i) & 1 for i in range(d)] for n in vecs]
        ans = []
        pool = [[0] * (d + 1) for _ in range(d)] + [v + [1] for v in vecs]
        for i in range(d):
            pool[i][i] = 1

        for i in range(d):  # zero out bit i
            for v in pool[d:]:
                if v[i] == 1:
                    break
            if v[i] == 0:
                v = pool[i]
            assert v[i] == 1  # found a vector with v[i] = 1, subtract it off from those with a 1 in the ith coordinate
            w = v[:]
            for v in pool:
                if v[i] == 1:
                    for j in range(d + 1):
                        v[j] ^= w[j]

        return [i for i in range(d) if pool[i][-1]]

    @staticmethod
    def rand_parity_problem(rand, d=63):
        secret = rand.sample(range(d), d // 2)
        num_vecs = d + 9
        vecs = [[rand.randrange(2) for _ in range(d)] for i in range(num_vecs)]
        for v in vecs:
            v[secret[0]] = (1 + sum([v[i] for i in secret[1:]])) % 2
        vecs = [sum(1 << i for i, b in enumerate(v) if b) for v in vecs]  # encode into ints
        return vecs

    def gen(self, target_num_instances):
        vecs = self.rand_parity_problem(self.random, d=63)
        self.add(dict(vecs=vecs), multiplier=10)

    def gen_random(self):
        d = self.random.randrange(2, self.random.choice([5, 10, 20, 100]))
        vecs = self.rand_parity_problem(
            self.random,
            d=d,
        )
        self.add(dict(vecs=vecs), multiplier=10 if d > 9 else 1)


class LearnParityWithNoise(PuzzleGenerator):
    """Learn parity with noise (*unsolved*)

    The fastest known algorithm to this
    [Parity learning problem](https://en.wikipedia.org/w/index.php?title=Parity_learning)
    runs in time $2^(d/(log d))$

    The example puzzle has small dimension so is easily solvable, but other instances are much harder.
    """

    @staticmethod
    def sat(inds: List[int], vecs=[26, 5, 32, 3, 15, 18, 31, 13, 24, 25, 34, 5, 15, 24, 16, 13, 0, 27, 37]):
        """
        Learning parity with noise: Given binary vectors, find the secret set $S$ of indices such that, for at least
        3/4 of the vectors, $$sum_{i \in S} x_i = 1 (mod 2)$$
        """
        return sum(sum((v >> i) & 1 for i in inds) % 2 for v in vecs) >= len(vecs) * 3 / 4

    @staticmethod
    def sol(vecs):
        # brute force
        d = 0  # decode vectors into arrays
        m = max(vecs)
        while m:
            m >>= 1
            d += 1
        vecs = [[(n >> i) & 1 for i in range(d)] for n in vecs]

        import random
        rand = random.Random(0)
        target = (len(vecs) * 3) // 4
        max_attempts = 10 ** 5
        for _ in range(max_attempts):
            ans = [i for i in range(d) if rand.randrange(2)]
            if sum(sum(v[i] for i in ans) % 2 for v in vecs) >= len(vecs) * 3 / 4:
                return ans

    @staticmethod
    def rand_parity_problem(rand, d=63):
        secret = rand.sample(range(d), d // 2)
        num_vecs = 2 * d + 5
        vecs = [[rand.randrange(2) for _ in range(d)] for i in range(num_vecs)]
        for v in vecs:
            v[secret[0]] = (1 + sum([v[i] for i in secret[1:]])) % 2
        mistakes = rand.sample(vecs, int(len(vecs) * rand.random() * 1 / 4))
        for v in mistakes:
            v[secret[0]] ^= 1  # flip bit in mistakes
        vecs = [sum(1 << i for i, b in enumerate(v) if b) for v in vecs]  # encode into ints
        return vecs

    def gen(self, target_num_instances):
        vecs = self.rand_parity_problem(self.random, d=63)
        self.add(dict(vecs=vecs), test=False, multiplier=1000)

    def gen_random(self):
        d = self.random.randrange(2, self.random.choice([11, 100]))  # number of dimensions
        vecs = self.rand_parity_problem(
            self.random,
            d=d
        )
        self.add(dict(vecs=vecs), test=d < 19, multiplier=1000 if d > 40 else 30 if d >= 19  else 1)

    if __name__ == "__main__":
        PuzzleGenerator.debug_problems()
