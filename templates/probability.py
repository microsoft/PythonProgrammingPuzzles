"""Probability problems"""

from problems import Problem
from typing import List


# Hint: subclass Problem.Debug for quick testing. Run make_dataset.py to make the dataset
# See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle for more info


class BirthdayParadox(Problem):
    """Find `n` such that the probability of two people having the same birthday in a group of `n` is near `1/2`.
    The year length is year_len (365 is earth, while Neptune year is 60,182)
    See [Birthday Problem](https://en.wikipedia.org/wiki/Birthday_problem (Mathematical Problems category))"""

    @staticmethod
    def sat(n: int, year_len=365):
        prob = 1.0
        for i in range(n):
            prob *= (year_len - i) / year_len
        return (prob - 0.5) ** 2 <= 1/year_len

    @staticmethod
    def sol(year_len):
        n = 1
        distinct_prob = 1.0
        best = (0.5, 1)  # (difference between probability and 1/2, n)
        while distinct_prob > 0.5:
            distinct_prob *= (year_len - n) / year_len
            n += 1
            best = min(best, (abs(0.5 - distinct_prob), n))

        return best[1]

    def safe_add(self, **inputs):
        if self.sat(self.sol(**inputs), **inputs):
            self.add(inputs)

    def gen(self, target_num_instances):
        self.safe_add(year_len=60182)  # Neptune year!
        year_len = 2
        while len(self.instances) < target_num_instances:
            self.safe_add(year_len=year_len)
            year_len += 1



class BirthdayParadoxMonteCarlo(BirthdayParadox):
    """A slower, Monte Carlo version of the above Birthday Paradox problem."""

    @staticmethod
    def sat(n: int, year_len=365):
        import random
        random.seed(0)
        K = 1000  # number of samples
        prob = sum(len({random.randrange(year_len) for i in range(n)}) < n for j in range(K)) / K
        return (prob - 0.5) ** 2 <= year_len





class BallotProblem(Problem):
    """Suppose a list of m 1's and n -1's are permuted at random. What is the probability that
    all of the cumulative sums are positive?
    The goal is to find `m` and `n` that make the probability of the ballot problem close to a
    specific probability `target_prob`.
    See the [Wikipedia article](https://en.wikipedia.org/wiki/Bertrand%27s_ballot_theorem) or
    or  [Addario-Berry L., Reed B.A. (2008) Ballot Theorems, Old and New. In: Gyori E., Katona G.O.H., Lovász L.,
    Sági G. (eds) Horizons of Combinatorics. Bolyai Society Mathematical Studies, vol 17.
    Springer, Berlin, Heidelberg.](https://doi.org/10.1007/978-3-540-77200-2_1)
    """

    @staticmethod
    def sat(counts: List[int], target_prob=0.5):
        m, n = counts  # m = num 1's, n = num -1's
        probs = [1.0] + [0.0] * n  # probs[n] is probability for current m, starting with m = 1
        for i in range(2, m + 1):  # compute probs using dynamic programming for m = i
            old_probs = probs
            probs = [1.0] + [0.0] * n
            for j in range(1, min(n + 1, i)):
                probs[j] = (
                        j / (i + j) * probs[j - 1]  # last element is a -1 so use probs
                        +
                        i / (i + j) * old_probs[j]  # last element is a 1 so use old_probs, m = i - 1
                )
        return abs(probs[n] - target_prob) < 1e-6

    @staticmethod
    def sol(target_prob):
        for m in range(1, 10000):
            n = round(m * (1 - target_prob) / (1 + target_prob))
            if abs(target_prob - (m - n) / (m + n)) < 1e-6:
                return [m, n]

    def gen_random(self):
        m = self.random.randrange(1, self.random.choice([10, 100, 200, 300, 400, 500, 1000]))
        n = self.random.randrange(1, m + 1)
        target_prob = (m - n) / (m + n)
        self.add(dict(target_prob=target_prob))


class BinomialProbabilities(Problem):
    """Find `a`, `b` so that the probability of seeing `a` heads and `b` tails on `n = a + b` coin flips
    is the given `target_prob`.
    See [Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution)"""

    @staticmethod
    def sat(counts: List[int], p=0.5, target_prob=1 / 16.0):
        from itertools import product
        a, b = counts
        n = a + b
        prob = (p ** a) * ((1-p) ** b)
        tot = sum([prob for sample in product([0, 1], repeat=n) if sum(sample) == a])
        return abs(tot - target_prob) < 1e-6

    @staticmethod
    def sol(p, target_prob):
        probs = [1.0]
        q = 1 - p
        while len(probs) < 20:
            probs = [(p * a + q * b) for a, b in zip([0] + probs, probs + [0])]
            answers = [i for i, p in enumerate(probs) if abs(p - target_prob) < 1e-6]
            if answers:
                return [answers[0], len(probs) - 1 - answers[0]]

    def gen_random(self):
        probs = [1.0]
        p = self.random.random()
        q = 1 - p
        for n in range(self.random.randrange(1, 11)):
            probs = [(p * a + q * b) for a, b in zip([0] + probs, probs + [0])]
        target_prob = self.random.choice(probs)
        self.add(dict(p=p, target_prob=target_prob))



class ExponentialProbability(Problem):
    """Find stopping probability, so that the probability of stopping in `steps` or fewer time steps
    is the given `target_prob`.
    See [Exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution)"""

    @staticmethod
    def sat(p_stop: float, steps=10, target_prob=0.5):
        prob = sum(p_stop*(1-p_stop)**t for t in range(steps))
        return abs(prob - target_prob) < 1e-6

    @staticmethod
    def sol(steps, target_prob):
        return 1 - (1 - target_prob) ** (1.0/steps)

    def gen_random(self):
        steps = self.random.randrange(1, 100)
        target_prob = self.random.random()
        self.add(dict(steps=steps, target_prob=target_prob))



if __name__ == "__main__":
    Problem.debug_problems()
