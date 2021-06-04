"""
Hard problems from game theory.
"""

from problems import Problem, register, get_problems
from typing import List


@register
class Nash(Problem):
    """Compute a [Nash equilibrium](https://en.wikipedia.org/wiki/Nash_equilibrium) for a given
     [bimatrix game](https://en.wikipedia.org/wiki/Bimatrix_game). While this problem was known to be
     PPAD-hard in general. In fact the challenge is be much easier for an approximate
     [eps-equilibrium](https://en.wikipedia.org/wiki/Epsilon-equilibrium) and of course for small games."""

    @staticmethod
    def sat(strategies: List[List[float]],
            A=[[-1., -3.], [0., -2.]],  # P1's payoffs (prisoner dilemma example)
            B=[[-1., 0.], [-3., -2.]],  # P2's payoffs
            eps=0.01):  # error tolerance
        m, n = len(A), len(A[0])
        p, q = strategies
        assert len(B) == m and all(len(row) == n for row in A + B), "inputs are a bimatrix game"
        assert len(p) == m and len(q) == n, "solution is a pair of strategies"
        assert sum(p) == sum(q) == 1.0 and min(p + q) >= 0.0, "strategies must be non-negative and sum to 1"
        v = sum(A[i][j] * p[i] * q[j] for i in range(m) for j in range(n))
        w = sum(B[i][j] * p[i] * q[j] for i in range(m) for j in range(n))
        return (all(sum(A[i][j] * q[j] for j in range(n)) <= v + eps for i in range(m)) and
                all(sum(B[i][j] * p[i] for i in range(m)) <= w + eps for j in range(n)))

    @staticmethod
    def sol(A, B, eps):
        NUM_ATTEMPTS = 100

        def sat(strategies: List[List[float]], A, B, eps):
            m, n = len(A), len(A[0])
            p, q = strategies
            assert len(B) == m and all(len(row) == n for row in A + B), "inputs are a bimatrix game"
            assert len(p) == m and len(q) == n, "solution is a pair of strategies"
            assert sum(p) == sum(q) == 1.0 and min(p + q) >= 0.0, "strategies must be non-negative and sum to 1"
            v = sum(A[i][j] * p[i] * q[j] for i in range(m) for j in range(n))
            w = sum(B[i][j] * p[i] * q[j] for i in range(m) for j in range(n))
            return (all(sum(A[i][j] * q[j] for j in range(n)) <= v + eps for i in range(m)) and
                    all(sum(B[i][j] * p[i] for i in range(m)) <= w + eps for j in range(n)))

        import random
        r = random.Random(0)
        dims = len(A), len(A[0])
        # possible speedup: remove dominated strategies
        for _attempt in range(NUM_ATTEMPTS):
            strategies = []
            for d in dims:
                s = [max(0.0, r.random() - 0.5) for _ in range(d)]
                tot = sum(s) + 1e-6
                for i in range(d):
                    s[i] = (1.0 - sum(s[:-1])) if i == d - 1 else (s[i] / tot)  # to ensure sum is exactly 1.0
                strategies.append(s)
            if sat(strategies, A, B, eps):
                return strategies

    def gen_random(self):
        m = self.random.randrange(2, 10)
        n = self.random.randrange(2, 10)
        A, B = [[[self.random.random() for _i in range(m)] for _j in range(n)] for _k in range(2)]
        eps = self.random.choice([0.5, 0.1, 0.01])
        solved = self.sol(A, B, eps) is not None
        self.add(dict(A=A, B=B, eps=eps), test=solved)


@register
class ZeroSum(Problem):
    """Compute minimax optimal strategies for a given
     [zero-sum game](https://en.wikipedia.org/wiki/Zero-sum_game). This problem is known to be equivalent to
     Linear Programming. Note that the provided instances are all quite easy---harder solutions could readily
     be made by decreasing the accuracy tolerance `eps` at which point the solution we provided would fail and
     more efficient algorithms would be needed."""

    @staticmethod
    def sat(strategies: List[List[float]],
            A = [[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]], # P1's payoffs (rock-paper-scissors example)
            eps=0.1):  # error tolerance
        m, n = len(A), len(A[0])
        p, q = strategies
        assert all(len(row) == n for row in A), "inputs are a matrix"
        assert len(p) == m and len(q) == n, "solution is a pair of strategies"
        assert sum(p) == sum(q) == 1.0 and min(p + q) >= 0.0, "strategies must be non-negative and sum to 1"
        v = sum(A[i][j] * p[i] * q[j] for i in range(m) for j in range(n))
        return (all(sum(A[i][j] * q[j] for j in range(n)) <= v + eps for i in range(m)) and
                all(sum(A[i][j] * p[i] for i in range(m)) >= v - eps for j in range(n)))

    @staticmethod
    def sol(A, eps):
        MAX_ITER = 10**4
        m, n = len(A), len(A[0])
        a = [0 for _i in range(m)]
        b = [0 for _j in range(n)]

        for count in range(1, MAX_ITER):
            i_star = max(range(m), key=lambda i: sum(A[i][j] * b[j] for j in range(n)))
            j_star = min(range(n), key=lambda j: sum(A[i][j] * a[i] for i in range(m)))
            a[i_star] += 1
            b[j_star] += 1
            p = [x / (count + 1e-6) for x in a]
            p[-1] = 1 - sum(p[:-1])  # rounding issues
            q = [x / (count + 1e-6) for x in b]
            q[-1] = 1 - sum(q[:-1])  # rounding issues

            v = sum(A[i][j] * p[i] * q[j] for i in range(m) for j in range(n))
            if (all(sum(A[i][j] * q[j] for j in range(n)) <= v + eps for i in range(m)) and
                    all(sum(A[i][j] * p[i] for i in range(m)) >= v - eps for j in range(n))):
                return [p, q]

    def gen_random(self):
        m = self.random.randrange(2, 10)
        n = self.random.randrange(2, 10)
        A = [[self.random.random() for _i in range(m)] for _j in range(n)]
        eps = self.random.choice([0.5, 0.1, 0.01])
        solved = self.sol(A, eps) is not None
        self.add(dict(A=A, eps=eps), test=solved)


if __name__ == "__main__":
    for problem in get_problems(globals()):
        problem.test()
