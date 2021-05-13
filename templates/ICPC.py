"""
Problems inspired by the [International Collegiate Programming Contest](https://icpc.global) (ICPC).
"""

from problems import Problem, register, get_problems
from typing import List


@register
class BiPermutations(Problem):
    """There are two rows of objects. Given the length-n integer arrays of prices and heights of objects in each
    row, find a permutation of both rows so that the permuted prices are non-decreasing in each row and
    so that the first row is taller than the second row.

    Inspired by
    [ICPC 2019 Problem A: Azulejos](https://icpc.global/newcms/worldfinals/problems/2019%20ACM-ICPC%20World%20Finals/icpc2019.pdf)."""

    @staticmethod
    def sat(perms: List[List[int]],
            prices0=[7, 7, 9, 5, 3, 7, 1, 2],
            prices1=[5, 5, 5, 4, 2, 5, 1, 1],
            heights0=[2, 4, 9, 3, 8, 5, 5, 4],
            heights1=[1, 3, 8, 1, 5, 4, 4, 2]):
        n = len(prices0)
        perm0, perm1 = perms
        assert sorted(perm0) == sorted(perm1) == list(range(n)), "Solution must be two permutations"
        for i in range(n - 1):
            assert prices0[perm0[i]] <= prices0[perm0[i + 1]], "Permuted prices must be nondecreasing (row 0)"
            assert prices1[perm1[i]] <= prices1[perm1[i + 1]], "Permuted prices must be nondecreasing (row 1)"
        return all(heights0[i] > heights1[j] for i, j in zip(perm0, perm1))

    @staticmethod
    def sol(prices0=[7, 7, 9, 5, 3, 7, 1, 2],
            prices1=[5, 5, 5, 4, 2, 5, 1, 1],
            heights0=[2, 4, 9, 3, 8, 5, 5, 4],
            heights1=[1, 3, 8, 1, 5, 4, 4, 2]):
        n = len(prices0)
        prices = [prices0, prices1]
        orders = [sorted(range(n), key=lambda i: (prices0[i], heights0[i])),
                  sorted(range(n), key=lambda i: (prices1[i], -heights1[i]))]
        jumps = [1, 1]  # next price increase locations
        for i in range(n):
            for r, (p, o) in enumerate(zip(prices, orders)):
                while jumps[r] < n and p[o[jumps[r]]] == p[o[i]]:
                    jumps[r] += 1

            to_fix = orders[jumps[0] < jumps[1]]
            j = i
            while heights0[orders[0][i]] <= heights1[orders[1][i]]:
                j += 1
                to_fix[i], to_fix[j] = to_fix[j], to_fix[i]

        return orders

    def gen_random(self):
        n = self.random.randint(2, self.random.choice([10, 20, 100]))
        P = sorted(self.random.choices(range(1 + n // 10), k=n))  # non-decreasing prices
        H = [self.random.randint(1, 10) for _ in range(n)]
        perm1 = list(range(n))
        self.random.shuffle(perm1)
        prices1 = [P[i] for i in perm1]
        heights1 = [H[i] for i in perm1]

        P = sorted(self.random.choices(range(1 + n // 10), k=n))  # non-decreasing prices
        H = [h + self.random.randint(1, 5) for h in H]  # second row taller than first
        perm0 = list(range(n))
        self.random.shuffle(perm0)
        prices0 = [P[i] for i in perm0]
        heights0 = [H[i] for i in perm0]
        self.add(dict(prices0=prices0, heights0=heights0, prices1=prices1, heights1=heights1))


@register
class OptimalBridges(Problem):
    """
    You are to choose locations for bridge bases from among a given set of mountain peaks located at
    `xs, ys`, where `xs` and `ys` are lists of n integers of the same length. Your answer should be a sorted
    list of indices starting at 0 and ending at n-1. The goal is to minimize building costs such that the bridges
    are feasible. The bridges are all semicircles placed on top of the pillars. The feasibility constraints are that:
    * The bridges may not extend above a given height `H`. Mathematically, if the distance between the two xs
    of adjacent pillars is d, then the semicircle will have radius `d/2` and therefore the heights of the
    selected mountain peaks must both be at most `H - d/2`.
    *  The bridges must clear all the mountain peaks, which means that the semicircle must lie above the tops of the
    peak. See the code for how this is determined mathematically.
    * The total cost of all the bridges must be at most `thresh`, where the cost is parameter alpha * (the sum of
    all pillar heights) + beta * (the sum of the squared diameters)

    Inspired by
    [ICPC 2019 Problem B: Bridges](https://icpc.global/newcms/worldfinals/problems/2019%20ACM-ICPC%20World%20Finals/icpc2019.pdf)"""

    def sat(indices: List[int],
            H=60,
            alpha=18,
            beta=2,
            xs=[0, 10, 20, 30, 50, 80, 100, 120, 160, 190, 200],
            ys=[0, 30, 10, 30, 50, 40, 10, 20, 20, 55, 10],
            thresh=26020):
        assert sorted({0, len(xs) - 1, *indices}) == indices, f"Ans. should be sorted list [0, ..., {len(xs) - 1}]"
        cost = alpha * (H - ys[0])
        for i, j in zip(indices, indices[1:]):
            a, b, r = xs[i], xs[j], (xs[j] - xs[i]) / 2
            assert max(ys[i], ys[j]) + r <= H, "Bridge too tall"
            assert all(ys[k] <= H - r + ((b - xs[k]) * (xs[k] - a)) ** 0.5 for k in range(i + 1, j)), \
                "Bridge too short"
            cost += alpha * (H - ys[j]) + beta * (b - a) ** 2
        return cost <= thresh

    # adapted from https://github.com/SnapDragon64/ACMFinalsSolutions/blob/master/finals2019/beautifulbridgesDK.cc
    def sol(H, alpha, beta, xs, ys, thresh):  # thresh is ignored
        n = len(xs)
        cost = [-1] * n
        prior = [n] * n
        cost[0] = beta * (H - ys[0])
        for i in range(n):
            if cost[i] == -1:
                continue
            min_d = 0
            max_d = 2 * (H - ys[i])
            for j in range(i + 1, n):
                d = xs[j] - xs[i]
                h = H - ys[j]
                if d > max_d:
                    break
                if 2 * h <= d:
                    min_d = max(min_d, 2 * d + 2 * h - int((8 * d * h) ** 0.5))
                max_d = min(max_d, 2 * d + 2 * h + int((8 * d * h) ** 0.5))
                if min_d > max_d:
                    break
                if min_d <= d <= max_d:
                    new_cost = cost[i] + alpha * h + beta * d * d
                    if cost[j] == -1 or cost[j] > new_cost:
                        cost[j] = new_cost
                        prior[j] = i
        rev_ans = [n - 1]
        while rev_ans[-1] != 0:
            rev_ans.append(prior[rev_ans[-1]])
        return rev_ans[::-1]

    def gen_random(self):
        H = 10 ** 5
        L = self.random.choice([10, 20, 50, 100, 1000])
        n = self.random.randrange(2, L)
        alpha = self.random.randrange(L)
        beta = self.random.randrange(L)
        m = self.random.randrange(1, n)
        keys = [0] + sorted(self.random.sample(range(1, H), m - 1)) + [H]
        assert len(keys) == m + 1
        dists = [keys[i + 1] - keys[i] for i in range(m)]
        assert len(dists) == m
        heights = [self.random.randint(0, H - (max([dists[max(0, i - 1)], dists[min(m - 1, i)]]) + 1) // 2)
                   for i in range(m + 1)]
        xs = []
        ys = []
        for i in range(m + 1):
            xs.append(keys[i])
            ys.append(heights[i])
            for _ in range(int(1 / self.random.random())):
                if i >= m or len(xs) + m + 1 - i >= L or xs[-1] == keys[i + 1]:
                    break
                x = self.random.randint(xs[-1], keys[i + 1] - 1)
                xs.append(x)
                c = (keys[i + 1] + keys[i]) / 2
                r = (keys[i + 1] - keys[i]) / 2
                y = self.random.randint(0, int(H - r + (r ** 2 - (x - c) ** 2) ** 0.5))
                ys.append(y)

        indices = OptimalBridges.sol(H, alpha, beta, xs, ys, None)  # compute min-cost, thresh is ignored
        cost = alpha * (H - ys[0])
        for i, j in zip(indices, indices[1:]):
            a, b, r = xs[i], xs[j], (xs[j] - xs[i]) / 2
            assert max(ys[i], ys[j]) + r <= H, "Bridge too tall"
            assert all(
                ys[k] <= H - r + ((b - xs[k]) * (xs[k] - a)) ** 0.5 for k in range(i + 1, j)), "Bridge too short"
            cost += alpha * (H - ys[j]) + beta * (b - a) ** 2
        thresh = cost
        self.add(dict(H=H, alpha=alpha, beta=beta, xs=xs, ys=ys, thresh=thresh))


if __name__ == "__main__":
    for problem in get_problems(globals()):
        problem.test()
