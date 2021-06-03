"""Classic chess problems"""

from problems import Problem, register, get_problems
from typing import List


@register
class EightQueensOrFewer(Problem):
    """Eight (or fewer) Queens Puzzle

    Position min(m, n) <= 8 queens on an m x n chess board so that no pair is attacking each other. Hint:
    a brute force approach works on this puzzle.

    See the MoreQueens puzzle for another (longer but clearer) equivalent definition of sat

    See Wikipedia entry on
    [Eight Queens puzzle](https://en.wikipedia.org/w/index.php?title=Eight_queens_puzzle)."""

    @staticmethod
    def sat(squares: List[List[int]], m=8, n=8):
        k = min(m, n)
        assert all(i in range(m) and j in range(n) for i, j in squares) and len(squares) == k
        return 4 * k == len({t for i, j in squares for t in [('row', i), ('col', j), ('SE', i + j), ('NE', i - j)]})

    @staticmethod
    def sol(m, n):  # brute force
        k = min(m, n)

        from itertools import permutations
        for p in permutations(range(k)):
            if 4 * k == len(
                    {t for i, j in enumerate(p) for t in [('row', i), ('col', j), ('SE', i + j), ('NE', i - j)]}):
                return [[i, j] for i, j in enumerate(p)]

    def gen_random(self):
        m, n = [self.random.randrange(4, self.random.choice([10, 100])) for _ in range(2)]
        if min(m, n) <= 8:
            self.add(dict(m=m, n=n))


@register
class MoreQueens(Problem):
    """More Queens Puzzle

    Position min(m, n) > 8 queens on an m x n chess board so that no pair is attacking each other. A brute force
    approach will not work on many of these problems. Here, we use a different

    See Wikipedia entry on
    [Eight Queens puzzle](https://en.wikipedia.org/w/index.php?title=Eight_queens_puzzle)."""

    @staticmethod
    def sat(squares: List[List[int]], m=9, n=9):
        k = min(m, n)
        assert all(i in range(m) and j in range(n) for i, j in squares), "queen off board"
        assert len(squares) == k, "Wrong number of queens"
        assert len({i for i, j in squares}) == k, "Queens on same row"
        assert len({j for i, j in squares}) == k, "Queens on same file"
        assert len({i + j for i, j in squares}) == k, "Queens on same SE diagonal"
        assert len({i - j for i, j in squares}) == k, "Queens on same NE diagonal"
        return True

    @staticmethod
    def sol(m, n):
        t = min(m, n)
        ans = []
        if t % 2 == 1:  # odd k, put a queen in the lower right corner (and decrement k)
            ans.append([t - 1, t - 1])
            t -= 1
        if t % 6 == 2:  # do something special for 8x8, 14x14 etc:
            ans += [[i, (2 * i + t // 2 - 1) % t] for i in range(t // 2)]
            ans += [[i + t // 2, (2 * i - t // 2 + 2) % t] for i in range(t // 2)]
        else:
            ans += [[i, 2 * i + 1] for i in range(t // 2)]
            ans += [[i + t // 2, 2 * i] for i in range(t // 2)]
        return ans

    def gen_random(self):
        m, n = [self.random.randrange(4, self.random.choice([10, 100])) for _ in range(2)]
        if min(m, n) > 8:
            self.add(dict(m=m, n=n))


@register
class KnightsTour(Problem):
    """Knights Tour

    Find an (open) tour of knight moves on an m x n chess-board that visits each square once.

    See Wikipedia entry on [Knight's tour](https://en.wikipedia.org/w/index.php?title=Knight%27s_tour)"""

    @staticmethod
    def sat(tour: List[List[int]], m=8, n=8):
        assert all({abs(i1 - i2), abs(j1 - j2)} == {1, 2} for [i1, j1], [i2, j2] in zip(tour, tour[1:])), 'legal moves'
        return sorted(tour) == [[i, j] for i in range(m) for j in range(n)]  # cover every square once

    @staticmethod
    def sol(m, n):  # using Warnsdorff's heuristic, breaking ties randomly and restarting 10 times
        import random
        for seed in range(10):
            r = random.Random(seed)
            ans = [(0, 0)]
            free = {(i, j) for i in range(m) for j in range(n)} - {(0, 0)}

            def possible(i, j):
                moves = [(i + s * a, j + t * b) for (a, b) in [(1, 2), (2, 1)] for s in [-1, 1] for t in [-1, 1]]
                return [z for z in moves if z in free]

            while True:
                if not free:
                    return [[a, b] for (a, b) in ans]
                candidates = possible(*ans[-1])
                if not candidates:
                    break
                ans.append(min(candidates, key=lambda z: len(possible(*z)) + r.random()))
                free.remove(ans[-1])

    def gen(self, num_target_problems):
        count = 0
        for n in [9, 8, 7, 6, 5] + list(range(10, 100)):
            m = n
            if self.sol(m, n):
                self.add(dict(m=m, n=n))
                count += 1
                if count == num_target_problems:
                    return

    def gen_random(self):
        m, n = [self.random.randrange(5, self.random.choice([10, 100])) for _ in range(2)]
        if self.sol(m, n):
            self.add(dict(m=m, n=n))


@register
class UncrossedKnightsPath(Problem):
    """Uncrossed Knights Path (known solvable, but no solution given)

    Find long (open) tour of knight moves on an m x n chess-board whose edges don't cross.
    The goal of these problems is to match the nxn_records from [http://ukt.alex-black.ru/](http://ukt.alex-black.ru/)
    (accessed 2020-11-29).

    A more precise description is in this
    [Wikipedia article](https://en.wikipedia.org/w/index.php?title=Longest_uncrossed_knight%27s_path)."""

    nxn_records = {3: 2, 4: 5, 5: 10, 6: 17, 7: 24, 8: 35, 9: 47, 10: 61, 11: 76, 12: 94, 13: 113, 14: 135, 15: 158,
                   16: 183, 17: 211, 18: 238, 19: 268, 20: 302, 21: 337, 22: 375, 23: 414}

    @staticmethod
    def sat(path: List[List[int]], m=8, n=8, target=35):
        def legal_move(m):
            (a, b), (i, j) = m
            return {abs(i - a), abs(j - b)} == {1, 2}

        def legal_quad(m1, m2):  # non-overlapping test: parallel or bounding box has (width - 1) * (height - 1) >= 5
            (i1, j1), (i2, j2) = m1
            (a1, b1), (a2, b2) = m2
            return (len({(i1, j1), (i2, j2), (a1, b1), (a2, b2)}) < 4  # adjacent edges in path, ignore
                    or (i1 - i2) * (b1 - b2) == (j1 - j2) * (a1 - a2)  # parallel
                    or (max(a1, a2, i1, i2) - min(a1, a2, i1, i2)) * (max(b1, b2, j1, j2) - min(b1, b2, j1, j2)) >= 5
                    # far
                    )

        assert all(i in range(m) and j in range(n) for i, j in path), "move off board"
        assert len({(i, j) for i, j in path}) == len(path), "visited same square twice"

        moves = list(zip(path, path[1:]))
        assert all(legal_move(m) for m in moves), "illegal move"
        assert all(legal_quad(m1, m2) for m1 in moves for m2 in moves), "intersecting move pair"

        return len(path) >= target

    def gen(self, target_num_problems):
        for count, n in enumerate(self.nxn_records):
            if len(self.instances) >= target_num_problems:
                return
            self.add(dict(m=n, n=n, target=self.nxn_records[n]))

    def gen_random(self):
        m, n = [self.random.randrange(3, self.random.choice([10, 100])) for _ in range(2)]
        k = min(m, n)
        if k in self.nxn_records:
            target = self.random.randrange(self.nxn_records[k])
            self.add(dict(m=m, n=n, target=target))  # solved by someone

@register
class UNSOLVED_UncrossedKnightsPath(UncrossedKnightsPath):
    """Uncrossed Knights Path (open problem, unsolved)

    Find long (open) tour of knight moves on an m x n chess-board whose edges don't cross.
    The goal of these problems is to *beat* the nxn_records from
    [http://ukt.alex-black.ru/](http://ukt.alex-black.ru/)
    (accessed 2020-11-29).

    A more precise description is in this
    [Wikipedia article](https://en.wikipedia.org/w/index.php?title=Longest_uncrossed_knight%27s_path)."""

    unsolved_nxn_records = {10: 61, 11: 76, 12: 94, 13: 113, 14: 135, 15: 158,
                            16: 183, 17: 211, 18: 238, 19: 268, 20: 302, 21: 337, 22: 375, 23: 414}

    def gen(self, target_num_problems):
        for count, n in enumerate(self.nxn_records):
            if len(self.instances) >= target_num_problems:
                return
            self.add(dict(m=n, n=n, target=self.nxn_records[n] + 1))  # Note the +1 means breaking the record!

if __name__ == "__main__":
    for problem in get_problems(globals()):
        problem.test()
