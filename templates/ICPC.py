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
    [ICPC 2019 Problem A: Azulejos](https://icpc.global/newcms/worldfinals/problems/2019%20ACM-ICPC%20World%20Finals/icpc2019.pdf)
    which is 2,287 characters."""

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
    [ICPC 2019 Problem B: Bridges](https://icpc.global/newcms/worldfinals/problems/2019%20ACM-ICPC%20World%20Finals/icpc2019.pdf)
    which is 3,003 characters.
    """

    @staticmethod
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
    @staticmethod
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


@register
class CheckersPosition(Problem):
    """You are given a partial transcript a checkers game. Find an initial position such that the transcript
    would be a legal set of moves. The board positions are [x, y] pairs with 0 <= x, y < 8 and x + y even.
    There are two players which we call -1 and 1 for convenience, and player 1 must move first in transcript.
    The initial position is represented as a list [x, y, piece] where piece means:
    * 0 is empty square
    * 1 or -1 is piece that moves only in the y = 1 or y = -1 dir, respectively
    * 2 or -2 is king for player 1 or player 2 respectively

    Additional rules:
    * You must jump if you can, and you must continue jumping until one can't any longer.
    * You cannot start the position with any non-kings on your last rank.
    * Promotion happens after the turn ends

    Inspired by
    [ICPC 2019 Problem C: Checks Post Facto](https://icpc.global/newcms/worldfinals/problems/2019%20ACM-ICPC%20World%20Finals/icpc2019.pdf)
    """

    @staticmethod
    def sat(position: List[List[int]], transcript=[[[3, 3], [5, 5], [3, 7]], [[5, 3], [6,4]]]):
        board = {(x, y): 0 for x in range(8) for y in range(8) if (x + y) % 2 == 0}  # empty board, 0 = empty
        for x, y, p in position:
            assert -2 <= p <= 2 and board[x, y] == 0  # -1, 1 is regular piece, -2, 2 is king
            board[x, y] = p

        def has_a_jump(x, y):
            p = board[x, y]  # piece to move
            deltas = [(dx, dy) for dx in [-1, 1] for dy in [-1, 1] if dy != -p]  # don't check backwards for non-kings
            return any(board.get((x + 2 * dx, y + 2 * dy)) == 0 and board[x + dx, y + dy] * p < 0 for dx, dy in deltas)

        sign = 1  # player 1 moves first
        for move in transcript:
            start, end = tuple(move[0]), tuple(move[-1])
            p = board[start]  # piece to move
            assert p * sign > 0, "Moving square must be non-empty and players must be alternate signs"
            assert all(board[x, y] == 0 for x, y in move if [x, y] != move[0]), "Moved to an occupied square"

            for (x1, y1), (x2, y2) in zip(move, move[1:]):
                assert abs(p) != 1 or (y2 - y1) * p > 0, "Non-kings can only move forward (in direction of sign)"
                if abs(x2 - x1) == 1:  # non-jump
                    assert not any(has_a_jump(*a) for a in board if board[a] * p > 0), "Must make a jump if possible"
                    break
                mid = ((x1 + x2) // 2, (y1 + y2) // 2)
                assert board[mid] * p < 0, "Can only jump over piece of opposite sign"
                board[mid] = 0
            board[start], board[end] = 0, p
            assert abs(x2 - x1) == 1 or not has_a_jump(*end)
            if abs(p) == 1 and any(y in {0, 7} for x, y in move[1:]):
                board[end] *= 2  # king me at the end of turn after any jumps are done!
            sign *= -1

        return True

    @staticmethod
    def sol(transcript):
        START_PLAYER = 1  # assumed

        class InitOpts:
            def __init__(self, x, y):
                self.x, self.y = x, y
                self.opts = {-2, -1, 0, 1, 2}
                if y == 0:
                    self.opts.remove(-1)
                if y == 7:
                    self.opts.remove(1)
                self.promoted = 2 ** 63  # on which step was it promoted t >= 0
                self.jumped = 2 ** 63  # on which step was it jumped t >= 0

        # def board2str(board):  # for debugging
        #     mapping = ".bBWw"
        #     ans = ""
        #     for y in range(7, -1, -1):
        #         ans += "".join(" " if (x+y)%2 else mapping[board[x,y]] for x in range(8)) + "\n"
        #     return ans

        init_opts = {(x, y): InitOpts(x, y) for x in range(8) for y in range(8) if (x + y) % 2 == 0}
        # board = {(x, y): (1 if y < 3 else -1 if y > 4 else 0) for x in range(8) for y in range(8) if
        #          (x + y) % 2 == 0}  # new board

        transcript = [[tuple(a) for a in move] for move in transcript]

        permuted_opts = init_opts.copy()
        sign = START_PLAYER
        for t, move in enumerate(transcript):
            start, end = tuple(move[0]), tuple(move[-1])
            p = permuted_opts[start]  # opts to move
            assert p.jumped >= t
            p.opts -= {-sign, -2 * sign, 0}
            if any((y2 - y1) * sign < 0 for (x1, y1), (x2, y2) in zip(move, move[1:])):  # backward move!
                if p.promoted >= t:
                    p.opts -= {sign}  # must be a king!

            for a, b in zip(move, move[1:]):
                if permuted_opts[b].jumped >= t:
                    permuted_opts[b].opts -= {-2, -1, 1, 2}  # must be empty
                assert permuted_opts[a].jumped >= t
                permuted_opts[a], permuted_opts[b] = permuted_opts[b], permuted_opts[a]
                # board[a], board[b] = board[b], board[a]
                (x1, y1), (x2, y2) = a, b
                if abs(x2 - x1) == 2:  # jump
                    mid = ((x1 + x2) // 2, (y1 + y2) // 2)
                    assert permuted_opts[mid].jumped >= t
                    permuted_opts[mid].opts -= {0, sign, 2 * sign}  # Can only jump over piece of opposite sign
                    permuted_opts[mid].jumped = t
                    # board[mid] = 0

            if any(y in {0, 7} for x, y in move[1:]):
                if p.promoted > t:
                    p.promoted = t
                # if abs(board[x2, y2]) == 1:
                #     board[x2, y2] *= 2

            sign *= -1

        for y in range(7, -1, -1):
            for x in range(8):
                if (x, y) in init_opts:
                    s = init_opts[x, y].opts
                    if {1, 2} <= s:
                        s.remove(2)
                    if {-1, -2} <= s:
                        s.remove(-2)

        def helper():  # returns True if success and store everything, otherwise None
            my_opts = init_opts.copy()
            sign = START_PLAYER  # player 1 always starts

            for t, move in enumerate(transcript):
                if abs(move[0][0] - move[1][0]) == 1:  # not a jump
                    check_no_jumps = [a for a, p in my_opts.items() if p.jumped >= t and p.opts <= {sign, 2 * sign}]
                else:
                    for a, b in zip(move, move[1:]):
                        my_opts[a], my_opts[b] = my_opts[b], my_opts[a]
                    check_no_jumps = [b]

                for x, y in check_no_jumps:
                    p = my_opts[x, y]
                    [o] = p.opts
                    assert o * sign > 0
                    dys = [o] if (abs(o) == 1 and p.promoted >= t) else [-1, 1]  # only check forward jumps
                    for dx in [-1, 1]:
                        for dy in dys:
                            target_o = my_opts.get((x + 2 * dx, y + 2 * dy))
                            if target_o is not None and (0 in target_o.opts or target_o.jumped < t):
                                mid_o = my_opts[x + dx, y + dy]
                                if mid_o.jumped > t and mid_o.opts <= {-sign, -2 * sign}:  # ok if jumped at t
                                    if target_o.jumped < t or target_o.opts == {0}:
                                        return False
                                    old_opts = target_o.opts
                                    for v in target_o.opts:
                                        if v != 0:
                                            target_o.opts = {v}
                                            h = helper()
                                            if h:
                                                return True
                                    target_o.opts = old_opts
                                    return False

                if abs(move[0][0] - move[1][0]) == 1:  # not a jump
                    a, b = move[0], move[1]
                    my_opts[a], my_opts[b] = my_opts[b], my_opts[a]

                sign *= -1
            return True

        res = helper()
        assert res

        def get_opt(opts):
            if 0 in opts.opts:
                return 0
            assert len(opts.opts) == 1
            return list(opts.opts)[0]

        return [[x, y, get_opt(opts)] for (x, y), opts in init_opts.items()]

    def gen_random(self):
        full_transcript = self.random_game_transcript()
        n = len(full_transcript)
        a = self.random.randrange(0, n + 1, 2)
        b = self.random.randrange(n + 1)
        transcript = full_transcript[a:b]  # won't add duplicates so empty transcript is only added once
        self.add(dict(transcript=transcript))

    def random_game_transcript(self):
        START_PLAYER = 1  # assumed

        transcript = []
        board = {(x, y): (1 if y < 3 else -1 if y > 4 else 0) for x in range(8) for y in range(8) if
                 (x + y) % 2 == 0}  # new board

        def get_jumps(x, y):
            p = board[x, y]  # piece to move
            return [(x + 2 * dx, y + 2 * dy)
                    for dx, dy in deltas(x, y)
                    if board.get((x + 2 * dx, y + 2 * dy)) == 0 and board[x + dx, y + dy] * p < 0]

        def deltas(x, y):
            p = board[x, y]  # piece to move
            assert p != 0
            return [(dx, dy)
                    for dx in [-1, 1]
                    for dy in [-1, 1]
                    if dy != -p]  # don't check backwards for non-kings

        def make_random_move(sign):  # returns True if move was made else False if no legal move
            pieces = [a for a in board if board[a] * sign > 0]
            jumps = [[a, b] for a in pieces for b in get_jumps(*a)]
            if jumps:
                move = self.random.choice(jumps)
                while True:
                    (x1, y1), (x2, y2) = move[-2:]
                    mid = ((x1 + x2) // 2, (y1 + y2) // 2)
                    assert board[mid] * sign < 0, "Can only jump over piece of opposite sign"
                    board[x2, y2] = board[x1, y1]
                    board[x1, y1] = 0
                    board[mid] = 0
                    try:
                        move.append(self.random.choice(get_jumps(*move[-1])))
                    except IndexError:
                        break
            else:
                candidates = [[(x, y), (x + dx, y + dy)]
                              for x, y in pieces for dx, dy in deltas(x, y)
                              if board.get((x + dx, y + dy)) == 0]
                if not candidates:
                    return False
                move = self.random.choice(candidates)
                board[move[0]], board[move[1]] = 0, board[move[0]]

            transcript.append(move)
            end = move[-1]
            if abs(board[end]) == 1 and any(y in {0, 7} for x, y in move[1:]):
                board[end] *= 2  # promotion to king, king me, at the end of turn after any jumps are done!
            return True

        transcript = []
        sign = START_PLAYER
        while make_random_move(sign):
            sign *= -1

        return [[list(a) for a in move] for move in transcript]  # convert to lists


if __name__ == "__main__":
    for problem in get_problems(globals()):
        problem.test()
