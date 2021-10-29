"""Classic puzzles"""

from puzzle_generator import PuzzleGenerator
from typing import List


# See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle to learn about adding puzzles


class TowersOfHanoi(PuzzleGenerator):
    """[Towers of Hanoi](https://en.wikipedia.org/w/index.php?title=Tower_of_Hanoi)

    In this classic version one must move all 8 disks from the first to third peg."""

    @staticmethod
    def sat(moves: List[List[int]]):
        """
        Eight disks of sizes 1-8 are stacked on three towers, with each tower having disks in order of largest to
        smallest. Move [i, j] corresponds to taking the smallest disk off tower i and putting it on tower j, and it
        is legal as long as the towers remain in sorted order. Find a sequence of moves that moves all the disks
        from the first to last towers.
        """
        rods = ([8, 7, 6, 5, 4, 3, 2, 1], [], [])
        for [i, j] in moves:
            rods[j].append(rods[i].pop())
            assert rods[j][-1] == min(rods[j]), "larger disk on top of smaller disk"
        return rods[0] == rods[1] == []

    @staticmethod
    def sol():
        def helper(m, i, j):
            if m == 0:
                return []
            k = 3 - i - j
            return helper(m - 1, i, k) + [[i, j]] + helper(m - 1, k, j)

        return helper(8, 0, 2)


class TowersOfHanoiArbitrary(PuzzleGenerator):
    """[Towers of Hanoi](https://en.wikipedia.org/w/index.php?title=Tower_of_Hanoi)

    In this version one must transform a given source state to a target state."""

    @staticmethod
    def sat(moves: List[List[int]],
            source=[[0, 7], [4, 5, 6], [1, 2, 3, 8]],
            target=[[0, 1, 2, 3, 8], [4, 5], [6, 7]]):
        """
        A state is a partition of the integers 0-8 into three increasing lists. A move is pair of integers i, j in
        {0, 1, 2} corresponding to moving the largest number from the end of list i to list j, while preserving the
        order of list j. Find a sequence of moves that transform the given source to target states.
        """
        state = [s[:] for s in source]

        for [i, j] in moves:
            state[j].append(state[i].pop())
            assert state[j] == sorted(state[j])

        return state == target

    @staticmethod
    def sol(source, target):
        state = {d: i for i, tower in enumerate(source) for d in tower}
        final = {d: i for i, tower in enumerate(target) for d in tower}
        disks = set(state)
        assert disks == set(final) and all(isinstance(i, int) for i in state) and len(source) == len(target) >= 3
        ans = []

        def move(d, i):  # move disk d to tower i
            if state[d] == i:
                return
            for t in range(3):  # first tower besides i, state[d]
                if t != i and t != state[d]:
                    break
            for d2 in range(d + 1, max(disks) + 1):
                if d2 in disks:
                    move(d2, t)
            ans.append([state[d], i])
            state[d] = i

        for d in range(min(disks), max(disks) + 1):
            if d in disks:
                move(d, final[d])

        return ans

    def gen_random(self):
        n = self.random.randrange(4, 18)
        source, target = [[[] for _ in range(3)] for _ in range(2)]
        for d in range(n):
            self.random.choice(source).append(d)
            self.random.choice(target).append(d)
        self.add(dict(source=source, target=target))


class LongestMonotonicSubstring(PuzzleGenerator):
    """This is a form of the classic
    [Longest increasing subsequence](https://en.wikipedia.org/wiki/Longest_increasing_subsequence) problem
    where the goal is to find a substring with characters in sorted order."""

    @staticmethod
    def sat(x: List[int], length=13, s="Dynamic programming solves this puzzle!!!"):
        """
        Remove as few characters as possible from s so that the characters of the remaining string are alphebetical.
        Here x is the list of string indices that have not been deleted.
        """
        return all(s[x[i]] <= s[x[i + 1]] and x[i + 1] > x[i] >= 0 for i in range(length - 1))

    @staticmethod
    def sol(length, s):
        # O(N^2) method. Todo: add binary search solution which is O(n log n)
        if s == "":
            return []
        n = len(s)
        dyn = []  # list of (seq length, seq end, prev index)
        for i in range(n):
            try:
                dyn.append(max((length + 1, i, e) for length, e, _ in dyn if s[e] <= s[i]))
            except ValueError:
                dyn.append((1, i, -1))  # sequence ends at i
        _length, i, _ = max(dyn)
        backwards = [i]
        while dyn[i][2] != -1:
            i = dyn[i][2]
            backwards.append(i)
        return backwards[::-1]

    def gen_random(self):
        n = self.random.randrange(self.random.choice([10, 100, 1000]))  # a length between 1-10 or 1-100 or 1-1000
        length = self.random.randrange(n + 1)
        rand_chars = [chr(self.random.randrange(32, 124)) for _ in range(n)]
        li = sorted(rand_chars[:length])
        for i in range(length, n):
            li.insert(self.random.randrange(i + 1), rand_chars[i])
        s = "".join(li)
        self.add(dict(length=length, s=s))


class LongestMonotonicSubstringTricky(PuzzleGenerator):
    """The same as the above problem, but with a twist!"""

    @staticmethod
    def sat(x: List[int], length=20, s="Dynamic programming solves this classic job-interview puzzle!!!"):
        """Find the indices of the longest substring with characters in sorted order"""
        return all(s[x[i]] <= s[x[i + 1]] and x[i + 1] > x[i] for i in range(length - 1))

    @staticmethod
    def sol(length, s):
        # O(N^2) method. Todo: add binary search solution which is O(n log n)
        if s == "":
            return []
        n = len(s)
        dyn = []  # list of (seq length, seq end, prev index)
        for i in range(-n, n):
            try:
                dyn.append(max((length + 1, i, e) for length, e, _ in dyn if s[e] <= s[i]))
            except ValueError:
                dyn.append((1, i, None))  # sequence ends at i
        _length, i, _ = max(dyn)
        backwards = [i]
        while dyn[n + i][2] is not None:
            i = dyn[n + i][2]
            backwards.append(i)
        return backwards[::-1]

    def gen_random(self):
        n = self.random.randrange(self.random.choice([10, 100, 1000]))  # a length between 1-10 or 1-100 or 1-1000
        length = self.random.randrange(n + 1)
        rand_chars = [chr(self.random.randrange(32, 124)) for _ in range(n)]
        li = sorted(rand_chars[:length])
        a, b = li[:length // 2][::-1], li[length // 2:][::-1]
        li = []
        while a or b:
            li.append(self.random.choice([c for c in (a, b) if c]).pop())
        for i in range(length, n):
            li.insert(self.random.randrange(i + 1), rand_chars[i])
        s = "".join(li)
        self.add(dict(length=length, s=s))


class Quine(PuzzleGenerator):
    """[Quine](https://en.wikipedia.org/wiki/Quine_%28computing%29)"""

    @staticmethod
    def sat(quine: str):
        """Find a string that when evaluated as a Python expression is that string itself."""
        return eval(quine) == quine

    @staticmethod
    def sol():
        return "(lambda x: f'({x})({chr(34)}{x}{chr(34)})')(\"lambda x: f'({x})({chr(34)}{x}{chr(34)})'\")"


class RevQuine(PuzzleGenerator):
    """Reverse [Quine](https://en.wikipedia.org/wiki/Quine_%28computing%29). The solution we give is from GPT3."""

    @staticmethod
    def sat(rev_quine: str):
        """Find a string that, when reversed and evaluated gives you back that same string."""
        return eval(rev_quine[::-1]) == rev_quine

    @staticmethod
    def sol():
        return "rev_quine"[::-1]  # thanks GPT-3!


class BooleanPythagoreanTriples(PuzzleGenerator):
    """[Boolean Pythagorean Triples Problem](https://en.wikipedia.org/wiki/Boolean_Pythagorean_triples_problem)"""

    @staticmethod
    def sat(colors: List[int], n=100):
        """
        Color the first n integers with one of two colors so that there is no monochromatic Pythagorean triple.
        A monochromatic Pythagorean triple is a triple of numbers i, j, k such that i^2 + j^2 = k^2 that
        are all assigned the same color. The input, colors, is a list of 0/1 colors of length >= n.
        """
        assert set(colors) <= {0, 1} and len(colors) >= n
        squares = {i ** 2: colors[i] for i in range(1, len(colors))}
        return not any(c == d == squares.get(i + j) for i, c in squares.items() for j, d in squares.items())

    @staticmethod
    def sol(n):
        sqrt = {i * i: i for i in range(1, n)}
        trips = [(sqrt[i], sqrt[j], sqrt[i + j]) for i in sqrt for j in sqrt if i < j and i + j in sqrt]
        import random
        random.seed(0)
        sol = [random.randrange(2) for _ in range(n)]
        done = False
        while not done:
            done = True
            random.shuffle(trips)
            for i, j, k in trips:
                if sol[i] == sol[j] == sol[k]:
                    done = False
                    sol[random.choice([i, j, k])] = 1 - sol[i]
        return sol

    def gen(self, target_num_instances):
        for n in [7824] + list(range(target_num_instances)):
            if self.num_generated_so_far() == target_num_instances:
                return
            self.add(dict(n=n), test=n <= 100)


class ClockAngle(PuzzleGenerator):
    """[Clock Angle Problem](https://en.wikipedia.org/wiki/Clock_angle_problem), easy variant"""

    @staticmethod
    def sat(hands: List[int], target_angle=45):
        """Find clock hands = [hour, min] such that the angle is target_angle degrees."""
        h, m = hands
        assert 0 < h <= 12 and 0 <= m < 60
        hour_angle = 30 * h + m / 2
        minute_angle = 6 * m
        return abs(hour_angle - minute_angle) in [target_angle, 360 - target_angle]

    @staticmethod
    def sol(target_angle):
        for h in range(1, 13):
            for m in range(60):
                hour_angle = 30 * h + m / 2
                minute_angle = 6 * m
                if abs(hour_angle - minute_angle) % 360 in [target_angle, 360 - target_angle]:
                    return [h, m]

    def gen_random(self):
        target_angle = self.random.randrange(0, 360)
        if self.sol(target_angle):
            self.add(dict(target_angle=target_angle))


class Kirkman(PuzzleGenerator):
    """[Kirkman's problem](https://en.wikipedia.org/wiki/Kirkman%27s_schoolgirl_problem)"""

    @staticmethod
    def sat(daygroups: List[List[List[int]]]):
        """
        Arrange 15 people into groups of 3 each day for seven days so that no two people are in the same group twice.
        """
        assert len(daygroups) == 7
        assert all(len(groups) == 5 and {i for g in groups for i in g} == set(range(15)) for groups in daygroups)
        assert all(len(g) == 3 for groups in daygroups for g in groups)
        return len({(i, j) for groups in daygroups for g in groups for i in g for j in g}) == 15 * 15

    @staticmethod
    def sol():
        from itertools import combinations
        import random
        rand = random.Random(0)
        days = [[list(range(15)) for _2 in range(2)] for _ in range(7)]  # each day is pi, inv
        counts = {(i, j): (7 if j in range(k, k + 3) else 0)
                  for k in range(0, 15, 3)
                  for i in range(k, k + 3)
                  for j in range(15) if j != i
                  }

        todos = [pair for pair, count in counts.items() if count == 0]
        while True:
            pair = rand.choice(todos)  # choose i and j to make next to each other on some day
            if rand.randrange(2):
                pair = pair[::-1]

            a, u = pair
            pi, inv = rand.choice(days)
            assert pi[inv[a]] == a and pi[inv[u]] == u
            bases = [3 * (inv[i] // 3) for i in pair]
            (b, c), (v, w) = [[x for x in pi[b: b + 3] if x != i] for i, b in zip(pair, bases)]
            if rand.randrange(2):
                b, c, = c, b
            # current (a, b, c) (u, v, w). consider swap of u with b to make (a, u, c) (b, v, w)

            new_pairs = [(a, u), (c, u), (b, v), (b, w)]
            old_pairs = [(u, v), (u, w), (b, a), (b, c)]
            gained = sum(counts[p] == 0 for p in new_pairs)
            lost = sum(counts[p] == 1 for p in old_pairs)
            if rand.random() <= 100 ** (gained - lost):
                for p in new_pairs:
                    counts[p] += 1
                    counts[p[::-1]] += 1
                for p in old_pairs:
                    counts[p] -= 1
                    counts[p[::-1]] -= 1
                pi[inv[b]], pi[inv[u]], inv[b], inv[u] = u, b, inv[u], inv[b]
                todos = [pair for pair, count in counts.items() if count == 0]
                if len(todos) == 0:
                    return [[pi[k:k + 3] for k in range(0, 15, 3)] for pi, _inv in days]

        # return [[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8, 13], [4, 9, 14]], # wikipedia solution
        #         [[0, 1, 4], [2, 3, 6], [7, 8, 11], [9, 10, 13], [12, 14, 5]],
        #         [[1, 2, 5], [3, 4, 7], [8, 9, 12], [10, 11, 14], [13, 0, 6]],
        #         [[4, 5, 8], [6, 7, 10], [11, 12, 0], [13, 14, 2], [1, 3, 9]],
        #         [[2, 4, 10], [3, 5, 11], [6, 8, 14], [7, 9, 0], [12, 13, 1]],
        #         [[4, 6, 12], [5, 7, 13], [8, 10, 1], [9, 11, 2], [14, 0, 3]],
        #         [[10, 12, 3], [11, 13, 4], [14, 1, 7], [0, 2, 8], [5, 6, 9]]]


class MonkeyAndCoconuts(PuzzleGenerator):
    """[The Monkey and the Coconuts](https://en.wikipedia.org/wiki/The_monkey_and_the_coconuts)"""

    @staticmethod
    def sat(n: int):
        """
        Find the number of coconuts to solve the following riddle:
            There is a pile of coconuts, owned by five men. One man divides the pile into five equal piles, giving the
            one left over coconut to a passing monkey, and takes away his own share. The second man then repeats the
            procedure, dividing the remaining pile into five and taking away his share, as do the third, fourth, and
            fifth, each of them finding one coconut left over when dividing the pile by five, and giving it to a monkey.
            Finally, the group divide the remaining coconuts into five equal piles: this time no coconuts are left over.
            How many coconuts were there in the original pile?
                                              Quoted from https://en.wikipedia.org/wiki/The_monkey_and_the_coconuts
        """
        for i in range(5):
            assert n % 5 == 1
            n -= 1 + (n - 1) // 5
        return n > 0 and n % 5 == 1

    @staticmethod
    def sol():
        m = 1
        while True:
            n = m
            for i in range(5):
                if n % 5 != 1:
                    break
                n -= 1 + (n - 1) // 5
            if n > 0 and n % 5 == 1:
                return m
            m += 5


class No3Colinear(PuzzleGenerator):
    """[No three-in-a-line](https://en.wikipedia.org/wiki/No-three-in-line_problem)"""

    @staticmethod
    def sat(coords: List[List[int]], side=10, num_points=20):
        """Find num_points points in an side x side grid such that no three points are collinear."""
        for i1 in range(len(coords)):
            x1, y1 = coords[i1]
            assert 0 <= x1 < side and 0 <= y1 < side
            for i2 in range(i1):
                x2, y2 = coords[i2]
                for i3 in range(i2):
                    x3, y3 = coords[i3]
                    assert x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) != 0
        return len({(a, b) for a, b in coords}) == len(coords) >= num_points

    @staticmethod
    def sol(side, num_points):
        from itertools import combinations
        assert side <= 5 or side == 10, "Don't know how to solve other sides"

        def test(coords):
            return all(p[0] * (q[1] - r[1]) + q[0] * (r[1] - p[1]) + r[0] * (p[1] - q[1])
                       for p, q, r in combinations(coords, 3))

        if side <= 5:
            grid = [[i, j] for i in range(side) for j in range(side)]
            return next(list(coords) for coords in combinations(grid, num_points) if test(coords))

        if side == 10:
            def mirror(coords):  # rotate to all four corners
                return [[a, b] for x, y in coords for a in [x, side - 1 - x] for b in [y, side - 1 - y]]

            grid = [[i, j] for i in range(side // 2) for j in range(side // 2)]
            return next(list(mirror(coords)) for coords in combinations(grid, side // 2) if
                        test(coords) and test(mirror(coords)))

    def gen(self, target_num_instances):
        for easy in range(47):
            for side in range(47):
                if self.num_generated_so_far() == target_num_instances:
                    return
                test = side < 5 or side == 10
                num_points = 1 if side == 1 else 2 * side
                if num_points >= easy:
                    num_points -= easy
                    self.add(dict(side=side, num_points=num_points), test=test)


class PostageStamp(PuzzleGenerator):
    """[Postage stamp problem](https://en.wikipedia.org/wiki/Postage_stamp_problem)"""

    @staticmethod
    def sat(stamps: List[int], target=80, max_stamps=4, options=[10, 32, 8]):
        """Find a selection of at most max_stamps stamps whose total worth is the target value."""
        for s in stamps:
            assert s in options
        return len(stamps) <= max_stamps and sum(stamps) == target

    @staticmethod
    def sol(target, max_stamps, options):
        from itertools import combinations_with_replacement
        for n in range(max_stamps + 1):
            for c in combinations_with_replacement(options, n):
                if sum(c) == target:
                    return list(c)

    def gen_random(self):
        max_stamps = self.random.randrange(1, 10)
        options = [self.random.randrange(1, 100) for _ in range(self.random.randrange(1, 10))]
        target = sum(self.random.choices(options, k=self.random.randrange(1, max_stamps + 1)))
        self.add(dict(target=target, max_stamps=max_stamps, options=options))


class Sudoku(PuzzleGenerator):
    """The classic game of [Sudoku](https://en.wikipedia.org/wiki/Sudoku)"""

    @staticmethod
    def sat(x: str, puz='____9_2___7__________1_8_4____2_78____4_____1____69____2_8___5__6__3_7___49______'):
        """Find the unique valid solution to the Sudoku puzzle"""
        assert all(c == "_" or c == s for (c, s) in zip(puz, x))

        full = set('123456789')
        for i in range(9):
            assert {x[i] for i in range(9 * i, 9 * i + 9)} == full, "invalid row"
            assert {x[i] for i in range(i, i + 81, 9)} == full, "invalid column"
            assert {x[9 * a + b + i + 26 * (i % 3)] for a in range(3) for b in range(3)} == full, "invalid square"

        return True

    @staticmethod
    def solve(puz):
        """Simple depth-first backtracking solver that branches at the square with fewest possibilities"""
        sets = [{int(c)} if c != '_' else set(range(1, 10)) for c in puz]

        groups = []
        for i in range(9):
            groups.append(list(range(9 * i, 9 * i + 9)))
            groups.append(list(range(i, i + 81, 9)))
            groups.append([9 * a + b + i + 26 * (i % 3) for a in range(3) for b in range(3)])

        inv = [[] for i in range(81)]
        for g in groups:
            for i in g:
                inv[i].append(g)

        def reduce():
            """Reduce possibilities and return False if it's clearly impossible to solve, True otherwise.
            Repeatedly applies two types of logic:
            * When an entry has a single possibility, remove that value from all 20 neighbors
            * When a row/col/square has only one entry with k as a possibility, fill in that possibility
            """
            done = False
            while not done:
                done = True
                for i in range(81):
                    new = sets[i] - {k for g in inv[i] for j in g if j != i and len(sets[j]) == 1 for k in sets[j]}
                    if not new:
                        return False
                    if len(sets[i]) != len(new):
                        sets[i] = new
                        done = False

                for g in groups:
                    for k in range(1, 10):
                        possibilities = [i for i in g if k in sets[i]]
                        if not possibilities:
                            return False
                        if len(possibilities) == 1:
                            i = possibilities[0]
                            if len(sets[i]) > 1:
                                done = False
                                sets[i] = {k}

            return True

        ans = []

        counter = 0

        def solve_helper():
            nonlocal sets, ans, counter
            counter += 1
            assert len(ans) <= 1, "Sudoku puzzle should have a unique solution"
            old_sets = sets[:]
            if reduce():
                if all(len(s) == 1 for s in sets):
                    ans.append("".join(str(list(s)[0]) for s in sets))
                else:
                    smallest_set = min(range(81), key=lambda i: len(sets[i]) if len(sets[i]) > 1 else 10)
                    for v in sorted(sets[smallest_set]):
                        sets[smallest_set] = {v}
                        solve_helper()

            sets = old_sets

        solve_helper()
        assert ans, "No solution found"
        return ans[0]

    @staticmethod
    def print_board(board):
        """Helpful method used for development"""
        for i in range(9):
            for j in range(9):
                print(board[9 * i + j], end=" " if j == 2 or j == 5 else "")
            print()
            if i == 2 or i == 5:
                print()

    @staticmethod
    def print_sets(sets):
        """Helpful method used for development"""
        ans = ""
        for i in range(9):
            for j in range(9):
                ans += " " + "".join(str(k) if k in sets[9 * i + j] else "_" for k in range(1, 10))
                if j == 2 or j == 5:
                    ans += "  | "
            if i == 8:
                print(ans)
                return
            ans += "\n"
            if i == 2 or i == 5:
                ans += "\n"

    @staticmethod
    def gen_sudoku_puzzle(rand):

        groups = []
        for i in range(9):
            groups.append(list(range(9 * i, 9 * i + 9)))
            groups.append(list(range(i, i + 81, 9)))
            groups.append([9 * a + b + i + 26 * (i % 3) for a in range(3) for b in range(3)])

        inv = [[] for i in range(81)]
        for g in groups:
            for i in g:
                inv[i].append(g)

        def solve(puz):
            """Basically the same as our solver above except that it returns a list of (up to 2) solutions."""
            sets = [{int(c)} if c != '_' else set(range(1, 10)) for c in puz]

            def reduce():
                """Reduce possibilities and return False if it's clearly impossible to solve, True otherwise.
                Repeatedly applies two types of logic:
                * When an entry has a single possibility, remove that value from all 20 neighbors
                * When a row/col/square has only one entry with k as a possibility, fill in that possibility
                """
                done = False
                while not done:
                    done = True
                    for i in range(81):
                        new = sets[i] - {k for g in inv[i] for j in g if j != i and len(sets[j]) == 1 for k in sets[j]}
                        if not new:
                            return False
                        if len(sets[i]) != len(new):
                            sets[i] = new
                            done = False

                    for g in groups:
                        for k in range(1, 10):
                            possibilities = [i for i in g if k in sets[i]]
                            if not possibilities:
                                return False
                            if len(possibilities) == 1:
                                i = possibilities[0]
                                if len(sets[i]) > 1:
                                    done = False
                                    sets[i] = {k}

                return True

            ans = []

            counter = 0

            def solve_helper():
                nonlocal sets, ans, counter
                counter += 1
                if len(ans) > 1:
                    return
                old_sets = sets[:]
                if reduce():
                    if all(len(s) == 1 for s in sets):
                        ans.append("".join(str(list(s)[0]) for s in sets))
                    else:
                        smallest_set = min(range(81), key=lambda i: len(sets[i]) if len(sets[i]) > 1 else 10)
                        pi = sorted(sets[smallest_set])
                        rand.shuffle(pi)
                        for v in pi:
                            sets[smallest_set] = {v}
                            solve_helper()

                sets = old_sets

            solve_helper()
            return ans

        x = ["_"] * 81
        perm = list("123456789")
        rand.shuffle(perm)
        x[:9] == perm
        x = list(solve(x)[0])

        done = False
        while not done:
            done = True
            pi = list([i for i in range(81) if x[i] != "_"])
            rand.shuffle(pi)
            for i in pi:
                old = x[i]
                x[i] = "_"
                ans = solve("".join(x))
                assert ans
                if len(ans) > 1:
                    x[i] = old
                else:
                    done = False
            # print()
            # Sudoku.print_board(x)
            # print("                    ", 81-x.count("_"))

        return "".join(x)

    def gen_random(self):

        puz = None
        for attempt in range(10 if self.num_generated_so_far() < 10 else 1):
            puz2 = Sudoku.gen_sudoku_puzzle(self.random)
            if puz is None or puz2.count("_") > puz.count("_"):
                puz = puz2

        self.add(dict(puz=puz))


class SquaringTheSquare(PuzzleGenerator):
    """[Squaring the square](https://en.wikipedia.org/wiki/Squaring_the_square)
    Wikipedia gives a minimal [solution with 21 squares](https://en.wikipedia.org/wiki/Squaring_the_square)
    due to Duijvestijn (1978).
    """

    @staticmethod
    def sat(xy_sides: List[List[int]]):
        """
        Partition a square into smaller squares with unique side lengths. A perfect squared path has distinct sides.
        xy_sides is a List of (x, y, side)
        """
        n = max(x + side for x, y, side in xy_sides)
        assert len({side for x, y, side in xy_sides}) == len(xy_sides) > 1
        for x, y, s in xy_sides:
            assert 0 <= y < y + s <= n and 0 <= x
            for x2, y2, s2 in xy_sides:
                assert s2 <= s or x2 >= x + s or x2 + s2 <= x or y2 >= y + s or y2 + s2 <= y

        return sum(side ** 2 for x, y, side in xy_sides) == n ** 2

    @staticmethod
    def sol():
        return [[0, 0, 50], [0, 50, 29], [0, 79, 33], [29, 50, 25], [29, 75, 4], [33, 75, 37], [50, 0, 35],
                [50, 35, 15], [54, 50, 9], [54, 59, 16], [63, 50, 2], [63, 52, 7], [65, 35, 17], [70, 52, 18],
                [70, 70, 42], [82, 35, 11], [82, 46, 6], [85, 0, 27], [85, 27, 8], [88, 46, 24], [93, 27, 19]]


class NecklaceSplit(PuzzleGenerator):
    """[Necklace Splitting Problem](https://en.wikipedia.org/wiki/Necklace_splitting_problem)"""

    @staticmethod
    def sat(n: int, lace="bbrbrbbbbbbrrrrrrrbrrrrbbbrbrrbbbrbrrrbrrbrrbrbbrrrrrbrbbbrrrbbbrbbrbbbrbrbb"):
        """
        Find a split dividing the given red/blue necklace in half at n so that each piece has an equal number of
        reds and blues.
        """
        sub = lace[n: n + len(lace) // 2]
        return n >= 0 and lace.count("r") == 2 * sub.count("r") and lace.count("b") == 2 * sub.count("b")

    @staticmethod
    def sol(lace):
        if lace == "":
            return 0
        return next(n for n in range(len(lace) // 2) if lace[n: n + len(lace) // 2].count("r") == len(lace) // 4)

    def gen_random(self):
        m = 2 * self.random.randrange(self.random.choice([10, 100, 1000]))
        lace = ["r", "b"] * m
        self.random.shuffle(lace)
        lace = "".join(lace)
        self.add(dict(lace=lace))


class PandigitalSquare(PuzzleGenerator):
    """[Pandigital](https://en.wikipedia.org/wiki/Pandigital_number) Square"""

    @staticmethod
    def sat(n: int):
        """Find an integer whose square has all digits 0-9 once."""
        s = str(n * n)
        for i in "0123456789":
            assert s.count(i) == 1
        return True

    @staticmethod
    def sol():
        for n in range(10 ** 5):
            if sorted([int(s) for s in str(n * n)]) == list(range(10)):
                return n


class AllPandigitalSquares(PuzzleGenerator):
    """All [Pandigital](https://en.wikipedia.org/wiki/Pandigital_number) Squares"""

    @staticmethod
    def sat(nums: List[int]):
        """Find all 174 integers whose 10-digit square has all digits 0-9 just once."""
        return [sorted([int(s) for s in str(n * n)]) for n in set(nums)] == [list(range(10))] * 174

    @staticmethod
    def sol():
        return [i for i in range(-10 ** 5, 10 ** 5) if sorted([int(s) for s in str(i * i)]) == list(range(10))]


# MAYBE: add a version of TowersOfHanoiArbitrary where one has to find the fewest moves (maybe with more than 3 pegs)

class CardGame24(PuzzleGenerator):
    """[24 Game](https://en.wikipedia.org/wiki/24_Game)

    In this game one is given four numbers from the range 1-13 (Ace-King) and one needs to combine them with
        + - * / (and parentheses)
    to make the number 24.
    The solution to this tricky example is `7 * (3 + 3 / 7)`
    """

    @staticmethod
    def sat(expr: str, nums=[3, 7, 3, 7]):
        """Find a formula with two 3's and two 7's and + - * / (and parentheses) that evaluates to 24."""
        assert len(nums) == 4 and 1 <= min(nums) and max(nums) <= 13, "hint: nums is a list of four ints in 1..13"
        expr = expr.replace(" ", "")  # ignore whitespace
        digits = ""
        for i in range(len(expr)):
            if i == 0 or expr[i - 1] in "+*-/(":
                assert expr[i] in "123456789(", "Expr cannot contain **, //, or unary -"
            assert expr[i] in "1234567890()+-*/", "Expr can only contain `0123456789()+-*/`"
            digits += expr[i] if expr[i] in "0123456789" else " "
        assert sorted(int(s) for s in digits.split()) == sorted(nums), "Each number must occur exactly once"
        return abs(eval(expr) - 24.0) < 1e-6

    @staticmethod
    def sol(nums):
        def helper(pairs):
            if len(pairs) == 2:
                (x, s), (y, t) = pairs
                ans = {
                    x + y: f"{s}+{t}",
                    x - y: f"{s}-({t})",
                    y - x: f"{t}-({s})",
                    x * y: f"({s})*({t})"
                }
                if y != 0:
                    ans[x / y] = f"({s})/({t})"
                if x != 0:
                    ans[y / x] = f"({t})/({s})"
                return ans
            ans = {y: t
                   for i in range(len(pairs))
                   for x_s in helper(pairs[:i] + pairs[i + 1:]).items()
                   for y, t in helper([x_s, pairs[i]]).items()}
            if len(pairs) == 3:
                return ans
            ans.update({z: u
                        for i in range(1, 4)
                        for x_s in helper([pairs[0], pairs[i]]).items()
                        for y_t in helper(pairs[1:i] + pairs[i + 1:]).items()
                        for z, u in helper([x_s, y_t]).items()
                        })
            return ans

        derivations = helper([(n, str(n)) for n in nums])
        for x in derivations:
            if abs(x - 24.0) < 1e-6:
                return derivations[x]

    def gen_random(self):
        nums = [self.random.randint(1, 13) for _ in range(4)]
        if self.sol(nums):
            self.add({"nums": nums})


class Easy63(PuzzleGenerator):
    '''An easy puzzle to make 63 using two 8's and one 1's.'''

    @staticmethod
    def sat(s: str):
        """Find a formula using two 8s and two 1's and -+*/ that evaluates to 1."""
        return set(s) <= set("18-+*/") and s.count("8") == 2 and s.count("1") == 1 and eval(s) == 63

    @staticmethod
    def sol():
        return "8*8-1"


class Harder63(PuzzleGenerator):
    '''An harder puzzle to make 63 using three 8's and one 1's.'''

    @staticmethod
    def sat(s: str):
        """Find an expression using two 8s and two 1's and -+*/ that evaluates to 1."""
        return set(s) <= set("18-+*/") and s.count("8") == 3 and s.count("1") == 1 and eval(s) == 63

    @staticmethod
    def sol():
        return "8*8-1**8"


class WaterPouring(PuzzleGenerator):
    """[Water pouring puzzle](https://en.wikipedia.org/w/index.php?title=Water_pouring_puzzle&oldid=985741928)"""

    @staticmethod
    def sat(moves: List[List[int]], capacities=[8, 5, 3], init=[8, 0, 0], goal=[4, 4, 0]):
        """
        Given an initial state of water quantities in jugs and jug capacities, find a sequence of moves (pouring
        one jug into another until it is full or the first is empty) to reaches the given goal state.
        moves is list of [from, to] pairs
        """
        state = init.copy()

        for [i, j] in moves:
            assert min(i, j) >= 0, "Indices must be non-negative"
            assert i != j, "Cannot pour from same state to itself"
            n = min(capacities[j], state[i] + state[j])
            state[i], state[j] = state[i] + state[j] - n, n

        return state == goal

    @staticmethod
    def sol(capacities, init, goal):
        from collections import deque
        num_jugs = len(capacities)
        start = tuple(init)
        target = tuple(goal)
        trails = {start: ([], start)}
        queue = deque([tuple(init)])
        while target not in trails:
            state = queue.popleft()
            for i in range(num_jugs):
                for j in range(num_jugs):
                    if i != j:
                        n = min(capacities[j], state[i] + state[j])
                        new_state = list(state)
                        new_state[i], new_state[j] = state[i] + state[j] - n, n
                        new_state = tuple(new_state)
                        if new_state not in trails:
                            queue.append(new_state)
                            trails[new_state] = ([i, j], state)
        ans = []
        state = target
        while state != start:
            move, state = trails[state]
            ans.append(move)
        return ans[::-1]

    def gen_random(self):

        def random_reachable(capacities: List[int], init: List[int]):
            num_jugs = len(capacities)
            reachables = set()
            queue = {tuple(init)}
            while queue:
                state = queue.pop()
                if state not in reachables:
                    reachables.add(state)
                    for i in range(num_jugs):
                        for j in range(num_jugs):
                            if i != j:
                                n = min(capacities[j], state[i] + state[j])
                                new_state = list(state)
                                new_state[i], new_state[j] = state[i] + state[j] - n, n
                                new_state = tuple(new_state)
                                queue.add(new_state)
            return list(self.random.choice(sorted(reachables)))
            # sorted ensures same result if run twice despite use of sets

        capacities = [self.random.randrange(1, 1000) for _ in range(3)]
        init = [self.random.randrange(1, c + 1) for c in capacities]
        goal = random_reachable(capacities, init)
        self.add(dict(init=init, goal=goal, capacities=capacities))


class VerbalArithmetic(PuzzleGenerator):  # updated because the answer was given away in the docstring! OMG
    """
    Find a substitution of digits for characters to make the numbers add up in a sum like this:
    SEND + MORE = MONEY

    The first digit in any number cannot be 0. In this example the solution is `9567 + 1085 = 10652`.
    See [Wikipedia article](https://en.wikipedia.org/wiki/Verbal_arithmetic)
    """

    @staticmethod
    def sat(li: List[int], words=["SEND", "MORE", "MONEY"]):
        """
        Find a list of integers corresponding to the given list of strings substituting a different digit for each
        character, so that the last string corresponds to the sum of the previous numbers.
        """
        assert len(li) == len(words) and all(i > 0 and len(str(i)) == len(w) for i, w in zip(li, words))
        assert len({c for w in words for c in w}) == len({(d, c) for i, w in zip(li, words) for d, c in zip(str(i), w)})
        return sum(li[:-1]) == li[-1]

    @staticmethod
    def sol(words):
        pi = list(range(10))  # permutation
        letters = []
        order = {}
        steps = []
        tens = 1
        for col in range(1, 1 + max(len(w) for w in words)):
            for w in words:
                is_tot = (w is words[-1])
                if len(w) >= col:
                    c = w[-col]
                    if c in order:
                        if is_tot:
                            kind = "check"
                        else:
                            kind = "seen"
                    else:
                        if is_tot:
                            kind = "derive"
                        else:
                            kind = "add"
                        order[c] = len(letters)
                        letters.append(c)
                    steps.append((kind, order[c], tens))
            tens *= 10

        inits = [any(w[0] == c for w in words) for c in letters]

        def helper(pos, delta):  # on success, returns True and pi has the correct values
            if pos == len(steps):
                return delta == 0

            kind, i, tens = steps[pos]

            if kind == "seen":
                return helper(pos + 1, delta + tens * pi[i])

            if kind == "add":
                for j in range(i, 10):
                    if pi[j] != 0 or not inits[i]:  # not adding a leading 0
                        pi[i], pi[j] = pi[j], pi[i]
                        if helper(pos + 1, delta + tens * pi[i]):
                            return True
                        pi[i], pi[j] = pi[j], pi[i]
                return False
            if kind == "check":
                delta -= tens * pi[i]
                return (delta % (10 * tens)) == 0 and helper(pos + 1, delta)

            assert kind == "derive"
            digit = (delta % (10 * tens)) // tens
            if digit == 0 and inits[i]:
                return False  # would be a leading 0
            j = pi.index(digit)
            if j < i:
                return False  # already used
            pi[i], pi[j] = pi[j], pi[i]
            if helper(pos + 1, delta - tens * digit):
                return True
            pi[i], pi[j] = pi[j], pi[i]
            return False

        assert helper(0, 0)
        return [int("".join(str(pi[order[c]]) for c in w)) for w in words]

    _fixed = [
        ["FORTY", "TEN", "TEN", "SIXTY"],
        ["GREEN", "ORANGE", "COLORS"]
    ]

    def gen(self, target_num_instances):
        for words in self._fixed:
            self.add(dict(words=words))

    def gen_random(self):
        alpha = list("abcdefghijklmnopqrstuvwxyz")
        n = self.random.randrange(2, 10)
        nums = [self.random.randrange(10000) for _ in range(n)]
        nums.append(sum(nums))
        self.random.shuffle(alpha)
        words = ["".join(alpha[int(d)] for d in str(i)) for i in nums]
        self.add(dict(words=words))  # , test=False)


class SlidingPuzzle(PuzzleGenerator):
    """
    [Sliding puzzle](https://en.wikipedia.org/wiki/15_puzzle)
    The 3-, 8-, and 15-sliding puzzles are classic examples of A* search.
    The problem is NP-hard but the puzzles can all be solved with A* and an efficient representation.
    """

    @staticmethod
    def sat(moves: List[int], start=[[5, 0, 2, 3], [1, 9, 6, 7], [4, 14, 8, 11], [12, 13, 10, 15]]):
        """
        In this puzzle, you are given a board like:
        1 2 5
        3 4 0
        6 7 8

        and your goal is to transform it to:
        0 1 2
        3 4 5
        6 7 8

        by a sequence of swaps with the 0 square (0 indicates blank). The starting configuration is given by a 2d list
        of lists and the answer is represented by a list of integers indicating which number you swap with 0. In the
        above example, an answer would be [1, 2, 5]
        """

        locs = {i: [x, y] for y, row in enumerate(start) for x, i in enumerate(row)}  # locations, 0 stands for blank
        for i in moves:
            assert abs(locs[0][0] - locs[i][0]) + abs(locs[0][1] - locs[i][1]) == 1
            locs[0], locs[i] = locs[i], locs[0]
        return all(locs[i] == [i % len(start[0]), i // len(start)] for i in locs)

    @staticmethod
    def sol(start):
        from collections import defaultdict
        import math
        d = len(start)
        N = d * d
        assert all(len(row) == d for row in start)

        def get_state(
                li):  # state is an integer with 4 bits for each slot and the last 4 bits indicate where the blank is
            ans = 0
            for i in li[::-1] + [li.index(0)]:
                ans = (ans << 4) + i
            return ans

        start = get_state([i for row in start for i in row])
        target = get_state(list(range(N)))

        def h(state):  # manhattan distance
            ans = 0
            for i in range(N):
                state = (state >> 4)
                n = state & 15
                if n != 0:
                    ans += abs(i % d - n % d) + abs(i // d - n // d)
            return ans

        g = defaultdict(lambda: math.inf)
        g[start] = 0  # shortest p ath lengths
        f = {start: h(start)}  # f[s] = g[s] + h(s)
        backtrack = {}

        todo = {start}
        import heapq
        heap = [(f[start], start)]

        neighbors = [[i for i in [b - 1, b + 1, b + d, b - d] if i in range(N) and (b // d == i // d or b % d == i % d)]
                     for b in range(N)]

        def next_state(s, blank, i):
            assert blank == (s & 15)
            v = (s >> (4 * i + 4)) & 15
            return s + (i - blank) + (v << (4 * blank + 4)) - (v << (4 * i + 4))

        while todo:
            (dist, s) = heapq.heappop(heap)
            if f[s] < dist:
                continue
            if s == target:
                # compute path
                ans = []
                while s != start:
                    s, i = backtrack[s]
                    ans.append((s >> (4 * i + 4)) & 15)
                return ans[::-1]

            todo.remove(s)

            blank = s & 15
            score = g[s] + 1
            for i in neighbors[blank]:
                s2 = next_state(s, blank, i)

                if score < g[s2]:
                    # paths[s2] = paths[s] + [s[i]]
                    g[s2] = score
                    backtrack[s2] = (s, i)
                    score2 = score + h(s2)
                    f[s2] = score2
                    todo.add(s2)
                    heapq.heappush(heap, (score2, s2))

    def gen_random(self):
        d = self.random.randint(2, 4)
        N = d * d
        state = list(range(N))
        num_moves = self.random.randrange(100)
        for _ in range(num_moves):
            blank = state.index(0)
            delta = self.random.choice([-1, 1, d, -d])

            i = blank + delta
            if i not in range(N) or delta == 1 and i % d == 0 or delta == -1 and blank % d == 0:
                continue

            state[i], state[blank] = state[blank], state[i]
        start = [list(state[i:i + d]) for i in range(0, N, d)]
        self.add(dict(start=start))


if __name__ == "__main__":
    PuzzleGenerator.debug_problems()
