"""Problems inspired by [codeforces](https://codeforces.com)."""

from problems import Problem
from typing import List


# Hint: subclass Problem.Debug for quick testing. Run make_dataset.py to make the dataset
# See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle for more info


class IsEven(Problem):
    """Determine if n can be evenly divided into two equal numbers. (Easy)

    Inspired by [Codeforces Problem 4 A](https://codeforces.com/problemset/problem/4/A)
    """

    @staticmethod
    def sat(b: bool, n=10):
        i = 0
        while i <= n:
            if i + i == n:
                return b == True
            i += 1
        return b == False

    @staticmethod
    def sol(n):
        return n % 2 == 0

    def gen(self, target_num_instances):
        n = 0
        while len(self.instances) < target_num_instances:
            self.add(dict(n=n))
            n += 1


class Abbreviate(Problem):
    """Abbreviate strings longer than a given length by replacing everything but the first and last characters by
    an integer indicating how many characters there were in between them.

    Inspired by [Codeforces Problem 71 A](https://codeforces.com/problemset/problem/71/A)
    """

    @staticmethod
    def sat(s: str, word="antidisestablishmentarianism", max_len=10):
        if len(word) <= max_len:
            return word == s
        return int(s[1:-1]) == len(word[1:-1]) and word[0] == s[0] and word[-1] == s[-1]

    @staticmethod
    def sol(word, max_len):
        if len(word) <= max_len:
            return word
        return f"{word[0]}{len(word) - 2}{word[-1]}"

    def gen_random(self):
        word = self.random.pseudo_word(min_len=3, max_len=30)
        max_len = self.random.randrange(5, 15)
        self.add(dict(word=word, max_len=max_len))


class SquareTiles(Problem):
    """Find a minimal list of corner locations for a×a tiles that covers [0, m] × [0, n] and does not double-cover
    squares.

    Sample Input:
    m = 10
    n = 9
    a = 5
    target = 4

    Sample Output:
    [[0, 0], [0, 5], [5, 0], [5, 5]]

    Inspired by [Codeforces Problem 1 A](https://codeforces.com/problemset/problem/1/A)
    """

    @staticmethod
    def sat(corners: List[List[int]], m=10, n=9, a=5, target=4):
        covered = {(i + x, j + y) for i, j in corners for x in range(a) for y in range(a)}
        assert len(covered) == len(corners) * a * a, "Double coverage"
        return len(corners) <= target and covered.issuperset({(x, y) for x in range(m) for y in range(n)})

    @staticmethod
    def sol(m, n, a, target):
        return [[x, y] for x in range(0, m, a) for y in range(0, n, a)]

    def gen_random(self):
        a = self.random.randrange(1, 11)
        m = self.random.randrange(1, self.random.choice([10, 100, 1000]))
        n = self.random.randrange(1, self.random.choice([10, 100, 1000]))
        target = len(self.sol(m, n, a, None)) + self.random.randrange(5)  # give a little slack
        self.add(dict(a=a, m=m, n=n, target=target))


class EasyTwos(Problem):
    """
    Given a list of lists of triples of integers, return True for each list with a total of at least 2 and False for
    each other list.

    Inspired by [Codeforces Problem 231 A](https://codeforces.com/problemset/problem/231/A)"""

    @staticmethod
    def sat(lb: List[bool], trips=[[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]):
        return len(lb) == len(trips) and all(
            (b is True) if sum(s) >= 2 else (b is False) for b, s in zip(lb, trips))

    @staticmethod
    def sol(trips):
        return [sum(s) >= 2 for s in trips]

    def gen_random(self):
        trips = [[self.random.randrange(2) for _ in range(3)] for _ in range(self.random.randrange(20))]
        self.add(dict(trips=trips))


class DecreasingCountComparison(Problem):
    """
    Given a list of non-increasing integers and given an integer k, determine how many positive integers in the list
    are at least as large as the kth.

    Inspired by [Codeforces Problem 158 A](https://codeforces.com/problemset/problem/158/A)
    """

    @staticmethod
    def sat(n: int, scores=[100, 95, 80, 70, 65, 9, 9, 9, 4, 2, 1], k=6):
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), "Hint: scores are non-decreasing"
        return all(s >= scores[k] and s > 0 for s in scores[:n]) and all(s < scores[k] or s <= 0 for s in scores[n:])

    @staticmethod
    def sol(scores, k):
        threshold = max(scores[k], 1)
        return sum(s >= threshold for s in scores)

    def gen_random(self):
        n = min(len(self.instances) + 1, 100)
        max_score = self.random.randrange(50)
        scores = sorted([self.random.randrange(max_score + 1) for _ in range(n)], reverse=True)
        k = self.random.randrange(len(scores))
        self.add(dict(scores=scores, k=k))


class VowelDrop(Problem):
    """Given an alphabetic string s, remove all vowels (aeiouy/AEIOUY), insert a "." before each remaining letter
    (consonant), and make everything lowercase.

    Sample Input:
    s = "Problems"

    Sample Output:
    .p.r.b.l.m.s

    Inspired by [Codeforces Problem 118 A](https://codeforces.com/problemset/problem/118/A)
    """

    @staticmethod
    def sat(t: str, s="Problems"):
        i = 0
        for c in s.lower():
            if c in "aeiouy":
                continue
            assert t[i] == ".", f"expecting `.` at position {i}"
            i += 1
            assert t[i] == c, f"expecting `{c}`"
            i += 1
        return i == len(t)

    @staticmethod
    def sol(s):
        return "".join("." + c for c in s.lower() if c not in "aeiouy")

    def gen_random(self):
        s = "".join([self.random.choice([c.upper(), c]) for c in self.random.pseudo_word(max_len=50)])
        self.add(dict(s=s))


class DominoTile(Problem):
    """Tile an m x n checkerboard with 2 x 1 tiles. The solution is a list of fourtuples [i1, j1, i2, j2] with i2 == i1
    and j2 == j1 + 1 or i2 == i1 + 1 and j2 == j1 with no overlap.

    Inspired by [Codeforces Problem 50 A](https://codeforces.com/problemset/problem/50/A)
    """

    @staticmethod
    def sat(squares: List[List[int]], m=10, n=5, target=50):
        covered = []
        for i1, j1, i2, j2 in squares:
            assert (0 <= i1 <= i2 < m) and (0 <= j1 <= j2 < n) and (j2 - j1 + i2 - i1 == 1)
            covered += [(i1, j1), (i2, j2)]
        return len(set(covered)) == len(covered) == target

    @staticmethod
    def sol(m, n, target):
        if m % 2 == 0:
            ans = [[i, j, i + 1, j] for i in range(0, m, 2) for j in range(n)]
        elif n % 2 == 0:
            ans = [[i, j, i, j + 1] for i in range(m) for j in range(0, n, 2)]
        else:
            ans = [[i, j, i + 1, j] for i in range(1, m, 2) for j in range(n)]
            ans += [[0, j, 0, j + 1] for j in range(0, n - 1, 2)]
        return ans

    def gen_random(self):
        m, n = [self.random.randrange(1, 50) for _ in range(2)]
        target = m * n - (m * n) % 2
        self.add(dict(m=m, n=n, target=target))


class IncDec(Problem):
    """This straightforward problem is a little harder than the Codeforces one.
    Given a sequence of operations "++x", "x++", "--x", "x--", and a target value, find initial value so that the
    final value is the target value.

    Sample Input:
    ops = ["x++", "--x", "--x"]
    target = 12

    Sample Output:
    13

    Inspired by [Codeforces Problem 282 A](https://codeforces.com/problemset/problem/282/A)
    """

    @staticmethod
    def sat(n: int, ops=["x++", "--x", "--x"], target=19143212):
        for op in ops:
            if op in ["++x", "x++"]:
                n += 1
            else:
                assert op in ["--x", "x--"]
                n -= 1
        return n == target

    @staticmethod
    def sol(ops, target):
        return target - ops.count("++x") - ops.count("x++") + ops.count("--x") + ops.count("x--")

    def gen_random(self):
        target = self.random.randrange(10 ** 5)
        num_ops = self.random.randrange(self.random.choice([10, 100, 1000]))
        ops = [self.random.choice(["x++", "++x", "--x", "x--"]) for _ in range(num_ops)]
        n = self.sol(ops, target)
        self.add(dict(ops=ops, target=target))


class CompareInAnyCase(Problem):
    """Ignoring case, compare s, t lexicographically. Output 0 if they are =, -1 if s < t, 1 if s > t.

    Inspired by [Codeforces Problem 112 A](https://codeforces.com/problemset/problem/112/A)
    """

    @staticmethod
    def sat(n: int, s="aaAab", t="aAaaB"):
        if n == 0:
            return s.lower() == t.lower()
        if n == 1:
            return s.lower() > t.lower()
        if n == -1:
            return s.lower() < t.lower()
        return False

    @staticmethod
    def sol(s, t):
        if s.lower() == t.lower():
            return 0
        if s.lower() > t.lower():
            return 1
        return -1

    def mix_case(self, word):
        return "".join([self.random.choice([c.upper(), c.lower()]) for c in word])

    def gen_random(self):
        s = self.mix_case(self.random.pseudo_word())
        if self.random.randrange(3):
            t = self.mix_case(s[:self.random.randrange(len(s) + 1)] + self.random.pseudo_word())
        else:
            t = self.mix_case(s)
        self.add(dict(s=s, t=t))


class SlidingOne(Problem):
    """We are given a 5x5 bimatrix with a single 1 like:

    0 0 0 0 0
    0 0 0 0 1
    0 0 0 0 0
    0 0 0 0 0
    0 0 0 0 0

    Find a (minimal) sequence of row and column swaps to move the 1 to the center. A move is a string
    in "0"-"4" indicating a row swap and "a"-"e" indicating a column swap

    Inspired by [Codeforces Problem 263 A](https://codeforces.com/problemset/problem/263/A)
    """

    @staticmethod
    def sat(s: str,
            matrix=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            max_moves=3):
        matrix = [m[:] for m in matrix]  # copy
        for c in s:
            if c in "01234":
                i = "01234".index(c)
                matrix[i], matrix[i + 1] = matrix[i + 1], matrix[i]
            if c in "abcde":
                j = "abcde".index(c)
                for row in matrix:
                    row[j], row[j + 1] = row[j + 1], row[j]

        return len(s) <= max_moves and matrix[2][2] == 1

    @staticmethod
    def sol(matrix, max_moves):
        i = [sum(row) for row in matrix].index(1)
        j = matrix[i].index(1)
        ans = ""
        while i > 2:
            ans += str(i - 1)
            i -= 1
        while i < 2:
            ans += str(i)
            i += 1
        while j > 2:
            ans += "abcde"[j - 1]
            j -= 1
        while j < 2:
            ans += "abcde"[j]
            j += 1
        return ans

    def gen(self, target_num_instances):
        for i in range(5):
            for j in range(5):
                if len(self.instances) == target_num_instances:
                    return
                matrix = [[0] * 5 for _ in range(5)]
                matrix[i][j] = 1
                max_moves = abs(2 - i) + abs(2 - j)
                self.add(dict(matrix=matrix, max_moves=max_moves))


class SortPlusPlus(Problem):
    """Sort numbers in a sum of digits, e.g., 1+3+2+1 -> 1+1+2+3

    Inspired by [Codeforces Problem 339 A](https://codeforces.com/problemset/problem/339/A)
    """

    @staticmethod
    def sat(s: str, inp="1+1+3+1+3+2+2+1+3+1+2"):
        return all(s.count(c) == inp.count(c) for c in inp + s) and all(s[i - 2] <= s[i] for i in range(2, len(s), 2))

    @staticmethod
    def sol(inp):
        return "+".join(sorted(inp.split("+")))

    def gen_random(self):
        inp = "+".join(self.random.choice("123") for _ in range(self.random.randrange(50)))
        self.add(dict(inp=inp))


class CapitalizeFirstLetter(Problem):
    """Capitalize first letter of word

    Inspired by [Codeforces Problem 281 A](https://codeforces.com/problemset/problem/281/A)
    """

    @staticmethod
    def sat(s: str, word="konjac"):
        for i in range(len(word)):
            if i == 0:
                if s[i] != word[i].upper():
                    return False
            else:
                if s[i] != word[i]:
                    return False
        return True

    @staticmethod
    def sol(word):
        return word[0].upper() + word[1:]

    def gen_random(self):
        word = self.random.pseudo_word()
        self.add(dict(word=word))


class LongestSubsetString(Problem):
    """You are given a string consisting of a's, b's and c's, find any longest substring containing no repeated
    consecutive characters.

    Sample Input:
    `"abbbc"`

    Sample Output:
    `"abc"`

    Inspired by [Codeforces Problem 266 A](https://codeforces.com/problemset/problem/266/A)
    """

    @staticmethod
    def sat(t: str, s="abbbcabbac", target=7):
        i = 0
        for c in t:
            while c != s[i]:
                i += 1
            i += 1
        return len(t) >= target and all(t[i] != t[i + 1] for i in range(len(t) - 1))

    @staticmethod
    def sol(s, target):  # target is ignored
        return s[:1] + "".join([b for a, b in zip(s, s[1:]) if b != a])

    def gen_random(self):
        n = self.random.randrange(self.random.choice([10, 100, 1000]))
        s = "".join([self.random.choice("abc") for _ in range(n)])
        target = len(self.sol(s, target=None))
        self.add(dict(s=s, target=target))


# Ignoring inappropriate problem http://codeforces.com/problemset/problem/236/A

class FindHomogeneousSubstring(Problem):
    """You are given a string consisting of 0's and 1's. Find an index after which the subsequent k characters are
    all 0's or all 1's.

    Sample Input:
    s = 0000111111100000, k = 5

    Sample Output:
    4
    (or 5 or 6 or 11)

    Inspired by [Codeforces Problem 96 A](https://codeforces.com/problemset/problem/96/A)
    """

    @staticmethod
    def sat(n: int, s="0000111111100000", k=5):
        return s[n:n + k] == s[n] * k

    @staticmethod
    def sol(s, k):
        return s.index("0" * k if "0" * k in s else "1" * k)

    @staticmethod
    def sol2(s, k):
        import re
        return re.search(r"([01])\1{" + str(k - 1) + "}", s).span()[0]

    @staticmethod
    def sol3(s, k):
        if "0" * k in s:
            return s.index("0" * k)
        else:
            return s.index("1" * k)

    @staticmethod
    def sol4(s, k):
        try:
            return s.index("0" * k)
        except:
            return s.index("1" * k)

    def gen_random(self):
        k = self.random.randrange(1, 20)
        n = self.random.randrange(1, self.random.choice([10, 100, 1000]))
        s = "".join([self.random.choice("01") for _ in range(n)])
        if not ("0" * k in s or "1" * k in s):
            i = self.random.randrange(n + 1)
            s = s[:i] + self.random.choice(['0', '1']) * k + s[i:]
        self.add(dict(s=s, k=k))


class Triple0(Problem):
    """Find the missing triple of integers to make them all add up to 0 coordinatewise

    Inspired by [Codeforces Problem 630 A](https://codeforces.com/problemset/problem/69/A)
    """

    @staticmethod
    def sat(delta: List[int], nums=[[1, 2, 3], [9, -2, 8], [17, 2, 50]]):
        return all(sum(vec[i] for vec in nums) + delta[i] == 0 for i in range(3))

    @staticmethod
    def sol(nums):
        return [-sum(vec[i] for vec in nums) for i in range(3)]

    def gen_random(self):
        nums = [[self.random.randrange(-100, 100) for _ in range(3)] for _i in range(self.random.randrange(10))]
        self.add(dict(nums=nums))


class TotalDifference(Problem):
    """Find n such that n + a == b * (the sum of the first c integers)

    Inspired by [Codeforces Problem 546 A](https://codeforces.com/problemset/problem/546/A)
    """

    @staticmethod
    def sat(n: int, a=17, b=100, c=20):
        return n + a == sum([b * i for i in range(c)])

    @staticmethod
    def sol(a, b, c):
        return -a + sum([b * i for i in range(c)])

    def gen_random(self):
        a, b, c = [self.random.randrange(1, 100) for _ in range(3)]
        self.add(dict(a=a, b=b, c=c))


class TripleDouble(Problem):
    """Find n such that n + a == b * (the sum of the first c integers)

    Inspired by [Codeforces Problem 791 A](https://codeforces.com/problemset/problem/791/A)
    """

    @staticmethod
    def sat(n: int, v=17, w=100):
        for i in range(n):
            assert v <= w
            v *= 3
            w *= 2
        return v > w

    @staticmethod
    def sol(v, w):
        i = 0
        while v <= w:
            v *= 3
            w *= 2
            i += 1
        return i

    def gen_random(self):
        w = self.random.randrange(2, 10 ** 9)
        v = self.random.randrange(1, w)
        self.add(dict(v=v, w=w))


class RepeatDec(Problem):
    """Find the result of applying the following operation to integer m, n times: if the last digit is zero, remove
    the zero, otherwise subtract 1.

    Inspired by [Codeforces Problem 977 A](https://codeforces.com/problemset/problem/977/A)
    """

    @staticmethod
    def sat(res: int, m=1234578987654321, n=4):
        for i in range(n):
            m = (m - 1 if m % 10 else m // 10)
        return res == m

    @staticmethod
    def sol(m, n):
        for i in range(n):
            m = (m - 1 if m % 10 else m // 10)
        return m

    def gen_random(self):
        m = self.random.randrange(2, 10 ** 20)
        n = self.random.randrange(1, 10)
        self.add(dict(m=m, n=n))


class ShortestDecDelta(Problem):
    """Find a the shortest sequence of integers going from 1 to n where each difference is at most 10. Do not include
    1 or n in the sequence.

    Inspired by [Codeforces Problem 617 A](https://codeforces.com/problemset/problem/617/A)
    """

    @staticmethod
    def sat(li: List[int], n=149, upper=14):
        return len(li) <= upper and all(abs(a - b) <= 10 for a, b in zip([1] + li, li + [n]))

    @staticmethod
    def sol(n, upper):
        m = 1
        ans = []
        while True:
            m = min(n, m + 10)
            if m >= n:
                return ans
            ans.append(m)

    def gen_random(self):
        n = self.random.randrange(1, 10 ** 6)
        upper = len(self.sol(n, None))
        self.add(dict(n=n, upper=upper))


class MaxDelta(Problem):
    """Given a sequence of integer pairs, p_i, m_i, where \sum p_i-m_i = 0, find the maximum value, over t, of
    p_{t+1} + \sum_{i=1}^t p_i - m_i

    Inspired by [Codeforces Problem 116 A](https://codeforces.com/problemset/problem/116/A)
    """

    @staticmethod
    def sat(n: int, pairs=[[3, 0], [17, 1], [9254359, 19], [123, 9254359], [0, 123]]):
        assert sum(p - m for p, m in pairs) == 0, "oo"
        tot = 0
        success = False
        for p, m in pairs:
            tot -= m
            tot += p
            assert tot <= n
            if tot == n:
                success = True
        return success

    @staticmethod
    def sol(pairs):
        tot = 0
        n = 0
        for p, m in pairs:
            tot += p - m
            if tot > n:
                n = tot
        return n

    def gen_random(self):
        tot = 0
        pairs = []
        while self.random.randrange(10):
            m = self.random.randrange(tot + 1)
            p = self.random.randrange(10 ** 6)
            tot += p - m
            pairs.append([p, m])
        pairs.append([0, tot])
        self.add(dict(pairs=pairs))


class CommonCase(Problem):
    """Given a word, replace it either with an upper-case or lower-case depending on whether or not it has more
    capitals or lower-case letters. If it has strictly more capitals, use upper-case, otherwise, use lower-case.

    Inspired by [Codeforces Problem 59 A](https://codeforces.com/problemset/problem/59/A)
    """

    @staticmethod
    def sat(s_case: str, s="CanYouTellIfItHASmoreCAPITALS"):
        caps = 0
        for c in s:
            if c != c.lower():
                caps += 1
        return s_case == (s.upper() if caps > len(s) // 2 else s.lower())

    @staticmethod
    def sol(s):
        """
        This is a trivial puzzle, especially if the AI realizes that it can can just copy the solution from
        the problem"""
        caps = 0
        for c in s:
            if c != c.lower():
                caps += 1
        return (s.upper() if caps > len(s) // 2 else s.lower())  # duh, just take sat and return the answer checked for

    def gen_random(self):
        s = "".join([c.upper() if self.random.random() > 0.5 else c.lower() for c in self.random.pseudo_word(1, 30)])
        self.add(dict(s=s))


class Sssuubbstriiingg(Problem):
    """
    Find increasing indices to make the substring "substring"

    Inspired by [Codeforces Problem 58 A](https://codeforces.com/problemset/problem/58/A)
    """

    @staticmethod
    def sat(inds: List[int], string="Sssuubbstriiingg"):
        return inds == sorted(inds) and "".join(string[i] for i in inds) == "substring"

    @staticmethod
    def sol(string):
        target = "substring"
        j = 0
        ans = []
        for i in range(len(string)):
            while string[i] == target[j]:
                ans.append(i)
                j += 1
                if j == len(target):
                    return ans

    def gen_random(self):
        chars = list("substring")
        for _ in range(self.random.randrange(20)):
            i = self.random.randrange(len(chars) + 1)
            ch = self.random.choice("   abcdefghijklmnopqrstuvwxyz    ABCDEFGHIJKLMNOPQRSTUVWXYZ  ")
            chars.insert(i, ch)
        string = "".join(chars)
        self.add(dict(string=string))


class Sstriiinggssuubb(Problem):
    """
    Find increasing indices to make the substring "intelligent" (with a surprise twist)

    Inspired by [Codeforces Problem 58 A](https://codeforces.com/problemset/problem/58/A)
    """

    @staticmethod
    def sat(inds: List[int], string="enlightenment"):
        return inds == sorted(inds) and "".join(string[i] for i in inds) == "intelligent"

    @staticmethod
    def sol(string):
        target = "intelligent"
        j = 0
        ans = []
        for i in range(-len(string), len(string)):
            while string[i] == target[j]:
                ans.append(i)
                j += 1
                if j == len(target):
                    return ans

    def gen_random(self):
        chars = list("inteligent")
        i = self.random.randrange(len(chars))
        a, b = chars[:i][::-1], chars[i:][::-1]
        chars = []
        while a and b:
            chars.append(self.random.choice([a, b]).pop())
        while (a or b):
            chars.append((a or b).pop())
        for _ in range(self.random.randrange(20)):
            i = self.random.randrange(len(chars) + 1)
            ch = self.random.choice("   abcdefghijklmnopqrstuvwxyz    ABCDEFGHIJKLMNOPQRSTUVWXYZ  ")
            chars.insert(i, ch)
        string = "".join(chars)
        self.add(dict(string=string))


class Moving0s(Problem):
    """
    Find a sequence of 0's and 1's so that, after n_steps of swapping each adjacent (0, 1), target target sequence
    is achieved.

    Inspired by [Codeforces Problem 266 B](https://codeforces.com/problemset/problem/266/B)
    """

    @staticmethod
    def sat(seq: List[int], target=[1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], n_steps=4):
        s = seq[:]  # copy
        for step in range(n_steps):
            for i in range(len(seq) - 1):
                if (s[i], s[i + 1]) == (0, 1):
                    (s[i], s[i + 1]) = (1, 0)
        return s == target

    @staticmethod
    def sol(target, n_steps):
        s = target[:]  # copy
        for step in range(n_steps):
            for i in range(len(target) - 2, -1, -1):
                if (s[i], s[i + 1]) == (1, 0):
                    (s[i], s[i + 1]) = (0, 1)
        return s

    def gen_random(self):
        seq = [self.random.randrange(2) for _ in range(self.random.randrange(3, 20))]
        n_steps = self.random.randrange(len(seq))
        target = seq[:]  # copy
        for step in range(n_steps):
            for i in range(len(seq) - 1):
                if (target[i], target[i + 1]) == (0, 1):
                    (target[i], target[i + 1]) = (1, 0)
        self.add(dict(target=target, n_steps=n_steps))


class Factor47(Problem):
    """
    Find a integer factor of n whose decimal representation consists only of 7's and 4's.

    Inspired by [Codeforces Problem 122 A](https://codeforces.com/problemset/problem/122/A)
    """

    @staticmethod
    def sat(d: int, n=6002685529):
        return n % d == 0 and set(str(d)) <= {"4", "7"}

    @staticmethod
    def sol(n):
        def helper(so_far, k):
            if k > 0:
                return helper(so_far * 10 + 4, k - 1) or helper(so_far * 10 + 7, k - 1)
            return (n % so_far == 0) and so_far

        for length in range(1, len(str(n)) // 2 + 2):
            ans = helper(0, length)
            if ans:
                return ans

    def gen_random(self):
        length = self.random.randrange(1, 14)
        d = int("".join(self.random.choice("47") for _ in range(length)))
        n = self.random.randrange(1, 10 ** length) * d
        if self.sol(n) == d:
            self.add(dict(n=n))


class Count47(Problem):
    """
    Find a number bigger than n whose decimal representation has k 4's and 7's where k's decimal representation
    consists only of 4's and 7's

    Inspired by [Codeforces Problem 110 A](https://codeforces.com/problemset/problem/110/A)
    """

    @staticmethod
    def sat(d: int, n=123456789):
        return d > n and set(str(str(d).count("4") + str(d).count("7"))) <= {"4", "7"}

    @staticmethod
    def sol(n):
        return int("4444" + "0" * (len(str(n)) - 3))

    def gen_random(self):
        n = self.random.randrange(10 ** self.random.randrange(2, 30))
        self.add(dict(n=n))


class MaybeReversed(Problem):
    """
    Either reverse a string or don't based on the reversed flag

    Inspired by [Codeforces Problem 41 A](https://codeforces.com/problemset/problem/41/A)
    """

    @staticmethod
    def sat(s: str, target="reverse me", reverse=True):
        return (s[::-1] == target) == reverse

    @staticmethod
    def sol(target, reverse):
        return target[::-1] if reverse else target + "x"

    def gen_random(self):
        reverse = self.random.choice([True, False])
        target = self.random.pseudo_word()
        self.add(dict(target=target, reverse=reverse))


class MinBigger(Problem):
    """
    Find minimal counts whose sum is greater than half the total

    Inspired by [Codeforces Problem 160 A](https://codeforces.com/problemset/problem/160/A)
    """

    @staticmethod
    def sat(taken: List[int], val_counts=[[4, 3], [5, 2], [9, 3], [13, 13], [8, 11], [56, 1]], upper=11):
        advantage = 0
        for i, (val, count) in zip(taken, val_counts):
            assert 0 <= i <= count
            advantage += val * i - val * count / 2
        return len(taken) == len(val_counts) and advantage > 0 and sum(taken) <= upper

    @staticmethod
    def sol(val_counts, upper):
        n = len(val_counts)
        pi = sorted(range(n), key=lambda i: val_counts[i][0])
        needed = sum(a * b for a, b in val_counts) / 2 + 0.1
        ans = [0] * n
        while needed > 0:
            while val_counts[pi[-1]][1] == ans[pi[-1]]:
                pi.pop()
            i = pi[-1]
            ans[i] += 1
            needed -= val_counts[i][0]
        return ans

    def gen_random(self):
        val_counts = [[self.random.randrange(1, 100) for _ in "vc"] for i in range(self.random.randrange(1, 10))]
        upper = sum(self.sol(val_counts, None))
        self.add(dict(val_counts=val_counts, upper=upper))


class Dada(Problem):
    """
    Find a string with a given number of a's and d's

    Inspired by [Codeforces Problem 734 A](https://codeforces.com/problemset/problem/734/A)
    """

    @staticmethod
    def sat(s: str, a=5129, d=17):
        return s.count("a") == a and s.count("d") == d and len(s) == a + d

    @staticmethod
    def sol(a, d):
        return "a" * a + "d" * d

    def gen_random(self):
        a = self.random.randrange(10 ** 4)
        d = self.random.randrange(10 ** 4)
        self.add(dict(a=a, d=d))


class DistinctDigits(Problem):
    """
    Find a string with a given number of a's and d's

    Inspired by [Codeforces Problem 271 A](https://codeforces.com/problemset/problem/271/A)
    """

    @staticmethod
    def sat(nums: List[int], a=100, b=1000, count=1):
        return sum(len(str(n)) == len(set(str(n))) and a <= n <= b for n in set(nums)) >= count

    @staticmethod
    def sol(a, b, count):
        return [n for n in range(a, b + 1) if len(str(n)) == len(set(str(n)))]

    def gen_random(self):
        b = self.random.randrange(1, 10 ** 3)
        a = self.random.randrange(b)
        count = len(self.sol(a, b, None))
        self.add(dict(a=a, b=b, count=count))


class EasySum(Problem):
    """
    Add up 1 or 2 for numbers in a list depending on whether they exceed a threshold

    Inspired by [Codeforces Problem 677 A](https://codeforces.com/problemset/problem/677/A)
    """

    @staticmethod
    def sat(tot: int, nums=[2, 8, 25, 18, 99, 11, 17, 16], thresh=17):
        return tot == sum(1 if i < thresh else 2 for i in nums)

    @staticmethod
    def sol(nums, thresh):
        return sum(1 if i < thresh else 2 for i in nums)

    def gen_random(self):
        nums = [self.random.randrange(100) for _ in range(self.random.randrange(30))]
        thresh = self.random.randrange(1, 100)
        self.add(dict(nums=nums, thresh=thresh))


# TO ADD:  133A 467A 136A 344A 1030A 318A 158B 705A 580A 486A 61A 200B 131A
# 479A 405A 469A 208A 148A 228A 337A 144A 443A 1328A 25A 268A 520A 785A 996A 141A 1335A 492B 230A 339B 451A 4C 510A 230B
# 189A 750A 581A 155A 1399A 1352A 1409A 472A 732A 1154A 427A 455A 1367A 1343B 466A 723A 432A 758A 500A 1343A 313A 1353B
# 490A 1374A 1360A 1399B 1367B 703A 460A 1360B 489C 379A'


class FivePowers(Problem):
    """What are the last two digits of 5^n?

    Inspired by [Codeforces Problem 630 A](https://codeforces.com/problemset/problem/630/A)
    """

    @staticmethod
    def sat(s: str, n=7):
        return int(str(5 ** n)[:-2] + s) == 5 ** n

    @staticmethod
    def sol(n):
        return ("1" if n == 0 else "5" if n == 1 else "25")

    def gen(self, target_num_instances):
        for n in range(target_num_instances):
            self.add(dict(n=n))


class CombinationLock(Problem):
    """Shortest Combination Lock Path

    Given a starting a final lock position, find the (minimal) intermediate states, where each transition
    involves increasing or decreasing a single digit (mod 10), e.g.

    start = "012"
    combo = "329"
    output: ['112', '212', '312', '322', '321', '320']

    Inspired by [Codeforces Problem 540 A](https://codeforces.com/problemset/problem/540/A)
    """

    @staticmethod
    def sat(states: List[str], start="012", combo="329", target_len=6):
        assert all(len(s) == len(start) for s in states) and all(c in "0123456789" for s in states for c in s)
        for a, b in zip([start] + states, states + [combo]):
            assert sum(i != j for i, j in zip(a, b)) == 1
            assert all(abs(int(i) - int(j)) in {0, 1, 9} for i, j in zip(a, b))

        return len(states) <= target_len

    @staticmethod
    def sol(start, combo, target_len):
        n = len(start)
        ans = []
        a, b = [[int(c) for c in x] for x in [start, combo]]
        for i in range(n):
            while a[i] != b[i]:
                a[i] = (a[i] - 1 if (a[i] - b[i]) % 10 < 5 else a[i] + 1) % 10
                if a != b:
                    ans.append("".join(str(i) for i in a))
        return ans

    def gen_random(self):
        n = self.random.randrange(1, 11)
        start, combo = tuple("".join(str(self.random.randrange(10)) for i in range(n)) for _ in range(2))
        if start != combo:
            target_len = len(self.sol(start, combo, target_len=None))
            self.add(dict(start=start, combo=combo, target_len=target_len))


class CombinationLockObfuscated(CombinationLock):
    """An obfuscated version of CombinationLock above"""

    @staticmethod
    def sat(states: List[str], start="012", combo="329", target_len=6):
        return all(sum((int(a[i]) - int(b[i])) ** 2 % 10 for i in range(len(start))) == 1
                   for a, b in zip([start] + states, states[:target_len] + [combo]))


class InvertPermutation(Problem):
    """Find a string that, when a given permutation of characters is applied, has a given result.

    Inspired by [Codeforces Problem 474 A](https://codeforces.com/problemset/problem/474/A)
    """

    @staticmethod
    def sat(s: str, perm="qwertyuiopasdfghjklzxcvbnm", target="hello are you there?"):
        return "".join((perm[(perm.index(c) + 1) % len(perm)] if c in perm else c) for c in s) == target

    @staticmethod
    def sol(perm, target):
        return "".join((perm[(perm.index(c) - 1) % len(perm)] if c in perm else c) for c in target)

    def gen_random(self):
        perm = "qwertyuiopasdfghjklzxcvbnm"
        target = " ".join(self.random.pseudo_word() for _ in range(self.random.randrange(1, 10)))
        self.add(dict(perm=perm, target=target))


class SameDifferent(Problem):
    """
    Given a list of integers and a target length, create of the given length such that:
    * The first list must be all the same numbers.
    * The second must be all different.
    * The two lists together comprise a sublist of all the list items

    Inspired by [Codeforces Problem 1335 C](https://codeforces.com/problemset/problem/1335/C)
    """

    @staticmethod
    def sat(lists: List[List[int]], items=[5, 4, 9, 4, 5, 5, 5, 1, 5, 5], length=4):
        a, b = lists
        assert len(set(a)) == len(a) == len(b) == length and len(set(b)) == 1 and set(a + b) <= set(items)
        i = b[0]
        return (a + b).count(i) <= items.count(i)

    @staticmethod
    def sol(items, length):
        from collections import Counter
        [[a, count]] = Counter(items).most_common(1)
        assert count >= length
        seen = {a}
        dedup = [i for i in items if i not in seen and not seen.add(i)]
        return [(dedup + [a])[:length], [a] * length]

    def gen_random(self):
        items = [self.random.randrange(10) for _ in range(self.random.randrange(5, 100))]
        from collections import Counter
        count = Counter(items).most_common(1)[0][1]
        n = len(set(items))
        length = (count - 1) if count == n else min(count, n)
        self.add(dict(items=items, length=length))


class OnesAndTwos(Problem):
    """Find a sequence of 1's and 2's of a given length that that adds up to n

    Inspired by [Codeforces Problem 476 A](https://codeforces.com/problemset/problem/476/A)"""

    @staticmethod
    def sat(seq: List[int], n=10000, length=5017):
        return set(seq) <= {1, 2} and sum(seq) == n and len(seq) == length

    @staticmethod
    def sol(n, length):
        return [2] * (n - length) + [1] * (2 * length - n)

    def gen_random(self):
        n = self.random.randrange(10 ** self.random.randrange(5))
        length = self.random.randrange((n + 1) // 2, n + 1)
        self.add(dict(n=n, length=length))


class MinConsecutiveSum(Problem):
    """Find a sequence of k consecutive indices whose sum is minimal

    Inspired by [Codeforces Problem 363 B](https://codeforces.com/problemset/problem/363/B)"""

    @staticmethod
    def sat(start: int, k=3, upper=6, seq=[17, 1, 2, 65, 18, 91, -30, 100, 3, 1, 2]):
        return 0 <= start <= len(seq) - k and sum(seq[start:start + k]) <= upper

    @staticmethod
    def sol(k, upper, seq):
        return min(range(len(seq) - k + 1), key=lambda start: sum(seq[start:start + k]))

    def gen_random(self):
        k = self.random.randrange(1, 11)
        n = self.random.randrange(k, k + 10 ** self.random.randrange(3))
        seq = [self.random.randrange(-100, 100) for _ in range(n)]
        upper = min(sum(seq[start:start + k]) for start in range(n - k + 1))
        self.add(dict(k=k, upper=upper, seq=seq))


class MaxConsecutiveSum(Problem):
    """Find a sequence of k consecutive indices whose sum is maximal

    Inspired by [Codeforces Problem 363 B](https://codeforces.com/problemset/problem/363/B)"""

    @staticmethod
    def sat(start: int, k=3, lower=150, seq=[3, 1, 2, 65, 18, 91, -30, 100, 0, 19, 52]):
        return 0 <= start <= len(seq) - k and sum(seq[start:start + k]) >= lower

    @staticmethod
    def sol(k, lower, seq):
        return max(range(len(seq) - k + 1), key=lambda start: sum(seq[start:start + k]))

    def gen_random(self):
        k = self.random.randrange(1, 11)
        n = self.random.randrange(k, k + 10 ** self.random.randrange(3))
        seq = [self.random.randrange(-100, 100) for _ in range(n)]
        lower = max(sum(seq[start:start + k]) for start in range(n - k + 1))
        self.add(dict(k=k, lower=lower, seq=seq))


class MaxConsecutiveProduct(Problem):
    """Find a sequence of k consecutive indices whose product is maximal, possibly looping around

    Inspired by [Codeforces Problem 363 B](https://codeforces.com/problemset/problem/363/B)"""

    @staticmethod
    def sat(start: int, k=3, lower=100000, seq=[91, 1, 2, 64, 18, 91, -30, 100, 3, 65, 18]):
        prod = 1
        for i in range(start, start + k):
            prod *= seq[i]
        return prod >= lower

    @staticmethod
    def sol(k, lower, seq):
        def prod(start):
            ans = 1
            for i in range(start, start + k):
                ans *= seq[i]
            return ans

        return max(range(-len(seq), len(seq) - k + 1), key=prod)

    def gen_random(self):
        k = self.random.randrange(1, 11)
        n = self.random.randrange(k, k + 10 ** self.random.randrange(3))
        seq = [self.random.randrange(-100, 100) for _ in range(n)]

        def prod(start):
            ans = 1
            for i in range(start, start + k):
                ans *= seq[i]
            return ans

        lower = max(prod(i) for i in range(-len(seq), len(seq) - k + 1))

        self.add(dict(k=k, lower=lower, seq=seq))


class DistinctOddSum(Problem):
    """Find n distinct positive odd integers that sum to tot

    Inspired by [Codeforces Problem 1327 A](https://codeforces.com/problemset/problem/1327/A)"""

    @staticmethod
    def sat(nums: List[int], tot=12345, n=5):
        return len(nums) == len(set(nums)) == n and sum(nums) == tot and all(i >= i % 2 > 0 for i in nums)

    @staticmethod
    def sol(tot, n):
        return list(range(1, 2 * n - 1, 2)) + [tot - sum(range(1, 2 * n - 1, 2))]

    def gen_random(self):
        n = self.random.randrange(1, 100)
        tot = sum(self.random.sample(range(1, max(2 * n + 2, 1000), 2), n))

        self.add(dict(tot=tot, n=n))


class MinRotations(Problem):
    """
    We begin with the string `"a...z"`

    An `r`-rotation of a string means shifting it to the right (positive) or left (negative) by `r` characters and
    cycling around. Given a target string of length n, find the n rotations that put the consecutive characters
    of that string at the beginning of the r-rotation, with minimal sum of absolute values of the `r`'s.

    For example if the string was `'dad'`, the minimal rotations would be `[3, -3, 3]` with a total of `9`.

    Inspired by [Codeforces Problem 731 A](https://codeforces.com/problemset/problem/731/A)"""

    @staticmethod
    def sat(rotations: List[int], target='dad', upper=9):
        s = "abcdefghijklmnopqrstuvwxyz"
        assert len(rotations) == len(target)
        for r, c in zip(rotations, target):
            s = s[r:] + s[:r]
            assert s[0] == c

        return sum(abs(r) for r in rotations) <= upper

    @staticmethod
    def sol(target, upper):
        s = "abcdefghijklmnopqrstuvwxyz"
        ans = []
        for c in target:
            i = s.index(c)
            r = min([i, i - len(s)], key=abs)
            ans.append(r)
            s = s[r:] + s[:r]
            assert s[0] == c
        return ans

    def gen_random(self):
        target = self.random.pseudo_word()
        upper = sum(abs(r) for r in self.sol(target, None))
        self.add(dict(target=target, upper=upper))


if __name__ == "__main__":
    Problem.debug_problems()
