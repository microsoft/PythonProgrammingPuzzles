"""Problems inspired by [codeforces](https://codeforces.com)."""

from problems import Problem, register, get_problems
from typing import List


@register
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

    def gen(self, target_num_problems):
        n = 0
        while len(self.instances) < target_num_problems:
            self.add(dict(n=n))
            n += 1


@register
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

    def sol(word, max_len):
        if len(word) <= max_len:
            return word
        return f"{word[0]}{len(word) - 2}{word[-1]}"

    def gen_random(self):
        word = self.random.pseudo_word(min_len=3, max_len=30)
        max_len = self.random.randrange(5, 15)
        self.add(dict(word=word, max_len=max_len))


@register
class SquareTiles(Problem):
    """Find a minimal list of corner locations for a×a tiles that covers [0, m] × [0, n] 
    and does not double-cover squares.    
    
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


@register
class EasyTwos(Problem):
    """
    Given a list of lists of triples of integers, return True for each list with a total of at least 2 and False
    for each other list.

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


@register
class DecreasingCountComparison(Problem):
    """
    Given a list of non-increasing integers and given an integer k, determine how many positive integers in the list
    are at least as large as the kth.

    Inspired by [Codeforces Problem 158 A](https://codeforces.com/problemset/problem/158/A)"""

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

    Inspired by [Codeforces Problem 118 A](https://codeforces.com/problemset/problem/118/A)"""

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


@register
class DominoTile(Problem):
    """Tile an m x n checkerboard with 2 x 1 tiles. The solution is a list of fourtuples [i1, j1, i2, j2]
    with i2 == i1 and j2 == j1 + 1 or i2 == i1 + 1 and j2 == j1 with no overlap.

    Inspired by [Codeforces Problem 50 A](https://codeforces.com/problemset/problem/50/A)"""

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


@register
class IncDec(Problem):
    """This straightforward problem is a little harder than the Codeforces one.
    Given a sequence of operations "++x",
    "x++", "--x", "x--", and a target value, find initial value so that the final value is the target value.

    Sample Input:
    ops = ["x++", "--x", "--x"]
    target = 12

    Sample Output:
    13

    Inspired by [Codeforces Problem 282 A](https://codeforces.com/problemset/problem/282/A)"""

    @staticmethod
    def sat(n: int, ops=["x++", "--x", "--x"], target=12):
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


@register
class CompareInAnyCase(Problem):
    """Ignoring case, compare s, t lexicographically. Output 0 if they are =, -1 if s < t, 1 if s > t.

    Inspired by [Codeforces Problem 112 A](https://codeforces.com/problemset/problem/112/A)"""

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


@register
class SlidingOne(Problem):
    """We are given a 5x5 bimatrix with a single 1 like:
    0 0 0 0 0
    0 0 0 0 1
    0 0 0 0 0
    0 0 0 0 0
    0 0 0 0 0

    Find a (minimal) sequence of row and column swaps to move the 1 to the center. A move is a string
    in "0"-"4" indicating a row swap and "a"-"e" indicating a column swap

    Inspired by [Codeforces Problem 263 A](https://codeforces.com/problemset/problem/263/A)"""

    @staticmethod
    def sat(s: str,
            matrix=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            max_moves=3):
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

    def gen(self, target_num_problems):
        for i in range(5):
            for j in range(5):
                if len(self.instances) == target_num_problems:
                    return
                matrix = [[0] * 5 for _ in range(5)]
                matrix[i][j] = 1
                max_moves = abs(2 - i) + abs(2 - j)
                self.add(dict(matrix=matrix, max_moves=max_moves))


@register
class SortPlusPlus(Problem):
    """Sort numbers in a sum of digits, e.g., 1+3+2+1 -> 1+1+2+3

    Inspired by [Codeforces Problem 339 A](https://codeforces.com/problemset/problem/339/A)"""

    @staticmethod
    def sat(s: str, inp="1+1+3+1+3+2+2+1+3+1+2"):
        return all(s.count(c) == inp.count(c) for c in inp + s) and all(s[i - 2] <= s[i] for i in range(2, len(s), 2))

    @staticmethod
    def sol(inp):
        return "+".join(sorted(inp.split("+")))

    def gen_random(self):
        inp = "+".join(self.random.choice("123") for _ in range(self.random.randrange(50)))
        self.add(dict(inp=inp))


@register
class CapitalizeFirstLetter(Problem):
    """Capitalize first letter of word

    Inspired by [Codeforces Problem 281 A](https://codeforces.com/problemset/problem/281/A)"""

    @staticmethod
    def sat(s: str, word="konjac"):
        for i in range(len(s)):
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


@register
class LongestSubsetString(Problem):
    """You are given a string consisting of a's, b's and c's, find any longest substring containing no
    repeated consecutive characters.

    Sample Input:
    abbbc

    Sample Output:
    abc

    Inspired by [Codeforces Problem 266 A](https://codeforces.com/problemset/problem/266/A)"""

    @staticmethod
    def sat(t: str, s="abbbcabbac", target=7):
        i = 0
        for c in t:
            while c != s[i]:
                i += 1
            i += 1
        return len(t) >= target

    @staticmethod
    def sol(s, target): # target is ignored
        return s[:1] + "".join([b for a, b in zip(s, s[1:]) if b != a])

    def gen_random(self):
        n = self.random.randrange(self.random.choice([10, 100, 1000]))
        s = "".join([self.random.choice("abc") for _ in range(n)])
        target = len(self.sol(s, target=None))
        self.add(dict(s=s, target=target))


@register
class FindHomogeneousSubstring(Problem):
    """You are given a string consisting of 0's and 1's. Find an index after which the subsequent k characters are
    all 0's or all 1's.

    Sample Input:
    s = 0000111111100000, k = 5

    Sample Output:
    4
    (or 5 or 6 or 11)

    Inspired by [Codeforces Problem 96 A](https://codeforces.com/problemset/problem/96/A)"""

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


@register
class FivePowers(Problem):
    """What are the last two digits of 5^n?

    Inspired by [Codeforces Problem 630 A](https://codeforces.com/problemset/problem/630/A)"""

    @staticmethod
    def sat(s: str, n=7):
        return int(str(5 ** n)[:-2] + s) == 5 ** n

    @staticmethod
    def sol(n):
        return ("1" if n == 0 else "5" if n == 1 else "25")

    def gen(self, target_num_problems):
        for n in range(target_num_problems):
            self.add(dict(n=n))


@register
class CombinationLock(Problem):
    """Shortest Combination Lock Path

    Given a starting a final lock position, find the (minimal) intermediate states, where each transition
    involves increasing or decreasing a single digit (mod 10)
    e.g.
    start = "012"
    combo = "329"

    output: ['112', '212', '312', '322', '321', '320']

    Inspired by [Codeforces Problem 540 A](https://codeforces.com/problemset/problem/540/A)"""

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


@register
class CombinationLockObfuscated(CombinationLock):
    """An obfuscated version of CombinationLock above"""

    @staticmethod
    def sat(states: List[str], start="012", combo="329", target_len=6):
        return all(sum((int(a[i]) - int(b[i])) ** 2 % 10 for i in range(len(start))) == 1
                   for a, b in zip([start] + states, states[:target_len] + [combo]))


@register
class InvertPermutation(Problem):
    """Find a string that, when a given permutation of characters is applied, has a given result.

    Inspired by [Codeforces Problem 474 A](https://codeforces.com/problemset/problem/474/A)"""

    @staticmethod
    def sat(s: str, perm="qwertyuiopasdfghjklzxcvbnm", target="hello are you there?"):
        return "".join((perm[(perm.index(c)+1) % len(perm)] if c in perm else c) for c in s) == target

    @staticmethod
    def sol(perm, target):
        return "".join((perm[(perm.index(c) - 1) % len(perm)] if c in perm else c) for c in target)

    def gen_random(self):
        perm = "qwertyuiopasdfghjklzxcvbnm"
        target = " ".join(self.random.pseudo_word() for _ in range(self.random.randrange(1, 10)))
        self.add(dict(perm=perm, target=target))

if __name__ == "__main__":
    for problem in get_problems(globals()):
        problem.test()
