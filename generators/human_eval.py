"""Problems inspired by [HumanEval dataset](https://github.com/openai/human-eval) described
in the [codex paper](https://arxiv.org/abs/2107.03374), specifically,
[this](https://github.com/openai/human-eval/blob/fa06031e684fbe1ee429c7433809460c159b66ad/data/HumanEval.jsonl.gz)
version released 7/7/21."""

from puzzle_generator import PuzzleGenerator
from typing import List

"""
Some that came out especially nicely as puzzles:
ParenthesesPermutation
Derivative
Frac/ClosestInteger
HeronTriangle
RomanNumerals
ClosestPalindrome
WildSort
Intersperse
SimplifyProductFraction
Fib4
MinSquaredDeviation
DiffChars
RotateString
EvaluateOperators
Grader
Median
TripleZeroSum
PrimeFib

Some that weren't such natural puzzles:
CircularShiftNum
ReplaceMe
MinSubArraySum
Buckets
OddEvenSum
FindStrangeSum
EvenSqure
StrongestExtension
HungryRabbits
ReverseCase
MatchBrackets
ListTotal
BelowThreshold
RemoveVowels
"""


# See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle to learn about adding puzzles



class FindCloseElements(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#0"""

    @staticmethod
    def sat(pair: List[float], nums=[0.17, 21.3, 5.0, 9.0, 11.0, 4.99, 17.0, 17.0, 12.4, 6.8]):
        """
        Given a list of numbers, find the two closest distinct numbers in the list.

        Sample Input:
        [1.2, 5.23, 0.89, 21.0, 5.28, 1.2]

        Sample Output:
        [5.23, 5.28]
        """
        a, b = pair
        assert a in nums and b in nums and a != b
        return abs(a - b) == min(x - y for x in nums for y in nums if x > y)

    @staticmethod
    def sol(nums):
        s = sorted(set(nums))
        return min([[a, b] for a, b in zip(s, s[1:])], key=lambda x: x[1] - x[0])

    def gen_random(self):
        nums = [self.random.uniform(-10, 10) for _ in range(self.random.randrange(2, 10))]
        nums.append(self.random.choice(nums))
        self.random.shuffle(nums)
        self.add(dict(nums=nums))


class SeparateParenGroups(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#1"""

    @staticmethod
    def sat(ls: List[str], combined='() (()) ((() () ())) (() )'):
        """
        Given a string consisting of whitespace and groups of matched parentheses, split it
        into groups of perfectly matched parentheses without any whitespace.

        Sample Input:
        '( ()) ((()()())) (()) ()'

        Sample Output:
        ['(())', '((()()()))', '(())', '()']
        """
        for s in ls:
            assert s.count("(") == s.count(")")
            assert all(s[:i].count("(") > s[:i].count(")") for i in range(1, len(s)))  # s is not further divisible
        return ''.join(ls) == combined.replace(' ', '')

    @staticmethod
    def sol(combined):
        cur = ''
        ans = []
        depth = 0
        for c in combined.replace(' ', ''):
            cur += c
            if c == '(':
                depth += 1
            else:
                assert c == ')'
                depth -= 1
                if depth == 0:
                    ans.append(cur)
                    cur = ''
        return ans

    def gen_random(self):
        depth = 0
        combined = ''
        while depth > 0 or self.random.random() > 0.2:
            c = self.random.choice('()) ' if depth > 0 else '( ')
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            combined += c
        self.add(dict(combined=combined))


class Frac(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#2"""

    @staticmethod
    def sat(x: float, v=523.12892):
        """
        Given a floating point number, find its fractional part.

        Sample Input:
        4.175

        Sample Output:
        0.175
        """
        return 0 <= x < 1 and (v - x).is_integer()

    @staticmethod
    def sol(v):
        return v % 1.0

    def gen_random(self):
        v = self.random.uniform(-100, 100)
        self.add(dict(v=v))


class FirstNegCumulative(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#3"""

    @staticmethod
    def sat(firsts: List[int], balances=[[2, 7, -2, 4, 3, -15, 10, -45, 3], [3, 4, -17, -1], [100, -100, -101], [-1]]):
        """
        Given a list of numbers which represent bank deposits and withdrawals, find the *first* negative balance.

        Sample Input:
        [12, -5, 3, -99, 14, 88, -99]

        Sample Output:
        -89
        """
        for i, bals in enumerate(balances):
            total = 0
            for b in bals:
                total += b
                if total < 0:
                    assert total == firsts[i]
                    break
        return True

    @staticmethod
    def sol(balances):
        firsts = []
        for bals in balances:
            total = 0
            for b in bals:
                total += b
                if total < 0:
                    firsts.append(total)
                    break
        return firsts

    def gen_random(self):
        balances = [
            [self.random.randrange(-10 ** 10, 10 ** 10) for _ in range(self.random.randrange(1, 11))]
            for _ in range(10)
        ]
        balances = [bals for bals in balances if any(sum(bals[:i + 1]) < 0 for i in range(len(bals)))]
        self.add(dict(balances=balances))


class MinSquaredDeviation(PuzzleGenerator):
    """
    Loosely inspired by [HumanEval](https://github.com/openai/human-eval) \\#4

    The HumanEval problem was simply to compute the mean absolute deviation. This problem is more interesting.
    It requires minimizing the sum of squared deviations, which turns out to be the mean `mu`. Moreover, if
    `mu` is the mean of the numbers then a simple calculation shows that:

    `sum((mu - n) ** 2 for n in nums) == sum((m - n) ** 2 for m in nums for n in nums) / (2 * len(nums))`

    We use 0.501 rather than 1/2 to deal with rounding errors.
    """

    @staticmethod
    def sat(x: float, nums=[12, -2, 14, 3, -15, 10, -45, 3, 30]):
        """
        Given a list of numbers, find x that minimizes mean squared deviation.

        Sample Input:
        [4, -5, 17, -9, 14, 108, -9]

        Sample Output:
        17.14285
        """
        return sum((n - x) ** 2 for n in nums) * len(nums) <= sum((m - n) ** 2 for m in nums for n in nums) * .5 + 1e-4

    @staticmethod
    def sol(nums):
        return sum(nums) / len(nums)  # mean minimizes mean squared deviation

    def gen_random(self):
        length = self.random.randrange(1, 11)
        nums = [self.random.randrange(-100, 100) for _ in range(length)]
        self.add(dict(nums=nums))


class Intersperse(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#5"""

    @staticmethod
    def sat(li: List[int], nums=[12, 23, -2, 5, 0], sep=4):
        """
        Given a list of numbers and a number to inject, create a list containing that number in between each pair of
        adjacent numbers.

        Sample Input:
        [8, 14, 21, 17, 9, -5], 3

        Sample Output:
        [8, 3, 14, 3, 21, 3, 17, 3, 9, 3, -5]
        """
        return li[::2] == nums and li[1::2] == [sep] * (len(nums) - 1)

    @staticmethod
    def sol(nums, sep):
        ans = [sep] * (2 * len(nums) - 1)
        ans[::2] = nums
        return ans

    def gen_random(self):
        length = self.random.randrange(10)
        nums = [self.random.randrange(100) for _ in range(length)]
        sep = self.random.randrange(100)
        self.add(dict(nums=nums, sep=sep))


class DeepestParens(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#6"""

    @staticmethod
    def sat(depths: List[int], parens='() (()) ((()()())) (((((((())))))))'):
        """
        Given a string consisting of groups of matched nested parentheses separated by parentheses,
        compute the depth of each group.

        Sample Input:
        '(()) ((()()())) (()) ()'

        Sample Output:
        [2, 3, 2, 1]
        """
        groups = parens.split()
        for depth, group in zip(depths, groups):
            budget = depth
            success = False
            for c in group:
                if c == '(':
                    budget -= 1
                    if budget == 0:
                        success = True
                    assert budget >= 0
                else:
                    assert c == ')'
                    budget += 1
            assert success

        return len(groups) == len(depths)

    @staticmethod
    def sol(parens):
        def max_depth(s):
            m = 0
            depth = 0
            for c in s:
                if c == '(':
                    depth += 1
                    m = max(m, depth)
                else:
                    assert c == ')'
                    depth -= 1
            assert depth == 0
            return m

        return [max_depth(s) for s in parens.split()]

    def gen_random(self):
        def gen_group():
            ans = ''
            depth = 0
            while depth > 0 or ans == '' or self.random.random() > 0.2:
                c = self.random.choice('())') if depth > 0 else '('
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                ans += c
            return ans

        parens = " ".join(gen_group() for _ in range(self.random.randrange(6)))
        self.add(dict(parens=parens))


class FindContainers(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#7"""

    @staticmethod
    def sat(containers: List[str], strings=['cat', 'dog', 'shatter', 'bear', 'at', 'ta'], substring='at'):
        """
        Find the strings in a list containing a given substring

        Sample Input:
        ['cat', 'dog', 'bear'], 'a'

        Sample Output:
        ['cat', 'bear']
        """
        i = 0
        for s in strings:
            if substring in s:
                assert containers[i] == s
                i += 1
        return i == len(containers)

    @staticmethod
    def sol(strings, substring):
        return [s for s in strings if substring in s]

    def gen_random(self):
        substring = self.random.pseudo_word(min_len=0, max_len=3)

        def gen():
            n = self.random.choice([1, 2])
            return substring.join([self.random.pseudo_word(min_len=0, max_len=5) for _ in range(n)])

        strings = [gen() for _ in range(self.random.randrange(6))]
        self.add(dict(strings=strings, substring=substring))


class SumProduct(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#8"""

    @staticmethod
    def sat(nums: List[int], tot=14, prod=99):
        """
        Find a list of numbers with a given sum and a given product.

        Sample Input:
        12, 32

        Sample Output:
        [2, 8, 2]
        """
        assert sum(nums) == tot
        p = 1
        for n in nums:
            p *= n
        return p == prod

    @staticmethod
    def sol(tot, prod):
        ans = [prod]
        while sum(ans) > tot:
            ans += [-1, -1]
        ans += [1] * (tot - sum(ans))
        return ans

    def gen_random(self):
        tot = self.random.randrange(-100, 100)
        prod = self.random.randrange(-100, 100)
        self.add(dict(tot=tot, prod=prod))


class RollingMax(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#9"""

    @staticmethod
    def sat(maxes: List[int], nums=[1, 4, 3, -6, 19]):
        """
        Find a list whose ith element is the maximum of the first i elements of the input list.

        Sample Input:
        [2, 8, 2]

        Sample Output:
        [2, 8, 8]
        """
        assert len(maxes) == len(nums)
        for i in range(len(nums)):
            if i > 0:
                assert maxes[i] == max(maxes[i - 1], nums[i])
            else:
                assert maxes[0] == nums[0]
        return True

    @staticmethod
    def sol(nums):
        return [max(nums[:i]) for i in range(1, len(nums) + 1)]

    @staticmethod
    def sol2(nums):
        ans = []
        if nums:
            m = nums[0]
            for n in nums:
                m = max(n, m)
                ans.append(m)
        return ans

    def gen_random(self):
        nums = [self.random.randrange(-100, 100) for _ in range(self.random.randrange(10))]
        self.add(dict(nums=nums))


class PalindromeContaining(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#10"""

    @staticmethod
    def sat(ans: str, s="so easy", length=20):
        """
        Find a palindrome of a given length containing a given string.

        Sample Input:
        "abba", 6

        Sample Output:
        "cabbac"
        """
        return ans == ans[::-1] and len(ans) == length and s in ans

    @staticmethod
    def sol(s, length):
        ls = list(s)
        for i in range(length - len(s) + 1):
            arr = ['x'] * length
            arr[i:i + len(s)] = ls
            a = length - i - 1
            b = length - (i + len(s)) - 1
            if b == -1:
                b = None
            arr[a:b:-1] = ls
            if arr == arr[::-1]:
                ans = "".join(arr)
                if s in ans:
                    return ans
        assert False, "shouldn't reach here"

    def gen_random(self):
        part = "".join([self.random.choice("ab") for _ in range(self.random.randrange(20))])
        pal = part + self.random.choice([part, part[:-1]])[::-1]
        n = self.random.randrange(len(pal) + 1)
        m = self.random.randrange(n + 1)
        s = pal[m:n]
        self.add(dict(s=s, length=len(pal)))


class BinaryStrXOR(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#11"""

    @staticmethod
    def sat(str_num: str, nums=["100011101100001", "100101100101110"]):
        """
        Find a the XOR of two given strings interpreted as binary numbers.

        Sample Input:
        "0001", "1011"

        Sample Output:
        "1010"
        """
        a, b = nums
        return int(str_num, 2) == int(a, 2) ^ int(b, 2)

    @staticmethod
    def sol(nums):
        a, b = nums
        ans = int(a, 2) ^ int(b, 2)
        return format(ans, "b")

    def gen_random(self):
        nums = [format(self.random.randrange(1024), "b") for _ in range(2)]
        self.add(dict(nums=nums))


# In the HumanEval dataset, tie breaking needs to be specified because each problem must have a unique answer
class LongestStr(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#12"""

    @staticmethod
    def sat(ans: str, words=["these", "are", "some", "pretty", "long", "words"]):
        """
        Find the longest of a list of strings

        Sample Input:
        ["cat", "dog", "sheep", "chimp"]

        Sample Output:
        "sheep"
        """
        return ans in words and all(len(ans) >= len(w) for w in words)

    @staticmethod
    def sol(words):
        return max(words, key=len)

    def gen_random(self):
        words = [self.random.pseudo_word() for _ in range(self.random.randrange(1, 10))]
        self.add(dict(words=words))


class CertifiedGCD(PuzzleGenerator):
    """
    Inspired by [HumanEval](https://github.com/openai/human-eval) \\#13
    """

    @staticmethod
    def sat(ans: List[int], m=200004931, n=66679984):
        """
        Find the greatest common divisor of two integers m, n and a certificate a, b such that m*a + n*b = gcd

        Sample Input:
        20, 30

        Sample Output:
        10, -1, 1
        """
        gcd, a, b = ans
        return m % gcd == n % gcd == 0 and a * m + b * n == gcd and gcd > 0

    @staticmethod
    def sol(m, n):
        """
        Derivation of solution below
        Recursive solution guarantees a * (big % small) + b * small == gcd
        Let d = big // small so (big % small) == big - small * d
        gives a * (big - small * d) + b * small == gcd
        or equivalently (b - a * d) * small + a * big == gcd
        """

        def gcd_cert(small, big):
            """Returns gcd, a, b, such that small * a + big * b == gcd"""
            assert 0 < small <= big
            if big % small == 0:
                return [small, 1, 0]
            gcd, a, b = gcd_cert(big % small, small)
            return [gcd, b - a * (big // small), a]

        if m < n:
            return gcd_cert(m, n)
        gcd, a, b = gcd_cert(n, m)
        return [gcd, b, a]

    def gen_random(self):
        factor, r1, r2 = [1 + self.random.randrange(10 ** self.random.randrange(10)) for _ in range(3)]
        m = r1 * factor
        n = r2 * factor
        self.add(dict(m=m, n=n))


class AllPrefixes(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#14"""

    @staticmethod
    def sat(prefixes: List[str], s="donesezichethofalij"):
        """
        Find all prefixes of a given string

        Sample Input:
        "aabcd"

        Sample Output:
        ["", "a", "aa", "aab", "aabc", "aabcd"]
        """
        return all(s.startswith(p) for p in prefixes) and len(set(prefixes)) > len(s)

    @staticmethod
    def sol(s):
        return [s[:i] for i in range(len(s) + 1)]

    def gen_random(self):
        s = self.random.pseudo_word(min_len=0, max_len=30)
        self.add(dict(s=s))


class SpaceyRange(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#15"""

    @staticmethod
    def sat(ans: str, n=15):
        """
        Find a string consisting of the non-negative integers up to n inclusive

        Sample Input:
        4

        Sample Output:
        '0 1 2 3 4'
        """
        return [int(i) for i in ans.split(' ')] == list(range(n + 1))

    @staticmethod
    def sol(n):
        return ' '.join(str(i) for i in range(n + 1))

    def gen_random(self):
        n = self.random.randrange(10 ** 5)
        self.add(dict(n=n))


class DistinctChars(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#16"""

    @staticmethod
    def sat(ans: List[str], s='The quick brown fox jumps over the lazy dog!', n=28):
        """
        Find the set of distinct characters in a string, ignoring case

        Sample Input:
        'HELlo', 4

        Sample Output:
        ['h', 'e', 'l', 'o']
        """
        assert all(ans.count(c.lower()) == 1 for c in s)
        assert all(c == c.lower() for c in ans)
        assert all(c in s.lower() for c in ans)
        return True

    @staticmethod
    def sol(s, n):
        return list(set(s.lower()))

    def gen_random(self):
        s = self.random.string()
        s = s[0].upper() + s[1:]
        n = len(set(s.lower()))
        self.add(dict(s=s, n=n))


class ParseMusic(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#17"""

    @staticmethod
    def sat(beats: List[int], score="o o o| o| .| .| .| o| o| o o o| .|"):
        """
        Parse a string of notes to beats, 'o'=4, 'o|'=2, '.|'=1

        Example input:
        'o o .| o|'

        Example output:
        [4, 4, 1, 2]
        """
        return " ".join({1: '.|', 2: 'o|', 4: 'o'}[b] for b in beats) == score

    @staticmethod
    def sol(score):
        mapping = {'.|': 1, 'o|': 2, 'o': 4}
        return [mapping[note] for note in score.split()]

    def gen_random(self):
        n = self.random.randrange(12)
        score = ' '.join(self.random.choice(['.|', 'o|', 'o']) for _ in range(n))
        self.add(dict(score=score))


class OverlappingCount(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#18"""

    @staticmethod
    def sat(ans: List[int], s='Bananannanaannanaanananananana', sub='anan', count=7):
        """
        Find occurrences of a substring in a parent string *including overlaps*

        Sample Input:
        'helllo', 'll'

        Sample Output:
        [2, 3]
        """
        return all(sub == s[i:i + len(sub)] and i >= 0 for i in ans) and len(set(ans)) >= count

    @staticmethod
    def sol(s, sub, count):
        ans = []
        for i in range(len(s) + 1):
            if s[i:i + len(sub)] == sub:
                ans.append(i)
        return ans

    def gen_random(self):
        s = self.random.pseudo_word(max_len=100)
        j = self.random.randrange(1, len(s) + 1)
        i = self.random.randrange(j)
        sub = s[i:j]
        count = len(self.sol(s, sub, None))
        self.add(dict(s=s, sub=sub, count=count))


class SortNumbers(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#19"""

    @staticmethod
    def sat(ans: str, s="six one four three two nine eight"):
        """
        Sort numbers based on strings

        Sample input
        ---
        "six one four"

        Sample output
        ---
        "one four six"
        """
        nums = 'zero one two three four five six seven eight nine'.split()
        return [nums.index(x) for x in ans.split(" ")] == sorted([nums.index(x) for x in s.split(" ")])

    @staticmethod
    def sol(s):
        nums = 'zero one two three four five six seven eight nine'.split()
        arr = [nums.index(x) for x in s.split()]
        arr.sort()
        ans = " ".join([nums[i] for i in arr])
        return ans

    def gen_random(self):
        nums = 'zero one two three four five six seven eight nine'.split()
        n = self.random.randrange(3, 9)
        ans = ""
        for _ in range(n):
            ans += self.random.choice(nums) + " "
        ans = ans[:-1]
        s = ans
        self.add(dict(s=s))


class FindClosePair(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#20"""

    @staticmethod
    def sat(inds: List[int], nums=[0.31, 21.3, 5.0, 9.0, 11.0, 5.01, 17.2]):
        """
        Given a list of numbers, find the indices of the closest pair.

        Sample Input:
        [1.2, 5.25, 0.89, 21.0, 5.23]

        Sample Output:
        [4, 1]
        """
        a, b = inds
        assert a != b and a >= 0 and b >= 0
        for i in range(len(nums)):
            for j in range(i):
                assert abs(nums[i] - nums[j]) >= abs(nums[b] - nums[a])
        return True

    @staticmethod
    def sol(nums):
        best = [0, 1]
        best_score = abs(nums[1] - nums[0])
        for i in range(len(nums)):
            for j in range(i):
                score = abs(nums[i] - nums[j])
                if score < best_score:
                    best_score = score
                    best = [i, j]
        return best

    def gen_random(self):
        nums = [self.random.uniform(-10, 10) for _ in range(self.random.randrange(2, 10))]
        if self.random.random() < 0.2:
            nums.append(nums[0])
        self.random.shuffle(nums)
        self.add(dict(nums=nums))


class Rescale(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#21"""

    @staticmethod
    def sat(ans: List[float], nums=[13.0, 17.0, 17.0, 15.5, 2.94]):
        """
        Rescale and shift numbers so that they cover the range [0, 1]

        Sample input
        ---
        [18.5, 17.0, 18.0, 19.0, 18.0]

        Sample output
        ---
        [0.75, 0.0, 0.5, 1.0, 0.5]
        """
        assert min(ans) == 0.0 and max(ans) == 1.0
        a = min(nums)
        b = max(nums)
        for i in range(len(nums)):
            x = a + (b - a) * ans[i]
            assert abs(nums[i] - x) < 1e-6
        return True

    @staticmethod
    def sol(nums):
        nums = nums.copy()

        a = min(nums)
        b = max(nums)
        if b - a == 0:
            return [0.0] + [1.0] * (len(nums) - 1)
        for i in range(len(nums)):
            nums[i] = (nums[i] - a) / (b - a)
        return nums

    def gen_random(self):
        nums = [self.random.heavy_tail_float() for _ in range(self.random.randrange(2, 10))]
        if self.random.random() < 0.2:
            nums = [nums[0]] * len(nums)
        self.add(dict(nums=nums))


class FilterInts(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#22"""

    @staticmethod
    def sat(candidates: List[str], int_indices=[2, 4, 7, 9, 101]):
        """
        Find a list of strings where the only valid integers are at the given indices

        Sample input
        ---
        [2, 4, 5]

        Sample output
        ---
        ["cat", "2.7", "2", "", "3", "-17", "free"]
        """
        for i in int_indices:
            int(candidates[i])
        for i, s in enumerate(candidates):
            if i not in int_indices:
                try:
                    int(s)
                    return False
                except ValueError:
                    pass
        return True

    @staticmethod
    def sol(int_indices):
        if not int_indices:
            return []
        ans = [""] * (1 + max(abs(i) for i in int_indices))
        for i in int_indices:
            ans[i] = "17"
        return ans

    def gen_random(self):
        int_indices = [self.random.randrange(100) for _ in range(self.random.randrange(10))]
        self.add(dict(int_indices=int_indices))


class StrLength(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#23"""

    @staticmethod
    def sat(lengths: List[int], strs=["pneumonoultramicroscopicsilicovolcanoconiosis", " ", "foo", "2.5"]):
        """
        Find the lengths of a list of non-empty strings

        Sample input
        ---
        ["foo", "bars"]

        Sample output
        ---
        [3, 4]
        """
        for length, s in zip(lengths, strs):
            try:
                s[length]
                return False
            except IndexError:
                s[length - 1]
        return len(lengths) == len(strs)

    @staticmethod
    def sol(strs):
        return [len(s) for s in strs]

    def gen_random(self):
        strs = [self.random.string(min_len=1, max_len=50) for _ in range(10)]
        self.add(dict(strs=strs))


class LargestDivisor(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#24"""

    @staticmethod
    def sat(d: int, n=123456):
        """
        Find the largest integer divisor of a number n that is less than n

        Sample input
        ---
        1000

        Sample output
        ---
        500
        """
        return n % d == 0 and d < n and all(n % e for e in range(d + 1, n))

    @staticmethod
    def sol(n):
        return next(d for d in range(n - 1, 0, -1) if n % d == 0)

    def gen_random(self):
        n = self.random.randrange(1, 10 ** 5)
        self.add(dict(n=n))


class PrimeFactorization(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#25"""

    @staticmethod
    def sat(factors: List[int], n=123456, num_factors=8):
        """
        Factor number n into a given number of non-trivial factors

        Sample input
        ---
        1000, 6

        Sample output
        ---
        [2, 2, 2, 5, 5, 5]
        """
        assert len(factors) == num_factors
        prod = 1
        for d in factors:
            prod *= d
            assert d > 1
        return prod == n

    @staticmethod
    def sol(n, num_factors):
        if num_factors == 0:
            return []
        if num_factors == 1:
            return [n]
        ans = []
        for d in range(2, n):
            while n % d == 0:
                n //= d
                ans.append(d)
                if len(ans) == num_factors - 1:
                    ans.append(n)
                    return ans
        assert False

    def gen_random(self):
        num_factors = self.random.randrange(10)
        n = 2 ** num_factors
        for _ in range(self.random.randrange(10)):
            n *= self.random.choice([3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
            num_factors += 1
        self.add(dict(n=n, num_factors=num_factors))


class Dedup(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#26"""

    @staticmethod
    def sat(ans: List[int], li=[2, 19, 2, 53, 1, 1, 2, 44, 17, 0, 19, 31]):
        """
        Remove duplicates from a list of integers, preserving order

        Sample input
        ---
        [1, 3, 2, 9, 2, 1, 55]

        Sample output
        ---
        [1, 3, 2, 9, 55]
        """
        return set(ans) == set(li) and all(li.index(ans[i]) < li.index(ans[i + 1]) for i in range(len(ans) - 1))

    @staticmethod
    def sol(li):
        seen = set()
        ans = []
        for n in li:
            if n not in seen:
                ans.append(n)
                seen.add(n)
        return ans

    def gen_random(self):
        n = self.random.randrange(20)
        li = [self.random.randrange(10) for _ in range(n)]
        self.add(dict(li=li))


class FlipCase(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#27"""

    @staticmethod
    def sat(ans: str, s="FlIp ME!"):
        """
        Flip case

        Sample input
        ---
        'cAt'

        Sample output
        ---
        'CaT'
        """
        return len(ans) == len(s) and all({c, d} == {d.upper(), d.lower()} for c, d in zip(ans, s))

    @staticmethod
    def sol(s):
        return "".join(c.lower() if c.upper() == c else c.upper() for c in s)

    def gen_random(self):
        w = self.random.string()
        s = "".join(self.random.choice([c.upper(), c.lower()] * 5 + [' ', '!', '3']) for c in w)
        self.add(dict(s=s))


class CatStrings(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#28"""

    @staticmethod
    def sat(cat: str, strings=["Will", "i", "am", "Now", "here"]):
        """
        Concatenate a list of strings

        Sample input
        ---
        ['cat', 'dog', 'bird']

        Sample output
        ---
        'catdogbird'
        """
        i = 0
        for s in strings:
            for c in s:
                assert cat[i] == c
                i += 1
        return i == len(cat)

    @staticmethod
    def sol(strings):
        return "".join(strings)

    def gen_random(self):
        strings = [self.random.pseudo_word() for _ in range(self.random.randrange(10))]
        self.add(dict(strings=strings))


class FindExtensions(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#29"""

    @staticmethod
    def sat(extensions: List[str], strings=['cat', 'dog', 'shatter', 'donut', 'at', 'todo'], prefix='do'):
        """
        Find the strings in a list starting with a given prefix

        Sample Input:
        ['cat', 'car', 'fear', 'center'], 'ca'

        Sample Output:
        ['cat', 'car']
        """
        i = 0
        for s in strings:
            if s.startswith(prefix):
                assert extensions[i] == s
                i += 1
        return i == len(extensions)

    @staticmethod
    def sol(strings, prefix):
        return [s for s in strings if s.startswith(prefix)]

    def gen_random(self):
        prefix = self.random.pseudo_word(min_len=0, max_len=3)

        def gen():
            return self.random.choice(["", prefix]) + self.random.pseudo_word(min_len=0, max_len=5)

        strings = [gen() for _ in range(self.random.randrange(6))]
        self.add(dict(strings=strings, prefix=prefix))


class FindPositives(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#30"""

    @staticmethod
    def sat(positives: List[int], nums=[2, 2342, -2, 32, -8, -5, 2342, 0, -9, 44, 11]):
        """
        Find the positive integers in a list

        Sample Input:
        [-1, 3, 19, -2, 0, 44, 0, 44, 11]

        Sample Output:
        [3, 19, 44, 44, 11]
        """
        stack = positives[::-1]
        for n in nums:
            assert n <= 0 or n == stack.pop()
        return stack == []

    @staticmethod
    def sol(nums):
        return [i for i in nums if i > 0]

    def gen_random(self):
        nums = [self.random.randrange(-100, 100) for _ in range(self.random.randrange(10))]
        self.add(dict(nums=nums))


class FermatComposites(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#31"""

    @staticmethod
    def sat(certificates: List[int], nums=[1449, 14, 21, 105, 217]):
        """
        Find Fermat composite certificates for a list of numbers > 1

        Sample Input:
        [1469]

        Sample Output:
        [3]  # because (3 ** 1468) % 1469 != 1
        """
        return all(pow(cert, n - 1, n) > 1 for cert, n in zip(certificates, nums)) and len(certificates) == len(nums)

    @staticmethod
    def sol(nums):
        return [next(i for i in range(2, n) if pow(i, n - 1, n) > 1) for n in nums]

    def gen_random(self):
        nums = []
        for _ in range(self.random.randrange(10)):
            a, b = [self.random.randrange(3, 10 ** 5, 2) for _ in range(2)]
            if not self.random.randrange(10):
                a += 1
            nums.append(a * b)
        self.add(dict(nums=nums))


class OddDegreePolynomialRoot(PuzzleGenerator):
    """
    Polynomials of odd degree always have a real solution.

    Inspired by [HumanEval](https://github.com/openai/human-eval) \\#32
    """

    @staticmethod
    def sat(root: float, coeffs=[1, 2, 3, 17]):
        """
        Find a real root of an odd degree polynomial from its coefficients

        Sample Input:
        [1, 0, 8]

        Sample Output:
        -2.0  # 1*(-2.0)^3 + 8 == 0
        """
        return abs(sum(coeff * (root ** i) for i, coeff in enumerate(coeffs))) < 1e-4

    @staticmethod
    def sol(coeffs):
        def p(x):
            return sum(coeff * (x ** i) for i, coeff in enumerate(coeffs))

        for attempt in range(100):
            a, b = -(10 ** attempt), (10 ** attempt)
            p_a, p_b = p(a), p(b)
            while p_a * p_b <= 0:
                mid = (a + b) / 2
                p_mid = p(mid)
                if abs(p_mid) < 1e-4:
                    return mid
                assert mid not in [a, b]
                if p_mid * p_a > 0:
                    a, p_a = mid, p_mid
                else:
                    b, p_b = mid, p_mid

        assert False, "Root finder failed on 100 attempts"

    def gen_random(self):
        degree = self.random.randrange(1, 10, 2)
        coeffs = [self.random.randrange(-10, 10) for _ in range(degree)]
        coeffs.append(self.random.randrange(1, 10))
        self.add(dict(coeffs=coeffs))


# slightly modified for convenience
class TwoThirdsSorted(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#33"""

    @staticmethod
    def sat(li: List[int], orig=[1, -2, 3, 17, 8, 4, 12, 3, 18, 5, -29, 0, 0]):
        """
        Start with a list of integers, keep every third element in place and otherwise sort the list

        Sample Input:
        [8, 0, 7, 2, 9, 4, 1, 2, 8, 3]

        Sample Output:
        [8, 0, 2, 2, 4, 8, 1, 8, 9, 3]
        """
        assert orig[::3] == li[::3], "Keep every third entry fixed"
        assert sorted(li) == sorted(orig), "Not even a permutation"
        assert all(li[i] <= li[i + 1] for i in range(1, len(li) - 1, 3))
        assert all(li[i] <= li[i + 2] for i in range(2, len(li) - 2, 3))
        return True

    @staticmethod
    def sol(orig):
        n = len(orig)
        your_list = orig[::3]
        sub = orig[:]
        for i in range(int((len(sub) + 2) / 3)):
            sub.pop((2 * i))
        sub = sorted(sub)
        answ = []
        for i in range(int(n / 3)):
            answ.append(your_list[i])
            answ.append(sub[i * 2])
            answ.append(sub[i * 2 + 1])
        if n % 3 == 1:
            answ.append(your_list[-1])
        if n % 3 == 2:
            answ.append(your_list[-1])
            answ.append(sub[-1])
        return answ

    def gen_random(self):
        list_length = self.random.randrange(20)
        orig = []
        for _ in range(list_length):
            orig.append(self.random.randrange(-10, 10))
        self.add(dict(orig=orig))


class UniqueSorted(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#34"""

    @staticmethod
    def sat(li: List[int], orig=[1, 1, 3, 2, 0, 8, 32, -4, 0]):
        """
        Find an increasing sequence consisting of the elements of the original list.

        Sample Input:
        [8, 0, 7, 2, 9, 4, 4, -2, 8, 3]

        Sample Output:
        [-2, 0, 2, 3, 4, 7, 8, 9]
        """
        for i in range(len(li) - 1):
            assert li[i] < li[i + 1]
            assert li[i] in orig
        for n in orig:
            assert n in li
        return True

    @staticmethod
    def sol(orig):
        my_list = sorted(set(orig))
        return my_list

    def gen_random(self):
        list_length = self.random.randrange(20)
        orig = []
        for _ in range(list_length):
            orig.append(self.random.randrange(-10, 10))
        self.add(dict(orig=orig))


class MaxInt(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#35"""

    @staticmethod
    def sat(m: int, hello=[1, 31, 3, 2, 0, 18, 32, -4, 2, -1000, 3502145, 3502145, 21, 18, 2, 60]):
        """
        Find the largest integer in a sequence

        Sample Input:
        [8, 0, 1, 4, 9, 3, 4, -2, 8, 3]

        Sample Output:
        9
        """
        return m in hello and not any(m < i for i in hello)

    @staticmethod
    def sol(hello):
        return max(hello)

    def gen_random(self):
        list_length = self.random.randrange(1, 20)
        hello = []
        for _ in range(list_length):
            hello.append(self.random.randrange(-10, 10))
        self.add(dict(hello=hello))


class SevenElevenThirteen(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#36"""

    @staticmethod
    def sat(li: List[List[int]], n=19723, lower=1000):
        """
        Find all 7's in integers less than n that are divisible by 11 or 13

        Sample Input:
        79, 3

        Sample Output:
        [[77, 0], [77, 1], [78, 0]]
        """
        assert len({(i, j) for i, j in li}) >= lower, "not enough 7's (ignoring duplicates)"
        return all(str(i)[j] == '7' and (i % 11 == 0 or i % 13 == 0) and 0 <= i < n and 0 <= j for i, j in li)

    @staticmethod
    def sol(n, lower):
        return [[i, j] for i in range(n) if (i % 11 == 0 or i % 13 == 0) for j, c in enumerate(str(i)) if c == '7']

    def gen(self, target_num_instances):
        lower = 0
        n = 0
        while self.num_generated_so_far() < target_num_instances:
            if n % 11 == 0 or n % 13 == 0:
                lower += str(n).count('7')
            n += 1
            if self.random.randrange(10) == 0:
                self.add(dict(n=n, lower=lower))


# Since this human-eval problem #37 is very similar to TwoThirdsSorted #33, we use a different approach to sat
class HalfSorted(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#37"""

    @staticmethod
    def sat(li: List[int], orig=[1, 6, 3, 41, 19, 4, 12, 3, 18, 5, -29, 0, 19521]):
        """
        Start with a list of integers, keep every other element in place and otherwise sort the list

        Sample Input:
        [8, 0, 7, 2, 9, 4, 1, 2, 8, 3]

        Sample Output:
        [1, 0, 2, 2, 4, 8, 8, 8, 9, 3]
        """
        return orig[1::2] == li[1::2] and li[::2] == sorted(orig[::2])

    @staticmethod
    def sol(orig):
        n = len(orig)
        odds = orig[1::2]
        evens = sorted(orig[::2])
        ans = []
        for i in range(len(evens)):
            ans.append(evens[i])
            if i < len(odds):
                ans.append(odds[i])
        return ans

    def gen_random(self):
        list_length = self.random.randrange(20)
        orig = []
        for _ in range(list_length):
            orig.append(self.random.randrange(-10, 10))
        self.add(dict(orig=orig))


class ThreeCycle(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#38"""

    @staticmethod
    def sat(s: str, target="Hello world"):
        """
        Given a target string, find a string s such that when each group of three consecutive characters is cycled
        forward one character, you achieve the target string.

        Sample Input:
        "This is a test"

        Sample Output:
        'hiT is aste st'
        """

        def cycle3(trip):
            return trip if len(trip) != 3 else trip[2] + trip[:2]

        return target == "".join(cycle3(s[i: i + 3]) for i in range(0, len(s), 3))

    @staticmethod
    def sol(target):
        def un_cycle3(trip):
            return trip if len(trip) != 3 else trip[1:3] + trip[0]

        return "".join(un_cycle3(target[i: i + 3]) for i in range(0, len(target), 3))

    def gen_random(self):
        target = self.random.pseudo_word(max_len=30)
        self.add(dict(target=target))


class PrimeFib(PuzzleGenerator):
    """
    Inspired by [HumanEval](https://github.com/openai/human-eval) \\#39

    Ira Gessel observed that n is a Fibonacci number if and if either 5 n^2 - 4 or 5 n^2 + 4 is a perfect square
    """

    @staticmethod
    def sat(n: int, lower=123456):
        """
        Find a prime Fibonacci number bigger than a certain threshold, using Ira Gessel's test for Fibonacci numbers.

        Sample Input:
        10

        Sample Output:
        11
        """
        assert any((i ** 0.5).is_integer() for i in [5 * n * n - 4, 5 * n * n + 4]), "n must be a Fibonacci number"
        assert all(n % i for i in range(2, int(n ** 0.5) + 1)), "n must be prime"
        return n > lower

    @staticmethod
    def sol(lower):
        m, n = 2, 3
        while True:
            m, n = n, (m + n)
            if n > lower and all(n % i for i in range(2, int(n ** 0.5) + 1)):
                return n

    def gen_random(self):
        lower = self.random.randrange(2 ** self.random.randrange(20))
        self.add(dict(lower=lower))


class TripleZeroSum(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#40
    
    Similar to but harder than PairZeroSum \#43.
    
    This is a version of the classic [3SUM](https://en.wikipedia.org/wiki/3SUM) problem.
    """

    @staticmethod
    def sat(inds: List[int], nums=[12, 6, 41, 15, -10452, 18242, 10440, 6, 6, 6, 6]):
        """
        Find the indices of three numbers that sum to 0 in a list.

        --- Example input ---
        [1, 2, 4, -3, 5]

        --- Example output ---
        [0, 1, 3]
        """
        return len(inds) == 3 and sum(nums[i] for i in inds) == 0

    @staticmethod
    def sol(nums):
        # \tilde{O}(n^2) algorithm
        inv = {n: i for i, n in enumerate(nums)}  # note that later duplicates will override earlier entries
        for i, n in enumerate(nums):
            if inv[n] == i:
                del inv[n]
            if any((-m - n) in inv for m in nums[:i]):  # found solution!
                j, m = next((j, m) for j, m in enumerate(nums) if (-m - n) in inv)
                k = inv[-m - n]
                return sorted([i, j, k])

    def gen_random(self):
        nums = [self.random.randrange(-100, 100) for _ in range(self.random.randrange(2, 10))]
        nums.append(-sum(nums[:2]))
        self.random.shuffle(nums)
        self.add(dict(nums=nums))


class NumPasses(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#41"""

    @staticmethod
    def sat(count: int, n=981):
        """
        Given n cars traveling East and n cars traveling West on a road, how many passings will there be?
        A passing is when one car passes another. The East-bound cars all begin further West than the West-bound cars.

        --Sample input--
        2

        --Sample output--
        4
        """
        for i in range(n):
            for j in range(n):
                count -= 1
        return count == 0

    @staticmethod
    def sol(n):
        return n ** 2

    def gen_random(self):
        n = self.random.randrange(1000)
        self.add(dict(n=n))


class ListInc(PuzzleGenerator):
    """
    Increment each element of a list by 1

    Inspired by [HumanEval](https://github.com/openai/human-eval) \\#42
    """

    @staticmethod
    def sat(new_list: List[int], old_list=[321, 12, 532, 129, 9, -12, 4, 56, 90, 0]):
        """
        Decrement each element of new_list by 1 and check that it's old_list

        Sample Input:
        [17, 15, 99]

        Sample Output:
        [18, 16, 100]
        """
        return [i - 1 for i in new_list] == old_list

    @staticmethod
    def sol(old_list):
        return [i + 1 for i in old_list]

    def gen_random(self):
        old_list = [self.random.randrange(100) for _ in range(self.random.randrange(10))]
        self.add(dict(old_list=old_list))


class PairZeroSum(PuzzleGenerator):
    """
    Inspired by [HumanEval](https://github.com/openai/human-eval) \\#43

    Similar to TripleZeroSum \#40
    """

    @staticmethod
    def sat(inds: List[int], nums=[12, -10452, 18242, 10440, 81, 241, 525, -18242, 91, 20]):
        """
        Find the indices of two numbers that sum to 0 in a list.

        Sample Input:
        [1, -4, -4, 7, -3]

        Sample Output:
        [1, 2]
        """
        a, b = inds
        return nums[a] + nums[b] == 0 and a >= 0 and b >= 0

    @staticmethod
    def sol(nums):
        s = set(nums)
        for i in s:
            if -i in s:
                return [nums.index(i), nums.index(-i)]

    def gen_random(self):
        n = self.random.randrange(1, 100)
        nums = [self.random.randrange(-n, n) for _ in range(n)]
        if 0 not in nums:
            nums.append(-self.random.choice(nums))
        self.random.shuffle(nums)
        self.add(dict(nums=nums))


class ChangeBase(PuzzleGenerator):
    """
    Inspired by [HumanEval](https://github.com/openai/human-eval) \\#44
    """

    @staticmethod
    def sat(s: str, n=142, base=7):
        """
        Write n in the given base as a string

        Sample Input:
        n=23, base=12

        Sample Output:
        '1A'
        """
        return int(s, base) == n

    @staticmethod
    def sol(n, base):
        assert 2 <= base <= 10
        ans = ""
        while n:
            ans = str(n % base) + ans
            n //= base
        return ans or "0"

    def gen_random(self):
        n = self.random.randrange(1, 10 ** 7)
        base = self.random.randrange(2, 11)
        self.add(dict(n=n, base=base))


class TriangleArea(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#45"""

    @staticmethod
    def sat(height: int, area=1319098728582, base=45126):
        """
        Find the height of a triangle given the area and base. It is guaranteed that the answer is an integer.

        Sample Input:
        area = 6, base = 3

        Sample Output:
        4
        """
        return base * height == 2 * area

    @staticmethod
    def sol(area, base):
        return (2 * area) // base

    def gen_random(self):
        base = self.random.randrange(1, 10 ** self.random.randrange(1, 10))
        height = self.random.randrange(1, 10 ** self.random.randrange(1, 10))
        area = (base * height) // 2
        if base * height == 2 * area:
            self.add(dict(area=area, base=base))


class Fib4(PuzzleGenerator):
    """
    Inspired by [HumanEval](https://github.com/openai/human-eval) \\#46

    Almost identical to problem 63
    """

    @staticmethod
    def sat(init: List[int], target=2021):
        """
        Define a four-wise Fibonacci sequence to be a sequence such that each number is the sum of the previous
        four. Given a target number, find an initial four numbers such that the 100th number in the sequence is the
        given target number.

        Sample Input:
        0

        Sample Output:
        [0, 0, 0, 0]
        """
        a, b, c, d = init
        for i in range(99):
            a, b, c, d = b, c, d, (a + b + c + d)
        return a == target

    @staticmethod
    def sol(target):
        nums = [target, 0, 0, 0]
        for i in range(99):
            x = nums[3] - sum(nums[:3])  # x is such that x + nums[:3] == nums[3]
            nums = [x] + nums[:3]
        return nums

    def gen_random(self):
        target = self.random.randrange(10 ** self.random.randrange(10))
        self.add(dict(target=target))


class Median(PuzzleGenerator):
    """
    One definition of the median is a number that minimizes the sum of absolute deviations.

    Inspired by [HumanEval](https://github.com/openai/human-eval) \\#47
    """

    @staticmethod
    def sat(x: int, nums=[132666041, 237412, 28141, -12, 11939, 912414, 17], upper=133658965):
        """
        Find an integer that minimizes the sum of absolute deviations with respect to the given numbers.

        Sample Input:
        [3, 6, 1, 2, 5, 4, 100], upper=105

        Sample Output:
        4
        """
        dev = sum(n - x for n in nums)
        return dev <= upper

    @staticmethod
    def sol(nums, upper):
        return sorted(nums)[len(nums) // 2] if nums else 0

    def gen_random(self):
        nums = [self.random.randrange(-10 ** 10, 10 ** 10) for _ in range(self.random.randrange(10))]
        x = sorted(nums)[len(nums) // 2] if nums else 0
        upper = sum(n - x for n in nums)
        self.add(dict(nums=nums, upper=upper))


class Palindrome(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#48"""

    @staticmethod
    def sat(pals: List[bool], strs=["palindrome", "madamimadam", "", "foo", "eyes", "(-:-)"]):
        """
        Test whether the given strings are palindromes

        Sample Input:
        ["aba", "no"]

        Sample Output:
        [True, False]
        """
        return all(pals[i] == (s == s[::-1]) for i, s in enumerate(strs))

    @staticmethod
    def sol(strs):
        return [s == s[::-1] for s in strs]

    def gen_random(self):
        strs = []
        for _ in range(self.random.randrange(10)):
            s = self.random.pseudo_word()
            strs.append(self.random.choice([s, s + s[::-1], s[:-1] + s[::-1]]))

        self.add(dict(strs=strs))


class LittleFermat(PuzzleGenerator):
    """Harder but loosely inspired by [HumanEval](https://github.com/openai/human-eval) \\#49"""

    @staticmethod
    def sat(exp_poly: List[int], d=74152093423, poly=[1, 6, 3, 1, 0, 4, 4]):
        """
        Fermat's little theorem implies that any polynomial can be written equivalently as a degree p-1
        polynomial (mod p).
        Given the p coefficients of a polynomial poly, compute a polynomial equivalent to poly^d (mod p).

        Sample Input:
        d=2, poly=[1, 0, 0, 1, 0]  # 1 + x^3

        Sample Output:
        [1, 0, 1, 2, 0]  # 1+ x^2 + 2x^3 because (1 + x^3)^2 = 1 + 2x^3 + x^6 and x^6 = x^2 (mod 5)
        """
        p = len(poly)
        assert p > 2 and all(p % i for i in range(2, p)), "Hint: p is a prime > 2"

        def val(coeffs, n):  # evaluate polynomial mod p
            return sum(c * pow(n, i, p) for i, c in enumerate(coeffs)) % p

        return all(val(exp_poly, n) == pow(val(poly, n), d, p) for n in range(p))

    @staticmethod
    def sol(d, poly):
        """
        Use repeated squaring to exponentiate polynomial
        """
        p = len(poly)

        def prod(poly1, poly2):  # multiply two polynomials mod p
            ans = [0] * p
            for i, a in enumerate(poly1):
                for j, b in enumerate(poly2):
                    e = (i + j) % (p - 1)
                    if e == 0 and i + j > 1:
                        e = p - 1
                    ans[e] = (ans[e] + a * b) % p
            return ans

        ans = [1] + [0] * (p - 1)
        while d:
            if d % 2:
                ans = prod(ans, poly)
            poly = prod(poly, poly)
            d //= 2
        # for i in range(d):
        #     ans = prod(ans, poly)
        return ans


def gen_random(self):
    p = self.random.choice([3, 5, 7, 11])
    poly = [self.random.randrange(p) for _ in range(p)]
    d = self.random.randrange(2 ** self.random.randrange(100))
    self.add(dict(d=d, poly=poly))


class ShiftChars(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#50"""

    @staticmethod
    def sat(orig: str, result="Hello, world!", shift=7):
        """
        Find a string which, when each character is shifted (ascii incremented) by shift, gives the result.

        Sample Input:
        result='very good', shift=-1

        Sample Output:
        'wfsz!hppe'
        """
        n = len(result)
        assert len(orig) == n
        return all(ord(orig[i]) + shift == ord(result[i]) for i in range(n))

    @staticmethod
    def sol(result, shift):
        return "".join(chr(ord(c) - shift) for c in result)

    def gen_random(self):
        result = self.random.pseudo_word()
        shift = self.random.randrange(-11, 11)
        self.add(dict(result=result, shift=shift))


def random_case_word(rand, **args):
    return "".join(rand.choice([c.lower(), c.upper()]) for c in rand.pseudo_word(**args))


class RemoveVowels(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#51

    Related to FindVowels \\#54"""

    @staticmethod
    def sat(txt: str, text="Hello, world!"):
        """
        Remove the vowels from the original string.

        Sample Input:
        "very good"

        Sample Output:
        'vry gd'
        """
        n = 0
        for c in text:
            if c.lower() not in "aeiou":
                assert txt[n] == c
                n += 1
        assert n == len(txt)
        return True

    @staticmethod
    def sol(text):
        return "".join(c for c in text if c.lower() not in "aeiou")

    def gen_random(self):
        text = random_case_word(self.random)
        self.add(dict(text=text))


class BelowThreshold(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#52"""

    @staticmethod
    def sat(indexes: List[int], nums=[0, 2, 17, 4, 4213, 322, 102, 29, 15, 39, 55], thresh=100):
        """
        Find the indexes of numbers below a given threshold

        Sample Input:
        nums=[4, 7, 11, 5], threshold=10

        Sample Output:
        [0, 1, 3]
        """
        j = 0
        for i, n in enumerate(nums):
            if n < thresh:
                assert indexes[j] == i
                j += 1
        assert j == len(indexes)
        return True

    @staticmethod
    def sol(nums, thresh):
        return [i for i, n in enumerate(nums) if n < thresh]

    def gen_random(self):
        thresh = self.random.randrange(-100, 100)
        nums = [self.random.randrange(-100, 100) for _ in range(self.random.randrange(10))]
        self.add(dict(nums=nums, thresh=thresh))


class ListTotal(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#53"""

    @staticmethod
    def sat(n: int, nums=[10, 42, 17, 9, 1315182, 184, 102, 29, 15, 39, 755]):
        """
        Find the number which when appended to the list makes the total 0

        Sample Input:
        [1, 2, 3]

        Sample Output:
        -6
        """
        return sum(nums + [-n]) == 0

    @staticmethod
    def sol(nums):
        return sum(nums)

    def gen_random(self):
        m = 10 ** self.random.randrange(10)
        nums = [self.random.randrange(-m, m) for _ in range(self.random.randrange(10))]
        self.add(dict(nums=nums))


class DiffChars(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#54"""

    @staticmethod
    def sat(c: str, a="the quick brown fox jumped over the lazy dog", b="how vexingly quick daft zebras jump"):
        """
        Find a character in one string that is not in the other.

        Sample Input:
        'Do you like green eggs and ham?', 'I do not like green eggs and ham.'

        Sample Output:
        't'  # or .?yI
        """
        return (c in a) != (c in b)

    @staticmethod
    def sol(a, b):
        return sorted(set(a).symmetric_difference(b))[0]

    def gen_random(self):
        a = self.random.pseudo_word()
        b = self.random.choice([self.random.pseudo_word(), "".join(sorted(a + "m"))])
        if set(a) != set(b):
            self.add(dict(a=a, b=b))


class Fibonacci(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#55"""

    @staticmethod
    def sat(nums: List[int], n=1402):
        """
        Find the first n Fibonacci numbers

        Sample Input:
        4

        Sample Output:
        [1, 1, 2, 3]
        """
        return nums[0] == nums[1] == 1 and all(nums[i + 2] == nums[i + 1] + nums[i] for i in range(n - 2))

    @staticmethod
    def sol(n):
        ans = [1, 1]
        while len(ans) < n:
            ans.append(ans[-1] + ans[-2])
        return ans

    def gen_random(self):
        n = self.random.randrange(12000)
        self.add(dict(n=n))


class MatchBrackets(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#56"""

    @staticmethod
    def sat(matches: List[int], brackets="<<>><<<><>><<>>>"):
        """
        Find the index of the matching brackets for each character in the string

        Sample Input:
        "<><>"

        Sample Output:
        [1, 0, 3, 2]
        """
        for i in range(len(brackets)):
            j = matches[i]
            c = brackets[i]
            assert brackets[j] != c and matches[j] == i and all(i < matches[k] < j for k in range(i + 1, j))
        return len(matches) == len(brackets)

    @staticmethod
    def sol(brackets):
        matches = [-1] * len(brackets)
        opens = []
        for i, c in enumerate(brackets):
            if c == "<":
                opens.append(i)
            else:
                assert c == ">"
                j = opens.pop()
                matches[i] = j
                matches[j] = i
        return matches

    def gen_random(self):
        depth = 0
        brackets = ''
        while depth > 0 or self.random.random() > 0.2:
            c = self.random.choice('<>>' if depth > 0 else '<')
            if c == '<':
                depth += 1
            elif c == '>':
                depth -= 1
            brackets += c
        self.add(dict(brackets=brackets))


class Monotonic(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#57"""

    @staticmethod
    def sat(direction: str, nums=[2, 4, 17, 29, 31, 1000, 416629]):
        """
        Determine the direction ('increasing' or 'decreasing') of monotonic sequence nums

        Sample Input:
        [1, 2, 5]

        Sample Output:
        "increasing"
        """
        if direction == "increasing":
            return all(nums[i] < nums[i + 1] for i in range(len(nums) - 1))
        if direction == "decreasing":
            return all(nums[i + 1] < nums[i] for i in range(len(nums) - 1))

    @staticmethod
    def sol(nums):
        return "increasing" if len(nums) > 1 and nums[1] > nums[0] else "decreasing"

    def gen_random(self):
        nums = sorted({self.random.randrange(1000) for _ in range(self.random.randrange(10))},
                      reverse=self.random.choice([True, False]))
        self.add(dict(nums=nums))


class CommonNumbers(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#58"""

    @staticmethod
    def sat(common: List[int], a=[2, 416629, 2, 4, 17, 29, 31, 1000], b=[31, 2, 4, 17, 29, 41205]):
        """
        Find numbers common to a and b

        Sample Input:
        [1, 2, 3], [3, 4, 5]

        Sample Output:
        [3]
        """
        return all((i in common) == (i in a and i in b) for i in a + b + common)

    @staticmethod
    def sol(a, b):
        return sorted(set(a).intersection(set(b)))

    def gen_random(self):
        common, a, b = [[self.random.randrange(1000) for _ in range(self.random.randrange(10))] for _2 in range(3)]
        a += common
        b += common
        self.random.shuffle(a)
        self.random.shuffle(b)
        self.add(dict(a=a, b=b))


class LargestPrimeFactor(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#59"""

    @staticmethod
    def sat(p: int, n=101076):
        """
        Find the largest prime factor of n.

        Sample Input:
        125

        Sample Output:
        5
        """

        def is_prime(m):
            return all(m % i for i in range(2, m - 1))

        return is_prime(p) and n % p == 0 and p > 0 and all(n % i or not is_prime(i) for i in range(p + 1, n))

    @staticmethod
    def sol(n):
        def is_prime(m):
            return all(m % i for i in range(2, m - 1))

        return next(n // i for i in range(1, n) if n % i == 0 and is_prime(n // i))

    def gen_random(self):
        n = self.random.randrange(2, 100 * 1000)
        self.add(dict(n=n))


class CumulativeSums(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#60"""

    @staticmethod
    def sat(sums: List[int], n=104):
        """
        Find the sums of the integers from 1 to n

        Sample Input:
        3

        Sample Output:
        [0, 1, 3, 6]
        """
        return all(sums[i + 1] - sums[i] == i for i in range(n)) and sums[0] == 0

    @staticmethod
    def sol(n):
        ans = [0]
        for i in range(n):
            ans.append(ans[-1] + i)
        return ans

    def gen_random(self):
        n = self.random.randrange(20 * 1000)
        self.add(dict(n=n))


class ParenDepth(PuzzleGenerator):
    """
    Inspired by [HumanEval](https://github.com/openai/human-eval) \\#61

    Note that problems 61 and 56 are essentially the same
    """

    @staticmethod
    def sat(matches: List[int], parens="((())()(()()))(())"):
        """
        Find the index of the matching parentheses for each character in the string

        Sample Input:
        "()((()))"

        Sample Output:
        [1, 0, 7, 6, 5, 4, 3, 2]
        """
        for i, (j, c) in enumerate(zip(matches, parens)):
            assert parens[j] != c and matches[j] == i and all(i < matches[k] < j for k in range(i + 1, j))
        return len(matches) == len(parens)

    @staticmethod
    def sol(parens):
        matches = [-1] * len(parens)
        opens = []
        for i, c in enumerate(parens):
            if c == "(":
                opens.append(i)
            else:
                assert c == ")"
                j = opens.pop()
                matches[i] = j
                matches[j] = i
        return matches

    def gen_random(self):
        depth = 0
        parens = ''
        while depth > 0 or self.random.random() > 0.3:
            c = self.random.choice('())' if depth > 0 else '(')
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            parens += c
        self.add(dict(parens=parens))


class Derivative(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#62

    This puzzle gives the raw definition of a derivative in terms of small changes in x.
    """

    @staticmethod
    def sat(derivative: List[int], poly=[2, 1, 0, 4, 19, 231, 0, 5]):
        """
        Find the derivative of the given polynomial, with coefficients in order of increasing degree

        Sample Input:
        [3, 4, 1] # 3 + 4x + x^2

        Sample Output:
        [2, 4]   # 4 + 2x^2
        """

        def val(poly, x):
            return sum(coeff * (x ** i) for i, coeff in enumerate(poly))

        return all(abs(val(poly, x + 1e-8) - val(poly, x) - 1e-8 * val(derivative, x)) < 1e-4 for x in range(len(poly)))

    @staticmethod
    def sol(poly):
        return [i * poly[i] for i in range(1, len(poly))]

    def gen_random(self):
        poly = [self.random.randrange(-10, 10) for _ in range(self.random.randrange(10))]
        self.add(dict(poly=poly))


class Fib3(PuzzleGenerator):
    """
    Inspired by [HumanEval](https://github.com/openai/human-eval) \\#63

    Almost identical to problem 46
    """

    @staticmethod
    def sat(init: List[int], target=124156):
        """
        Define a triple-Fibonacci sequence to be a sequence such that each number is the sum of the previous
        three. Given a target number, find an initial triple such that the 17th number in the sequence is the
        given target number.

        Sample Input:
        0

        Sample Output:
        [0, 0, 0]
        """
        a, b, c = init
        for i in range(16):
            a, b, c = b, c, (a + b + c)
        return a == target

    @staticmethod
    def sol(target):
        nums = [target, 0, 0]
        for i in range(16):
            x = nums[-1] - sum(nums[:-1])  # x is such that x + nums[:3] == nums[3]
            nums = [x] + nums[:-1]
        return nums

    def gen_random(self):
        target = self.random.randrange(2 ** self.random.randrange(15))
        self.add(dict(target=target))


class FindVowels(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#64

    Very similar to RemoveVowels \\#51
    """

    @staticmethod
    def sat(vowels: List[str], texts=["Hello, world!", "Goodbye, world!"]):
        """
        Find the vowels from each of the original texts (y counts as a vowel at the end of the word)

        Sample Input:
        ["You can do it!", "CAT"]

        Sample Output:
        ["ouaoi", "A"]
        """
        for v, t in zip(vowels, texts):
            i = 0
            for j, c in enumerate(t):
                if c.lower() in "aeiou" or c.lower() == 'y' and j == len(t) - 1:
                    assert v[i] == c
                    i += 1
            assert i == len(v)
        return len(vowels) == len(texts)

    @staticmethod
    def sol(texts):
        return ["".join(c for c in text if c.lower() in "aeiou") + (text[-1] if text[-1].lower() == "y" else "")
                for text in texts]

    def gen_random(self):
        texts = [random_case_word(self.random) for _ in range(self.random.randrange(10))]
        self.add(dict(texts=texts))


class CircularShiftNum(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#65"""

    @staticmethod
    def sat(shifted: str, n=124582369835, shift=3):
        """
        Shift the decimal digits n places to the left, wrapping the extra digits around. If shift > the number of
        digits of n, reverse the string.

        n=12345 shift=2 => '34512'
        """
        if shift > len(str(n)):
            return n == int(shifted[::-1])
        return n == int(shifted[-shift:] + shifted[:-shift])

    @staticmethod
    def sol(n, shift):
        s = str(n)
        if shift > len(s):
            return s[::-1]
        return s[shift:] + s[:shift]

    def gen_random(self):
        n = self.random.randrange(10 ** self.random.randrange(30))
        shift = self.random.randrange(31)
        self.add(dict(n=n, shift=shift))


class CharSum(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#66"""

    @staticmethod
    def sat(tot: int, s="Add ME uP AND YOU WILL GET A BIG NUMBER!"):
        """
        Compute the sum of the ASCII values of the upper-case characters in the string.

        Sample Input:
        ARt

        Sample Output:
        147 # = 65 + 82
        """
        for c in s:
            if c.isupper():
                tot -= ord(c)
        return tot == 0

    @staticmethod
    def sol(s):
        return sum(ord(c) for c in s if c.isupper())

    def gen_random(self):
        s = self.random.string(min_len=0)
        self.add(dict(s=s))


class MissingBananas(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#67"""

    @staticmethod
    def sat(bananas: int, bowl="5024 apples and 12189 oranges", total=12491241):
        """
        Determine how many bananas are necessary to reach a certain total amount of fruit

        bowl="3 apples and 4 oranges", total=12 => 5
        """
        bowl += f" and {bananas} bananas"
        return sum([int(s) for s in bowl.split() if s.isdigit()]) == total

    @staticmethod
    def sol(bowl, total):
        apples, oranges = [int(s) for s in bowl.split() if s.isdigit()]
        return total - apples - oranges

    def gen_random(self):
        digits = self.random.choice([1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        apples, oranges, bananas = [self.random.randrange(10 ** digits) for _ in range(3)]
        bowl = f"{apples} apples and {oranges} oranges"
        total = apples + oranges + bananas
        self.add(dict(bowl=bowl, total=total))


class SmallestEven(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#68"""

    @staticmethod
    def sat(val_index: List[int], nums=[125123, 422323, 141, 5325, 812152, 9, 42145, 5313, 421, 812152]):
        """
        Given an array of nums representing a branch on a binary tree, find the minimum even value and its index.
        In the case of a tie, return the smallest index. If there are no even numbers, the answer is [].

        Sample Input:
        [1, 7, 4, 6, 10, 11, 14]

        Sample Output:
        [4, 2]
        """
        if val_index == []:
            return all(n % 2 == 1 for n in nums)
        v, i = val_index
        assert v % 2 == 0 and nums[i] == v
        return all(n > v or n % 2 == 1 for n in nums[:i]) and all(n >= v or n % 2 == 1 for n in nums[i:])

    @staticmethod
    def sol(nums):
        if any(n % 2 == 0 for n in nums):
            return min([v, i] for i, v in enumerate(nums) if v % 2 == 0)
        else:
            return []

    def gen_random(self):
        digits = self.random.randrange(10)
        nums = [self.random.randrange(10 ** digits) for _ in range(self.random.randrange(5))]
        self.add(dict(nums=nums))


class GreatestHIndex(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#69"""

    @staticmethod
    def sat(h: int, seq=[3, 1, 4, 17, 5, 17, 2, 1, 41, 32, 2, 5, 5, 5, 5]):
        """
        Find the h-index, the largest positive number h such that that h occurs in the sequence at least h times.
        h = -1 if there is no such positive number.

        Sample Input:
        [1, 2, 2, 3, 3, 3, 4, 4]

        Sample Output:
        3
        """
        for i in seq:
            assert not (i > 0 and i > h and seq.count(i) >= i)
        return h == -1 or seq.count(h) >= h > 0

    @staticmethod
    def sol(seq):
        return max([-1] + [i for i in seq if i > 0 and seq.count(i) >= i])

    def gen_random(self):
        seq = [self.random.randrange(10) for _ in range(self.random.randrange(100))]
        self.add(dict(seq=seq))


class WildSort(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#70"""

    @staticmethod
    def sat(strange: List[int], li=[30, 12, 42, 717, 45, 317, 200, -1, 491, 32, 15]):
        """
        Find the following strange sort of li: the first element is the smallest, the second is the largest of the
        remaining, the third is the smallest of the remaining, the fourth is the smallest of the remaining, etc.

        Sample Input:
        [1, 2, 7, 3, 4, 5, 6]

        Sample Output:
        [1, 7, 2, 6, 3, 5, 4]
        """
        assert sorted(strange) == sorted(li), "Must be a permutation"
        return all(n == (min, max)[i % 2](strange[i:]) for i, n in enumerate(strange))

    @staticmethod
    def sol(li):
        s = sorted(li)
        i = 0
        j = len(li) - 1
        ans = []
        while i <= j:
            if len(ans) % 2:
                ans.append(s[j])
                j -= 1
            else:
                ans.append(s[i])
                i += 1
        return ans

    def gen_random(self):
        li = [self.random.randrange(10) for _ in range(self.random.randrange(20))]
        self.add(dict(li=li))


class HeronTriangle(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#71

    That problem essentially asks for Heron's formula for the area of a triangle in terms of its three sides.
    In our version, we consider the related problem (also solved by Heron's formula) of finding 2d coordinates
    of a triangle with the given sides. If one knows the area, this is a straightforward calculation.
    """

    @staticmethod
    def sat(coords: List[List[float]], sides=[8.9, 10.8, 17.0]):
        """
        Find the coordinates of a triangle with the given side lengths

        Sample Input:
        [3.0, 4.0, 5.0

        Sample Output:
        [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]
        """
        assert len(coords) == 3
        sides2 = [((x - x2) ** 2 + (y - y2) ** 2) ** 0.5 for i, (x, y) in enumerate(coords) for x2, y2 in coords[:i]]
        return all(abs(a - b) < 1e-6 for a, b in zip(sorted(sides), sorted(sides2)))

    @staticmethod
    def sol(sides):
        a, b, c = sorted(sides)

        s = sum(sides) / 2  # semi-perimeter
        area = (s * (s - a) * (s - b) * (s - c)) ** 0.5  # Heron's formula

        y = 2 * area / a  # height
        x = (c ** 2 - y ** 2) ** 0.5
        return [[0.0, 0.0], [a, 0.0], [x, y]]

    def gen_random(self):
        sides = sorted([self.random.random() * 100 for _ in range(3)])
        if sides[0] + sides[1] > sides[2]:
            self.add(dict(sides=sides))


class InvestigateCrash(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#72"""

    @staticmethod
    def sat(problem: int, weights=[1, 2, 5, 2, 1, 17], max_weight=100):
        """
        An object will "fly" if its weights are a palindrome and sum to <= max_weight. The given object won't fly.
        You have to determine why. Find index where the weights aren't a palindrome or -1 if weights are too big.

        weights=[77, 40], max_weight=100 => -1

        weights=[1,2,3], max_weight=50   => 0 # because 1 != 3
        """
        if problem == -1:
            return sum(weights) > max_weight
        return weights[problem] != weights[- 1 - problem]

    @staticmethod
    def sol(weights, max_weight):
        if sum(weights) > max_weight:
            return -1
        return next(i for i, w in enumerate(weights) if weights[-i - 1] != weights[i])

    def gen_random(self):
        weights = [self.random.randrange(100) for _ in range(self.random.randrange(1, 10))]
        weights += self.random.choice([weights, weights[:-1]])[::-1]
        if self.random.random() < 0.8:
            weights[self.random.randrange(len(weights))] = self.random.randrange(100)

        max_weight = sum(weights) + self.random.randrange(-10, 100)
        if sum(weights) > max_weight or weights != weights[::-1]:
            self.add(dict(weights=weights, max_weight=max_weight))


class ClosestPalindrome(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#73"""

    @staticmethod
    def sat(pal: str, s="palindromordinals"):
        """
        Find the closest palindrome

        Sample Input:
        "cat"

        Sample Output:
        "tat"
        """
        assert pal == pal[::-1] and len(pal) == len(s)
        return sum(a != b for a, b in zip(pal, s)) == sum(a != b for a, b in zip(s, s[::-1])) // 2

    @staticmethod
    def sol(s):
        n = len(s)
        return s[:(n + 1) // 2] + s[:n // 2][::-1]

    def gen_random(self):
        w = self.random.pseudo_word()
        n = len(w)
        s = w[:(n + 1) // 2] + "".join(self.random.choice([c, self.random.char()]) for c in w[:n // 2])
        self.add(dict(s=s))


class NarrowerList(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#74"""

    @staticmethod
    def sat(li: List[str], lists=[["this", "list", "is", "narrow"], ["I", "am", "shorter but wider"]]):
        """
        Find the list that has fewer total characters (including repetitions)

        Sample Input:
        [["sh", "ort"], ["longest"]]

        Sample Output:
        [["sh", "ort"]
        """
        width = sum(len(s) for s in li)
        for li2 in lists:
            assert width <= sum(len(s) for s in li2)
        return li in lists

    @staticmethod
    def sol(lists):
        return min(lists, key=lambda x: sum(len(i) for i in x))

    def gen_random(self):
        num_lists = self.random.randrange(1, 5)
        lists = [[self.random.pseudo_word() for _ in range(self.random.randrange(2, 5))] for _ in range(num_lists)]
        self.add(dict(lists=lists))


class ThreePrimes(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#75"""

    @staticmethod
    def sat(factors: List[List[int]]):
        """
        Find all 247 integers <= 1000 that are the product of exactly three primes.
        Each integer should represented as the list of its three prime factors.
        [[2, 2, 2], [2, 2, 3],  [2, 2, 5], ...
        """
        primes = set(range(2, 1000))
        for n in range(2, 1000):
            if n in primes:
                primes.difference_update(range(2 * n, 1000, n))
        assert all(p in primes for f in factors for p in f), "all factors must be prime"
        nums = {p * q * r for p, q, r in factors}
        return max(nums) < 1000 and len(nums) == 247

    @staticmethod
    def sol():
        primes = set(range(2, 1000))
        for n in range(2, 1000):
            if n in primes:
                primes.difference_update(range(2 * n, 1000, n))
        return [[p, q, r] for p in primes for q in primes if p <= q for r in primes if q <= r and p * q * r < 1000]


class IntegerLog(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#76"""

    @staticmethod
    def sat(x: int, a=3, n=1290070078170102666248196035845070394933441741644993085810116441344597492642263849):
        """Find an integer exponent x such that a^x = n
        Sample Input:
        a=2, n=1024

        Sample Output:
        x = 10
        """
        return a ** x == n

    @staticmethod
    def sol(a, n):
        m = 1
        x = 0
        while m != n:
            x += 1
            m *= a
        return x

    def gen_random(self):
        a = self.random.randrange(1, 10)
        n = a ** self.random.randrange(255)
        self.add(dict(a=a, n=n))


class CubeRoot(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#77

    We made it harder by giving very large n for which `round(n ** (1/3))`
    """

    @staticmethod
    def sat(x: int, n=42714774173606970182754018064350848294149432972747296768):
        """Find an integer that when cubed is n

        Sample Input:
        21

        Sample Output:
        3
        """
        return x ** 3 == n

    @staticmethod
    def sol(n):
        # Using Newton's method
        m = abs(n)
        x = round(abs(n) ** (1 / 3))
        while x ** 3 != m:
            x += (m - x ** 3) // (3 * x ** 2)
        return -x if n < 0 else x

    def gen_random(self):
        digits = self.random.randrange(30)
        n = self.random.randrange(-10 ** digits, 10 ** digits) ** 3
        self.add(dict(n=n))


class HexPrimes(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#78"""

    @staticmethod
    def sat(primes: List[bool], n="A4D4455214122CE192CCBE3"):
        """Determine which characters of a hexidecimal correspond to prime numbers

        Sample Input:
        "123ABCD"

        Sample Output:
        [False, True, True, False, True, False True]
        """
        return all(primes[i] == (c in "2357BD") for i, c in enumerate(n))

    @staticmethod
    def sol(n):
        return [c in "2357BD" for c in n]

    def gen_random(self):
        digits = self.random.randrange(30)
        n = hex(self.random.randrange(10 ** digits))[2:]
        self.add(dict(n=n))


class Binarize(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#79"""

    @staticmethod
    def sat(b: str, n=5324680297138495285):
        """Write n base 2 followed and preceded by 'bits'
        Sample Input:
        2

        Sample Output:
        bits10bits
        """
        assert b[:4] == b[-4:] == 'bits'
        inside = b[4:-4]
        assert all(c in "01" for c in inside)
        assert inside[0] == "1" or len(inside) == 1
        m = 0
        for c in inside:
            m = 2 * m + int(c)
        return m == n

    @staticmethod
    def sol(n):
        s = bin(n)[2:]
        return f'bits{s}bits'

    def gen_random(self):
        digits = self.random.randrange(30)
        n = self.random.randrange(10 ** digits)
        self.add(dict(n=n))


class NearbyDuplicates(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#80"""

    @staticmethod
    def sat(indices: List[int], s="I am an unhappy string!"):
        """A string is happy if every three consecutive characters are distinct. Find two indices making s unhappy.
        Sample Input:
        "street"

        Sample Output:
        [3, 4]
        """
        i, j = indices
        return s[i] == s[j] and 0 <= i < j < i + 3

    @staticmethod
    def sol(s):
        for i in range(len(s) - 2):
            if s[i] == s[i + 1]:
                return [i, i + 1]
            if s[i] == s[i + 2]:
                return [i, i + 2]

    def gen_random(self):
        a = self.random.string(min_len=1)
        s = a + self.random.choice(["", self.random.char()]) + a[-1] + self.random.string(min_len=1)
        self.add(dict(s=s))


class Grader(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#81"""

    @staticmethod
    def sat(grades: List[str], gpas=[2.8, 3.1, 4.0, 2.2, 3.1, 2.5, 0.9]):
        """
        Convert GPAs to letter grades according to the following table:
        4.0: A+
        3.7: A
        3.4: A-
        3.0: B+
        2.7: B
        2.4: B-
        2.0: C+
        1.7: C
        1.4: C-
        below: F

        Sample input: [4.0, 3.5, 3.8]
        Sample output: ['A+', 'A-', 'A']
        """
        assert len(grades) == len(gpas)
        letters = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'F']
        scores = [4.0, 3.7, 3.4, 3.0, 2.7, 2.4, 2.0, 1.7, 1.4, 0.0]
        for grade, gpa in zip(grades, gpas):
            i = letters.index(grade)
            assert gpa >= scores[i]
            assert i == 0 or gpa <= scores[i - 1]
        return True

    @staticmethod
    def sol(gpas):
        letters = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'F']
        scores = [4.0, 3.7, 3.4, 3.0, 2.7, 2.4, 2.0, 1.7, 1.4, 0.0]
        ans = []
        for gpa in gpas:
            i = 0
            while gpa < scores[i]:
                i += 1
            ans.append(letters[i])
        return ans

    def gen_random(self):
        gpas = [self.random.random() * 4.0 for _ in range(self.random.randrange(10))]
        self.add(dict(gpas=gpas))


class FactorString(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#82"""

    @staticmethod
    def sat(factor: str, s="catscatcatscatcatscat"):
        """Find a string which when repeated more than once gives s
        Sample Input:
        "haha"

        Sample Output:
        "ha"
        """
        return len(factor) < len(s) and s == factor * (len(s) // len(factor))

    @staticmethod
    def sol(s):
        n = len(s)
        return next(s[:i] for i in range(1, len(s)) if s == s[:i] * (n // i))

    def gen_random(self):
        s = self.random.pseudo_word() * self.random.randrange(2, 10)
        self.add(dict(s=s))


class OneEnded(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#83"""

    @staticmethod
    def sat(nums: List[int], n=5):
        """Find all n-digit integers that start or end with 1

        1 => [1]"""
        count = 18 * (10 ** (n - 2)) if n > 1 else 1
        strs = {str(n) for n in nums}
        return len(strs) == count and all(s.startswith("1") or s.endswith("1") and len(s) == n for s in strs)

    @staticmethod
    def sol(n):
        ans = []
        for i in range(10 ** (n - 1), 10 ** n):
            assert len(str(i)) == n
            if str(i).startswith("1") or str(i).endswith("1"):
                ans.append(i)
        return ans

    examples = [dict(n=i) for i in range(1, 7)]


class BitSum(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#84"""

    @staticmethod
    def sat(n: int, b=107, s=25):
        """Find an b-bit integer with a bit-sum of s

        b=3, s=2 => 5 # 5 is 101 in binary
        """
        n_str = bin(n)[2:]  # n in binary
        return len(n_str) == b and sum(int(i) for i in n_str) == s

    @staticmethod
    def sol(b, s):
        return int("1" * s + "0" * (b - s), 2)

    def gen_random(self):
        b = self.random.randrange(1, 1000)
        s = self.random.randrange(1, b + 1)
        self.add(dict(b=b, s=s))


class EvenOddSum(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#85

    Very similar to OddEvenSum \#121
    """

    @staticmethod
    def sat(even_odd_sum: int, nums=[2341, 125146894, 12521, -12451293476325, 535284623934, 132974693614350]):
        """Find the sum of the even elements that are at odd indices

        [1, 2, 8, 3, 9, 4] => 6
        """
        for i in nums[1::2]:
            if i % 2 == 0:
                even_odd_sum -= i
        return even_odd_sum == 0

    @staticmethod
    def sol(nums):
        return sum(i for i in nums[1::2] if i % 2 == 0)

    def gen_random(self):
        nums = [self.random.randrange(-100, 100) for _ in range(self.random.randrange(20))]
        self.add(dict(nums=nums))


class AntiShuffle(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#86"""

    @staticmethod
    def sat(s: str, orig="Hello world!!!"):
        """Create a new string by taking s, and word by word rearranging its characters in ascii order
        Sample input:
        'maltos wow'

        Sample output:
        'almost oww'
        """
        for a, b in zip(s.split(' '), orig.split(' ')):
            for i in range(len(a) - 1):
                assert a[i] <= a[i + 1], "characters must s-words be in increasing order"
            assert len(a) == len(b) and all(a.count(c) == b.count(c) for c in b), "must have same chars"
        return len(s) == len(orig)

    @staticmethod
    def sol(orig):
        return " ".join("".join(sorted(w)) for w in orig.split(' '))

    def gen(self, target_num_instances):
        self.add(dict(orig="YOU CAN rearrange my letters, yes you can!"))
        self.add(dict(orig="caN you handlE LONGGGGGGGGGGGG strings?"))
        self.add(dict(orig="how bout    spaces and weird punctuation!?$%@#%"))

    def gen_random(self):
        orig = " ".join(self.random.pseudo_word() for _ in range(self.random.randrange(5)))
        self.add(dict(orig=orig))


class UnevenFind(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#87"""

    @staticmethod
    def sat(indices: List[List[int]], uneven=[[1, 3, 2, 32, 17], [17, 2, 48, 17], [], [9, 35, 4], [3, 17]], target=17):
        """Find the indices of all occurrences of target in the uneven matrix
        Sample input:
        uneven=[[2, 3, 2], [], [9, 2]], target=2

        Sample output:
        [[0, 0], [0, 2], [2, 1]]
        """
        for i, j in indices:
            assert uneven[i][j] == target
        for i, row in enumerate(uneven):
            for j, n in enumerate(row):
                assert n != target or [i, j] in indices
        return True

    @staticmethod
    def sol(uneven, target):
        return [[i, j] for i, row in enumerate(uneven) for j, n in enumerate(row) if n == target]

    def gen_random(self):
        target = self.random.randrange(100)
        uneven = [[self.random.choice([target, self.random.randrange(100)]) for _ in range(self.random.randrange(10))]
                  for _2 in range(self.random.randrange(10))]
        self.add(dict(uneven=uneven, target=target))


class UpDownSort(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#88"""

    @staticmethod
    def sat(up_down: List[int], nums=[17, 2, 3, 523, 18, -2, 0, 2, -1]):
        """Reorder nums in increasing/decreasing order based on whether the first plus last element is even/odd

        Sample input:
        [1, 7, 4]

        Sample output:
        [1, 4, 7] # because 1 + 4 is odd

        Sample input:
        [1, 7, 5]

        Sample output:
        [8, 5, 1] # because 1 + 5 is even
        """
        assert all(up_down.count(i) == nums.count(i) for i in set(up_down + nums)), "not a reordering"
        increasing_sign = 1 if ((nums[0] + nums[-1]) % 2 == 1) else -1
        return all((up_down[i + 1] - up_down[i]) * increasing_sign >= 0 for i in range(len(up_down) - 1))

    @staticmethod
    def sol(nums):
        return sorted(nums, reverse=(False if (nums[0] + nums[-1]) % 2 else True))


class SubstitutionCypher(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#89"""

    @staticmethod
    def sat(encrypted: str, orig="Hello, world!"):
        """Apply a substitution cypher in which each character is advanced by two multiplied by two places.

        'substitution cypher' => 'wyfwxmxyxmsr$g}tliv'
        """
        assert len(encrypted) == len(orig)
        return all(chr(ord(a) - 2 * 2) == b for a, b in zip(encrypted, orig))

    @staticmethod
    def sol(orig):
        return "".join(chr(ord(b) + 2 * 2) for b in orig)

    def gen_random(self):
        orig = " ".join(self.random.pseudo_word() for _ in range(self.random.randrange(5)))
        self.add(dict(orig=orig))


class SecondSmallestUnique(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#90"""

    @staticmethod
    def sat(n: int, nums=[17, -1023589211, -293485382500, 31, -293485382500, 105762, 94328103589]):
        """Find the second smallest unique number in the list nums.

        Sample input:
        [2, 5, 2, 7, 9]

        Sample output:
        5
        """
        assert n in nums
        return len({i for i in nums if i <= n}) == 2

    @staticmethod
    def sol(nums):
        return sorted(set(nums))[1]

    def gen_random(self):
        nums = [self.random.randrange(-10, 10) for _ in range(self.random.randrange(2, 10))]
        if len(set(nums)) >= 2:
            self.add(dict(nums=nums))


class FindBored(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#91"""

    @staticmethod
    def sat(boring: List[str], text="This is not boring. I am boring! I am sooo tired."):
        """A bored sentence starts with the word "I". Find all bored sentences in s. Sentence delimiters are '.!?'

        --- Example input ---
        'I wrote this. You read it? I think I am so cool. In another time, I would be lame.'

        --- Example output ---
        ['I wrote this', ' I think I am so cool']

        """
        sentences = text.replace("!", ".").replace("?", ".").split(".")
        boring_and_exciting = boring + [s for s in sentences if s.split()[:1] != ["I"]]
        return sorted(boring_and_exciting) == sorted(sentences)

    @staticmethod
    def sol(text):
        return [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.split()[:1] == ["I"]]

    def gen_random(self):
        text = ""
        while self.random.random() < 0.75:
            length = self.random.randrange(6)
            words = self.random.choice([[], ["I"]]) + [self.random.pseudo_word() for _ in range(length)]
            text += " ".join(words) + self.random.choice(".!?")
        self.add(dict(text=text))


class IdentifyZeroTrips(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#92"""

    @staticmethod
    def sat(zero_sums: List[bool], trips=[[1253532, -3920635, 332], [-24, 18, 6], [0, 5, -5], [1, 1, 1], [-20, 17, 4]]):
        """Determine which triples sum to zero

        --- Example input ---
        [1, 2, 4, -3, 5]

        --- Example output ---
        [0, 1, 3]
        """
        return len(zero_sums) == len(trips) and all(z == ((a + b + c) == 0) for z, (a, b, c) in zip(zero_sums, trips))

    @staticmethod
    def sol(trips):
        return [sum(t) == 0 for t in trips]

    def gen_random(self):
        trips = [[self.random.randrange(-10, 11) for _ in range(3)] for _ in range(self.random.randrange(2, 20))]
        for t in trips:
            if self.random.randrange(2):
                t[-1] = t[0] + t[1]
        self.add(dict(trips=trips))


class WeirdDecodeVowels(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#93"""

    @staticmethod
    def sat(s: str, target="Hello, world!"):
        """Find string s that, when case is flipped gives target where vowels are replaced by chars two later.
        --- Example input ---
        'THIS is a TEST'

        --- Example output ---
        'thks KS C tgst'
        """
        subs = {ord(c): ord(c) + 2 for c in "aeiouAEIOU"}
        return s.swapcase() == target.translate(subs)

    @staticmethod
    def sol(target):
        subs = {ord(c): ord(c) + 2 for c in "aeiouAEIOU"}
        return target.translate(subs).swapcase()

    def gen(self, target_num_instances):
        self.add(dict(target="This is a good test"))
        self.add(dict(target=""))
        self.add(dict(target="That last test was a bad test!"))
        self.add(dict(target="pneumonoultramicroscopicsilicovolanoconiosis"))

    def gen_random(self):
        target = " ".join(self.random.pseudo_word() for _ in range(self.random.randrange(1, 4)))
        self.add(dict(target=target))


class LargestPrimeDigitSum(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#94"""

    @staticmethod
    def sat(ans: List[int], nums=[23, 17, 201, 14, 10473, 43225, 421, 423, 11, 10, 2022, 342157]):
        """Find the index of the largest prime in the list and the sum of its digits

        --- Example input ---
        [2, 4, 7, 19, 21]

        --- Example output ---
        [3, 10]
        """
        i, digit_sum = ans
        n = nums[i]

        def is_prime(n):
            return n > 1 and all(n % j for j in range(2, int(n ** 0.5) + 1))

        return is_prime(n) and all(m <= n for m in nums if is_prime(m)) and digit_sum == sum(int(c) for c in str(n))

    @staticmethod
    def sol(nums):
        def is_prime(n):
            return n > 1 and all(n % j for j in range(2, int(n ** 0.5) + 1))

        n, i = max((n, i) for i, n in enumerate(nums) if is_prime(n))
        return [i, sum(int(c) for c in str(n))]

    def gen_random(self):
        nums = [self.random.randrange(1, 10 ** self.random.randrange(1, 8)) for _ in range(10)]

        def is_prime(n):
            return n > 1 and all(n % j for j in range(2, int(n ** 0.5) + 1))

        if any(is_prime(n) for n in nums):
            self.add(dict(nums=nums))


class OddCase(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#95"""

    @staticmethod
    def sat(different: str, d={"cat": "CAT", "tree": "T", "pick me": "not", "OK": "red", "blah": "blah", "z": "Z"}):
        """Find the dictionary key whose case is different than all other keys

        --- Example input ---
        {"red": "", "GREEN": "", "blue": "orange"}

        --- Example output ---
        "GREEN"
        """
        return different in d and all(k.islower() != different.islower() for k in d if k != different)

    @staticmethod
    def sol(d):
        for different in d:
            if all(k.islower() != different.islower() for k in d if k != different):
                return different

    def gen_random(self):
        mostly_upper = self.random.choice([True, False])
        trans = lambda x: (x.upper() if mostly_upper else x)
        d = {trans(self.random.pseudo_word()): self.random.pseudo_word() for _ in range(10)}
        mostly_upper = not mostly_upper
        d[trans(self.random.pseudo_word())] = self.random.pseudo_word()
        self.add(dict(d=d))


class PrimesUpTo(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#96"""

    @staticmethod
    def sat(primes: List[int], n=1234):
        """Find all primes up to n

        --- Example input ---
        9

        --- Example output ---
        [2, 3, 5, 7]
        """
        assert all(1 < p for p in primes) and all(p % q for p in primes for q in primes if q < p)
        return len({i for p in primes for i in range(p, n, p)}) == max(n - 2, 0)

    @staticmethod
    def sol(n):
        primes = []
        candidates = set(range(2, n))
        for i in range(2, n):
            if i in candidates:
                primes.append(i)
                candidates.difference_update(range(i, n, i))
        return primes

    def gen(self, target_num_instances):
        self.add(dict(n=10))
        self.add(dict(n=1000))
        self.add(dict(n=-1))
        self.add(dict(n=10000))

    def gen_random(self):
        n = self.random.randrange(20 * 1000)
        self.add(dict(n=n))


class UnitsProduct(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#97"""

    @staticmethod
    def sat(prod: int, nums=[17, 24, 39, 15, 11, 201, 97, 65, 18]):
        """Find the product of the units digits in the numbers

        [12, 34] => 8
        """
        if not all(nums):
            return prod == 0
        for n in nums:
            k = abs(n % 10)
            if k == 0:
                return prod == 0
            assert prod % k == 0
            prod //= k
        return prod == 1

    @staticmethod
    def sol(nums):
        prod = 1
        for n in nums:
            prod *= abs(n % 10)
        return prod

    def gen_random(self):
        nums = [self.random.randrange(-100, 100) for _ in range(10)]
        self.add(dict(nums=nums))


class UppercaseEven(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#98"""

    @staticmethod
    def sat(positions: List[int], s="ThIs is A tEsT, Or *IS* iT?"):
        """Find the positions of all uppercase vowels (not counting Y) in even indices

        "EAT here NOW" => [0, 10]
        """
        assert all(s[i] in "AEIOU" for i in positions)
        return all(i in positions or c not in "AEIOU" or i % 2 == 1 for i, c in enumerate(s))

    @staticmethod
    def sol(s):
        return [i for i, c in enumerate(s) if i % 2 == 0 and c in "AEIOU"]

    def gen_random(self):
        s = "".join([self.random.choice([c.lower(), c.upper()]) for c in self.random.pseudo_word()])
        self.add(dict(s=s))


class ClosestInteger(PuzzleGenerator):
    """
    Inspired by [HumanEval](https://github.com/openai/human-eval) \\#99

    Since we can tolerate more than one answer per puzzle, we do not need to specify a tie-breaking rule.
    """

    @staticmethod
    def sat(n: int, x=329437923.5):
        """Round to nearest integer

        --- input ---
        3.7

        --- output ---
        4
        """
        return abs(n - x) <= 0.5

    @staticmethod
    def sol(x):
        return round(x)

    def gen_random(self):
        x = self.random.heavy_tail_float(lower=-1e20, upper=1e20, median_dev=1e6)
        if self.random.randrange(10) == 0:
            x = int(x) - 0.5
        self.add(dict(x=x))


class StonePiles(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#100"""

    @staticmethod
    def sat(li: List[int], n=909):
        """We are making n stone piles! The first pile has n stones. If n is even, then all piles have an even
        number of stones. If n is odd, all piles have an odd number of stones. Each pile must more stones
        than the previous pile but as few as possible. Return the number of stones in each pile.

        2 => [2, 4]
        """
        return li[0] == n and len(li) == n and all(b - a == 2 for a, b in zip(li, li[1:]))

    @staticmethod
    def sol(n):
        return [n + 2 * i for i in range(n)]

    def gen_random(self):
        n = self.random.randrange(10 ** 5)
        self.add(dict(n=n))


class CompleteSplit(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#101"""

    @staticmethod
    def sat(splits: List[List[str]], string="Hello, world!  You look like you're on turtles."):
        """
        Split a string of words separated by commas and spaces into 2 lists: words and separators

        Sample input: "Hi there, Anna"
        Sample output: [["Hi", "there", "Anna"], [" ", ", "]]
        """
        words, separators = splits
        assert len(words) == len(separators) + 1
        merged = []
        for w, s in zip(words, separators + [" "]):
            assert s.count(" ") + s.count(",") == len(s) > 0
            assert w.count(" ") + w.count(",") == 0
            merged += [w, s]
        return "".join(merged[:-1]) == string

    @staticmethod
    def sol(string):
        import re
        merged = re.split(r"([ ,]+)", string)
        return [merged[::2], merged[1::2]]

    def gen(self, target_num_instances):
        self.add(dict(string="    This is     a valley, so, so so,,,,"))
        self.add(dict(string=""))
        self.add(dict(string=" ,,,,, , , "))
        self.add(dict(string="Do not worry\nabout newlines\n!"))

    def gen_random(self):
        string = ""
        for _ in range(self.random.randrange(20)):
            string += self.random.choice([" ", ",", ".", self.random.pseudo_word()])
        self.add(dict(string=string))


class BiggestEven(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#102"""

    @staticmethod
    def sat(x: int, a=145, b=24126846790974):
        """Return the biggest even number between a and b inclusive, or -1 if there is no such number

        Example input:
        a=20, b=99

        Example output:
        98
        """
        if x == -1:
            return all(i % 2 == 1 for i in range(a, b + 1))
        return a <= x <= b and all(i % 2 == 1 for i in range(x + 1, b + 1))

    @staticmethod
    def sol(a, b):
        if a > b or (a == b and a % 2 == 1):
            return -1
        return b if b % 2 == 0 else b - 1

    def gen(self, target_num_instances):
        self.add(dict(a=17, b=17))
        self.add(dict(a=-10, b=-6))
        self.add(dict(a=100, b=84))
        self.add(dict(a=0, b=323523571223))

    def gen_random(self):
        a = self.random.randrange(10 ** self.random.randrange(10))
        b = self.random.randrange(10 ** self.random.randrange(10))
        self.add(dict(a=a, b=b))


class BinaryAverage(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#103"""

    @staticmethod
    def sat(s: str, a=-103252, b=10657):
        """Return the average of the numbers a through b rounded to nearest integer, in binary
        (or -1 if there are no such numbers)

        a=4, b=7 => '110' because the mean of 4, 5, 6 is 5 which is 110 in binary
        """
        n = int(s, 2)
        r = range(a, b)
        if len(r) == 0:
            return n == -1
        mu = sum(r) / len(r)
        return abs(mu - n) <= min(abs(mu - n - 1), abs(mu - n + 1))

    @staticmethod
    def sol(a, b):
        r = range(a, b)
        if len(r) == 0:
            return "-1"
        return bin(round(sum(r) / len(r)))

    def gen(self, target_num_instances):
        self.add(dict(a=70421, b=70421))
        self.add(dict(a=-10299, b=-10300))

    def gen_random(self):
        a = self.random.choice([-1, 1]) * self.random.randrange(10 ** self.random.randrange(5))
        b = self.random.randrange(10 ** self.random.randrange(6))
        self.add(dict(a=a, b=b))


class SortedOdds(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#104"""

    @staticmethod
    def sat(sub: List[int], nums=[17, 20, -100, 101, 423258, 19949, 0, 20174, 9351773, -11]):
        """Find the sublist of numbers with only odd digits in increasing order

        [17, 21, 18, 1, 4] => [1, 17, 21]
        """
        for i in range(len(sub)):
            n = sub[i]
            assert n == min(sub[i:])
            assert all(int(c) % 2 for c in str(abs(n)))  # all odd digits
            assert sub.count(n) == nums.count(n)

        for n in nums:
            if n not in sub:
                assert any(int(c) % 2 == 0 for c in str(abs(n)))

        return True

    @staticmethod
    def sol(nums):
        return sorted(n for n in nums if all(int(c) % 2 for c in str(abs(n))))

    def gen_random(self):
        nums = [self.random.randrange(-10 ** self.random.randrange(8), 10 ** self.random.randrange(8))
                for _ in range(self.random.randrange(20))]
        self.add(dict(nums=nums))


class BackwardsDigits(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#105"""

    @staticmethod
    def sat(backwards_digits: List[str], nums=[0, 2, 14, -2, 3, 8, 4, 5, 5, 7, 21, 101, 41, 2, 9, 6]):
        """Return the single digits in nums sorted backwards and converted to English words

        [2, 3, 4, 5, 17] => ['five', 'four', 'three', 'two']
        """
        digits = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
        li = [digits[s] for s in backwards_digits]
        for i, n in enumerate(li):
            assert n == max(li[i: i + 2])
            assert nums.count(n) == li.count(n)

        return all(n not in range(1, 10) or n in li for n in nums)

    @staticmethod
    def sol(nums):
        digits = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
        return [digits[n] for n in sorted(nums, reverse=True) if n in digits]

    def gen_random(self):
        nums = [self.random.randrange(-5, self.random.choice([12, 100])) for _ in range(self.random.randrange(20))]
        self.add(dict(nums=nums))


class AlternatingFactorials(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#106"""

    @staticmethod
    def sat(li: List[int], n=100):
        """Output a list of n integers, where the mth entry is m! if m is even or else (1+2+...+m)

        5 => [1, 2, 6, 9, 120]
        """
        assert len(li) == n
        for i, m in enumerate(li):
            if i < 2:
                assert m == i + 1
            elif i % 2 == 1:
                assert m == li[i - 2] + i + (i + 1)
            else:
                assert m == li[i - 2] * i * (i + 1)
        return True

    @staticmethod
    def sol(n):
        ans = []
        for i in range(n):
            if i < 2:
                m = i + 1
            elif i % 2 == 1:
                m = ans[i - 2] + i + (i + 1)
            else:
                m = ans[i - 2] * i * (i + 1)
            ans.append(m)

        return ans

    def gen_random(self):
        n = self.random.randrange(1000)
        self.add(dict(n=n))


class EvenPalindromeNumbers(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#107"""

    @staticmethod
    def sat(pals: List[int], n=1099, count=49):
        """Find all even palindromes up to n

        3 => [0, 2]
        """
        return all(0 <= i <= n and str(i) == str(i)[::-1] and i % 2 == 0 for i in pals) and len(set(pals)) >= count

    @staticmethod
    def sol(n, count):
        return [i for i in range(0, n + 1, 2) if str(i) == str(i)[::-1]]

    def gen_random(self):
        n = self.random.randrange(10 * 1000)
        count = len(self.sol(n, -1))
        self.add(dict(n=n, count=count))


class PositiveDigitSums(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#108"""

    @staticmethod
    def sat(pos: List[int], nums=[-804, 9124, -945, 2410, 0, 21, -123]):
        """Filter for the numbers in nums whose sum of digits is > 0, where the first digit can be negative.

        [12, -7, -102, -100] => [12, -102]
        """
        for n in pos + nums:
            s = str(n)
            if int(s[:2]) + sum(int(c) for c in s[2:]) <= 0:
                assert n not in pos
            else:
                assert pos.count(n) == nums.count(n)
        return True

    @staticmethod
    def sol(nums):
        def bad(n):
            s = str(n)
            return int(s[:2]) + sum(int(c) for c in s[2:]) <= 0

        return [n for n in nums if not bad(n)]

    def gen_random(self):
        nums = [self.random.randrange(-10 ** 5, 5 * 10 ** 4) for _ in range(self.random.randrange(1, 10))]
        self.add(dict(nums=nums))


class RotateSort(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#109

    This puzzle (and RotateString from #154) use the fact that a string is a rotation of r if it is a substring of r+r
    """

    @staticmethod
    def sat(original: List[int], arr=[2, 3, -1, -1, 0, 1, 1]):
        """
        An array is ring-sorted if it is a "rotation" of a non-decreasing list.
        Remove at most one element from arr to make it ring-sorted.

        [1, 2, 3, -1, 6, 0] => [1, 2, 3, -1, 0]
        """
        assert str(original)[1:-1] in str(sorted(original) * 2), "Not ring sorted"
        return any(original == arr[:i] + arr[i + 1:] for i in range(len(arr) + 1))

    @staticmethod
    def sol(arr):
        def sat(near):
            order_violations = 0
            erasures = 0
            for i, n in enumerate(near):
                if n < near[i - 1]:  # -1 when i =0 gives last element
                    order_violations += 1
                while n != arr[i + erasures]:
                    erasures += 1
            return order_violations <= 1 and erasures <= 1

        candidates = [arr] + [arr[:i] + arr[i + 1:] for i in range(len(arr))]
        return next(near for near in candidates if sat(near))

    def gen_random(self):
        n = self.random.randrange(10)
        original = sorted(self.random.randrange(10) for _ in range(n))
        i = self.random.randrange(n + 1)
        arr = original[i:] + original[:i]
        arr.insert(self.random.randrange(n + 1), self.random.randrange(10))
        self.add(dict(arr=arr))


class ParityExchange(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#110"""

    @staticmethod
    def sat(swaps: List[List[int]], nums1=[1, 3, 2, 4, 5, 8, 7, 11], nums2=[0, 7, 0, 8, 19, 4, 41, 43, 42]):
        """
        Find a sequence of swaps (indices into two lists) such that, after making those swaps, all numbers in the
        first list are even

        [1, 3, 4] [2, 4, 5] => [0, 1]
        """
        copy1 = nums1[:]
        copy2 = nums2[:]
        for i, j in swaps:
            copy1[i], copy2[j] = copy2[j], copy1[i]
        return all(n % 2 == 0 for n in copy1)

    @staticmethod
    def sol(nums1, nums2):
        odds = [i for i, n in enumerate(nums1) if n % 2 == 1]
        evens = [i for i, n in enumerate(nums2) if n % 2 == 0]
        return [[i, j] for i, j in zip(odds, evens)]

    def gen_random(self):
        nums1 = [self.random.randrange(-10, 10) for _ in range(self.random.randrange(10))]
        nums2 = [self.random.randrange(-10, 10) for _ in range(self.random.randrange(10))]
        if sum(n % 2 == 1 for i, n in enumerate(nums1)) <= sum(n % 2 == 0 for i, n in enumerate(nums2)):
            self.add(dict(nums1=nums1, nums2=nums2))


class CharCounts(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#111"""

    @staticmethod
    def sat(s: str, counts={'a': 4, 'b': 17, 'd': 101, 'e': 0, 'f': 12}):
        """Find a string consisting of space-separated characters with given counts

        {"f": 1, "o": 2} => "oof"
        """
        chars = s.split()
        for c in chars:
            assert chars.count(c) == counts[c]
        return len(chars) == sum(counts.values())

    @staticmethod
    def sol(counts):
        return " ".join(c for c, i in counts.items() for _ in range(i))

    def gen_random(self):
        alpha = "abcdefghijklmnopqrstuvwxyz"
        counts = {self.random.choice(alpha): self.random.randrange(10) for _ in range(self.random.randrange(10))}
        self.add(dict(counts=counts))


class DelPalindrome(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#112"""

    @staticmethod
    def sat(strings: List[str], a="this is a test", b="cat"):
        """
        Return a pair of a strings where the first string is the same as a with all the characters of b removed,
        and the second string is 'True' if this string is a palindrome otherwise 'False'.

        a="madam, I'm adam." b = "Yes, we're here." => ['madamImadam', 'True']
        """
        s, is_palindrome = strings
        i = 0
        for c in a:
            if c not in b:
                assert s[i] == c
                i += 1
        assert i == len(s)
        return is_palindrome == str(s == s[::-1])

    @staticmethod
    def sol(a, b):
        s = "".join(c for c in a if c not in b)
        return [s, str(s == s[::-1])]

    def gen_random(self):
        a = self.random.pseudo_word(max_len=50)
        b = self.random.pseudo_word(min_len=0)
        self.add(dict(a=a, b=b))


class ReplaceMe(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#113"""

    @staticmethod
    def sat(answers: List[str], lst=["234515", "21503", "2506236943"]):
        """For each string in lst, count the number of odd digits. Find a string with no t's such that replacing
        this number by t gives the string 'this is a test'

        ["123", "2"] => ["2his is a 2es2", "0his a 0es0"]
        """
        if len(answers) != len(lst):
            return False
        for a, s in zip(answers, lst):
            if "t" in a:
                return False
            num_odds = sum(int(i) % 2 for i in s)
            if a.replace(str(num_odds), "t") != "this is a test":
                return False
        return True

    @staticmethod
    def sol(lst):
        return ["this is a test".replace("t", str(sum(c in "13579" for c in s))) for s in lst]

    def gen_random(self):
        lst = [str(self.random.randrange(10 ** self.random.randrange(10))) for _ in range(self.random.randrange(10))]
        self.add(dict(lst=lst))


class MinSubArraySum(PuzzleGenerator):
    """
    Inspired by [HumanEval](https://github.com/openai/human-eval) \\#114

    This is harder than \#1114. The arrays here are chosen to be long enough that the brute-force n^2 algorithm takes
    while the O(n) algorithm takes milliseconds.
    """

    @staticmethod
    def sat(start_end: List[int], base=7, p=50741, upper=-4897754):
        """Find the start and end of the smallest-sum subarray of [(base^i mod p) - p/2 for i=start,..., end]

        base=3, p=7, upper =-3 => [0, 3]
        # because -3 is the sum of the elements [0:3] of [-2, 0, -1, 3, 1, 2, -2, 0, -1, 3 ...
        """
        start, end = start_end
        return sum(pow(base, i, p) - p // 2 for i in range(start, end)) <= upper

    @staticmethod
    def sol(base, p, upper):
        tot = 0
        best_tot = 0
        best_end = 0
        best_start = 0
        largest_cumulative_sum = 0
        largest_cumulative_sum_index = 0

        n = 1

        for i in range(p + 1):
            if tot > largest_cumulative_sum:
                largest_cumulative_sum = tot
                largest_cumulative_sum_index = i
            if tot - largest_cumulative_sum < best_tot:
                best_tot = tot - largest_cumulative_sum
                best_start = largest_cumulative_sum_index
                best_end = i

            tot += (n - p // 2)
            n = (n * base) % p

        return [best_start, best_end]

    @staticmethod
    def brute_force(base, p, upper):
        """too slow!"""
        nums = []
        n = 1
        for i in range(p):
            nums.append(n - p // 2)
            n = (n * base) % p

        return min([[i, j] for j in range(p + 1) for i in range(j + 1)], key=lambda ij: sum(nums[ij[0]:ij[1]]))

    def gen_random(self):
        p = self.random.randrange(2, 10 * 1000)
        base = self.random.randrange(1, p)

        tot = 0
        best_tot = 0
        largest_cumulative_sum = 0

        n = 1

        for i in range(p + 1):
            if tot > largest_cumulative_sum:
                largest_cumulative_sum = tot
            if tot - largest_cumulative_sum < best_tot:
                best_tot = tot - largest_cumulative_sum

            tot += (n - p // 2)
            n = (n * base) % p

        upper = best_tot

        self.add(dict(base=base, p=p, upper=upper))


class Buckets(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#115"""

    @staticmethod
    def sat(wells: List[List[List[int]]], grid=[[1, 1, 0, 1, 1], [0, 0, 0, 0, 0], [1, 1, 0, 0, 1]], capacity=2):
        """Given a grid, partition the 1's into groups of capacity [x, y] pairs, with at most one incomplete group"""
        grid2 = [[0 for _ in row] for row in grid]
        for group in wells:
            assert len(group) <= capacity
            for i, j in group:
                assert grid2[i][j] == 0
                grid2[i][j] = 1
        assert sum(len(group) != capacity for group in wells) <= 1  # at most one under-capacity group
        return grid2 == grid

    @staticmethod
    def sol(grid, capacity):
        ans = []
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                if val == 1:
                    if not ans or len(ans[-1]) == capacity:
                        ans.append([])
                    ans[-1].append([i, j])
        return ans

    def gen_random(self):
        m = self.random.randrange(1, 10)
        n = self.random.randrange(1, 10)
        grid = [[self.random.randrange(2) for _j in range(n)] for _i in range(m)]
        capacity = self.random.randrange(1, 10)
        self.add(dict(grid=grid, capacity=capacity))


class BinarySort(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#116"""

    @staticmethod
    def sat(ordered: List[int], arr=[4, 2, 3, -1, 15, 2, 6, 9, 5, 16, 1048576]):
        """Sort the numbers in arr based on the number of 1's in their binary representation.

        [1, 2, 3, 4, 6] => [1, 2, 4, 3, 6]
        """
        if sorted(ordered) != sorted(arr):
            return False  # not even a permutation
        return all(bin(a).count("1") <= bin(b).count("1") for a, b in zip(ordered, ordered[1:]))

    @staticmethod
    def sol(arr):
        return sorted(arr, key=lambda n: bin(n).count("1"))

    def gen_random(self):
        arr = [self.random.randrange(-100, 100) for _ in range(self.random.randrange(20))]
        self.add(dict(arr=arr))


class ConsonantFilter(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#117"""

    @staticmethod
    def sat(words: List[str], s="This is not a very hard puzzle", n=3):
        """Find all words in the string with n consonants

        Sample input:
        s="An eye for an I", n=1
        Sample output:
        ["An", "eye", "an"]
        """
        i = 0
        for w in s.split():
            num_consonants = 0
            for c in w.lower():
                if c not in "aeiou":
                    num_consonants += 1
            if num_consonants == n:
                if words[i] != w:
                    return False
                i += 1
        return i == len(words)

    @staticmethod
    def sol(s, n):
        return [w for w in s.split() if sum(c.lower() not in "aeiou" for c in w) == n]

    def gen_random(self):
        s = " ".join(self.random.pseudo_word() for _ in range(self.random.randrange(10)))
        n = self.random.randrange(7)
        self.add(dict(s=s, n=n))


class VowelSandwich(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#118"""

    @staticmethod
    def sat(ham: str, s="Any vowel is OK"):
        """Find any vowel sandwich, a string consisting of a vowel between two consonants, contained in s

        "sandwhich" => "hic"
        """
        vows = "aeiou"
        cons = "bcdfghjklmnpqrstvwxz"
        return ham in s and ham[0].lower() in cons and ham[1].lower() in vows and ham[2].lower() in cons

    @staticmethod
    def sol(s):
        vows = "aeiou"
        cons = "bcdfghjklmnpqrstvwxz"
        return next(s[i - 1:i + 2] for i in range(1, len(s) - 1)
                    if s[i].lower() in vows and s[i - 1].lower() in cons and s[i + 1].lower() in cons)

    def gen(self, target_num_instances):
        self.add(dict(s="wOwwwww!"))
        self.add(dict(s="do pyp you know ?"))

    def gen_random(self):
        vows = "aeiou"
        cons = "bcdfghjklmnpqrstvwxz"
        s = " ".join(self.random.pseudo_word() for _ in range(self.random.randrange(10)))
        if any(s[i].lower() in vows and s[i - 1].lower() in cons and s[i + 1].lower() in cons
               for i in range(1, len(s) - 1)):
            self.add(dict(s=s))


class ParenthesesPermutation(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#119
    
    This is harder version in which you need to find a permutation of many substrings. Brute force is too slow.
    """

    @staticmethod
    def sat(perm: str,
            s="))(  )()()() )))(( ))))((( )))))(((( ))))))))((((((( ))))))((((( )))))))(((((( )))))))))(((((((  (((((((((("):
        """The string s consists of groups of parentheses separated by spaces.
        Permute the groups such that the parentheses match.

        "( ) )(" => "( )( )"
        """
        assert sorted(perm.split()) == sorted(s.split()), "Must be a permutation of the space-delimited 'groups'"
        return all(perm[:i].count("(") >= perm[:i].count(")") for i in range(len(perm)))

    @staticmethod
    def sol(s):
        assert all(c in "( )" for c in s)
        parts = s.split()

        def min_depth(part):
            """Returns the lowest depth <= 0"""
            ans = 0
            depth = 0
            for c in part:
                if c == ")":
                    depth -= 1
                    ans = min(ans, depth)
                else:
                    depth += 1
            return ans

        def greedy_reorder(subs):
            """Reorder a bunch of parentheses substrings so as to maintain # ('s > # )'s """
            queue = subs[:]
            subs[:] = []
            height = 0
            while queue:
                best = max([s for s in queue if min_depth(s) + height >= 0], key=lambda s: s.count("(") - s.count(")"))
                height += best.count("(") - best.count(")")
                subs.append(best)
                queue.remove(best)

        lefts = [s for s in parts if s.count("(") >= s.count(")")]

        greedy_reorder(lefts)

        def mirror(sub):
            return "".join(")" if c == "(" else "(" for c in sub[::-1])

        rights = [mirror(s) for s in parts if s.count("(") < s.count(")")]  # mirror temporarily for reordering

        greedy_reorder(rights)
        return " ".join(lefts + [mirror(s) for s in rights[::-1]])

    def gen_random(self):
        parts = []
        depth = 0
        buffer = ''
        while depth > 0 or self.random.random() > 0.1:
            c = self.random.choice('()()()()())' if depth > 0 else '(')
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            buffer += c
            if self.random.randrange(10) == 0:
                parts.append(buffer)
                buffer = ''
        parts.append(buffer)
        self.random.shuffle(parts)
        s = " ".join(parts)
        self.add(dict(s=s))


class BiggestK(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#120"""

    @staticmethod
    def sat(biggest: List[int], k=7, nums=[31, 1, 2, -10, -2, 4, 17, 18, 20, 14, 20, 21, 18, 0]):
        """Find the largest k numbers

        k=2, [1, 2, 3, 4, 5, 5, 3, 5, 2] => [5, 5]
        """
        if len(biggest) != k:
            return False
        smallest = nums[:]
        for n in biggest:
            smallest.remove(n)
        return k == 0 or k == len(nums) or max(smallest) <= min(biggest)

    @staticmethod
    def sol(k, nums):
        return sorted(nums, reverse=True)[:k]

    def gen_random(self):
        length = self.random.randrange(20)
        nums = [self.random.randrange(-10, 100) for _ in range(length)]
        k = self.random.randrange(length + 1)
        self.add(dict(k=k, nums=nums))


class OddEvenSum(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#121

    Very similar to EvenOddSum from \#85"""

    @staticmethod
    def sat(tot: int, nums=[18, 42152, 125023521, -1221873620123, 17, 19]):
        """Find the sum of the odd elements that are at even indices

        [0, 1, 2, 3, 5, 6] => 5
        """
        for i in nums[::2]:
            if i % 2 == 1:
                tot -= i
        return tot == 0

    @staticmethod
    def sol(nums):
        return sum(i for i in nums[::2] if i % 2 == 1)

    def gen_random(self):
        nums = [self.random.randrange(-100, 100) for _ in range(self.random.randrange(20))]
        self.add(dict(nums=nums))


class LongEarlySum(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#122
    
    Changed slightly to make the answer not be a small integer. 
    """

    @staticmethod
    def sat(tot: int, k=5, nums=[1252, 125273523, 0, 42, 100, 214532, 2, 0, 11, 14]):
        """Find the sum of the numbers among the first k with more than 2 digits

        k=3, nums=[2, 102, 12, 1000] => 102
        """
        for n in nums[:k]:
            if len(str(abs(n))) > 2:
                tot -= n
        return tot == 0

    @staticmethod
    def sol(k, nums):
        return sum(n for n in nums[:k] if len(str(abs(n))) > 2)

    def gen_random(self):
        length = self.random.randrange(1, 20)
        nums = [self.random.randrange(-10 ** 10, 10 ** 10) for _ in range(length)]
        k = self.random.randrange(10)
        self.add(dict(k=k, nums=nums))


class OddCollatz(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#123"""

    @staticmethod
    def sat(odds: List[int], n=1243272912731):
        """Find the odd numbers in the collatz sequence starting at n

        3 => [3, 5, 1]  # because the Collatz sequence starting with 3 is [3, 10, 5, 16, 8, 4, 2, 1]
        """
        num_odds = 0
        while True:
            if n % 2 == 1:
                num_odds += 1
                if n not in odds:
                    return False
            if n <= 1:
                return num_odds == len(odds)
            n = (3 * n + 1) if n % 2 == 1 else n // 2

    @staticmethod
    def sol(n):
        ans = []
        while True:
            if n % 2 == 1:
                ans.append(n)
            if n <= 1:
                return ans
            n = (3 * n + 1) if n % 2 == 1 else n // 2

    def gen_random(self):
        n = self.random.randrange(1, 10 ** self.random.randrange(1, 20))
        self.add(dict(n=n))


class DateDiff(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#124"""

    @staticmethod
    def sat(s: str, target=-2075):
        """Find a valid date mm-dd-yyyy such that the date, viewed as a mathematical expression, evaluates to target

        -2029 => "10-18-2021" # because 10-18-2021 == -2029
        """
        assert all(c in "0123457689-" for c in s) and s[2] == s[5] == "-"
        m, d, y = [int(n) for n in s.split("-")]
        assert m in range(1, 13)
        assert d in range(1, 32)
        if m in [4, 6, 9, 11]:
            assert d <= 30
        if m == 2:
            assert d <= 29
        return m - d - y == target

    @staticmethod
    def sol(target):
        if target >= -30:
            return "12-01-" + str(11 - target).zfill(4)
        return "01-31-" + str(-30 - target).zfill(4)

    def gen(self, target_num_instances):
        self.add(dict(target=11))
        self.add(dict(target=-30))
        self.add(dict(target=-1999))
        self.add(dict(target=-10029))

    def gen_random(self):
        target = self.random.randrange(-10029, 12)
        self.add(dict(target=target))


class StrangeSplit(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#125"""

    @staticmethod
    def sat(lst: List[str], s="Hello, world!"):
        """Split s into strings if there is a space in s, otherwise split on commas if there is a comma, otherwise
        return the list of lowercase letters with odd order (order of a = 0, b = 1, etc.)

        "a b c" => ["a", "b", "c"]
        "a,b" => ["a", "b"]
        """
        if " " in s:
            return " ".join(lst) == s
        if "," in s:
            return ",".join(lst) == s
        return "".join(lst) == "".join(c for c in s if c.islower() and ord(c) % 2 == 0)

    @staticmethod
    def sol(s):
        if " " in s:
            return s.split(" ")
        if "," in s:
            return s.split(",")
        return [c for c in s if c.islower() and ord(c) % 2 == 0]

    def gen(self, target_num_instances):
        self.add(dict(s="Goodbye,spaces!"))
        self.add(dict(s="abcbcbbedfsgfakbfjghskbne[pewte"))

    def gen_random(self):
        words = [self.random.pseudo_word() for _ in range(self.random.randrange(10))]
        s = self.random.choice([" ", ",", ""]).join(words)
        self.add(dict(s=s))


class IncreasingViolation(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#126"""

    @staticmethod
    def sat(violation: List[int], nums=[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 17, 17, 18, 19, 20, 22, 24]):
        """
        Find the indices of two entries that show that the list is not in increasing order.
        If there are no violations (they are increasing), return an empty list.

        [1,2,3,0,4,5,6] => [1, 3]
        """
        if not violation:
            return all(nums[i] < nums[i + 1] for i in range(len(nums) - 1))
        i, j = violation
        return 0 <= i < j and nums[i] >= nums[j]

    @staticmethod
    def sol(nums):
        for i in range(len(nums) - 1):
            if nums[i] >= nums[i + 1]:
                return [i, i + 1]
        return []

    def gen_random(self):
        nums = sorted(self.random.randrange(100) for _ in range(self.random.randrange(2, 20)))
        if self.random.randrange(2) == 1:
            i = self.random.randrange(len(nums))
            nums.insert(i, self.random.randrange(nums[i] + 1))
        self.add(dict(nums=nums))


class PrimeIntervalIntersection(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#127"""

    @staticmethod
    def sat(interval2: List[int], interval1=[32157, 93210127]):
        """Find an interval whose intersection with a given interval has a width that is a prime integer.

        [7, 100] => [0, 10]  # because 10-7=3 is prime
        """
        intersection_width = min(interval1[1], interval2[1]) - max(interval1[0], interval2[0])
        return intersection_width > 1 and all(intersection_width % i for i in range(2, intersection_width))

    @staticmethod
    def sol(interval1):
        a, b = interval1
        assert b - a >= 2
        return [a, a + 2]

    def gen_random(self):
        a, b = [self.random.randrange(10 ** self.random.randrange(10)) * self.random.choice([-1, 1]) for _ in range(2)]
        if b - a >= 2:
            interval1 = [a, b]
            self.add(dict(interval1=interval1))


class ProductSigns(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#128
    
    Easy puzzle since the answer is computed in the puzzle, but it is okay to have a few trivial puzzles.
    """

    @staticmethod
    def sat(n: int, arr=[1, 7, -20052, 14, -3, -11, 1025235, 14]):
        """Find the sum of the magnitudes of the elements in the array with a sign that is equal to the product of
        the signs of the entries.

        [1, -2, 3] => -6  # negative because there is one negative
        """
        tot = 0

        for i in arr:
            if tot >= 0:
                tot += abs(i)
            else:
                tot -= abs(i)
            if i < 0:
                tot = -tot
            elif i == 0:
                tot = 0
                break

        return n == tot

    @staticmethod
    def sol(arr):
        tot = sum(abs(i) for i in arr)
        if all(arr):
            return tot if sum(i < 0 for i in arr) % 2 == 0 else -tot
        return 0

    def gen_random(self):
        arr = [self.random.randrange(-100, 100) for _ in range(self.random.randrange(20))]
        self.add(dict(arr=arr))


class LexPath(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#129"""

    @staticmethod
    def sat(path: List[int], k=10, edges=[[2, 4], [3], [4, 1], [4], [0]]):
        """Find the lexicographically smallest path of length k in graph with given edge matrix (and no dead ends)

        k=3, edges=[[1,3], [0, 3], [2], [3]] => [0, 1, 0] # because 0-1 and 1-0 are edges
        """

        def check(prefix):
            for i, j in zip(path, prefix):
                if i != j:
                    return i < j
            return len(prefix) >= k or all(check(prefix + [i]) for i in edges[prefix[-1]])

        return all(path[i] in edges[path[i - 1]] for i in range(1, k)) and all(check([i]) for i in range(len(edges)))

    @staticmethod
    def sol(k, edges):
        path = []
        while len(path) < k:
            path.append(min(edges[path[-1]]) if path else 0)
        return path

    def gen_random(self):
        n = self.random.randrange(2, 6)
        edges = [self.random.sample(range(n), self.random.randrange(1, n + 1)) for i in range(n)]
        k = self.random.randrange(20)
        self.add(dict(k=k, edges=edges))


class Tribonacci(PuzzleGenerator):
    """
    Inspired by [HumanEval](https://github.com/openai/human-eval) \\#130

    This puzzle is a bit harder because the definition is slightly different at seq[1].
    """

    @staticmethod
    def sat(seq: List[int], length=181):
        """Find a sequence where seq[n] == 1 + n / 2 for even n, and
        seq[n] == seq[n - 1] + seq[n - 2] + seq[n + 1] for odd n < length."""
        return all(seq[n] == (seq[n - 1] + seq[n - 2] + seq[n + 1] if n % 2 else 1 + n // 2) for n in range(length))

    @staticmethod
    def sol(length):
        seq = []
        while len(seq) <= length:
            n = len(seq)
            if n % 2 == 0:
                seq.append(1 + n // 2)
            else:
                seq.append(sum(seq[-2:]) + (1 + (n + 1) // 2))
        return seq + [0]  # appending 0 at the end makes it easier so that seq[n-2] == 0 for n == 1

    def gen_random(self):
        length = self.random.randrange(1000)
        self.add(dict(length=length))


class OddProduct(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#131"""

    @staticmethod
    def sat(prod: int, n=14235764939971075543215213):
        """Return the product of the odd digits in n, or 0 if there aren't any

        12345 => 15
        """

        for c in str(n):
            i = int(c)
            if i % 2 == 1:
                assert prod % i == 0
                prod //= i
        return prod == any(int(c) % 2 for c in str(n))

    @staticmethod
    def sol(n):
        if any(int(c) % 2 for c in str(n)):
            prod = 1
            for c in str(n):
                if int(c) % 2 == 1:
                    prod *= int(c)
            return prod
        return 0

    def gen_random(self):
        n = self.random.randrange(10 ** self.random.randrange(20))
        self.add(dict(n=n))


class ValidBracketSubsequence(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#132"""

    @staticmethod
    def sat(valid: str, s="]]]]]]]]]]]]]]]]][][][][]]]]]]]]]]][[[][[][[[[[][][][]][[[[[[[[[[[[[[[[[["):
        """Find a valid substring of s that contains matching brackets, at least one of which is nested

        "]][][[]]]" => "[][[]]"
        """
        assert valid in s
        depths = [0]
        for c in valid:
            if c == "[":
                depths.append(depths[-1] + 1)
            elif c == "]":
                depths.append(depths[-1] - 1)
        return depths[-1] == 0 and min(depths) == 0 and max(depths) > 1

    @staticmethod
    def sol(s):
        left = []
        nested = False
        for i, c in enumerate(s):
            if c == "[":
                if len(left) == 2:
                    left = [left[1], i]
                    nested = False
                else:
                    left.append(i)
            elif c == "]":
                if not left:
                    continue
                if len(left) == 1 and nested:
                    return s[left[0]:i + 1]
                elif len(left) == 2:
                    nested = True
                left.pop()
        assert False

    @staticmethod
    def sol2(s):
        import re
        return re.search(r"\[(\[\])+\]", s).group(0)

    def gen_random(self):
        arr = [self.random.choice("[]") for _ in range(20)]
        match = "[" + "[]" * self.random.randrange(1, 10) + "]"
        arr.insert(self.random.randrange(len(arr) + 1), match)
        s = "".join(arr)
        self.add(dict(s=s))


class CeilingSquares(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#133"""

    @staticmethod
    def sat(running_squares: List[int], x=[201.1, 301.4, -18.1, 1244122.0, 10101.0101, 1e7]):
        """Round each float in x up to the next integer and return the running total of the integer squares

        [2.4, 3.7, 0.1] => [9, 25, 26]
        """
        for i, v in enumerate(x):
            ceiling = int(v) + (v > 0 and not v.is_integer())
            square = ceiling ** 2
            if running_squares[i] != square + (i > 0 and running_squares[i - 1]):
                return False

        return len(running_squares) == len(x)

    @staticmethod
    def sol(x):
        from math import ceil
        running_squares = []
        tot = 0
        for v in x:
            tot += ceil(v) ** 2
            running_squares.append(tot)
        return running_squares

    def gen_random(self):
        x = [self.random.uniform(-10, 10) for _ in range(self.random.randrange(2, 10))]
        self.add(dict(x=x))


class LastLetters(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#134"""

    @staticmethod
    def sat(y: List[bool], x=["Hello, world!", "cat", "", "a test", "test a", "i e", "o", "I O U", "You and I"]):
        """Determine, for each string in x, whether the last character is an isolated letter

        ["a b c", "abc"] => [True, False]
        """
        assert len(x) == len(y)
        for s, b in zip(x, y):
            if len(s.split(" ")[-1]) == 1:
                assert b == s[-1].isalpha()
            else:
                assert not b
        return True

    @staticmethod
    def sol(x):
        return [len(s.split(" ")[-1]) == 1 and s[-1].isalpha() for s in x]

    def gen_random(self):
        x = []
        for _ in range(self.random.randrange(30)):
            x.append(" ".join(self.random.pseudo_word() for _ in range(self.random.randrange(3))))
            if self.random.randrange(3):
                x[-1] += " " + self.random.char()

        self.add(dict(x=x))


class Drops(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#135"""

    @staticmethod
    def sat(drop_indexes: List[int], nums=[2, -1, 14, 8, 9, 9, 8, 4, 2, 4, 3, -100, 1000, 18, 4, -2, -3, -3, 1, 0]):
        """Find the indices for which the nums array drops.

        [1,2,3,0,2,4,1] => [3,6]
        """
        d = 0
        for i in range(1, len(nums)):
            if nums[i] < nums[i - 1]:
                assert drop_indexes[d] == i
                d += 1
        return d == len(drop_indexes)

    @staticmethod
    def sol(nums):
        return [i for i in range(1, len(nums)) if nums[i] < nums[i - 1]]

    def gen_random(self):
        nums = [self.random.randrange(-100, 100) for _ in range(self.random.randrange(30))]


class LargestNegSmallestPos(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#136"""

    @staticmethod
    def sat(extremes: List[int], nums=[-10, -4, 100, -40, 2, 2, 3, 17, -50, -25, 18, 41, 9, 11, 15]):
        """Find the largest negative ans smallest positive numbers (or 0 if none)

        [-2, -4, 14, 50] => [-2, 14]
        [3, 22] => [0, 3]
        """
        neg, pos = extremes
        if neg == 0:
            assert nums == [] or min(nums) >= 0
        else:
            assert neg < 0 and neg in nums and all(n >= 0 or n <= neg for n in nums)
        if pos == 0:
            assert nums == [] or max(nums) <= 0
        else:
            assert pos > 0 and pos in nums and all(n <= 0 or n >= pos for n in nums)
        return True

    @staticmethod
    def sol(nums):
        pos = [n for n in nums if n > 0]
        neg = [n for n in nums if n < 0]
        return [max(neg) if neg else 0, min(pos) if pos else 0]

    def gen_random(self):
        nums = [self.random.randint(-1000, 1000) for _ in range(self.random.randrange(20))]
        self.add(dict(nums=nums))


class LargestStringNum(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#137"""

    @staticmethod
    def sat(x: float, str_nums=["1,3", "-11", "17.5", "-11", "2", "2.2", "2,2", "4", "-18,18", "99.09"]):
        """Find the largest number where commas or periods are decimal points

        ["99,9", "100"] => 100.0
        """
        found = False
        for s in str_nums:
            y = float(s.replace(",", "."))
            assert y <= x
            if y == x:
                found = True
        return found

    @staticmethod
    def sol(str_nums):
        return max(float(s.replace(",", ".")) for s in str_nums)

    def gen_random(self):
        def rand():
            if self.random.randrange(2):
                return str(self.random.randrange(-100, 100))
            else:
                return str(self.random.uniform(-100, 100)).replace(".", self.random.choice(".,"))

        str_nums = [rand() for _ in range(self.random.randrange(1, 20))]
        self.add(dict(str_nums=str_nums))


class Even4Sum(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#138"""

    @staticmethod
    def sat(summands: List[int], n=1234567890):
        """Find four positive even integers whose sum is n

        100 => [22, 24, 26, 28]"""
        return sum(summands) == n and min(summands) > 0 and len(summands) == 4 and all(s % 2 == 0 for s in summands)

    @staticmethod
    def sol(n):
        return [2] * 3 + [n - 6]

    def gen(self, target_num_instances):
        self.add(dict(n=8))
        self.add(dict(n=10))
        self.add(dict(n=12))

    def gen_random(self):
        n = 2 * self.random.randrange(4, 10 ** self.random.randrange(1, 10))
        self.add(dict(n=n))


class InverseSuperFactorial(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#139"""

    @staticmethod
    def sat(nums: List[int], super_factorials=[1, 2, 1]):
        """The super-factorial of n is n! (n-1)! (n-2)! ... 1!. Invert a given list of super-factorials.

        [1, 2, 2, 12] => [1, 2, 2, 3]
        """
        for i, sf in enumerate(super_factorials):
            n = nums[i]
            for j in range(n, 0, -1):
                k = j ** (n - j + 1)
                assert sf % k == 0, f"{i} {sf} {j} {n}"
                sf //= k
            assert sf == 1
        return True

    @staticmethod
    def sol(super_factorials):
        queue = set(super_factorials)
        cache = {}
        n = 1
        fact = 1
        s_fact = 1
        while queue:
            fact *= n
            s_fact *= fact
            if s_fact in queue:
                queue.remove(s_fact)
                cache[s_fact] = n
            n += 1
        return [cache[sf] for sf in super_factorials]

    @staticmethod
    def superfactorial(n):
        i = 1
        fact = 1
        s_fact = 1
        for i in range(1, n + 1):
            fact *= i
            s_fact *= fact
        return s_fact

    def gen_random(self):
        super_factorials = [self.superfactorial(self.random.randrange(10)) for _ in range(11)]
        self.add(dict(super_factorials=super_factorials))


class ExpandSpaces(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#140"""

    @staticmethod
    def sat(orig: str, target="-Hello,_world!__This_is-so-easy!-"):
        """Find a string such that, when three or more spaces are compacted to a '-' and one or two spaces are
        replaced by underscores, leads to the target.

        "_o-k__?-" => "  o        k  ?     "
        """
        assert "_" not in orig and "-" not in orig
        new = ""
        space_count = 0
        for c in orig:
            if c == " ":
                space_count += 1
            else:
                new += ("-" if space_count > 2 else "_" * space_count)
                new += c
                space_count = 0
        new += ("-" if space_count > 2 else "_" * space_count)
        return new == target

    @staticmethod
    def sol(target):
        return target.replace("-", " " * 3).replace("_", " ")

    def gen_random(self):
        target = "".join(self.random.choice(["-", "_", self.random.char(), self.random.pseudo_word()])
                         for _ in range(self.random.randrange(10))).replace(" ", "")
        if "___" not in target and "-_" not in target and "_-" not in target and "--" not in target:
            self.add(dict(target=target))


class FilenameOK(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#141"""

    @staticmethod
    def sat(valids: List[str], filenames=["cat.txt", "!jog.dll", "31F9.html", "Is this okay?.txt", ".exe", ""]):
        """Return a list of Yes/No strings that determine whether candidate filename is valid. A valid filename
        should end in .txt, .exe, or .dll, and should have at most three digits, no additional periods

        ["train.jpg", "doc10234.txt", "3eadme.txt"] = ["No", "No", "Yes"]
        """
        assert len(valids) == len(filenames)
        for v, f in zip(valids, filenames):
            n_digits = sum(c.isdigit() for c in f)
            if v == "Yes":
                prefix, ext = f.split(".")
                assert ext in ["txt", "dll", "exe"] and prefix[0].isalpha() and n_digits < 4
            else:
                assert v == "No"
                assert f.split(".")[1:] not in [['txt'], ['dll'], ['exe']] or not f[0].isalpha() or n_digits > 3
        return True

    @staticmethod
    def sol(filenames):
        return ["Yes" if
                f.split(".")[1:] in [['txt'], ['dll'], ['exe']] and f[0].isalpha() and sum(c.isdigit() for c in f) < 4
                else "No"
                for f in filenames]

    def gen_random(self):
        filenames = [
            self.random.char() + self.random.pseudo_word() + self.random.char() + self.random.choice([
                ".txt", ".dll", ".exe", ".mp4", ".tar.zip"])
            for _ in range(self.random.randrange(20))
        ]
        self.add(dict(filenames=filenames))


class FindStrangeSum(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#142"""

    @staticmethod
    def sat(lst: List[int], tot=1125181293221):
        """Find a list of integers such that tot is the sum of (n^2 if 3 | n, else n^3 if 4 | n, else n)"""
        return sum(n ** 2 if n % 3 == 0 else n ** 3 if n % 4 == 0 else n for n in lst) == tot

    @staticmethod
    def sol(tot):
        residue = (tot - 1) % 12
        return [1] * residue + [tot - residue]

    def gen_random(self):
        tot = self.random.choice([-1, 1]) * self.random.randrange(0, 10 ** self.random.randrange(10))
        self.add(dict(tot=tot))


class PrimeWords(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#143"""

    @staticmethod
    def sat(primes: str, s="This is a test of whether you would want to do such strange puzzles"):
        """Find the string consisting of all the words whose lengths are prime numbers

        "A bird in the hand is worth two in the bush" => "in the is worth two in the"
        """

        def is_prime(n):
            return n > 1 and all(n % j for j in range(2, int(n ** 0.5) + 1))

        prime_words = primes.split()
        i = 0
        for word in s.split():
            if is_prime(len(word)):
                assert prime_words[i] == word
                i += 1

        return i == len(prime_words)

    @staticmethod
    def sol(s):
        def is_prime(n):
            return n > 1 and all(n % j for j in range(2, int(n ** 0.5) + 1))

        return " ".join(w for w in s.split() if is_prime(len(w)))

    def gen_random(self):
        s = " ".join(self.random.pseudo_word() for _ in range(self.random.randrange(10)))
        self.add(dict(s=s))


class SimplifyProductFraction(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#144"""

    @staticmethod
    def sat(z: str, x="-8142432/763083", y="66/-13474", max_len=18):
        """Write x * y as the shortest equivalent fraction using at most max_len chars

        x="-2/3", y="-3/8", max_len=3 => "1/4"
        """
        [[a, b], [c, d], [u, v]] = [[int(n) for n in s.split("/")] for s in [x, y, z]]
        return a * c * v == b * d * u and len(z) <= max_len

    @staticmethod
    def sol(x, y, max_len):
        [[a, b], [c, d]] = [[int(n) for n in s.split("/")] for s in [x, y]]
        num, den = a * c, b * d
        if num < 0 and den < 0:
            num, den = -num, -den
        if num == 0:
            return "0/1"

        def gcd(a, b):
            a, b = min(a, b), max(a, b)
            if b % a == 0:
                return a
            return gcd(b % a, a)

        d = gcd(abs(num), abs(den))
        return f'{num // d}/{den // d}'

    @staticmethod
    def naive(x, y):
        [[a, b], [c, d]] = [[int(n) for n in s.split("/")] for s in [x, y]]
        return f"{a * c}/{b * d}"

    def gen_random(self):
        a, b, c, d = [self.random.choice([-1, 1, 1]) * self.random.randrange(10 ** self.random.randrange(10)) for _ in
                      range(4)]
        if b == 0 or d == 0:
            return
        x, y = f"{a}/{b}", f"{c}/{d}"
        max_len = len(self.sol(x, y, None))
        bad_len = len(self.naive(x, y))
        if max_len < bad_len - 3:
            self.add(dict(x=x, y=y, max_len=max_len))


class SortByDigitSum(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#145"""

    @staticmethod
    def sat(ordered: List[int], nums=[1, 0, -1, -100, 10, 14, 235251, 11, 10000, 2000001, -155]):
        """Sort the numbers by the sum of their digits

        [17, 21, 0] => [0, 17, 21]
        """
        digit_sums = [sum(int(c) for c in str(n) if c != "-") for n in ordered]
        return sorted(ordered) == sorted(nums) and digit_sums == sorted(digit_sums)

    @staticmethod
    def sol(nums):
        return sorted(nums, key=lambda n: sum(int(c) for c in str(n) if c != "-"))

    def gen_random(self):
        nums = [self.random.randrange(-1000, 1000) for _ in range(self.random.randrange(10))]
        self.add(dict(nums=nums))


class BigOdds(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#146"""

    @staticmethod
    def sat(odds: List[int], nums=[204, 109, 203, 17, 45, 11, 21, 99, 909, 16, -33, 3, 17]):
        """Find the numbers that are greater than 10 and have odd first and last digits

        [73, 4, 72] => [73]
        """
        assert all(o > 10 and odds.count(o) == nums.count(o) and int(str(o)[i]) % 2 for o in odds for i in [-1, 0])
        return all(n in odds or n <= 10 or int(str(n)[0]) % 2 == 0 or int(str(n)[-1]) % 2 == 0 for n in nums)

    @staticmethod
    def sol(nums):
        return [n for n in nums if n > 10 and (int(str(n)[0]) * int(str(n)[-1])) % 2]

    def gen_random(self):
        nums = [self.random.randrange(-10 ** 3, 2 * 10 ** 4) for _ in range(self.random.randrange(10))]
        self.add(dict(nums=nums))


class Threeples(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#147"""

    @staticmethod
    def sat(trips: List[List[int]], a=[1, 0, -17, 42, 321, 36, 429, 35, 10, 923, 35, 18, 0, 17, 24, 32, 8], count=221):
        """Find all triples of increasing indices where the sum of the numbers is divisible by three

        a=[1, 2, 4, 8, 14, 10], count=2 => [[0, 2, 5], [1, 3, 4]] = > because 1 + 4 + 10, 2 + 8 + 14 are divisible by 3
        """
        assert len({tuple(t) for t in trips}) >= count
        return all(0 <= i < j < k and (a[i] + a[j] + a[k]) % 3 == 0 for i, j, k in trips)

    @staticmethod
    def sol(a, count):
        n = len(a)
        return [[i, j, k] for k in range(2, n) for j in range(k) for i in range(j) if (a[i] + a[j] + a[k]) % 3 == 0]

    def gen_random(self):
        a = [self.random.randrange(-1, 10) for _ in range(self.random.randrange(30))]
        count = len(self.sol(a, count=None))
        self.add(dict(a=a, count=count))


class PlanetRange(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#148"""

    @staticmethod
    def sat(planets_between: List[str], a="Mars", b="Neptune"):
        """Find all planets between the two given planets

        a="Jupiter", b="Pluto" => ["Saturn" "Uranus" "Neptune"]
        """
        assert " " not in "".join(planets_between)
        return " ".join([a] + planets_between + [b]) in "Venus Earth Mars Jupiter Saturn Uranus Neptune Pluto"

    @staticmethod
    def sol(a, b):
        planets = "Venus Earth Mars Jupiter Saturn Uranus Neptune Pluto".split()
        return planets[planets.index(a) + 1:planets.index(b)]

    def gen_random(self):
        planets = "Venus Earth Mars Jupiter Saturn Uranus Neptune Pluto".split()
        i = self.random.randrange(1, len(planets))
        j = self.random.randrange(i)
        b = planets[i]
        a = planets[j]
        self.add(dict(a=a, b=b))


class EvenWords(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#149"""

    @staticmethod
    def sat(evens: List[str], words=["The", "worm", "ate", "a", "bird", "imagine", "that", "!", "Absurd", "!!"]):
        """Find the even-length words and sort them by length.

        ["soup", "not", "splendid"] => ["soup", "splendid"]
        """
        lens = [len(w) for w in evens]
        assert all(lens[i] % 2 == 0 and lens[i] == max(lens[:i + 1]) and w in words for i, w in enumerate(evens))
        return all((len(w) % 2 == 1 or w in evens) for w in words)

    @staticmethod
    def sol(words):
        return sorted([w for w in words if len(w) % 2 == 0], key=lambda w: (len(w), w))

    def gen_random(self):
        words = [self.random.pseudo_word() for _ in range(self.random.randrange(1, 10))]
        self.add(dict(words=words))


class PrimeSel(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#150"""

    @staticmethod
    def sat(neighbors: List[int], nums=[14, 7, 11, 13, 7, 4, 19, 2, 55, 13, 31, 14, 2, 9, -7, 0, 88, 13, 13]):
        """Find a list of all numbers that are adjacent to a prime number in the list, sorted without duplicates

        [2, 17, 16, 0, 6, 4, 5] => [2, 4, 16, 17]"""

        def prime(m):
            return all(m % i for i in range(2, m - 1))

        goods = set()
        for i, n in enumerate(nums):
            if (i > 0 and prime(nums[i - 1])) or (i < len(nums) - 1 and prime(nums[i + 1])):
                goods.add(n)

        return set(neighbors) == goods and all(n == min(neighbors[i:]) for i, n in enumerate(neighbors))

    @staticmethod
    def sol(nums):
        def prime(m):
            return all(m % i for i in range(2, m - 1))

        return sorted({
            n for i, n in enumerate(nums)
            if (i > 0 and prime(nums[i - 1])) or (i < len(nums) - 1 and prime(nums[i + 1]))
        })

    def gen_random(self):
        nums = [self.random.randrange(-1, 20) for _ in range(self.random.randrange(30))]
        self.add(dict(nums=nums))


class EvenSqure(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#151"""

    @staticmethod
    def sat(tot: int, xs=[123.0, 872322.0, 542.2, -127.5, 18214.0, 3732.4, 12832.4, 23523800.0]):
        """Find the sum of the squares of the positive even integers

        [2.0, 3.0, 2.5, 4.0] => 20
        """
        for x in xs:
            if x.is_integer() and x > 0 and x % 2 == 0:
                tot -= int(x) ** 2

        return tot == 0

    @staticmethod
    def sol(xs):
        return sum(int(x) ** 2 for x in xs if x.is_integer() and x > 0 and x % 2 == 0)

    def gen_random(self):
        xs = [self.random.randrange(-10 ** 5, 3 * 10 ** 5) * self.random.choice([1.0, self.random.random()])
              for _ in range(self.random.randrange(10))]
        if self.sol(xs) > 0 or self.random.randrange(10) == 0:
            self.add(dict(xs=xs))


class ArrayDiff(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#152"""

    @staticmethod
    def sat(b: List[int], a=[1, 2, 3, 0, 4, 17, 2, 4, 5, 9, 8, 4], c=[1, 2, 3, 4, 0, 16, 2, 3, 5, 9, 8, 4]):
        """Find an array that when added to vector a gives array vector c

        [1, 2, 3], [4, 17, 5] => [3, 15, 2]
        """
        return len(b) == len(a) and all(i + j == k for i, j, k in zip(a, b, c))

    @staticmethod
    def sol(a, c):
        return [k - i for i, k in zip(a, c)]

    def gen_random(self):
        a = [self.random.randrange(-1, 20) for _ in range(self.random.randrange(30))]
        c = [self.random.randrange(-1, 20) for _ in range(len(a))]
        self.add(dict(a=a, c=c))


class StrongestExtension(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#153"""

    @staticmethod
    def sat(s: str, class_name="TestClass", extensions=["extEnd", "LOL", "SuPeRbLy", "v9ACLQWTEW", "PickMe", "AI"]):
        """Find the class_name.extension for the extension that has the largest #capitals - #lowercase letters"""
        assert s.startswith(class_name + ".")
        ext = s[len(class_name) + 1:]

        def case_delta(x: str):
            tot = 0
            for c in x:
                if c.isupper():
                    tot += 1
                elif c.islower():
                    tot -= 1
            return tot

        return ext in extensions and case_delta(ext) == max([case_delta(x) for x in extensions])

    @staticmethod
    def sol(class_name, extensions):
        def case_delta(x: str):
            tot = 0
            for c in x:
                if c.isupper():
                    tot += 1
                elif c.islower():
                    tot -= 1
            return tot

        return class_name + "." + max(extensions, key=case_delta)

    def gen_random(self):
        class_name = self.random.pseudo_word()
        class_name = class_name[0].upper() + class_name[1:]

        extensions = [random_case_word(self.random) for _ in range(self.random.randrange(5, 15))]
        self.add(dict(class_name=class_name, extensions=extensions))


class RotateString(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#154

    This puzzle (and RotateSort from #109) use the fact that a string is a rotation of r if it is a substring of r+r
    """

    @staticmethod
    def sat(r: str, s="light star", t="I love to look at the starlight!"):
        """Find a rotation of string s that is a substring of t

        Input Example:
        s="test", t="I love lattes"

        Output Example:
        "ttes"
        """
        return r in t and len(r) == len(s) and r in s + s

    @staticmethod
    def sol(s, t):
        return next(s[i:] + s[:i] for i in range(len(s)) if s[i:] + s[:i] in t)

    def gen_random(self):
        t = " ".join(self.random.pseudo_word() for _ in range(10))
        r = t[self.random.randrange(len(t) - 2):]
        r = r[:self.random.randrange(2, len(r) + 1)]
        i = self.random.randrange(len(r))
        s = r[i:] + r[:i]
        self.add(dict(s=s, t=t))


class EvenOddDigits(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#155"""

    @staticmethod
    def sat(n: int, evens=17, odds=3):
        """Find an integer n >= 0 with the given number of even and odd digits.

        evens=3, odds=4 => 2381695"""
        for c in str(n):
            if int(c) % 2 == 0:
                evens -= 1
            else:
                odds -= 1
        return evens == 0 and odds == 0

    @staticmethod
    def sol(evens, odds):
        return int("2" * evens + "1" * odds)

    def gen_random(self):
        evens = self.random.randrange(150)
        odds = self.random.randrange(150)
        if odds + evens:
            self.add(dict(evens=evens, odds=odds))


class RomanNumerals(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#156
    
    Do not add a reverse puzzle converting roman numerals to arabic numbers as it would give away the solution. 
    """

    @staticmethod
    def sat(roman: str, n=2414):
        """Convert integer 0 < n < 4000 to roman numerals, and make it lowercase

        11 => "xi"
        """
        key = {1000: 'm', 900: 'cm', 500: 'd', 400: 'cd',
               100: 'c', 90: 'xc', 50: 'l', 40: 'xl',
               10: 'x', 9: 'ix', 5: 'v', 4: 'iv',
               1: 'i'}
        m = 0
        for base in [1000, 100, 10, 1]:
            for mul in [9, 4, 5, 1, 1, 1]:  # up to three 1's, move on after 9 or 4
                val = base * mul
                if val in key and roman.startswith(key[val]):
                    m += val
                    roman = roman[len(key[val]):]
                    if mul == 9 or mul == 4:  # 9 or 4 can't be followed by anything else
                        break
        return m == n

    @staticmethod
    def sol(n):
        units = dict(m=1000, cm=900, d=500, cd=400, c=100, xc=90, l=50, xl=40, x=10, ix=9, v=5, iv=4, i=1)
        roman = ""
        for s, i in units.items():
            while n >= i:
                roman += s
                n -= i
        return roman

    def gen_random(self):
        n = self.random.randrange(1, 4000)
        self.add(dict(n=n))


class PythagoreanTriples(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#157"""

    @staticmethod
    def sat(triples: List[List[int]], n=920, m=799):
        """Find m Pythagorean triples a^2 + b^2 == c^2 for integers 0 < a < b < c <= n, in sorted order

        (n=6, m=1) => [[3, 4, 5]]
        """
        for a, b, c in triples:
            if not (a * a + b * b == c * c and 0 < a < b < c <= n):
                return False
        return triples == sorted(triples) and len(triples) >= m

    @staticmethod
    def sol(n, m):
        return [[a, b, int((a * a + b * b) ** 0.5)]
                for a in range(3, int(n / (2 ** 0.5)))
                for b in range(a + 1, int((n * n - a * a) ** 0.5) + 1)
                if ((a * a + b * b) ** 0.5).is_integer()]

    _cache = {}

    def gen_random(self):
        n = self.random.randrange(1, 1000)
        if n not in self._cache:
            self._cache[n] = len(self.sol(n, None))
        m = self._cache[n]
        self.add(dict(n=n, m=m))


class MostUnique(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#158"""

    @staticmethod
    def sat(s: str, pool=["cat", "catatatatctsa", "abcdefhijklmnop", "124259239185125", "", "foo", "unique"]):
        """Select a string from the pool with the most unique characters

        ["woooow", "cow"] => "cow"
        """
        assert s in pool
        n = len(set(s))
        for p in pool:
            assert len(set(p)) <= n
        return True

    @staticmethod
    def sol(pool):
        return max(pool, key=lambda x: len(set(x)))

    def gen_random(self):
        pool = [self.random.pseudo_word(min_len=0) for _ in range(1, self.random.randrange(2, 10))]
        self.add(dict(pool=pool))


class HungryRabbits(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#159"""

    @staticmethod
    def sat(results: List[List[int]], stats=[[2, 3, 18], [4, 9, 2], [2, 5, 7], [3, 8, 12], [4, 9, 106]]):
        """For each triple of eaten, need, stock return a pair of total appetite and remaining

        [[2, 5, 6], [3, 9, 22]] => [[7, 1], [12, 13]]
        """
        assert len(results) == len(stats)
        for (tot, remaining), (eaten, need, stock) in zip(results, stats):
            assert tot - eaten == min(need, stock)
            assert stock < need and remaining == 0 or stock >= need and remaining + need == stock
        return True

    @staticmethod
    def sol(stats):
        results = []
        for (eaten, need, stock) in stats:
            results.append([eaten + min(need, stock), max(0, stock - need)])
        return results

    def gen_random(self):
        stats = [[self.random.randrange(10) for _ in range(3)] for _ in range(self.random.randrange(10))]
        self.add(dict(stats=stats))


class EvaluateOperators(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#160"""

    @staticmethod
    def sat(ops: List[str], target=2021, nums=[4, 6, 2, 1, 1, 3, 9]):
        """Find a permutation of the operators +-*/^% which when inserted between nums evaluates to target

        target=3, nums=[7, 2, 3, 4, 5, 1, 6] => ["+", "*", "**", "%", "//", "-"]
                                                # because 7 + 2 * 3 ** 4 % 5 // 1 - 6 == 3
        """
        assert len(ops) == len(set(ops)) and set(ops) == {"**", "*", "+", "-", "//", "%"}
        expr = str(nums[0])
        for n, op in zip(nums[1:], ops):
            expr += op + str(n)
        return eval(expr) == target

    @staticmethod
    def sol(target, nums):
        from itertools import permutations
        for ops in permutations(["**", "*", "+", "-", "//", "%"]):
            expr = str(nums[0])
            for n, op in zip(nums[1:], ops):
                expr += op + str(n)
            try:
                if eval(expr) == target:
                    return list(ops)
            except (ZeroDivisionError, SyntaxError):
                pass
        assert False

    def gen_random(self):
        ops = ["**", "*", "+", "-", "//", "%"]
        nums = [self.random.randrange(1, 10) for _ in range(len(ops) + 1)]
        self.random.shuffle(ops)
        expr = str(nums[0])
        for n, op in zip(nums[1:], ops):
            expr += op + str(n)
        try:
            target = eval(expr)
        except ZeroDivisionError:
            return
        self.add(dict(target=target, nums=nums))


class ReverseCase(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#161"""

    @staticmethod
    def sat(rev: List[str], strs=["cat", "u8u", "12532", "", "191", "4tUn8", "ewrWQTEW", "i", "IoU"]):
        """Reverse the case of all strings. For those strings which contain no letters, reverse the strings.

        ["Test", "!@#"] => ["tEST", "#@!"]
        """
        assert len(rev) == len(strs)
        return all(r.swapcase() == s != r or r[::-1] == s == s.swapcase() for r, s in zip(rev, strs))

    @staticmethod
    def sol(strs):
        return [s.swapcase() if s.swapcase() != s else s[::-1] for s in strs]

    def gen_random(self):
        strs = [self.random.choice([random_case_word(self.random), str(self.random.randrange(-1000, 1000))])
                for _ in range(self.random.randrange(10))]
        self.add(dict(strs=strs))


class ZobristCollision(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#162"""

    @staticmethod
    def sat(positions: List[List[int]]):
        """Find a collision for the given Zobrist chess board hash: https://en.wikipedia.org/wiki/Zobrist_hashing

        Each of the two positions should be encoded as a list of 64 integers 0-12"""

        table = [[(i * 429436219 + j * 100239120) % 63491564 for j in range(13)] for i in range(64)]

        def zobrist(pos):
            h = 0
            for i in range(64):
                if pos[i]:
                    h ^= table[i][pos[i]]
            return h

        a, b = positions
        return zobrist(a) == zobrist(b) and a != b

    @staticmethod
    def sol():
        hashes = {}
        table = [[(i * 429436219 + j * 100239120) % 63491564 for j in range(13)] for i in range(64)]

        def zobrist(pos):
            h = 0
            for i in range(64):
                if pos[i]:
                    h ^= table[i][pos[i]]
            return h

        for i in range(1, 100000000):
            pos = [(i * 42 + ((i + 1) * j * 12589) % 54321) % 13 for j in range(64)]  # pseudo-random board
            h = zobrist(pos)
            if h in hashes:
                return [pos, hashes[h]]
            else:
                hashes[h] = pos


class EvenBetween(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#163"""

    @staticmethod
    def sat(ab: List[int], s="3298832990329923299432996329983300033002"):
        """Find integers [a, b] that are at least 5 apart and such that concatenating the even numbers
        between them gives the string s

        "32343638" => [31, 38]
        """
        return abs(ab[0] - ab[1]) > 4 and s == "".join(str(i) for i in range(min(ab), max(ab) + 1) if i % 2 == 0)

    @staticmethod
    def sol(s):
        for i in range(1, len(s)):
            n = int(s[:i])
            n -= (n + 1) % 2  # make n odd
            m = n + 1  # next even
            t = ""
            while len(t) < len(s):
                t += str(m)
                m += 2
            if s == t:
                return [n, m - 1]

        assert False

    def gen_random(self):
        a = self.random.randrange(100, 10 ** 5)
        b = a + self.random.randrange(5, 22)
        s = "".join(str(i) for i in range(a, b + 1) if i % 2 == 0)
        self.add(dict(s=s))


if __name__ == "__main__":
    PuzzleGenerator.debug_problems()


