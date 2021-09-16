"""Problems inspired by [HumanEval dataset](https://github.com/openai/human-eval) described
in the [codex paper](https://arxiv.org/abs/2107.03374), specifically,
[this](https://github.com/openai/human-eval/blob/fa06031e684fbe1ee429c7433809460c159b66ad/data/HumanEval.jsonl.gz)
version."""

from puzzle_generator import PuzzleGenerator
from typing import List


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
        assert a in nums and b in nums
        return abs(a - b) == min({abs(x - y) for x in nums for y in nums} - {0})

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
        assert ''.join(ls) == combined.replace(' ', '')
        for s in ls:  # check that s is not further divisible
            depth = 0
            for c in s[:-1]:
                if c == '(':
                    depth += 1
                else:
                    assert c == ')'
                    depth -= 1
                    assert depth >= 1
            assert depth == 1 and s[-1] == ')'
        return True

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
    def sat(n: int, balances=[2, 7, -2, 4, 3, -15, 10, -45, 3]):
        """
        Given a list of numbers which represent bank deposits and withdrawals, find the *first* negative balance.

        Sample Input:
        [12, -5, 3, -99, 14, 88, -99]

        Sample Output:
        -89
        """
        total = 0
        for b in balances:
            total += b
            if total < 0:
                return total == n

    @staticmethod
    def sol(balances):
        total = 0
        for b in balances:
            total += b
            if total < 0:
                return total
        assert False, "should not reach here"

    def gen_random(self):
        length = self.random.randrange(1, 11)
        while True:
            balances = [self.random.randrange(-10 ** 10, 10 ** 10) for _ in range(length)]
            if any(sum(balances[:i + 1]) < 0 for i in range(length)):
                self.add(dict(balances=balances))
                return


class NegCumulative_Trivial(PuzzleGenerator):
    """
    Inspired by [HumanEval](https://github.com/openai/human-eval) \\#3
    (see also FirstNegCumulative above which is not as trivial)
    This version is a more direct translation of the problem but it can of course
    be solved trivially just by trying both neg=True and neg=False
    """

    @staticmethod
    def sat(neg: bool, balances=[2, 7, -2, 4, 3, -15, 10, -45, 3]):
        """
        Given a list of numbers which represent bank deposits and withdrawals,
        determine if the cumulative sum is negative.

        Sample Input:
        [12, -5, 3, -99, 14, 88, -99]

        Sample Output:
        True
        """
        total = 0
        for b in balances:
            total += b
            if total < 0:
                return neg == True
        return neg == False

    @staticmethod
    def sol(balances):
        total = 0
        for b in balances:
            total += b
            if total < 0:
                return True
        return False

    def gen_random(self):
        length = self.random.randrange(1, 11)
        while True:
            balances = [self.random.randrange(-10 ** 10, 10 ** 10) for _ in range(length)]
            if any(sum(balances[:i + 1]) < 0 for i in range(length)):
                self.add(dict(balances=balances))
                return


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
        return sum((n - x) ** 2 for n in nums) <= sum((m - n) ** 2 for m in nums for n in nums) * 0.501 / len(nums)

    @staticmethod
    def sol(nums):
        return sum(nums) / len(nums)  # mean minimizes mean squared deviation

    def gen_random(self):
        length = self.random.randrange(1, 11)
        nums = [self.random.randrange(-10 ** 10, 10 ** 10) for _ in range(length)]
        self.add(dict(nums=nums))


class Intersperse(PuzzleGenerator):
    """
    Inspired by [HumanEval](https://github.com/openai/human-eval) \\#5

    The one-liner version is `li[::2] == nums and li[1::2] == [sep] * (len(li) - 1)`
    """

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
        assert len(li) == max(0, len(nums) * 2 - 1)
        for i, n in enumerate(nums):
            assert li[2 * i] == n
            if i > 0:
                assert li[2 * i - 1] == sep
        return True

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
    def sat(depths: List[int], parens='() (()) ((()()())) (())'):
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


class SumProduct_Trivial(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#8"""

    @staticmethod
    def sat(sum_prod: List[int], nums=[1, 3, 2, -6, 19]):
        """
        Find the sum and product of a list of numbers.

        Sample Input:
        [2, 8, 2]

        Sample Output:
        [12, 32]
        """
        p = 1
        for n in nums:
            p *= n
        return sum_prod == [sum(nums), p]

    @staticmethod
    def sol(nums):
        p = 1
        for n in nums:
            p *= n
        return [sum(nums), p]

    def gen_random(self):
        nums = [self.random.randrange(-100, 100) for _ in range(self.random.randrange(1, 5))]
        self.add(dict(nums=nums))


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


class PalindromeStartingWith(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#10"""

    @staticmethod
    def sat(ans: str, s="so easy", length=13):
        """
        Find a palindrome of a given length starting with a given string.

        Sample Input:
        "foo", 4

        Sample Output:
        "foof"
        """
        return ans == ans[::-1] and len(ans) == length and ans.startswith(s)

    @staticmethod
    def sol(s, length):
        return s[:length // 2] + ' ' * (length - len(s) * 2) + s[:(length + 1) // 2][::-1]

    def gen_random(self):
        part = "".join([self.random.choice("ab") for _ in range(self.random.randrange(20))])
        pal = part + self.random.choice([part, part[:-1]])[::-1]
        n = self.random.randrange(len(pal) + 1)
        s = pal[:n]
        self.add(dict(s=s, length=len(pal)))


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
    def sat(ans: List[int], m=1408862, n=2113293):
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
    def sat(indexes: List[int], li=["Hello", "5", "10", "bye"], num=2):
        """
        Find the indices of valid python integers in a list of strings

        Sample input
        ---
        ["18.5", "-1", "2+2", "7", "foo"]

        Sample output
        ---
        [1, 3]
        """
        [int(li[i]) for i in indexes]
        return len(set(indexes)) >= num and min(indexes) >= 0

    @staticmethod
    def sol(li, num):
        ans = []
        for i in range(len(li)):
            try:
                int(li[i])
                ans.append(i)
            except:
                pass
        return ans

    def gen_random(self):
        chars = "0123456789+-*'e. "
        ans = []
        length = self.random.randrange(10)
        for _ in range(length):
            ans.append("".join(self.random.choice(chars) for i in range(self.random.randrange(10))))


class StrLength(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#23"""

    @staticmethod
    def sat(length: int, s="pneumonoultramicroscopicsilicovolcanoconiosis"):
        """
        Find the length of a non-empty string

        Sample input
        ---
        "foo"

        Sample output
        ---
        3
        """
        try:
            s[length]
        except IndexError:
            s[length - 1]
            return True

    @staticmethod
    def sol(s):
        return len(s)

    def gen_random(self):
        s = self.random.string(min_len=1, max_len=50)
        self.add(dict(s=s))


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


class FermatComposite(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#31"""

    @staticmethod
    def sat(certificate: int, n=1449):
        """
        Find a Fermat composite certificate for a number n > 1

        Sample Input:
        1469

        Sample Output:
        3  # because (3 ** 1468) % 1469 != 1
        """
        return pow(certificate, n - 1, n) > 1

    @staticmethod
    def sol(n):
        return next(i for i in range(2, n) if pow(i, n - 1, n) > 1)

    def gen_random(self):
        a, b = [self.random.randrange(3, 10 ** 5, 2) for _ in range(2)]
        if not self.random.randrange(10):
            a += 1
        n = a * b
        self.add(dict(n=n))


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
    def sat(m: int, hello=[1, 31, 3, 2, 0, 18, 32, -4, 2, -1000, 35, 35, 21, 18, 2, 60]):
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
        while len(self.instances) < target_num_instances:
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
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#40"""

    @staticmethod
    def sat(inds: List[int], nums=[12, -10452, 18242, 10440]):
        """
        Find the indices of three numbers that sum to 0 in a list.
        """
        return len(inds) == 3 and sum(nums[i] for i in inds) == 0 and min(inds) >= 0

    @staticmethod
    def sol(nums):
        assert len(nums) == 4
        n = sum(nums)
        for i in range(4):
            if nums[i] == n:
                return [j for j in range(4) if j != i]

    def gen_random(self):
        nums = [self.random.randrange(-100, 100) for _ in range(3)]
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
    """

    @staticmethod
    def sat(inds: List[int], nums=[12, -10452, 18242, 10440, 81, 241, 525, -18242, 91, 20]):
        """
        Find the indices of two numbers that sum to 0 in a list.
        """
        a, b = inds
        return nums[a] + nums[b] == 0

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


class Palindrome_Trivial(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#48"""

    @staticmethod
    def sat(p: bool, s="This problem is trivial but common"):
        """
        Test whether the given string is a palindrome
        """
        return p == (s == s[::-1])

    @staticmethod
    def sol(s):
        return s == s[::-1]

    def gen_random(self):
        s = self.random.pseudo_word()
        s = self.random.choice([s, s + s[::-1], s[:-1] + s[::-1]])
        self.add(dict(s=s))


class LittleFermat(PuzzleGenerator):
    """Harder but loosely inspired by [HumanEval](https://github.com/openai/human-eval) \\#49"""

    @staticmethod
    def sat(exp_poly: List[int], d=74152093423, poly=[1, 6, 3, 1, 0, 4, 4]):
        """
        Fermat's little theorem implies that any polynomial can be written equivalently as a degree p-1
        polynomial (mod p).
        Given the p coefficients of a polynomial poly, compute a polynomial equivalent to poly^d (mod p).
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
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#51"""

    @staticmethod
    def sat(txt: str, text="Hello, world!"):
        """
        Remove the vowels from the original string.
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
        Find the indexes of numbers below a given threshold
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
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#62"""

    @staticmethod
    def sat(derivative: List[int], poly=[2, 1, 0, 4, 19, 231, 0, 5]):
        """
        Find the derivative of the given polynomial, with coefficients in order of increasing degree
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
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#64"""

    @staticmethod
    def sat(vowels: str, text="Hello, world!"):
        """
        Find the vowels from the original string.
        """
        i = 0
        for j, c in enumerate(text):
            if c.lower() in "aeiou" or c.lower() == 'y' and j == len(text) - 1:
                assert vowels[i] == c
                i += 1
        return i == len(vowels)

    @staticmethod
    def sol(text):
        return "".join(c for c in text if c.lower() in "aeiou") + (text[-1] if text[-1].lower() == "y" else "")

    def gen_random(self):
        text = random_case_word(self.random)
        self.add(dict(text=text))


class CircularShiftNum(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#65"""

    @staticmethod
    def sat(shifted: str, n=124582369835, shift=3):
        """
        Shift the decimal digits n places to the left, wrapping the extra digits around. If shift > the number of
        digits of n, reverse the string.
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


class DigitSum(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#66"""

    @staticmethod
    def sat(tot: int, s="Add ME uP AND YOU WILL GET A BIG NUMBER!"):
        """
        Compute the sum of the ASCII values of the upper-case characters in the string.
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
        """
        if val_index == []:
            return all(n % 2 == 1 for n in nums)
        v, i = val_index
        assert v % 2 == 0
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


class StrangeSort(PuzzleGenerator):
    """Inspired by [HumanEval](https://github.com/openai/human-eval) \\#70"""

    @staticmethod
    def sat(strange: List[int], li=[30, 12, 42, 717, 45, 317, 200, -1, 491, 32, 15]):
        """
        Find the following strange sort of li: the first element is the smallest, the second is the largest of the
        remaining, the third is the smallest of the remaining, the fourth is the smallest of the remaining, etc.
        """
        if len(li) < 2:
            return strange == li
        bounds = strange[:2]  # lower, upper
        for i, n in enumerate(strange):
            assert bounds[0] <= n <= bounds[1]
            bounds[i % 2] = n
        return sorted(strange) == sorted(li)  # permutation check

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
        """Find an integer exponent x such that a^x = n"""
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
        """Find an integer that when cubed is n"""
        return x ** 3 == n

    @staticmethod
    def sol(n):  # Using Newton's method
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
        """Determine which characters of a hexidecimal correspond to prime numbers"""
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
        """Write n base 2 followed and preceded by 'bits'"""
        assert b[:4] == b[-4:] == 'bits'
        inside = b[4:-4]
        assert all(c in "01" for c in inside)
        assert inside[0] == "1" or len(inside) == 1
        m = 0
        for c in inside:
            m = 2*m + int(c)
        return m == n

    @staticmethod
    def sol(n):
        s = bin(n)[2:]
        return f'bits{s}bits'

    def gen_random(self):
        digits = self.random.randrange(30)
        n = self.random.randrange(10 ** digits)
        self.add(dict(n=n))



if __name__ == "__main__":
    PuzzleGenerator.debug_problems()
