"""Problems inspired by [HumanEval dataset](https://github.com/openai/human-eval) described
in the [codex paper](https://arxiv.org/abs/2107.03374), specifically,
[this](https://github.com/openai/human-eval/blob/fa06031e684fbe1ee429c7433809460c159b66ad/data/HumanEval.jsonl.gz)
version."""

from problems import Problem
from typing import List


# Hint: subclass Problem.Debug for quick testing. Run make_dataset.py to make the dataset
# See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle for more info


class FindCloseElements(Problem):
    """
    Given a list of numbers and a threshold, find two distinct numbers in the list that
    are closer than the given threshold.

    Sample Input:
    [1.2, 5.23, 0.89, 21.0, 5.28], 0.1

    Sample Output:
    [5.23, 5.28]

    Inspired by [HumanEval](https://github.com/openai/human-eval)/0
    """

    @staticmethod
    def sat(pair: List[float], nums=[0.17, 21.3, 5.0, 9.0, 11.0, 4.99, 17.0], thresh=0.1):
        a, b = pair
        return a in nums and b in nums and 0 < abs(a - b) < thresh

    @staticmethod
    def sol(nums, thresh):
        s = sorted(set(nums))
        return min([[a, b] for a, b in zip(s, s[1:])], key=lambda x: x[1] - x[0])

    def gen_random(self):
        nums = [self.random.uniform(-10, 10) for _ in range(self.random.randrange(2, 10))]
        thresh = self.random.uniform(0.01, 1.0)
        nums.append(self.random.choice(nums) + self.random.uniform(-thresh, thresh))
        self.random.shuffle(nums)
        self.add(dict(nums=nums, thresh=thresh))


class SeparateParenGroups(Problem):
    """
    Given a string consisting of whitespace and groups of matched parentheses, split it
    into groups of perfectly matched parentheses without any whitespace.

    Sample Input:
    '( ()) ((()()())) (()) ()'

    Sample Output:
    ['(())', '((()()()))', '(())', '()']

    Inspired by [HumanEval](https://github.com/openai/human-eval)/1
    """

    @staticmethod
    def sat(ls: List[str], combined='() (()) ((() () ())) (() )'):
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


class Frac(Problem):
    """
    Given a floating point number, find its fractional part.

    Sample Input:
    4.175

    Sample Output:
    0.175

    Inspired by [HumanEval](https://github.com/openai/human-eval)/2
    """

    @staticmethod
    def sat(x: float, v=523.12892):
        return 0 <= x < 1 and (v - x).is_integer()

    @staticmethod
    def sol(v):
        return v % 1.0

    def gen_random(self):
        v = self.random.uniform(-100, 100)
        self.add(dict(v=v))


class FirstNegCumulative(Problem):
    """
    Given a list of numbers which represent bank deposits and withdrawals, find the *first* negative balance.

    Sample Input:
    [12, -5, 3, -99, 14, 88, -99]

    Sample Output:
    -89

    Inspired by [HumanEval](https://github.com/openai/human-eval)/3
    """

    @staticmethod
    def sat(n: int, balances=[2, 7, -2, 4, 3, -15, 10, -45, 3]):
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


# The following is a more direct translation of the problem but it can of course
# be solved trivially just by trying both neg=True and neg=False
class NegCumulative_Trivial(Problem):
    """
    Given a list of numbers which represent bank deposits and withdrawals,
    determine if the cumulative sum is negative.

    Sample Input:
    [12, -5, 3, -99, 14, 88, -99]

    Sample Output:
    True

    Inspired by [HumanEval](https://github.com/openai/human-eval)/3
    (see also FirstNegCumulative above which is not as trivial)
    """

    @staticmethod
    def sat(neg: bool, balances=[2, 7, -2, 4, 3, -15, 10, -45, 3]):
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


class MinRMS(Problem):
    """
    Given a list of numbers, find x whose root mean squared deviation is less than a given threshold.

    Sample Input:
    [4, -5, 17, -9, 14, 108, -9], 38.5

    Sample Output:
    17.14285

    Inspired by [HumanEval](https://github.com/openai/human-eval)/4
    """

    @staticmethod
    def sat(x: float, nums=[12, -2, 14, 3, -15, 10, -45, 3, 30], thresh=20.003):
        total = 0.0
        for n in nums:
            total += (n - x) ** 2
        return (total / len(nums)) ** 0.5 <= thresh

    @staticmethod
    def sol(nums, thresh):
        return sum(nums) / len(nums)  # mean minimizes RMS deviation

    def gen_random(self):
        length = self.random.randrange(1, 11)
        nums = [self.random.randrange(-10 ** 10, 10 ** 10) for _ in range(length)]
        mean = sum(nums) / len(nums)
        rms = (sum((i - mean) ** 2 for i in nums) / len(nums)) ** 0.5
        thresh = rms + 10.0 ** (self.random.randrange(-6, 3))
        self.add(dict(nums=nums, thresh=thresh))


class Intersperse(Problem):
    """
    Given a list of numbers and a number to inject, create a list containing that number in between each pair of
    adjacent numbers.

    Sample Input:
    [8, 14, 21, 17, 9, -5], 3

    Sample Output:
    [8, 3, 14, 3, 21, 3, 17, 3, 9, 3, -5]

    Inspired by [HumanEval](https://github.com/openai/human-eval)/5
    """

    @staticmethod
    def sat(li: List[int], nums=[12, 23, -2, 5, 0], sep=4):
        for i, n in enumerate(nums):
            assert li[2 * i] == n
            if i > 0:
                assert li[2 * i - 1] == sep
        return len(li) == max(0, len(nums) * 2 - 1)

    #  one-liner
    #
    # def sat(li: List[int], nums=[12, 23, -2, 5, 0], sep=4):
    #     return li[::2] == nums and set(li[1::2]) <= {sep}

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


class DeepestParens(Problem):
    """
    Given a string consisting of groups of matched nested parentheses separated by parentheses,
    compute the depth of each group.

    Sample Input:
    '(()) ((()()())) (()) ()'

    Sample Output:
    [2, 3, 2, 1]

    Inspired by [HumanEval](https://github.com/openai/human-eval)/6
    """

    @staticmethod
    def sat(depths: List[int], parens='() (()) ((()()())) (())'):
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


class FindContainers(Problem):
    """
    Find the strings in a list containing a given substring

    Sample Input:
    ['cat', 'dog', 'bear'], 'a'

    Sample Output:
    ['cat', 'bear']

    Inspired by [HumanEval](https://github.com/openai/human-eval)/7
    """

    @staticmethod
    def sat(containers: List[str], strings=['cat', 'dog', 'shatter', 'bear', 'at', 'ta'], substring='at'):
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


class SumProduct(Problem):
    """
    Find a list of numbers with a given sum and a given product.

    Sample Input:
    12, 32

    Sample Output:
    [2, 8, 2]

    Inspired by [HumanEval](https://github.com/openai/human-eval)/8
    """

    @staticmethod
    def sat(nums: List[int], tot=14, prod=99):
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


class SumProduct_Trivial(Problem):
    """
     Find the sum and product of a list of numbers.

     Sample Input:
     [2, 8, 2]

     Sample Output:
     [12, 32]

     Inspired by [HumanEval](https://github.com/openai/human-eval)/8
     """

    @staticmethod
    def sat(sum_prod: List[int], nums=[1, 3, 2, -6, 19]):
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


class RollingMax(Problem):
    """
     Find a list whose ith element is the maximum of the first i elements of the input list.

     Sample Input:
     [2, 8, 2]

     Sample Output:
     [2, 8, 8]

     Inspired by [HumanEval](https://github.com/openai/human-eval)/9
     """

    @staticmethod
    def sat(maxes: List[int], nums=[1, 4, 3, -6, 19]):
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


class PalindromeStartingWith(Problem):
    """
    Find a palindrome of a given length starting with a given string.

    Sample Input:
    "foo", 4

    Sample Output:
    "foof"

    Inspired by [HumanEval](https://github.com/openai/human-eval)/10
    """

    @staticmethod
    def sat(ans: str, s="so easy", length=20):
        return ans == ans[::-1] and len(ans) == length and ans.startswith(s)

    @staticmethod
    def sol(s, length):
        if length > len(s) * 2:
            return s + 'a' * (length - len(s) * 2) + s[::-1]
        if length % 2 == 0:
            return s[:length // 2] + s[:length // 2][::-1]
        else:
            return s[:length // 2] + s[:length // 2 + 1][::-1]

    def gen_random(self):
        part = "".join([self.random.choice("ab") for _ in range(self.random.randrange(20))])
        pal = part + self.random.choice([part, part[:-1]])[::-1]
        n = self.random.randrange(len(pal) + 1)
        s = pal[:n]
        self.add(dict(s=s, length=len(pal)))


class PalindromeContaining(Problem):
    """
    Find a palindrome of a given length containing a given string.

    Sample Input:
    "abba", 6

    Sample Output:
    "cabbac"

    Inspired by [HumanEval](https://github.com/openai/human-eval)/10
    """

    @staticmethod
    def sat(ans: str, s="so easy", length=20):
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


class BinaryStrXOR(Problem):
    """
    Find a the XOR of two given strings interpreted as binary numbers.

    Sample Input:
    "0001", "1011"

    Sample Output:
    "1010"

    Inspired by [HumanEval](https://github.com/openai/human-eval)/11
    """

    @staticmethod
    def sat(str_num: str, nums=["100011101100001", "100101100101110"]):
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
class LongestStr(Problem):
    """
    Find a the longest of a list of strings

    Sample Input:
    ["cat", "dog", "sheep", "chimp"]

    Sample Output:
    "sheep"

    Inspired by [HumanEval](https://github.com/openai/human-eval)/12
    """

    @staticmethod
    def sat(ans: str, words=["these", "are", "some", "pretty", "long", "words"]):
        return ans in words and all(len(ans) >= len(w) for w in words)

    @staticmethod
    def sol(words):
        return max(words, key=len)

    def gen_random(self):
        words = [self.random.pseudo_word() for _ in range(self.random.randrange(1, 10))]
        self.add(dict(words=words))


class CertifiedGCD(Problem):
    """
    Find the greatest common divisor of two integers m, n and a certificate a, b such that m*a + n*b = gcd

    Sample Input:
    20, 30

    Sample Output:
    10, -1, 1

    Inspired by [HumanEval](https://github.com/openai/human-eval)/13
    """

    @staticmethod
    def sat(ans: List[int], m=1408862, n=2113293):
        gcd, a, b = ans
        return m % gcd == n % gcd == 0 and a * m + b * n == gcd and gcd > 0

    # Derivation of solution below
    # Recursive solution guarantees a * (big % small) + b * small == gcd
    # Let d = big // small so (big % small) == big - small * d
    # gives a * (big - small * d) + b * small == gcd
    # or equivalently (b - a * d) * small + a * big == gcd

    @staticmethod
    def sol(m, n):
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


class AllPrefixes(Problem):
    """
    Find all prefixes of a given string

    Sample Input:
    "aabcd"

    Sample Output:
    ["", "a", "aa", "aab", "aabc", "aabcd"]

    Inspired by [HumanEval](https://github.com/openai/human-eval)/14
    """

    @staticmethod
    def sat(prefixes: List[str], s="donesezichethofalij"):
        return all(s.startswith(p) for p in prefixes) and len(set(prefixes)) > len(s)

    @staticmethod
    def sol(s):
        return [s[:i] for i in range(len(s) + 1)]

    def gen_random(self):
        s = self.random.pseudo_word(min_len=0, max_len=30)
        self.add(dict(s=s))


class SpaceyRange(Problem):
    """
    Find a string consisting of the non-negative integers up to n inclusive

    Sample Input:
    4

    Sample Output:
    '0 1 2 3 4'

    Inspired by [HumanEval](https://github.com/openai/human-eval)/15
    """

    @staticmethod
    def sat(ans: str, n=15):
        return [int(i) for i in ans.split(' ')] == list(range(n + 1))

    @staticmethod
    def sol(n):
        return ' '.join(str(i) for i in range(n + 1))

    def gen_random(self):
        n = self.random.randrange(10 ** 5)
        self.add(dict(n=n))


class DistinctChars(Problem):
    """
    Find the set of distinct characters in a string, ignoring case

    Sample Input:
    'HELlo', 4

    Sample Output:
    ['h', 'e', 'l', 'o']

    Inspired by [HumanEval](https://github.com/openai/human-eval)/16
    """

    @staticmethod
    def sat(ans: List[str], s='The quick brown fox jumps over the lazy dog!', n=28):
        return all(c.lower() in ans for c in s) and len(ans) <= 28

    @staticmethod
    def sol(s, n):
        return list(set(s.lower()))

    def gen_random(self):
        s = self.random.pseudo_word()
        s = s[0].upper() + s[1:]
        n = len(set(s.lower()))
        self.add(dict(s=s, n=n))


class ParseMusic(Problem):
    """
    Parse a string of notes to beats, 'o'=4, 'o|'=2, '.|'=1

    Example input:
    'o o .| o|'

    Example output:
    [4, 4, 1, 2]

    Inspired by [HumanEval](https://github.com/openai/human-eval)/17
    """

    @staticmethod
    def sat(beats: List[int], score="o o o| o| .| .| .| o| o| o o o| .|"):
        return " ".join({1: '.|', 2: 'o|', 4: 'o'}[b] for b in beats) == score

    @staticmethod
    def sol(score):
        mapping = {'.|': 1, 'o|': 2, 'o': 4}
        return [mapping[note] for note in score.split()]

    def gen_random(self):
        n = self.random.randrange(12)
        score = ' '.join(self.random.choice(['.|', 'o|', 'o']) for _ in range(n))
        self.add(dict(score=score))


class OverlappingCount(Problem):
    """
    Find occurrences of a substring in a parent string *including overlaps*

    Sample Input:
    'helllo', 'll'

    Sample Output:
    [2, 3]

    Inspired by [HumanEval](https://github.com/openai/human-eval)/18
    """

    @staticmethod
    def sat(ans: List[int], s='Bananannanaannanaanananananana', sub='anan', count=7):
        return all(sub == s[i:i + len(sub)] for i in ans) and len(set(ans)) >= count

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


class SortNumbers(Problem):
    """
    Sort numbers based on strings

    Sample input
    ---
    "six one four"

    Sample output
    ---
    "one four six"

    Inspired by [HumanEval](https://github.com/openai/human-eval)/19
    """

    @staticmethod
    def sat(ans: str, s="six one four"):
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


class FindClosePair(Problem):
    """
    Given a list of numbers, find the indices of the closest pair.

    Sample Input:
    [1.2, 5.25, 0.89, 21.0, 5.23]

    Sample Output:
    [4, 1]

    Inspired by [HumanEval](https://github.com/openai/human-eval)/20
    """

    @staticmethod
    def sat(inds: List[int], nums=[0.31, 21.3, 5.0, 9.0, 11.0, 5.01, 17.2]):
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


class Rescale(Problem):
    """
    Rescale and shift numbers so that they cover the range [0, 1]

    Sample input
    ---
    [18.5, 17.0, 18.0, 19.0, 18.0]

    Sample output
    ---
    [0.75, 0.0, 0.5, 1.0, 0.5]

    Inspired by [HumanEval](https://github.com/openai/human-eval)/21
    """

    @staticmethod
    def sat(ans: List[float], nums=[13.0, 17.0, 17.0, 15.5, 2.94]):
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


class FilterInts(Problem):
    """
    Find the indices of valid python integers in a list of strings

    Sample input
    ---
    ["18.5", "-1", "2+2", "7", "foo"]

    Sample output
    ---
    [1, 3]

    Inspired by [HumanEval](https://github.com/openai/human-eval)/22
    """

    @staticmethod
    def sat(indexes: List[int], li=["Hello", "5", "10", "bye"], num=2):
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


class StrLength(Problem):
    """
    Find the length of a non-empty string

    Sample input
    ---
    "foo"

    Sample output
    ---
    3

    Inspired by [HumanEval](https://github.com/openai/human-eval)/23
    """

    @staticmethod
    def sat(length: int, s="pneumonoultramicroscopicsilicovolcanoconiosis"):
        try:
            s[length]
        except IndexError:
            s[length - 1]
            return True

    @staticmethod
    def sol(s):
        return len(s)

    def gen_random(self):
        s = self.random.pseudo_word(min_len=1, max_len=50)
        self.add(dict(s=s))


class LargestDivisor(Problem):
    """
    Find the largest integer divisor of a number n that is less than n

    Sample input
    ---
    1000

    Sample output
    ---
    500

    Inspired by [HumanEval](https://github.com/openai/human-eval)/24
    """

    @staticmethod
    def sat(d: int, n=123456):
        return n % d == 0 and d < n and all(n % e for e in range(d + 1, n))

    @staticmethod
    def sol(n):
        return next(d for d in range(n - 1, 0, -1) if n % d == 0)

    def gen_random(self):
        n = self.random.randrange(1, 10 ** 5)
        self.add(dict(n=n))


class PrimeFactorization(Problem):
    """
    Factor number n into a given number of non-trivial factors

    Sample input
    ---
    1000, 6

    Sample output
    ---
    [2, 2, 2, 5, 5, 5]

    Inspired by [HumanEval](https://github.com/openai/human-eval)/25
    """

    @staticmethod
    def sat(factors: List[int], n=123456, num_factors=8):
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


class Dedup(Problem):
    """
    Remove duplicates from a list of integers, preserving order

    Sample input
    ---
    [1, 3, 2, 9, 2, 1, 55]

    Sample output
    ---
    [1, 3, 2, 9, 55]

    Inspired by [HumanEval](https://github.com/openai/human-eval)/26
    """

    @staticmethod
    def sat(ans: List[int], li=[2, 19, 2, 53, 1, 1, 2, 44, 17, 0, 19, 31]):
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


class FlipCase(Problem):
    """
    Flip case

    Sample input
    ---
    'cAt'

    Sample output
    ---
    'CaT'

    Inspired by [HumanEval](https://github.com/openai/human-eval)/27
    """

    @staticmethod
    def sat(ans: str, s="FlIp ME!"):
        return len(ans) == len(s) and all({c, d} == {d.upper(), d.lower()} for c, d in zip(ans, s))

    @staticmethod
    def sol(s):
        return "".join(c.lower() if c.upper() == c else c.upper() for c in s)

    def gen_random(self):
        w = self.random.pseudo_word()
        s = "".join(self.random.choice([c.upper(), c.lower()] * 5 + [' ', '!', '3']) for c in w)
        self.add(dict(s=s))


class CatStrings(Problem):
    """
    Concatenate a list of strings

    Sample input
    ---
    ['cat', 'dog', 'bird']

    Sample output
    ---
    'catdogbird'

    Inspired by [HumanEval](https://github.com/openai/human-eval)/28
    """

    @staticmethod
    def sat(cat: str, strings=["Will", "i", "am", "Now", "here"]):
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


class FindExtensions(Problem):
    """
    Find the strings in a list startings with a given prefix

    Sample Input:
    ['cat', 'car', 'fear', 'center'], 'ca'

    Sample Output:
    ['cat', 'car']

    Inspired by [HumanEval](https://github.com/openai/human-eval)/29
    """

    @staticmethod
    def sat(extensions: List[str], strings=['cat', 'dog', 'shatter', 'donut', 'at', 'ta'], prefix='do'):
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


class FindPositives(Problem):
    """
    Find the positive integers in a list

    Sample Input:
    [-1, 3, 19, -2, 0, 44, 0, 44, 11]

    Sample Output:
    [3, 19, 44, 44, 11]

    Inspired by [HumanEval](https://github.com/openai/human-eval)/30
    """

    @staticmethod
    def sat(positives: List[int], nums=[2, 2342, -2, 32, -8, -5, 2342, 0, -9, 44, 11]):
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


class FermatComposite(Problem):
    """
    Find a Fermat composite certificate for a number n > 1

    Sample Input:
    1469

    Sample Output:
    3  # because (3 ** 1468) % 1469 != 1

    Inspired by [HumanEval](https://github.com/openai/human-eval)/31
    """

    @staticmethod
    def sat(certificate: int, n=1449):
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


class OddDegreePolynomialRoot(Problem):
    """
    Find a real root of an odd degree polynomial from its coefficeints

    Sample Input:
    [1, 0, 8]

    Sample Output:
    -2.0  # 1*(-2.0)^3 + 8 == 0

    Inspired by [HumanEval](https://github.com/openai/human-eval)/32
    """

    @staticmethod
    def sat(root: float, coeffs=[1, 2, 3, 17]):
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
class TwoThirdsSorted(Problem):
    """
    Start with a list of integers, keep every third element in place and otherwise sort the list

    Sample Input:
    [8, 0, 7, 2, 9, 4, 1, 2, 8, 3]

    Sample Output:
    [8, 0, 2, 2, 4, 8, 1, 8, 9, 3]

    Inspired by [HumanEval](https://github.com/openai/human-eval)/33
    """

    @staticmethod
    def sat(li: List[int], orig=[1, -2, 3, 17, 8, 4, 12, 3, 18, 5, -29, 0, 0]):
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


class UniqueSorted(Problem):
    """
    Find an increasing sequence which contains all the elements of the original list.

    Sample Input:
    [8, 0, 7, 2, 9, 4, 4, -2, 8, 3]

    Sample Output:
    [-2, 0, 2, 3, 4, 7, 8, 9]

    Inspired by [HumanEval](https://github.com/openai/human-eval)/34
    """

    @staticmethod
    def sat(li: List[int], orig=[1, 1, 3, 2, 0, 8, 32, -4, 0]):
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


class MaxInt(Problem):
    """
    Find the largest integer in a sequence

    Sample Input:
    [8, 0, 1, 4, 9, 3, 4, -2, 8, 3]

    Sample Output:
    9

    Inspired by [HumanEval](https://github.com/openai/human-eval)/35
    """

    @staticmethod
    def sat(m: int, hello=[1, 32, 3, 2, 0, 18, 32, -4, 0]):
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


class SevenElevenThirteen(Problem):
    """
    Find all 7's in integers less than n that are divisible by 11 or 13

    Sample Input:
    79

    Sample Output:
    [[77, 0], [77, 1], [78, 0]]

    Inspired by [HumanEval](https://github.com/openai/human-eval)/36
    """

    @staticmethod
    def sat(li: List[List[int]], n=19723, lower=3):
        assert len({(i, j) for i, j in li}) >= lower, "not enough 7's (ignoring duplicates)"
        return all(str(i)[j] == '7' and (i % 11 == 0 or i % 13 == 0) and 0 <= i < n and 0 <= j for i, j in li)

    @staticmethod
    def sol(n, lower):
        return [[i, j] for i in range(n) if (i % 11 == 0 or i % 13 == 0) for j in range(len(str(i))) if
                str(i)[j] == '7']

    def gen(self, target_num_instances):
        lower = 0
        n = 0
        while len(self.instances) < target_num_instances:
            if n % 11 == 0 or n % 13 == 0:
                lower += str(n).count('7')
            n += 1
            if self.random.randrange(10) == 0:
                self.add(dict(n=n, lower=lower))


if __name__ == "__main__":
    Problem.debug_problems()
