"""Problems inspired by [HumanEval dataset](https://github.com/openai/human-eval) described
in the [codex paper](https://arxiv.org/abs/2107.03374)."""

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
            a = length - i-1
            b = length - (i + len(s))-1
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


# # This problem is quite different, only loosely inspired
# class MD5(Problem):
#     """
#     Find a string whose md5 hash ends in a certain 3 characters
#
#     Sample Input:
#     'c62'
#
#     Sample Output:
#     'Hello World'
#
#     Inspired by [HumanEval](https://github.com/openai/human-eval)/162
#     """
#
#     @staticmethod
#     def sat(s: str, end='82'):
#         import hashlib
#         return hashlib.md5(s.encode('ascii')).hexdigest().endswith(end)
#
#     @staticmethod
#     def sol(end):
#         import hashlib
#         for i in range(10**6):
#             if hashlib.md5(str(i).encode('ascii')).hexdigest().endswith(end):
#                 return str(i)
#         assert False
#
#     def gen_random(self):
#         import hashlib
#         s = self.random.pseudo_word()
#         n = self.random.choice([2,3,4])
#         end = hashlib.md5(s.encode('ascii')).hexdigest()[-n:]
#         self.add(dict(end=end))


if __name__ == "__main__":
    Problem.debug_problems()
