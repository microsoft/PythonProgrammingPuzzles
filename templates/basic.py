"""Problems testing basic knowledge -- easy to solve if you understand what is being asked"""

from problems import Problem
from typing import List


# Hint: subclass Problem.Debug for quick testing. Run make_dataset.py to make the dataset
# See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle for more info

class SumOfDigits(Problem):
    """Find a number that its digits sum to a specific value."""

    @staticmethod
    def sat(x: str, s=679):
        return s == sum([int(d) for d in x])

    @staticmethod
    def sol(s):
        return int(s / 9) * '9' + str(s % 9)

    def gen_random(self):
        s = self.random.randint(0, 10 ** 5)
        self.add(dict(s=s))


class FloatWithDecimalValue(Problem):
    """Create a float with a specific decimal."""

    @staticmethod
    def sat(z: float, v=9, d=0.0001):
        return int(z * 1 / d % 10) == v

    @staticmethod
    def sol(v, d):
        return v * d

    def gen_random(self):
        v = self.random.randint(0, 9)
        a = self.random.randint(-10 ** 2, 10 ** 2)
        while a == 0:
            a = self.random.randint(-10 ** 2, 10 ** 2)
        d = float(10 ** a)
        if not float((v * d) * 1 / d % 10) == v:
            # Some values won't be solved by the reference solution due to Python floats.
            return
        self.add(dict(v=v, d=d))


class ArithmeticSequence(Problem):
    """Create a list that is a subrange of an arithmetic sequence."""

    @staticmethod
    def sat(x: List[int], a=7, s=5, e=200):
        return x[0] == a and x[-1] <= e and (x[-1] + s > e) and all([x[i] + s == x[i + 1] for i in range(len(x) - 1)])

    @staticmethod
    def sol(a, s, e):
        return list(range(a, e + 1, s))

    def gen_random(self):
        a = self.random.randint(-10 ** 5, 10 ** 5)
        e = self.random.randint(a, 10 ** 6)
        s = self.random.randint(1, 10 ** 4)
        self.add(dict(a=a, e=e, s=s))


class GeometricSequence(Problem):
    """Create a list that is a subrange of an gemoetric sequence."""

    @staticmethod
    def sat(x: List[int], a=8, r=2, l=50):
        return x[0] == a and len(x) == l and all([x[i] * r == x[i + 1] for i in range(len(x) - 1)])

    @staticmethod
    def sol(a, r, l):
        return [a * r ** i for i in range(l)]

    def gen_random(self):
        a = self.random.randint(-10 ** 3, 10 ** 3)
        r = self.random.randint(1, 10 ** 1)
        l = self.random.randint(1, 10 ** 3)
        self.add(dict(a=a, r=r, l=l))


class LineIntersection(Problem):
    """Find the intersection of two lines.
       Solution should be a list of the (x,y) coordinates.
       Accuracy of fifth decimal digit is required."""

    @staticmethod
    def sat(e: List[int], a=2, b=-1, c=1, d=2021):
        x = e[0] / e[1]
        return abs(a * x + b - c * x - d) < 10 ** -5

    @staticmethod
    def sol(a, b, c, d):
        return [d - b, a - c]

    def gen_random(self):
        a = self.random.randint(-10 ** 8, 10 ** 8)
        b = self.random.randint(-10 ** 8, 10 ** 8)
        c = a
        while c == a:
            c = self.random.randint(-10 ** 8, 10 ** 8)
        d = self.random.randint(-10 ** 8, 10 ** 8)
        self.add(dict(a=a, b=b, c=c, d=d))


class IfProblem(Problem):
    """Simple if statement"""

    @staticmethod
    def sat(x: int, a=324554, b=1345345):
        if a < 50:
            return x + a == b
        else:
            return x - 2 * a == b

    @staticmethod
    def sol(a, b):
        if a < 50:
            return b - a
        else:
            return b + 2 * a

    def gen_random(self):
        a = self.random.randint(0, 100)
        b = self.random.randint(-10 ** 8, 10 ** 8)
        self.add(dict(a=a, b=b))


class IfProblemWithAnd(Problem):
    """Simple if statement with and clause"""

    @staticmethod
    def sat(x: int, a=9384594, b=1343663):
        if x > 0 and a > 50:
            return x - a == b
        else:
            return x + a == b

    @staticmethod
    def sol(a, b):
        if a > 50 and b > a:
            return b + a
        else:
            return b - a

    def gen_random(self):
        a = self.random.randint(0, 100)
        b = self.random.randint(0, 10 ** 8)
        self.add(dict(a=a, b=b))


class IfProblemWithOr(Problem):
    """Simple if statement with or clause"""

    @staticmethod
    def sat(x: int, a=253532, b=1230200):
        if x > 0 or a > 50:
            return x - a == b
        else:
            return x + a == b

    @staticmethod
    def sol(a, b):
        if a > 50 or b > a:
            return b + a
        else:
            return b - a

    def gen_random(self):
        a = self.random.randint(0, 100)
        b = self.random.randint(-10 ** 8, 10 ** 8)
        self.add(dict(a=a, b=b))


class IfCases(Problem):
    """Simple if statement with multiple cases"""

    @staticmethod
    def sat(x: int, a=4, b=54368639):
        if a == 1:
            return x % 2 == 0
        elif a == -1:
            return x % 2 == 1
        else:
            return x + a == b

    @staticmethod
    def sol(a, b):
        if a == 1:
            x = 0
        elif a == -1:
            x = 1
        else:
            x = b - a
        return x

    def gen_random(self):
        a = self.random.randint(-5, 5)
        b = self.random.randint(-10 ** 8, 10 ** 8)
        self.add(dict(a=a, b=b))


class ListPosSum(Problem):
    """Construct a list of non-negative integers that sum up to some value"""

    @staticmethod
    def sat(x: List[int], n=5, s=19):
        return len(x) == n and sum(x) == s and all([a > 0 for a in x])

    @staticmethod
    def sol(n, s):
        x = [1] * n
        x[0] = s - n + 1
        return x

    def gen_random(self):
        n = self.random.randint(1, 10 ** 4)
        s = self.random.randint(n, 10 ** 8)
        self.add(dict(n=n, s=s))


class ListDistinctSum(Problem):
    """Construct a list of distinct integers that sum up to some value"""

    @staticmethod
    def sat(x: List[int], n=4, s=2021):
        return len(x) == n and sum(x) == s and len(set(x)) == n

    @staticmethod
    def sol(n, s):
        a = 1
        x = []
        while len(x) < n - 1:
            x.append(a)
            a = -a
            if a in x:
                a += 1

        if s - sum(x) in x:
            x = [i for i in range(n - 1)]

        x = x + [s - sum(x)]
        return x

    def gen_random(self):
        n = self.random.randint(1, 10 ** 3)
        s = self.random.randint(n + 1, 10 ** 8)
        self.add(dict(n=n, s=s))


class ConcatStrings(Problem):
    """Concatenate list of characters"""

    @staticmethod
    def sat(x: str, s=["a", "b", "c", "d", "e", "f"], n=4):
        return len(x) == n and all([x[i] == s[i] for i in range(n)])

    @staticmethod
    def sol(s, n):
        return ''.join([s[i] for i in range(n)])

    def gen_random(self):
        n = self.random.randint(0, 25)
        extra = self.random.randint(0, 25)
        s = [self.random.char() for _ in range(n + extra)]
        self.add(dict(n=n, s=s))


class SublistSum(Problem):
    """Sum values of sublist by range specifications"""

    @staticmethod
    def sat(x: List[int], t=677, a=43, e=125, s=10):
        non_zero = [z for z in x if z != 0]
        return t == sum([x[i] for i in range(a, e, s)]) and len(set(non_zero)) == len(non_zero) and all(
            [x[i] != 0 for i in range(a, e, s)])

    @staticmethod
    def sol(t, a, e, s):
        x = [0] * e
        for i in range(a, e, s):
            x[i] = i
        correction = t - sum(x) + x[i]
        if correction in x:
            x[correction] = -1 * correction
            x[i] = 3 * correction
        else:
            x[i] = correction
        return x

    def gen_random(self):
        t = self.random.randint(1, 10 ** 8)
        a = self.random.randint(1, 100)
        e = self.random.randint(a, 10 ** 4)
        s = self.random.randint(1, 10)
        self.add(dict(t=t, a=a, e=e, s=s))


class CumulativeSum(Problem):
    """Number of values with cumulative sum less than target"""

    @staticmethod
    def sat(x: List[int], t=50, n=10):
        assert all([v > 0 for v in x])
        s = 0
        i = 0
        for v in sorted(x):
            s += v
            if s > t:
                return i == n
            i += 1
        return i == n

    @staticmethod
    def sol(t, n):
        return [1] * n + [t]

    def gen_random(self):
        n = self.random.randint(1, 10 ** 4)
        t = self.random.randint(n, 10 ** 10)
        self.add(dict(t=t, n=n))


class BasicStrCounts(Problem):
    """
    Find a string that has `count1` occurrences of `s1` and `count1` occurrences of `s1` and starts and ends with
    the same 10 characters
    """

    @staticmethod
    def sat(s: str, s1='a', s2='b', count1=50, count2=30):
        return s.count(s1) == count1 and s.count(s2) == count2 and s[:10] == s[-10:]

    @staticmethod
    def sol(s1, s2, count1, count2):
        if s1 == s2:
            ans = (s1 + "?") * count1
        elif s1.count(s2):
            ans = (s1 + "?") * count1
            ans += (s2 + "?") * (count2 - ans.count(s2))
        else:
            ans = (s2 + "?") * count2
            ans += (s1 + "?") * (count1 - ans.count(s1))
        return "?" * 10 + ans + "?" * 10

    def gen_random(self):
        s1 = self.random.pseudo_word(max_len=3)
        s2 = self.random.pseudo_word(max_len=3)
        count1 = self.random.randrange(100)
        count2 = self.random.randrange(100)
        inputs = dict(s1=s1, s2=s2, count1=count1, count2=count2)
        if self.sat(self.sol(**inputs), **inputs):
            self.add(inputs)


class ZipStr(Problem):
    """
    Find a string that contains all the `substrings` alternating, e.g., 'cdaotg' for 'cat' and 'dog'
    """

    @staticmethod
    def sat(s: str, substrings=["foo", "bar", "baz"]):
        return all(sub in s[i::len(substrings)] for i, sub in enumerate(substrings))

    @staticmethod
    def sol(substrings):
        m = max(len(s) for s in substrings)
        return "".join([(s[i] if i < len(s) else " ") for i in range(m) for s in substrings])

    def gen_random(self):
        substrings = [self.random.pseudo_word() for _ in range(self.random.randrange(1, 5))]
        self.add(dict(substrings=substrings))


class ReverseCat(Problem):
    """
    Find a string that contains all the `substrings` reversed and forward
    """

    @staticmethod
    def sat(s: str, substrings=["foo", "bar", "baz"]):
        return all(sub in s and sub[::-1] in s for sub in substrings)

    @staticmethod
    def sol(substrings):
        return "".join(substrings + [s[::-1] for s in substrings])

    def gen_random(self):
        substrings = [self.random.pseudo_word() for _ in range(self.random.randrange(1, 5))]
        self.add(dict(substrings=substrings))


class EngineerNumbers(Problem):
    """
    Find a list of `n` strings starting with `a` and ending with `b`
    """

    @staticmethod
    def sat(ls: List[str], n=100, a='bar', b='foo'):
        return len(ls) == len(set(ls)) == n and ls[0] == a and ls[-1] == b and ls == sorted(ls)

    @staticmethod
    def sol(n, a, b):
        return sorted([a] + [a + chr(0) + str(i) for i in range(n - 2)] + [b])

    def gen_random(self):
        a, b = sorted(self.random.pseudo_word() for _ in range(2))
        n = self.random.randrange(2, 100)
        if a != b:
            self.add(dict(n=n, a=a, b=b))


class PenultimateString(Problem):
    """Find the alphabetically second to last last string in a list."""

    @staticmethod
    def sat(s: str, strings=["cat", "dog", "bird", "fly", "moose"]):
        return s in strings and sum(t > s for t in strings) == 1

    @staticmethod
    def sol(strings):
        return sorted(strings)[-2]

    def gen_random(self):
        strings = [self.random.pseudo_word() for _ in range(10)]
        if self.sat(self.sol(strings), strings=strings):
            self.add(dict(strings=strings))


class PenultimateRevString(Problem):
    """Find the reversed version of the alphabetically second string in a list."""

    @staticmethod
    def sat(s: str, strings=["cat", "dog", "bird", "fly", "moose"]):
        return s[::-1] in strings and sum(t < s[::-1] for t in strings) == 1

    @staticmethod
    def sol(strings):
        return sorted(strings)[1][::-1]

    def gen_random(self):
        strings = [self.random.pseudo_word() for _ in range(10)]
        if self.sat(self.sol(strings), strings=strings):
            self.add(dict(strings=strings))


class CenteredString(Problem):
    """Find a substring of length `length` centered within `target`."""

    @staticmethod
    def sat(s: str, target="foobarbazwow", length=6):
        return target[(len(target) - length) // 2:(len(target) + length) // 2] == s

    @staticmethod
    def sol(target, length):
        return target[(len(target) - length) // 2:(len(target) + length) // 2]

    def gen_random(self):
        target = self.random.pseudo_word()
        length = self.random.randrange(len(target), 0, -1)
        self.add(dict(target=target, length=length))


if __name__ == "__main__":
    Problem.debug_problems()
