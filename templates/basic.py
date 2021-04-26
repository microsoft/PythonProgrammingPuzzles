"""Problems testing basic knowledge -- easy to solve if you understand what is being asked"""

from problems import Problem, register, get_problems
from typing import List


@register
class LineIntersection(Problem):
    """Find the intersection of two lines.
       Solution should be a list of the (x,y) coordinates.
       Accuracy of fifth decimal digit is required."""


    @staticmethod
    def sat(e: List[int], a=2, b=-1, c=1, d=-3):
        x = e[0] / e[1]
        return abs(a * x + b - c * x - d) < 10 ** -5

    @staticmethod
    def sol(a, b, c, d):
        return [d - b, a - c]

    def gen_random(self):
        a = self.random.randint(-10**8, 10**8)
        b = self.random.randint(-10**8, 10**8)
        c = a
        while c == a:
            c = self.random.randint(-10**8, 10**8)
        d = self.random.randint(-10**8, 10**8)
        self.add(dict(a=a, b=b, c=c, d=d))

@register
class IfProblem(Problem):
    """Simple if statement"""

    @staticmethod
    def sat(x: int, a=2, b=100):
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
        b = self.random.randint(-10**8, 10**8)
        self.add(dict(a=a, b=b))

@register
class IfProblemWithAnd(Problem):
    """Simple if statement with and clause"""

    @staticmethod
    def sat(x: int, a=2, b =100):
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
        b = self.random.randint(0, 10**8)
        self.add(dict(a=a, b=b))


@register
class IfProblemWithOr(Problem):
    """Simple if statement with or clause"""

    @staticmethod
    def sat(x: int, a=2, b=100):
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
        b = self.random.randint(-10**8, 10**8)
        self.add(dict(a=a, b=b))


@register
class IfCases(Problem):
    """Simple if statement with multiple cases"""

    @staticmethod
    def sat(x: int, a=1, b=100):
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
        b = self.random.randint(-10**8, 10**8)
        self.add(dict(a=a, b=b))


@register
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
        n = self.random.randint(1,10**4)
        s = self.random.randint(n, 10**8)
        self.add(dict(n=n, s=s))


@register
class ListDistinctSum(Problem):
    """Construct a list of distinct integers that sum up to some value"""

    @staticmethod
    def sat(x: List[int], n=4, s=1):
        return len(x) == n and sum(x) == s and len(set(x)) == n

    @staticmethod
    def sol(n, s):
        a = 1
        x = []
        while len(x) < n-1:
            x.append(a)
            a = -a
            if a in x:
                a += 1

        if s - sum(x) in x:
            x = [i for i in range(n-1)]

        x = x + [s - sum(x)]
        return x

    def gen_random(self):
        n = self.random.randint(1,10**3)
        s = self.random.randint(n+1, 10**8)
        self.add(dict(n=n, s=s))


@register
class ConcatStrings(Problem):
    """Concatenate list of characters"""

    @staticmethod
    def sat(x: str, s=["a", "b", "c", "d", "e", "f"], n=4):
        return len(x)==n and all([x[i] == s[i] for i in range(n)])

    @staticmethod
    def sol(s, n):
        return ''.join([s[i] for i in range(n)])

    def gen_random(self):
        n = self.random.randint(0,25)
        extra = self.random.randint(0,25)
        s = [self.random.char() for _ in range(n+extra)]
        self.add(dict(n=n, s=s))


@register
class SublistSum(Problem):
    """Sum values of sublist by range specifications"""

    @staticmethod
    def sat(x: List[int], t=677, a=43, e=125, s=10):
        non_zero = [z for z in x if z != 0]
        return t == sum([x[i] for i in range(a,e,s)]) and len(set(non_zero)) == len(non_zero) and all([x[i] != 0 for i in range(a,e,s)])

    @staticmethod
    def sol(t, a, e, s):
        x = [0] * e
        for i in range(a,e,s):
            x[i] = i
        correction = t - sum(x) + x[i]
        if correction in x:
            x[correction] = -1 * correction
            x[i] = 3 * correction
        else:
            x[i] = correction
        return x

    def gen_random(self):
        t = self.random.randint(1,10**8)
        a = self.random.randint(1,100)
        e = self.random.randint(a,10**4)
        s = self.random.randint(1,10)
        self.add(dict(t=t, a=a, e=e, s=s))


@register
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
        n = self.random.randint(1,10**4)
        t = self.random.randint(n,10**10)
        self.add(dict(t=t, n=n))


if __name__ == "__main__":
    for problem in get_problems(globals()):
        problem.test()
