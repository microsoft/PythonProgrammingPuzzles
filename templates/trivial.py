"""Trivial problems. Typically for any function, you can construct a trivial example.
For instance, for the len function you can ask for a string of len(s)==100 etc.
"""

from problems import Problem, register, get_problems
from typing import List


@register
class HelloWorld(Problem):
    """Trivial example, no solutions provided"""

    @staticmethod
    def sat(s: str):
        return s + 'world' == 'Hello world'


@register
class BackWorlds(Problem):
    """Two solutions, no inputs"""

    @staticmethod
    def sat(s: str):
        return s[::-1] + 'world' == 'Hello world'

    @staticmethod
    def sol():
        return ' olleH'

    @staticmethod
    def sol2():  # solution methods must begin with 'sol'
        return 'Hello '[::-1]


# With other inputs, the default values of the input are used to generate the first instance.
# You can run Uncat.get_example() to get the inputs, so you can then run
# assert Uncat.sat(Uncat.sol(**Uncat.get_example()))
@register
class StrAdd(Problem):
    """Solve simple string addition problem."""

    @staticmethod
    def sat(st: str, a='world', b='Hello world'):
        return st + a == b

    @staticmethod
    def sol(a, b):
        return b[:len(b) - len(a)]

    def gen_random(self):
        b = self.random.pseudo_word()
        a = b[self.random.randrange(len(b) + 1):]
        self.add({"a": a, "b": b})


assert StrAdd.sat(StrAdd.sol(**StrAdd.get_example()))

@register
class StrSetLen(Problem):
    """Find a string with a certain number of duplicate chars"""

    @staticmethod
    def sat(s: str, dups=1000):
        return len(set(s)) == len(s) - dups

    @staticmethod
    def sol(dups):
        return "a" * (dups + 1)

    def gen(self, target_num_problems):
        for dups in range(target_num_problems):
            if len(self.instances) == target_num_problems:
                return
            self.add(dict(dups=dups))

@register
class StrMul(Problem):
    """Solve string multiplication problem"""

    @staticmethod
    def sat(s: str, target='foofoofoofoo', n=2):
        return s * n == target

    @staticmethod
    def sol(target, n):
        if n == 0:
            return ''
        return target[:len(target) // n]

    def gen_random(self):
        s = self.random.pseudo_word() * self.random.randint(1, 3)
        n = self.random.randrange(10)
        target = s * n
        self.add(dict(target=target, n=n))


@register
class StrMul2(Problem):
    """Solve string multiplication problem"""

    @staticmethod
    def sat(n: int, target='foofoofoofoo', s='foofoo'):
        return s * n == target

    @staticmethod
    def sol(target, s):
        if len(s) == 0:
            return 1
        return len(target) // len(s)

    def gen_random(self):
        s = self.random.pseudo_word() * self.random.randint(1, 3)
        n = self.random.randrange(10)
        target = s * n
        self.add(dict(target=target, s=s))


@register
class StrLen(Problem):
    """Solve string length problem"""

    @staticmethod
    def sat(s: str, n=1000):
        return len(s) == n

    @staticmethod
    def sol(n):
        return 'a' * n

    def gen_random(self):
        n = self.random.randrange(self.random.choice([10, 100, 1000, 10000]))
        self.add(dict(n=n))


@register
class StrAt(Problem):
    """Solve str[i] problem"""

    @staticmethod
    def sat(i: int, s="cat", target="a"):
        return s[i] == target

    @staticmethod
    def sol(s, target):
        return s.index(target)

    def gen_random(self):
        s = self.random.pseudo_word() * self.random.randint(1, 3)
        target = self.random.choice(s)
        self.add(dict(s=s, target=target))


@register
class StrNegAt(Problem):
    """Solve str[-i] problem"""

    @staticmethod
    def sat(i: int, s="cat", target="a"):
        return s[i] == target and i < 0

    @staticmethod
    def sol(s, target):
        return - (len(s) - s.index(target))

    def gen_random(self):
        s = self.random.pseudo_word() * self.random.randint(1, 3)
        target = self.random.choice(s)
        self.add(dict(s=s, target=target))


@register
class StrSlice(Problem):
    """Solve string slice problem"""

    @staticmethod
    def sat(inds: List[int], s="hello world", target="do"):
        i, j, k = inds
        return s[i:j:k] == target

    @staticmethod
    def sol(s, target):
        from itertools import product
        for i, j, k in product(range(-len(s) - 1, len(s) + 1), repeat=3):
            try:
                if s[i:j:k] == target:
                    return [i, j, k]
            except (IndexError, ValueError):
                pass

    def gen_random(self):
        s = self.random.pseudo_word() * self.random.randint(1, 3)
        i, j, k = [self.random.randrange(-len(s) - 1, len(s) + 1) for _ in range(3)]
        try:
            target = s[i:j:k]
            self.add(dict(s=s, target=target))
        except (IndexError, ValueError):
            pass


@register
class StrIndex(Problem):
    """Solve str.index problem"""

    @staticmethod
    def sat(s: str, big_str="foobar", index=2):
        return big_str.index(s) == index

    @staticmethod
    def sol(big_str, index):
        return big_str[index:]

    def gen_random(self):
        big_str = self.random.pseudo_word(max_len=50)
        i = self.random.randrange(len(big_str))
        index = big_str.index(big_str[i:])
        self.add(dict(big_str=big_str, index=index))

@register
class StrIndex2(Problem):
    """Solve str.index problem"""

    @staticmethod
    def sat(big_str: str, sub_str="foobar", index=2):
        return big_str.index(sub_str) == index

    @staticmethod
    def sol(sub_str, index):
        i = ord('A')
        while chr(i) in sub_str:
            i += 1
        return chr(i)*index + sub_str

    def gen_random(self):
        sub_str = self.random.pseudo_word(max_len=50)
        index = self.random.randrange(1000)
        self.add(dict(sub_str=sub_str, index=index))


@register
class StrIn(Problem):
    """Solve str in problem"""

    @staticmethod
    def sat(s: str, a="hello", b="yellow", length=4):
        return len(s) == length and s in a and s in b

    @staticmethod
    def sol(a, b, length):
        for i in range(len(a)-length+1):
            if a[i:i+length] in b:
                return a[i:i+length]


    def gen_random(self):
        sub_str = self.random.pseudo_word()
        a = self.random.pseudo_word() + sub_str + self.random.pseudo_word()
        b = self.random.pseudo_word() + sub_str + self.random.pseudo_word()
        length = len(sub_str)
        self.add(dict(a=a, b=b, length=length))


@register
class StrIn2(Problem):
    """Solve str in problem"""

    @staticmethod
    def sat(substrings: List[str], s="hello", count=15):
        return len(substrings) == len(set(substrings)) >= count and all(sub in s for sub in substrings)

    @staticmethod
    def sol(s, count):
        return [""] + sorted({s[j:i] for i in range(len(s)+1) for j in range(i)})

    def gen_random(self):
        s = self.random.pseudo_word(max_len=50)
        count = len(self.sol(s, None))
        self.add(dict(s=s, count=count))


########################################
# int problems
########################################

@register
class IntNeg(Problem):
    """Solve unary negation problem"""

    @staticmethod
    def sat(x: int, a=93252338):
        return -x == a

    @staticmethod
    def sol(a):
        return - a

    def gen_random(self):
        a = self.random.randint(-10 ** 16, 10 ** 16)
        self.add(dict(a=a))


@register
class IntSum(Problem):
    """Solve sum problem"""

    @staticmethod
    def sat(x: int, a=1073258, b=72352549):
        return a + x == b

    @staticmethod
    def sol(a, b):
        return b - a

    def gen_random(self):
        a = self.random.randint(-10 ** 16, 10 ** 16)
        b = self.random.randint(-10 ** 16, 10 ** 16)
        self.add(dict(a=a, b=b))


@register
class IntSub(Problem):
    """Solve subtraction problem"""

    @staticmethod
    def sat(x: int, a=-382, b=14546310):
        return x - a == b

    @staticmethod
    def sol(a, b):
        return a + b

    def gen_random(self):
        m = 10 ** 16
        a = self.random.randint(-m, m)
        b = self.random.randint(-m, m)
        self.add(dict(a=a, b=b))


@register
class IntSub2(Problem):
    """Solve subtraction problem"""

    @staticmethod
    def sat(x: int, a=8665464, b=-93206):
        return a - x == b

    @staticmethod
    def sol(a, b):
        return a - b

    def gen_random(self):
        m = 10 ** 16
        a = self.random.randint(-m, m)
        b = self.random.randint(-m, m)
        self.add(dict(a=a, b=b))


@register
class IntMul(Problem):
    """Solve multiplication problem"""

    @staticmethod
    def sat(n: int, a=14302, b=5):
        return b * n + (a % b) == a

    @staticmethod
    def sol(a, b):
        return a // b

    def gen_random(self):
        m = 10 ** 6
        a = self.random.randint(-m, m)
        b = self.random.randint(-100, 100)
        if b != 0:
            self.add(dict(a=a, b=b))


@register
class IntDiv(Problem):
    """Solve division problem"""

    @staticmethod
    def sat(n: int, a=3, b=23463462):
        return b // n == a

    @staticmethod
    def sol(a, b):
        if a == 0:
            return 2 * b
        for n in [b // a, b // a - 1, b // a + 1]:
            if b // n == a:
                return n

    def gen_random(self):
        m = 10 ** 16
        n = self.random.randint(-m, m)
        b = self.random.randint(-m, m)
        if n != 0:
            a = b // n
            self.add(dict(a=a, b=b))


@register
class IntDiv2(Problem):
    """Solve division problem"""

    @staticmethod
    def sat(n: int, a=345346363, b=10):
        return n // b == a

    @staticmethod
    def sol(a, b):
        return a * b

    def gen_random(self):
        m = 10 ** 16
        a = self.random.randint(-m, m)
        b = self.random.randint(-m, m)
        if b != 0:
            self.add(dict(a=a, b=b))


@register
class SquareRoot(Problem):
    """Compute square root of number.
       The target has a round (integer) square root."""

    @staticmethod
    def sat(x: int, a=10201202001):
        return x ** 2 == a

    @staticmethod
    def sol(a):
        return int(a ** 0.5)

    def gen_random(self):
        z = self.random.randint(0, 2 ** 31)  # so square < 2 **64
        a = z ** 2
        self.add(dict(a=a))


@register
class NegSquareRoot(Problem):
    """Compute negative square root of number.
       The target has a round (integer) square root."""

    @staticmethod
    def sat(n: int, a=10000200001):
        return a == n * n and n < 0

    @staticmethod
    def sol(a):
        return -int(a ** 0.5)

    def gen_random(self):
        z = self.random.randint(0, 2 ** 31)
        a = z ** 2
        self.add(dict(a=a))


@register
class SquareRootFloat(Problem):
    """Compute square root of number.
       The target might not have a round solution.
       Accuracy of third decimal digit is required."""

    @staticmethod
    def sat(x: float, a=1020):
        return abs(x ** 2 - a) < 10 ** -3

    @staticmethod
    def sol(a):
        return a ** 0.5

    def gen_random(self):
        a = self.random.randint(0, 10 ** 10)
        self.add(dict(a=a))


@register
class NegSquareRootFloat(Problem):
    """Compute (negative) square root of number.
       The target might not have a round solution.
       Accuracy of third decimal digit is required."""

    @staticmethod
    def sat(x: float, a=1020):
        return abs(x ** 2 - a) < 10 ** -3 and x < 0

    @staticmethod
    def sol(a):
        return -a ** 0.5

    def gen_random(self):
        a = self.random.randint(0, 10 ** 10)
        self.add(dict(a=a))


if __name__ == "__main__":
    for problem in get_problems(globals()):
        problem.test(100)
