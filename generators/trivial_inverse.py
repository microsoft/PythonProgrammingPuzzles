"""Trivial problems. Typically for any function, you can construct a trivial example.
For instance, for the len function you can ask for a string of len(s)==100 etc.
"""

from puzzle_generator import PuzzleGenerator
from typing import List


# See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle to learn about adding puzzles


class HelloWorld(PuzzleGenerator):
    """Trivial example, no solutions provided"""
    @staticmethod
    def sat(s: str):
        """Find a string that when concatenated onto 'world' gives 'Hello world'."""
        return s + 'world' == 'Hello world'


class BackWorlds(PuzzleGenerator):
    """We provide two solutions"""

    @staticmethod
    def sat(s: str):
        """Find a string that when reversed and concatenated onto 'world' gives 'Hello world'."""
        return s[::-1] + 'world' == 'Hello world'

    @staticmethod
    def sol():
        return ' olleH'

    @staticmethod
    def sol2():
        # solution methods must begin with 'sol'
        return 'Hello '[::-1]


class StrAdd(PuzzleGenerator):
    @staticmethod
    def sat(st: str, a='world', b='Hello world'):
        """Solve simple string addition problem."""
        return st + a == b

    @staticmethod
    def sol(a, b):
        return b[:len(b) - len(a)]

    def gen_random(self):
        b = self.random.pseudo_word()
        a = b[self.random.randrange(len(b) + 1):]
        self.add({"a": a, "b": b})


class StrSetLen(PuzzleGenerator):
    @staticmethod
    def sat(s: str, dups=2021):
        """Find a string with dups duplicate chars"""
        return len(set(s)) == len(s) - dups

    @staticmethod
    def sol(dups):
        return "a" * (dups + 1)

    def gen(self, target_num_instances):
        for dups in range(target_num_instances):
            if len(self.instances) == target_num_instances:
                return
            self.add(dict(dups=dups))


class StrMul(PuzzleGenerator):
    @staticmethod
    def sat(s: str, target='foofoofoofoo', n=2):
        """Find a string which when repeated n times gives target"""
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


class StrMul2(PuzzleGenerator):
    @staticmethod
    def sat(n: int, target='foofoofoofoo', s='foofoo'):
        """Find n such that s repeated n times gives target"""
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


class StrLen(PuzzleGenerator):
    @staticmethod
    def sat(s: str, n=1000):
        """Find a string of length n"""
        return len(s) == n

    @staticmethod
    def sol(n):
        return 'a' * n

    def gen_random(self):
        n = self.random.randrange(self.random.choice([10, 100, 1000, 10000]))
        self.add(dict(n=n))


class StrAt(PuzzleGenerator):
    @staticmethod
    def sat(i: int, s="cat", target="a"):
        """Find the index of target in string s"""
        return s[i] == target

    @staticmethod
    def sol(s, target):
        return s.index(target)

    def gen_random(self):
        s = self.random.pseudo_word() * self.random.randint(1, 3)
        target = self.random.choice(s)
        self.add(dict(s=s, target=target))


class StrNegAt(PuzzleGenerator):
    @staticmethod
    def sat(i: int, s="cat", target="a"):
        """Find the index of target in s using a negative index."""
        return s[i] == target and i < 0

    @staticmethod
    def sol(s, target):
        return - (len(s) - s.index(target))

    def gen_random(self):
        s = self.random.pseudo_word() * self.random.randint(1, 3)
        target = self.random.choice(s)
        self.add(dict(s=s, target=target))


class StrSlice(PuzzleGenerator):
    @staticmethod
    def sat(inds: List[int], s="hello world", target="do"):
        """Find the three slice indices that give the specific target in string s"""
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


class StrIndex(PuzzleGenerator):
    @staticmethod
    def sat(s: str, big_str="foobar", index=2):
        """Find a string whose *first* index in big_str is index"""
        return big_str.index(s) == index

    @staticmethod
    def sol(big_str, index):
        return big_str[index:]

    def gen_random(self):
        big_str = self.random.pseudo_word(max_len=50)
        i = self.random.randrange(len(big_str))
        index = big_str.index(big_str[i:])
        self.add(dict(big_str=big_str, index=index))


class StrIndex2(PuzzleGenerator):
    @staticmethod
    def sat(big_str: str, sub_str="foobar", index=2):
        """Find a string whose *first* index of sub_str is index"""
        return big_str.index(sub_str) == index

    @staticmethod
    def sol(sub_str, index):
        i = ord('A')
        while chr(i) in sub_str:
            i += 1
        return chr(i) * index + sub_str

    def gen_random(self):
        sub_str = self.random.pseudo_word(max_len=50)
        index = self.random.randrange(1000)
        self.add(dict(sub_str=sub_str, index=index))


class StrIn(PuzzleGenerator):
    @staticmethod
    def sat(s: str, a="hello", b="yellow", length=4):
        """Find a string of length length that is in both strings a and b"""
        return len(s) == length and s in a and s in b

    @staticmethod
    def sol(a, b, length):
        for i in range(len(a) - length + 1):
            if a[i:i + length] in b:
                return a[i:i + length]

    def gen_random(self):
        sub_str = self.random.pseudo_word()
        a = self.random.pseudo_word() + sub_str + self.random.pseudo_word()
        b = self.random.pseudo_word() + sub_str + self.random.pseudo_word()
        length = len(sub_str)
        self.add(dict(a=a, b=b, length=length))


class StrIn2(PuzzleGenerator):
    @staticmethod
    def sat(substrings: List[str], s="hello", count=15):
        """Find a list of >= count distinct strings that are all contained in s"""
        return len(substrings) == len(set(substrings)) >= count and all(sub in s for sub in substrings)

    @staticmethod
    def sol(s, count):
        return [""] + sorted({s[j:i] for i in range(len(s) + 1) for j in range(i)})

    def gen_random(self):
        s = self.random.pseudo_word(max_len=50)
        count = len(self.sol(s, None))
        self.add(dict(s=s, count=count))


class StrCount(PuzzleGenerator):
    @staticmethod
    def sat(string: str, substring="a", count=10, length=100):
        """Find a string with a certain number of copies of a given substring and of a given length"""
        return string.count(substring) == count and len(string) == length

    @staticmethod
    def sol(substring, count, length):
        c = chr(1 + max(ord(c) for c in (substring or "a")))  # a character not in substring
        return substring * count + (length - len(substring) * count) * '^'

    def gen_random(self):
        substring = self.random.pseudo_word(max_len=self.random.randrange(1, 11))
        count = self.random.randrange(100)
        length = len(substring) * count + self.random.randrange(1000)
        self.add(dict(substring=substring, count=count, length=length))


class StrSplit(PuzzleGenerator):
    @staticmethod
    def sat(x: str, parts=["I", "love", "dumplings", "!"], length=100):
        """Find a string of a given length with a certain split"""
        return len(x) == length and x.split() == parts

    @staticmethod
    def sol(parts, length):
        joined = " ".join(parts)
        return joined + " " * (length - len(joined))

    def gen_random(self):
        parts = [self.random.pseudo_word() for _ in range(self.random.randrange(1, 6))]
        length = len(" ".join(parts)) + self.random.randrange(100)
        self.add(dict(parts=parts, length=length))


class StrSplitter(PuzzleGenerator):
    @staticmethod
    def sat(x: str, parts=["I", "love", "dumplings", "!", ""], string="I_love_dumplings_!_"):
        """Find a separator that when used to split a given string gives a certain result"""
        return string.split(x) == parts

    @staticmethod
    def sol(parts, string):
        if len(parts) <= 1:
            return string * 2
        length = (len(string) - len("".join(parts))) // (len(parts) - 1)
        start = len(parts[0])
        return string[start:start + length]

    def gen_random(self):
        x = self.random.pseudo_word()
        parts = [self.random.pseudo_word(min_len=0) for _ in range(1, self.random.randrange(6))]
        parts = [p for p in parts if x not in p]
        if not any(parts):
            return
        string = x.join(parts)
        self.add(dict(parts=parts, string=string))

class StrJoiner(PuzzleGenerator):
    @staticmethod
    def sat(x: str, parts=["I!!", "!love", "dumplings", "!", ""], string="I!!!!!love!!dumplings!!!!!"):
        """
        Find a separator that when used to join a given string gives a certain result.
        This is related to the previous problem but there are some edge cases that differ.
        """
        return x.join(parts) == string

    @staticmethod
    def sol(parts, string):
        if len(parts) <= 1:
            return ""
        length = (len(string) - len("".join(parts))) // (len(parts) - 1)
        start = len(parts[0])
        return string[start:start + length]

    def gen_random(self):
        x = self.random.pseudo_word()
        parts = [self.random.pseudo_word(min_len=0) for _ in range(1, self.random.randrange(6))]
        string = x.join(parts)
        self.add(dict(parts=parts, string=string))


class StrParts(PuzzleGenerator):
    @staticmethod
    def sat(parts: List[str], sep="!!", string="I!!!!!love!!dumplings!!!!!"):
        """Find parts that when joined give a specific string."""
        return sep.join(parts) == string and all(sep not in p for p in parts)

    @staticmethod
    def sol(sep, string):
        return string.split(sep)

    def gen_random(self):
        sep = self.random.pseudo_word()
        parts = [self.random.pseudo_word(min_len=0) for _ in range(1, self.random.randrange(6))]
        parts = [p for p in parts if sep not in p]
        string = sep.join(parts)
        self.add(dict(sep=sep, string=string))



########################################
# List problems
########################################


class ListSetLen(PuzzleGenerator):
    @staticmethod
    def sat(li: List[int], dups=42155):
        """Find a list with a certain number of duplicate items"""
        return len(set(li)) == len(li) - dups

    @staticmethod
    def sol(dups):
        return [1] * (dups + 1)

    def gen_random(self):
        self.add(dict(dups=self.random.randrange(10 ** 5)))


class ListMul(PuzzleGenerator):
    @staticmethod
    def sat(li: List[int], target=[17, 9, -1, 17, 9, -1], n=2):
        """Find a list that when multiplied n times gives the target list"""
        return li * n == target

    @staticmethod
    def sol(target, n):
        if n == 0:
            return []
        return target[:len(target) // n]

    def gen_random(self):
        li = [self.random.randrange(-10 ** 5, 10 ** 5) for _ in
              range(self.random.randrange(1, 10))] * self.random.randint(1, 3)
        n = self.random.randrange(10)
        target = li * n
        self.add(dict(target=target, n=n))


class ListLen(PuzzleGenerator):
    @staticmethod
    def sat(li: List[int], n=85012):
        """Find a list of a given length n"""
        return len(li) == n

    @staticmethod
    def sol(n):
        return [1] * n

    def gen_random(self):
        n = self.random.randrange(self.random.choice([10, 100, 1000, 10000]))
        self.add(dict(n=n))


class ListAt(PuzzleGenerator):
    @staticmethod
    def sat(i: int, li=[17, 31, 91, 18, 42, 1, 9], target=18):
        """Find the index of an item in a list. Any such index is fine."""
        return li[i] == target

    @staticmethod
    def sol(li, target):
        return li.index(target)

    def gen_random(self):
        li = [self.random.randrange(-10 ** 2, 10 ** 2) for _ in range(self.random.randrange(1, 20))]
        target = self.random.choice(li)
        self.add(dict(li=li, target=target))


class ListNegAt(PuzzleGenerator):
    @staticmethod
    def sat(i: int, li=[17, 31, 91, 18, 42, 1, 9], target=91):
        """Find the index of an item in a list using negative indexing."""
        return li[i] == target and i < 0

    @staticmethod
    def sol(li, target):
        return li.index(target) - len(li)

    def gen_random(self):
        li = [self.random.randrange(-10 ** 2, 10 ** 2) for _ in range(self.random.randrange(1, 20))]
        target = self.random.choice(li)
        self.add(dict(li=li, target=target))


class ListSlice(PuzzleGenerator):
    @staticmethod
    def sat(inds: List[int], li=[42, 18, 21, 103, -2, 11], target=[-2, 21, 42]):
        """Find three slice indices to achieve a given list slice"""
        i, j, k = inds
        return li[i:j:k] == target

    @staticmethod
    def sol(li, target):
        from itertools import product
        for i, j, k in product(range(-len(li) - 1, len(li) + 1), repeat=3):
            try:
                if li[i:j:k] == target:
                    return [i, j, k]
            except (IndexError, ValueError):
                pass

    def gen_random(self):
        li = [self.random.randrange(-10 ** 2, 10 ** 2) for _ in range(self.random.randrange(1, 20))]
        i, j, k = [self.random.randrange(-len(li) - 1, len(li) + 1) for _ in range(3)]
        try:
            target = li[i:j:k]
            if (target != [] and target != li) or self.random.randrange(50) == 0:
                self.add(dict(li=li, target=target))
        except (IndexError, ValueError):
            pass


class ListIndex(PuzzleGenerator):
    @staticmethod
    def sat(item: int, li=[17, 2, 3, 9, 11, 11], index=4):
        """Find the item whose first index in li is index"""
        return li.index(item) == index

    @staticmethod
    def sol(li, index):
        return li[index]

    def gen_random(self):
        li = [self.random.randrange(-10 ** 2, 10 ** 2) for _ in range(self.random.randrange(1, 20))]
        i = self.random.randrange(len(li))
        index = li.index(li[i])
        self.add(dict(li=li, index=index))


class ListIndex2(PuzzleGenerator):
    @staticmethod
    def sat(li: List[int], i=29, index=10412):
        """Find a list that contains i first at index index"""
        return li.index(i) == index

    @staticmethod
    def sol(i, index):
        return [i - 1] * index + [i]

    def gen_random(self):
        i = self.random.randrange(-10 ** 5, 10 ** 5)
        index = self.random.randrange(10 ** 5)
        self.add(dict(i=i, index=index))


class ListIn(PuzzleGenerator):
    @staticmethod
    def sat(s: str, a=['cat', 'dot', 'bird'], b=['tree', 'fly', 'dot']):
        """Find an item that is in both lists a and b"""
        return s in a and s in b

    @staticmethod
    def sol(a, b):
        return next(s for s in b if s in a)

    def gen_random(self):
        a = [self.random.pseudo_word() for _ in range(self.random.randrange(1, 100))]
        b = [self.random.pseudo_word() for _ in range(self.random.randrange(1, 100))]
        b.insert(self.random.randrange(len(b)), self.random.choice(a))
        self.add(dict(a=a, b=b))


########################################
# int problems
########################################

class IntNeg(PuzzleGenerator):
    @staticmethod
    def sat(x: int, a=93252338):
        """Solve a unary negation problem"""
        return -x == a

    @staticmethod
    def sol(a):
        return - a

    def gen_random(self):
        a = self.random.randint(-10 ** 16, 10 ** 16)
        self.add(dict(a=a))


class IntSum(PuzzleGenerator):
    @staticmethod
    def sat(x: int, a=1073258, b=72352549):
        """Solve a sum problem"""
        return a + x == b

    @staticmethod
    def sol(a, b):
        return b - a

    def gen_random(self):
        a = self.random.randint(-10 ** 16, 10 ** 16)
        b = self.random.randint(-10 ** 16, 10 ** 16)
        self.add(dict(a=a, b=b))


class IntSub(PuzzleGenerator):
    @staticmethod
    def sat(x: int, a=-382, b=14546310):
        """Solve a subtraction problem"""
        return x - a == b

    @staticmethod
    def sol(a, b):
        return a + b

    def gen_random(self):
        m = 10 ** 16
        a = self.random.randint(-m, m)
        b = self.random.randint(-m, m)
        self.add(dict(a=a, b=b))


class IntSub2(PuzzleGenerator):
    @staticmethod
    def sat(x: int, a=8665464, b=-93206):
        """Solve a subtraction problem"""
        return a - x == b

    @staticmethod
    def sol(a, b):
        return a - b

    def gen_random(self):
        m = 10 ** 16
        a = self.random.randint(-m, m)
        b = self.random.randint(-m, m)
        self.add(dict(a=a, b=b))


class IntMul(PuzzleGenerator):
    @staticmethod
    def sat(n: int, a=14302, b=5):
        """Solve a multiplication problem"""
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


class IntDiv(PuzzleGenerator):
    @staticmethod
    def sat(n: int, a=3, b=23463462):
        """Solve a division problem"""
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


class IntDiv2(PuzzleGenerator):
    @staticmethod
    def sat(n: int, a=345346363, b=10):
        """Find n that when divided by b is a"""
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


class IntSquareRoot(PuzzleGenerator):
    @staticmethod
    def sat(x: int, a=10201202001):
        """Compute an integer that when squared equals perfect-square a."""
        return x ** 2 == a

    @staticmethod
    def sol(a):
        return int(a ** 0.5)

    def gen_random(self):
        z = self.random.randint(0, 2 ** 31)  # so square < 2 **64
        a = z ** 2
        self.add(dict(a=a))


class IntNegSquareRoot(PuzzleGenerator):
    @staticmethod
    def sat(n: int, a=10000200001):
        """Find a negative integer that when squared equals perfect-square a."""
        return a == n * n and n < 0

    @staticmethod
    def sol(a):
        return -int(a ** 0.5)

    def gen_random(self):
        z = self.random.randint(0, 2 ** 31)
        a = z ** 2
        self.add(dict(a=a))


class FloatSquareRoot(PuzzleGenerator):
    @staticmethod
    def sat(x: float, a=1020):
        """Find a number that when squared is close to a."""
        return abs(x ** 2 - a) < 10 ** -3

    @staticmethod
    def sol(a):
        return a ** 0.5

    def gen_random(self):
        a = self.random.randint(0, 10 ** 10)
        self.add(dict(a=a))


class FloatNegSquareRoot(PuzzleGenerator):
    @staticmethod
    def sat(x: float, a=1020):
        """Find a negative number that when squared is close to a."""
        return abs(x ** 2 - a) < 10 ** -3 and x < 0

    @staticmethod
    def sol(a):
        return -a ** 0.5

    def gen_random(self):
        a = self.random.randint(0, 10 ** 10)
        self.add(dict(a=a))


if __name__ == "__main__":
    PuzzleGenerator.debug_problems()
