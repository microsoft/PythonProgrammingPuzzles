"""
Problems used for the study.
"""

from problems import Problem
from typing import List, Set


# Hint: subclass Problem.Debug for quick testing. Run make_dataset.py to make the dataset
# See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle for more info


class Study_1(Problem):
    "Find a string with 1000 'o's but no two adjacent 'o's."

    @staticmethod
    def sat(s: str):
        return s.count('o') == 1000 and s.count('oo') == 0

    @staticmethod
    def sol():
        return ('h' + 'o') * 1000


class Study_2(Problem):
    "Find a string with 1000 'o's, 100 pairs of adjacent 'o's and 801 copies of 'ho'."

    @staticmethod
    def sat(s: str):
        return s.count('o') == 1000 and s.count('oo') == 100 and s.count('ho') == 801

    @staticmethod
    def sol():
        return 'ho' * (800 + 1) + 'o' * (100 * 2 - 1)


class Study_3(Problem):
    "Find a permutation of [0, 1, ..., 998] such that the ith element is *not* i, for all i=0, 1, ..., 998."

    @staticmethod
    def sat(li: List[int]):
        return sorted(li) == list(range(999)) and all(li[i] != i for i in range(len(li)))

    @staticmethod
    def sol():
        return [((i + 1) % 999) for i in range(999)]


class Study_4(Problem):
    "Find a list of length 10 where the fourth element occurs exactly twice."

    @staticmethod
    def sat(li: List[int]):
        return len(li) == 10 and li.count(li[3]) == 2

    @staticmethod
    def sol():
        return list(range(10 // 2)) * 2


class Study_5(Problem):
    "Find a list integers such that the integer i occurs i times, for i = 0, 1, 2, ..., 9."

    @staticmethod
    def sat(li: List[int]):
        return all([li.count(i) == i for i in range(10)])

    @staticmethod
    def sol():
        return [i for i in range(10) for j in range(i)]


class Study_6(Problem):
    "Find an integer greater than 10^10 which is 4 mod 123."

    @staticmethod
    def sat(i: int):
        return i % 123 == 4 and i > 10 ** 10

    @staticmethod
    def sol():
        return 4 + 10 ** 10 + 123 - 10 ** 10 % 123


class Study_7(Problem):
    "Find a three-digit pattern  that occurs more than 8 times in the decimal representation of 8^2888."

    @staticmethod
    def sat(s: str):
        return str(8 ** 2888).count(s) > 8 and len(s) == 3

    @staticmethod
    def sol():
        s = str(8 ** 2888)
        return max({s[i: i + 3] for i in range(len(s) - 2)}, key=lambda t: s.count(t))


class Study_8(Problem):
    "Find a list of more than 1235 strings such that the 1234th string is a proper substring of the 1235th."

    @staticmethod
    def sat(ls: List[str]):
        return ls[1234] in ls[1235] and ls[1234] != ls[1235]

    @staticmethod
    def sol():
        return [''] * 1235 + ['a']


class Study_9(Problem):
    """Find a way to rearrange the letters in the pangram "The quick brown fox jumps over the lazy dog" to
    get the pangram "The five boxing wizards jump quickly". The answer should be represented as a list of index
    mappings.
    """

    @staticmethod
    def sat(li: List[int]):
        return ["The quick brown fox jumps over the lazy dog"[i] for i in li] == list(
            "The five boxing wizards jump quickly")

    @staticmethod
    def sol():
        return ['The quick brown fox jumps over the lazy dog'.index(t) for t in 'The five boxing wizards jump quickly']


class Study_10(Problem):
    "Find a palindrome of length greater than 11 in the decimal representation of 8^1818."

    @staticmethod
    def sat(s: str):
        return s in str(8 ** 1818) and s == s[::-1] and len(s) > 11

    @staticmethod
    def sol():
        s = str(8 ** 1818)
        return next(s[i: i + le]
                    for le in range(12, len(s) + 1)
                    for i in range(len(s) - le + 1)
                    if s[i: i + le] == s[i: i + le][::-1]
                    )


class Study_11(Problem):
    """Find a list of strings whose length (viewed as a string) is equal to the lexicographically largest element
    and is equal to the lexicographically smallest element."""

    @staticmethod
    def sat(ls: List[str]):
        return min(ls) == max(ls) == str(len(ls))

    @staticmethod
    def sol():
        return ['1']


class Study_12(Problem):
    """Find a list of 1,000 integers where every two adjacent integers sum to 9, and where the first
    integer plus 4 is 9."""

    @staticmethod
    def sat(li: List[int]):
        return all(i + j == 9 for i, j in zip([4] + li, li)) and len(li) == 1000

    @staticmethod
    def sol():
        return [9 - 4, 4] * (1000 // 2)


class Study_13(Problem):
    "Find a real number which, when you subtract 3.1415, has a decimal representation starting with 123.456."

    @staticmethod
    def sat(x: float):
        return str(x - 3.1415).startswith("123.456")

    @staticmethod
    def sol():
        return 123.456 + 3.1415


class Study_14(Problem):
    "Find a list of integers such that the sum of the first i integers is i, for i=0, 1, 2, ..., 19."

    @staticmethod
    def sat(li: List[int]):
        return all([sum(li[:i]) == i for i in range(20)])

    @staticmethod
    def sol():
        return [1] * 20


class Study_15(Problem):
    "Find a list of integers such that the sum of the first i integers is 2^i -1, for i = 0, 1, 2, ..., 19."

    @staticmethod
    def sat(li: List[int]):
        return all(sum(li[:i]) == 2 ** i - 1 for i in range(20))

    @staticmethod
    def sol():
        return [(2 ** i) for i in range(20)]


class Study_16(Problem):
    """Find a real number such that when you add the length of its decimal representation to it, you get 4.5.
    Your answer should be the string form of the number in its decimal representation."""

    @staticmethod
    def sat(s: str):
        return float(s) + len(s) == 4.5

    @staticmethod
    def sol():
        return str(4.5 - len(str(4.5)))


class Study_17(Problem):
    "Find a number whose decimal representation is *a longer string* when you add 1,000 to it than when you add 1,001."

    @staticmethod
    def sat(i: int):
        return len(str(i + 1000)) > len(str(i + 1001))

    @staticmethod
    def sol():
        return -1001


class Study_18(Problem):
    """Find a list of strings that when you combine them in all pairwise combinations gives the six strings:
    'berlin', 'berger', 'linber', 'linger', 'gerber', 'gerlin'
    """

    @staticmethod
    def sat(ls: List[str]):
        return [s + t for s in ls for t in ls if s != t] == 'berlin berger linber linger gerber gerlin'.split()

    @staticmethod
    def sol():
        seen = set()
        ans = []
        for s in 'berlin berger linber linger gerber gerlin'.split():
            t = s[:3]
            if t not in seen:
                ans.append(t)
                seen.add(t)
        return ans


class Study_19(Problem):
    """Find a set of integers whose pairwise sums make the set {0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 34}.
    That is find set S such that, { i + j | i, j in S } = {0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 34}.
    """

    @staticmethod
    def sat(si: Set[int]):
        return {i + j for i in si for j in si} == {0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 34}

    @staticmethod
    def sol():
        return {0, 1, 2, 3, 17}


class Study_20(Problem):
    """Find a list of integers, starting with 0 and ending with 128, such that each integer either differs from
    the previous one by one or is thrice the previous one."""

    @staticmethod
    def sat(li: List[int]):
        return all(j in {i - 1, i + 1, 3 * i} for i, j in zip([0] + li, li + [128]))

    @staticmethod
    def sol():
        return [1, 3, 4, 12, 13, 14, 42, 126, 127]


class Study_21(Problem):
    """Find a list integers containing exactly three distinct values, such that no integer repeats
    twice consecutively among the first eleven entries. (So the list needs to have length greater than ten.)
    """

    @staticmethod
    def sat(li: List[int]):
        return all([li[i] != li[i + 1] for i in range(10)]) and len(set(li)) == 3

    @staticmethod
    def sol():
        return list(range(3)) * 10


class Study_22(Problem):
    """Find a string s containing exactly five distinct characters which also contains as a substring every other
    character of s (e.g., if the string s were 'parrotfish' every other character would be 'profs')."""

    @staticmethod
    def sat(s: str):
        return s[::2] in s and len(set(s)) == 5

    @staticmethod
    def sol():
        return """abacadaeaaaaaaaaaa"""


class Study_23(Problem):
    """Find a list of characters which are aligned at the same indices of the three strings 'dee', 'doo', and 'dah!'."""

    @staticmethod
    def sat(ls: List[str]):
        return tuple(ls) in zip('dee', 'doo', 'dah!')

    @staticmethod
    def sol():
        return list(next(zip('dee', 'doo', 'dah!')))


class Study_24(Problem):
    """Find a list of integers with exactly three occurrences of seventeen and at least two occurrences of three."""

    @staticmethod
    def sat(li: List[int]):
        return li.count(17) == 3 and li.count(3) >= 2

    @staticmethod
    def sol():
        return [17] * 3 + [3] * 2


class Study_25(Problem):
    """Find a permutation of the string 'Permute me true' which is a palindrome."""

    @staticmethod
    def sat(s: str):
        return sorted(s) == sorted('Permute me true') and s == s[::-1]

    @staticmethod
    def sol():
        return """""".join(sorted('Permute me true'[1:])[::2] + ['P'] + sorted('Permute me true'[1:])[::2][::-1])


class Study_26(Problem):
    """Divide the decimal representation of 8^88 up into strings of length eight."""

    @staticmethod
    def sat(ls: List[str]):
        return "".join(ls) == str(8 ** 88) and all(len(s) == 8 for s in ls)

    @staticmethod
    def sol():
        return [str(8 ** 88)[i:i + 8] for i in range(0, len(str(8 ** 88)), 8)]


class Study_27(Problem):
    """Consider a digraph where each node has exactly one outgoing edge. For each edge (u, v), call u the parent and
    v the child. Then find such a digraph where the grandchildren of the first and second nodes differ but they
    share the same great-grandchildren. Represented this digraph by the list of children indices."""

    @staticmethod
    def sat(li: List[int]):
        return li[li[0]] != li[li[1]] and li[li[li[0]]] == li[li[li[1]]]

    @staticmethod
    def sol():
        return [1, 2, 3, 3]


class Study_28(Problem):
    """Find a set of one hundred integers between 0 and 999 which all differ by at least ten from one another."""

    @staticmethod
    def sat(si: Set[int]):
        return all(i in range(1000) and abs(i - j) >= 10 for i in si for j in si if i != j) and len(si) == 100

    @staticmethod
    def sol():
        return set(range(0, 1000, 10))


class Study_29(Problem):
    """Find a set of more than 995 integers between 0 and 999, inclusive, such that each pair of integers have
    squares that differ by at least 10."""

    @staticmethod
    def sat(si: Set[int]):
        return all(i in range(1000) and abs(i * i - j * j) >= 10 for i in si for j in si if i != j) and len(si) > 995

    @staticmethod
    def sol():
        return set(range(6, 1000)).union({0, 4})


class Study_30(Problem):
    """Define f(n) to be the residue of 123 times n mod 1000. Find a list of integers such that the first twenty one
    are between 0 and 999, inclusive, and are strictly increasing in terms of f(n)."""

    @staticmethod
    def sat(li: List[int]):
        return all([123 * li[i] % 1000 < 123 * li[i + 1] % 1000 and li[i] in range(1000) for i in range(20)])

    @staticmethod
    def sol():
        return sorted(range(1000), key=lambda n: 123 * n % 1000)[:21]

    @staticmethod
    def sol_surprisingly_short():
        return list(range(1000))[::8][::-1]


if __name__ == "__main__":
    Problem.debug_problems()
