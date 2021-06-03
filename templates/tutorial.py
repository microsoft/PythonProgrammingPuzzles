"""
A few example puzzles that were presented with solutions to participants of the study.
"""

from problems import Problem, register, get_problems
from typing import List, Set, Tuple


@register
class Tutorial1(Problem):
    """Find a string that when concatenated onto 'Hello ' gives 'Hello world'."""

    @staticmethod
    def sat(s: str):
        return "Hello " + s == "Hello world"

    @staticmethod
    def sol():
        return "world"


@register
class Tutorial2(Problem):
    """Find a string that when reversed and concatenated onto 'Hello ' gives 'Hello world'."""

    @staticmethod
    def sat(s: str):
        return "Hello " + s[::-1] == "Hello world"

    @staticmethod
    def sol():
        return "world"[::-1]


@register
class Tutorial3(Problem):
    """Find a list of two integers whose sum is 3."""

    @staticmethod
    def sat(x: List[int]):
        return len(x) == 2 and sum(x) == 3

    @staticmethod
    def sol():
        return [1, 2]


@register
class Tutorial4(Problem):
    """Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b'."""

    @staticmethod
    def sat(s: List[str]):
        return len(set(s)) == 1000 and all((x.count("a") > x.count("b")) and ('b' in x) for x in s)

    @staticmethod
    def sol():
        return ["a"*(i+2)+"b" for i in range(1000)]


@register
class Tutorial5(Problem):
    """Find an integer whose perfect square begins with 123456789 in its decimal representation."""

    @staticmethod
    def sat(n: int):
        return str(n * n).startswith("123456789")

    @staticmethod
    def sol():
        return int(int("123456789" + "0"*9) ** 0.5) + 1


# Not clear this last one is necessary/helpful
# @register
# class Tutorial6(Problem):
#     """Find a string corresponding to a decimal number whose negation is 1337."""
#
#     @staticmethod
#     def sat(s: str):
#         return -1 * int(s) == 1337
#
#     @staticmethod
#     def sol():
#         return str(-1337)

if __name__ == "__main__":
    for problem in get_problems(globals()):
        problem.test()
