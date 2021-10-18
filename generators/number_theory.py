"""Number theory problems"""

from puzzle_generator import PuzzleGenerator
from typing import List
import os, json


# See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle to learn about adding puzzles


class FermatsLastTheorem(PuzzleGenerator):
    """[Fermat's last theorem](https://en.wikipedia.org/w/index.php?title=Fermat%27s_Last_Theorem)

    Supposedly unsolvable, but how confident are really in the super-complicated proof?

    See [Wiles, Andrew. "Modular elliptic curves and Fermat's last theorem." Annals of mathematics 141.3 (1995): 443-551.](https://www.jstor.org/stable/2118559)"""

    @staticmethod
    def sat(nums: List[int]):
        """Find integers a,b,c > 0, n > 2, such such that a^n + b^n == c^n"""
        a, b, c, n = nums
        return (a ** n + b ** n == c ** n) and min(a, b, c) > 0 and n > 2


# see also class CertifyGCD in codex.py for an alternate puzzle without lower_bound

class GCD(PuzzleGenerator):
    """[Greatest Common Divisor](https://en.wikipedia.org/w/index.php?title=Greatest_common_divisor&oldid=990943381)
    (GCD)

    See also the [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm)"""

    @staticmethod
    def sat(n: int, a=15482, b=23223, lower_bound=5):
        """Find a large common divisor of two integers."""
        return a % n == 0 and b % n == 0 and n >= lower_bound

    @staticmethod
    def sol(a, b, lower_bound):
        m, n = min(a, b), max(a, b)
        while m > 0:
            m, n = n % m, m
        return n

    @staticmethod
    def sol_rec(a, b, lower_bound):
        def gcd(m, n):
            if m > n:
                return gcd(n, m)
            if m == 0:
                return n
            return gcd(n % m, m)

        return gcd(a, b)

    def gen_random(self):
        factor, r1, r2 = [1 + self.random.randrange(10 ** self.random.randrange(10)) for _ in range(3)]
        a = r1 * factor
        b = r2 * factor
        lower_bound = self.random.randrange(self.sol(a, b, 0) + 1)
        self.add(dict(a=a, b=b, lower_bound=lower_bound))


class GCD_multi(PuzzleGenerator):
    """[Greatest Common Divisor](https://en.wikipedia.org/w/index.php?title=Greatest_common_divisor&oldid=990943381)
    (GCD)

    See also the [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm)"""

    @staticmethod
    def sat(n: int, nums=[77410, 23223, 54187], lower_bound=2):
        """Find a large common divisor of the list of integers."""
        return all(i % n == 0 for i in nums) and n >= lower_bound

    @staticmethod
    def sol(nums, lower_bound):
        n = 0
        for i in nums:
            m, n = min(i, n), max(i, n)
            while m > 0:
                m, n = n % m, m
        return n

    def gen_random(self):
        k = self.random.randrange(2, 12)  # number of factors
        factor, *rs = [1 + self.random.randrange(10 ** self.random.randrange(10)) for _ in range(k + 1)]
        nums = [r * factor for r in rs]
        lower_bound = self.random.randrange(self.sol(nums, 0) + 1)

        self.add(dict(nums=nums, lower_bound=lower_bound))


class LCM(PuzzleGenerator):
    """[Least Common Multiple](https://en.wikipedia.org/wiki/Least_common_multiple)
    (LCM)

    See also the [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm)"""

    @staticmethod
    def sat(n: int, a=15, b=27, upper_bound=150):
        """Find a small common multiple of two integers."""
        return n % a == 0 and n % b == 0 and 0 < n <= upper_bound

    @staticmethod
    def sol(a, b, upper_bound):
        m, n = min(a, b), max(a, b)
        while m > 0:
            m, n = n % m, m
        return a * (b // n)

    def gen_random(self):
        factor, r1, r2 = [1 + self.random.randrange(10 ** self.random.randrange(10)) for _ in range(3)]
        a = r1 * factor
        b = r2 * factor
        upper_bound = int(self.sol(a, b, None) * (1 + self.random.random()))
        self.add(dict(a=a, b=b, upper_bound=upper_bound))


class LCM_multi(PuzzleGenerator):
    """[Least Common Multiple](https://en.wikipedia.org/wiki/Least_common_multiple)
    (LCM)

    See also the [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm)"""

    @staticmethod
    def sat(n: int, nums=[15, 27, 102], upper_bound=5000):
        """Find a small common multiple of a list of integers."""
        return all(n % i == 0 for i in nums) and 0 < n <= upper_bound

    @staticmethod
    def sol(nums, upper_bound):
        ans = 1
        for i in nums:
            m, n = min(i, ans), max(i, ans)
            while m > 0:
                m, n = n % m, m
            ans *= (i // n)
        return ans

    def gen_random(self):
        k = self.random.randrange(2, 12)  # number of factors
        factor, *rs = [1 + self.random.randrange(10 ** self.random.randrange(10)) for _ in range(k + 1)]
        nums = [r * factor for r in rs]
        upper_bound = int(self.sol(nums, None) * (1 + self.random.random()))
        self.add(dict(nums=nums, upper_bound=upper_bound))


class SmallExponentBigSolution(PuzzleGenerator):
    """Small exponent, big solution

    Problems have small b and target but solution is typically a large n.
    Some of them are really hard, for example, for `b=2, target=3`, the smallest solution is `n=4700063497`

    See [Richard K. Guy "The strong law of small numbers", (problem 13)](https://doi.org/10.2307/2322249)"""

    @staticmethod
    def sat(n: int, b=2, target=5):
        """Solve for n: b^n = target (mod n)"""
        return (b ** n) % n == target

    @staticmethod
    def sol(b, target):
        for n in range(1, 10 ** 5):
            if pow(b, n, n) == target:
                return n

    def gen(self, target_num_instances):
        self.add(dict(b=2, target=3), test=False)
        hard = {2: [1, 69], 3: [2, 14, 34, 56, 74], 4: [17, 53, 83, 87], 5: [58], 6: [29, 89],
                7: [36, 66, 86], 8: [49, 61, 91], 9: [8], 10: [9, 11, 29, 83]}
        for target, bs in hard.items():
            for b in bs:
                test = (self.sol(b, target) is not None)
                self.add(dict(b=b, target=target), test=test)
                if self.num_generated_so_far() >= target_num_instances:
                    return
        m = target_num_instances
        solved = {b: {pow(b, n, n) for n in range(1, 10 ** 5) if 1 < pow(b, n, n) < m} for b in range(2, 11)}
        for b, targets in solved.items():
            for target in range(2, 100):
                if target in targets and self.sol(b, target):
                    self.add(dict(b=b, target=target))
                    if self.num_generated_so_far() >= target_num_instances:
                        return

    def gen_random(self):
        pass


class ThreeCubes(PuzzleGenerator):
    """Sum of three cubes

    Given `n`, find integers `a`, `b`, `c` such that `a**3 + b**3 + c**3 = n`. This is unsolvable for `n % 9 in {4, 5}`.
    Conjectured to be true for all other n, i.e., `n % 9 not in {4, 5}`.
    `a`, `b`, `c` may be positive or negative

    See [wikipedia entry](https://en.wikipedia.org/wiki/Sums_of_three_cubes) or
    [Andrew R. Booker, Andrew V. Sutherland (2020). "On a question of Mordell."](https://arxiv.org/abs/2007.01209)
    """

    @staticmethod
    def sat(nums: List[int], target=983):
        """Given n, find integers a, b, c such that a^3 + b^3 + c^3 = n."""
        assert target % 9 not in [4, 5], "Hint"
        return len(nums) == 3 and sum([i ** 3 for i in nums]) == target

    @staticmethod
    def sol(target: int):
        assert target % 9 not in {4, 5}
        for i in range(20):
            for j in range(i + 1):
                for k in range(-20, j + 1):
                    n = i ** 3 + j ** 3 + k ** 3
                    if n == target:
                        return [i, j, k]
                    if n == -target:
                        return [-i, -j, -k]

    def gen(self, target_num_instances):
        targets = [114, 390, 579, 627, 633, 732, 921, 975]
        targets += [t for t in range(target_num_instances // 2) if t % 9 not in {4, 5}]
        for target in targets:
            self.add(dict(target=target), test=self.sol(target) is not None)
            if self.num_generated_so_far() >= target_num_instances:
                return

    def gen_random(self):
        digits = self.random.randrange(1, 10)
        target = self.random.randrange(10 ** digits)
        if target % 9 not in {4, 5}:
            self.add(dict(target=target), test=self.sol(target) is not None)


class FourSquares(PuzzleGenerator):
    """
    Sum of four squares

    [Lagrange's Four Square Theorem](https://en.wikipedia.org/w/index.php?title=Lagrange%27s_four-square_theorem)

    Given a non-negative integer `n`, a classic theorem of Lagrange says that `n` can be written as the sum of four
    integers. The problem here is to find them. This is a nice problem and we give an elementary solution
    that runs in time \tilde{O}(n),
    which is not "polynomial time" because it is not polynomial in log(n), the length of n. A poly-log(n)
    algorithm using quaternions is described in the book:
    ["Randomized algorithms in number theory" by Michael O. Rabin and Jeffery O. Shallit (1986)](https://doi.org/10.1002/cpa.3160390713)

    The first half of the problems involve small numbers and the second half involve some numbers up to 50 digits.
    """

    @staticmethod
    def sat(nums: List[int], n=12345):
        """Find four integers whose squares sum to n"""
        return len(nums) <= 4 and sum(i ** 2 for i in nums) == n

    @staticmethod
    def sol(n):
        m = n
        squares = {i ** 2: i for i in range(int(m ** 0.5) + 2) if i ** 2 <= m}
        sums_of_squares = {i + j: [a, b] for i, a in squares.items() for j, b in squares.items()}
        for s in sums_of_squares:
            if m - s in sums_of_squares:
                return sums_of_squares[m - s] + sums_of_squares[s]
        assert False, "Should never reach here"

    def gen(self, target_num_instances):
        for i in range(target_num_instances // 2):
            n = abs(i ** 2 - 1)
            self.add(dict(n=n))
            if self.num_generated_so_far() >= target_num_instances:
                return

    def gen_random(self):
        n = self.random.randrange(10 ** self.random.randrange(50))
        self.add(dict(n=n), test=(n < 10 ** 5))




class Factoring(PuzzleGenerator):
    """[Factoring](https://en.wikipedia.org/w/index.php?title=Integer_factorization) and
    [RSA challenge](https://en.wikipedia.org/w/index.php?title=RSA_numbers)

    *See class FermatComposite in codex.py for an easier composite test puzzle*

    The factoring problems require one to find any nontrivial factor of n, which is equivalent to factoring by a
    simple repetition process. Problems range from small (single-digit n) all the way to the "RSA challenges"
    which include several *unsolved* factoring problems put out by the RSA company. The challenge was closed in 2007,
    with hundreds of thousands of dollars in unclaimed prize money for factoring their given numbers. People
    continue to work on them, nonetheless, and only the first 22/53 have RSA challenges have been solved thusfar.

    From Wikipedia:

    RSA-2048 has 617 decimal digits (2,048 bits). It is the largest of the RSA numbers and carried the largest
    cash prize for its factorization, $200,000. The RSA-2048 may not be factorizable for many years to come,
    unless considerable advances are made in integer factorization or computational power in the near future.
    """

    DATA_PATH = os.path.join(os.path.dirname(__file__), "problem_constants", "rsa_challenges.json")
    MAX_TEST = 10 ** 16

    @staticmethod
    def sat(i: int, n=241864633):
        """Find a non-trivial factor of integer n"""
        return 1 < i < n and n % i == 0

    @staticmethod
    def sol(n):
        if n % 2 == 0:
            return 2

        for i in range(3, int(n ** 0.5) + 1, 2):
            if n % i == 0:
                return i

        assert False, "problem defined for composite n only"

    def gen(self, target_num_instances):
        with open(self.DATA_PATH) as f:
            challenges = json.load(f)  # format: ["challenge name", "n factor"] (if factor is known)
        numbers = [16] + [int(v.split()[0]) for name, v in challenges]
        for n in numbers:
            self.add(dict(n=n), test=n < self.MAX_TEST)
            if self.num_generated_so_far() >= target_num_instances:
                return

    def gen_random(self):
        # to make sure it's composite, we multiply two < d-digit odd numbers
        m = 10 ** self.random.randrange(1, 10)
        n = (2 * self.random.randrange(2, m) + 1) * (2 * self.random.randrange(2, m) + 1)
        self.add(dict(n=n), test=n < self.MAX_TEST)


class DiscreteLog(PuzzleGenerator):  # updated because the answer was given away in the docstring! OMG
    """Discrete Log

    The discrete logarithm problem is (given `g`, `t`, and `p`) to find n such that:

    `g ** n % p == t`

    From [Wikipedia article](https://en.wikipedia.org/w/index.php?title=Discrete_logarithm_records):

    "Several important algorithms in public-key cryptography base their security on the assumption
    that the discrete logarithm problem over carefully chosen problems has no efficient solution."

    The problem is *unsolved* in the sense that no known polynomial-time algorithm has been found.

    We include McCurley's discrete log challenge from
    [Weber D., Denny T. (1998) "The solution of McCurley's discrete log challenge."](https://link.springer.com/content/pdf/10.1007/BFb0055747.pdf)
    """

    @staticmethod
    def sat(n: int, g=44337, p=69337, t=38187):
        """Find n such that g^n is congruent to t mod n"""
        return pow(g, n, p) == t

    @staticmethod
    def sol(g, p, t):
        for n in range(p):
            if pow(g, n, p) == t:
                return n
        assert False, f"unsolvable discrete log problem g={g}, t={t}, p={p}"

    def gen(self, target_num_instances):
        mccurleys_discrete_log_challenge = {
            "g": 7,
            "p": (739 * (7 ** 149) - 736) // 3,
            "t": 127402180119973946824269244334322849749382042586931621654557735290322914679095998681860978813046595166455458144280588076766033781
        }
        self.add(mccurleys_discrete_log_challenge, test=False)

    def gen_random(self):
        p = self.random.randrange(3, 3 + 10 ** self.random.randrange(25), 2)
        g = self.random.randrange(2, p)
        n = self.random.randrange(p)
        t = pow(g, n, p)
        self.add(dict(g=g, p=p, t=t), test=p < 10 ** 6)


class GCD17(PuzzleGenerator):
    """
    According to [this article](https://primes.utm.edu/glossary/page.php?sort=LawOfSmall), the smallest
    solution is 8424432925592889329288197322308900672459420460792433
    """

    @staticmethod
    def sat(n: int):
        """Find n for which gcd(n^17+9, (n+1)^17+9) != 1"""
        i = n ** 17 + 9
        j = (n + 1) ** 17 + 9

        while i != 0:  # compute gcd using Euclid's algorithm
            (i, j) = (j % i, i)

        return n >= 0 and j != 1


class Znam(PuzzleGenerator):
    """
    [Znam's Problem](https://en.wikipedia.org/wiki/Zn%C3%A1m%27s_problem)

    For example [2, 3, 7, 47, 395] is a solution for k=5
    """

    @staticmethod
    def sat(li: List[int], k=5):
        """Find k positive integers such that each integer divides (the product of the rest plus 1)."""
        def prod(nums):
            ans = 1
            for i in nums:
                ans *= i
            return ans

        return min(li) > 1 and len(li) == k and all((1 + prod(li[:i] + li[i + 1:])) % li[i] == 0 for i in range(k))

    @staticmethod
    def sol(k):
        n = 2
        prod = 1
        ans = []
        while len(ans) < k:
            ans.append(n)
            prod *= n
            n = prod + 1
        return ans

    def gen(self, target_num_instances):
        k = 5
        while self.num_generated_so_far() < target_num_instances:
            self.add(dict(k=k), test=k < 18)
            k += 1


class CollatzCycleUnsolved(PuzzleGenerator):
    """Collatz Conjecture

    A solution to this problem would disprove the *Collatz Conjecture*, also called the *3n + 1 problem*,
    as well as the *Generalized Collatz Conjecture* (see the next problem).
    According to the [Wikipedia article](https://en.wikipedia.org/wiki/Collatz_conjecture):
    > Paul Erdos said about the Collatz conjecture: "Mathematics may not be ready for such problems."
    > He also offered US$500 for its solution. Jeffrey Lagarias stated in 2010 that the Collatz conjecture
    > "is an extraordinarily difficult problem, completely out of reach of present day mathematics."

    Consider the following process. Start with an integer `n` and repeatedly applying the operation:
    * if n is even, divide n by 2,
    * if n is odd, multiply n by 3 and add 1

    The conjecture is to that all `n > 0` eventually reach `n=1`. If this conjecture is false, then
    there is either a cycle or a sequence that increases without bound. This problem seeks a cycle.
    """

    @staticmethod
    def sat(n: int):
        """
        Consider the following process. Start with an integer `n` and repeatedly applying the operation:
        * if n is even, divide n by 2,
        * if n is odd, multiply n by 3 and add 1
        Find n > 4 which is part of a cycle of this process
        """
        m = n
        while n > 4:
            n = 3 * n + 1 if n % 2 else n // 2
            if n == m:
                return True


class CollatzGeneralizedUnsolved(PuzzleGenerator):
    """Generalized Collatz Conjecture

    This version, permits negative n and seek a cycle with a number of magnitude greater than 1000,
    which would disprove the Generalized conjecture that states that the only cycles are the known 5 cycles
    (which don't have positive integers).

    See the [Wikipedia article](https://en.wikipedia.org/wiki/Collatz_conjecture)
    """

    @staticmethod
    def sat(start: int):
        """
        Consider the following process. Start with an integer `n` and repeatedly applying the operation:
        * if n is even, divide n by 2,
        * if n is odd, multiply n by 3 and add 1
        Find n which is part of a cycle of this process that has |n| > 1000
        """
        n = start  # could be positive or negative ...
        while abs(n) > 1000:
            n = 3 * n + 1 if n % 2 else n // 2
            if n == start:
                return True


class CollatzDelay(PuzzleGenerator):
    """Collatz Delay

    Consider the following process. Start with an integer `n` and repeatedly applying the operation:
    * if n is even, divide n by 2,
    * if n is odd, multiply n by 3 and add 1
    Find `0 < n < upper` so that it takes exactly `t` steps to reach 1.


    For instance,
    the number `n=9780657630` takes 1,132 steps and the number `n=93,571,393,692,802,302` takes
    2,091 steps, according to the [Wikipedia article](https://en.wikipedia.org/wiki/Collatz_conjecture)

    Now, this problem can be solved trivially by taking exponentially large `n = 2 ** t` so we also bound the
    number of bits of the solution to be upper.

    See [this webpage](http://www.ericr.nl/wondrous/delrecs.html) for up-to-date records.
    """

    @staticmethod
    def sat(n: int, t=197, upper=20):
        """
        Consider the following process. Start with an integer `n` and repeatedly applying the operation:
        * if n is even, divide n by 2,
        * if n is odd, multiply n by 3 and add 1
        Find `0 < n < upper` so that it takes exactly `t` steps to reach 1.
        """
        m = n
        for i in range(t):
            if n <= 1:
                return False
            n = 3 * n + 1 if n % 2 else n // 2
        return n == 1 and m <= 2 ** upper

    @staticmethod
    def sol(t, upper):
        # Faster solution for simultaneously solving multiple problems is of course possible
        bound = t + 10
        while True:
            bound *= 2
            prev = {1}
            seen = set()
            for delay in range(t):
                seen.update(prev)
                curr = {2 * n for n in prev}
                curr.update({(n - 1) // 3 for n in prev if n % 6 == 4})
                prev = {n for n in curr if n <= bound} - seen
            if prev:
                return min(prev)

    def gen(self, target_num_instances):
        nums = [1000, 2000, 2283, 2337, 2350, 2500, 3000, 4000] + list(range(target_num_instances))

        for t in nums:
            if self.num_generated_so_far() < target_num_instances:
                self.add(dict(t=t, upper=t // 15 + self.random.randint(30, 100)), test=t <= 100)


class Lehmer(PuzzleGenerator):
    """Lehmer puzzle

    According to [The Strong Law of Large Numbers](https://doi.org/10.2307/2322249) Richard K. Guy states that
        D. H. & Emma Lehmer discovered that 2^n = 3 (mod n) for n = 4700063497,
        but for no smaller n > 1
    """

    @staticmethod
    def sat(n: int):
        """Find n  such that 2^n mod n = 3"""
        return pow(2, n, n) == 3

    @staticmethod
    def sol():
        return 4700063497


if __name__ == "__main__":
    PuzzleGenerator.debug_problems()
