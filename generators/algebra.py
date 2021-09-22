"""Roots of polynomials"""

from puzzle_generator import PuzzleGenerator
from typing import List


# See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle to learn about adding puzzles

class AlgebraGen:
    blah = 17
    taint_date = [2021, 4, 26]



class QuadraticRoot(AlgebraGen, PuzzleGenerator):
    """See [quadratic equations](https://en.wikipedia.org/wiki/Quadratic_formula)"""

    @staticmethod
    def sat(x: float, coeffs=[2.5, 1.3, -0.5]):
        """
        Find any (real) solution to:  a x^2 + b x + c where coeffs = [a, b, c].
        For example, since x^2 - 3x + 2 has a root at 1, sat(x = 1., coeffs = [1., -3., 2.]) is True.
        """
        a, b, c = coeffs
        return abs(a * x ** 2 + b * x + c) < 1e-6

    @staticmethod
    def sol(coeffs):
        a, b, c = coeffs
        if a == 0:
            ans = -c / b if b != 0 else 0.0
        else:
            ans = ((-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a))
        return ans

    @staticmethod
    def sol2(coeffs):
        a, b, c = coeffs
        if a == 0:
            ans = -c / b if b != 0 else 0.0
        else:
            ans = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
        return ans

    def gen_random(self):
        x, a, b = [self.random.heavy_tail_float() for _ in range(3)]
        c = -(a * x ** 2 + b * x)  # make sure it has a real-valued solution
        coeffs = [a, b, c]
        self.add(dict(coeffs=coeffs))


class AllQuadraticRoots(PuzzleGenerator):
    """See [quadratic equations](https://en.wikipedia.org/wiki/Quadratic_formula)."""

    @staticmethod
    def sat(roots: List[float], coeffs=[1.3, -0.5]):
        """Find all (real) solutions to: x^2 + b x + c (i.e., factor into roots), here coeffs = [b, c]"""
        b, c = coeffs
        r1, r2 = roots
        return abs(r1 + r2 + b) + abs(r1 * r2 - c) < 1e-6

    @staticmethod
    def sol(coeffs):
        b, c = coeffs
        delta = (b ** 2 - 4 * c) ** 0.5
        return [(-b + delta) / 2, (-b - delta) / 2]

    def gen_random(self):
        x, b = [self.random.heavy_tail_float() for _ in range(2)]
        c = -(x ** 2 + b * x)  # make sure it has a real-valued solution
        coeffs = [b, c]
        self.add(dict(coeffs=coeffs))


class CubicRoot(PuzzleGenerator):
    """See [cubic equation](https://en.wikipedia.org/wiki/Cubic_formula)."""

    @staticmethod
    def sat(x: float, coeffs=[2.0, 1.0, 0.0, 8.0]):
        """
        Find any (real) solution to: a x^3 + b x^2 + c x + d where coeffs = [a, b, c, d]
        For example, since (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6, sat(x = 1., coeffs = [-6., 11., -6.]) is True.
        """
        return abs(sum(c * x ** (3 - i) for i, c in enumerate(coeffs))) < 1e-6

    @staticmethod
    def sol(coeffs):
        a2, a1, a0 = [c / coeffs[0] for c in coeffs[1:]]
        p = (3 * a1 - a2 ** 2) / 3
        q = (9 * a1 * a2 - 27 * a0 - 2 * a2 ** 3) / 27
        delta = (q ** 2 + 4 * p ** 3 / 27) ** 0.5
        omega = (-(-1) ** (1 / 3))
        for cube in [(q + delta) / 2, (q - delta) / 2]:
            c = cube ** (1 / 3)
            for w in [c, c * omega, c * omega.conjugate()]:
                if w != 0:
                    x = complex(w - p / (3 * w) - a2 / 3).real
                    if abs(sum(c * x ** (3 - i) for i, c in enumerate(coeffs))) < 1e-6:
                        return x

    def gen_random(self):
        x, a, b, c = [self.random.heavy_tail_float() for _ in range(4)]
        d = -(a * x ** 3 + b * x ** 2 + c * x)  # make sure it has a real-valued solution
        coeffs = [a, b, c, d]
        if self.sol(coeffs) is not None:
            self.add(dict(coeffs=coeffs))


class AllCubicRoots(PuzzleGenerator):
    """See [cubic equation](https://en.wikipedia.org/wiki/Cubic_formula)."""

    @staticmethod
    def sat(roots: List[float], coeffs=[1.0, -2.0, -1.0]):
        """Find all 3 distinct real roots of x^3 + a x^2 + b x + c, i.e., factor into (x-r1)(x-r2)(x-r3).
        coeffs = [a, b, c]. For example, since (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6,
        sat(roots = [1., 2., 3.], coeffs = [-6., 11., -6.]) is True.
        """
        r1, r2, r3 = roots
        a, b, c = coeffs
        return abs(r1 + r2 + r3 + a) + abs(r1 * r2 + r1 * r3 + r2 * r3 - b) + abs(r1 * r2 * r3 + c) < 1e-6

    @staticmethod
    def sol(coeffs):
        a, b, c = coeffs
        p = (3 * b - a ** 2) / 3
        q = (9 * b * a - 27 * c - 2 * a ** 3) / 27
        delta = (q ** 2 + 4 * p ** 3 / 27) ** 0.5
        omega = (-(-1) ** (1 / 3))
        ans = []
        for cube in [(q + delta) / 2, (q - delta) / 2]:
            v = cube ** (1 / 3)
            for w in [v, v * omega, v * omega.conjugate()]:
                if w != 0.0:
                    x = complex(w - p / (3 * w) - a / 3).real
                    if abs(x ** 3 + a * x ** 2 + b * x + c) < 1e-4:
                        if not ans or min(abs(z - x) for z in ans) > 1e-6:
                            ans.append(x)
        if len(ans) == 3:
            return ans

    def gen_random(self):
        r1, r2, r3 = [self.random.heavy_tail_float() for _ in range(3)]
        coeffs = [-r1 - r2 - r3, r1 * r2 + r1 * r3 + r2 * r3, -r1 * r2 * r3]  # to ensure solvability
        if self.sol(coeffs) is not None:
            self.add(dict(coeffs=coeffs))  # won't add duplicates


if __name__ == "__main__":
    PuzzleGenerator.debug_problems()
