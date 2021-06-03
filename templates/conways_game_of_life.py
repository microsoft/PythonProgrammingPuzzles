"""Conway's Game of Life problems"""

from problems import Problem, register, get_problems
from typing import List

@register
class Oscillators(Problem):
    """Oscillators (including some unsolved, open problems)

    Find a pattern in [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
    that repeats with a certain period. This problem is *unsolved* for periods 19, 38, and 41.

    See
    [discussion](https://en.wikipedia.org/wiki/Oscillator_%28cellular_automaton%29#:~:text=Game%20of%20Life)
    in Wikipedia article on Cellular Automaton Oscillators.
    """

    @staticmethod
    def sat(init: List[List[int]], period=3):
        target = {x + y * 1j for x, y in init}  # complex numbers encode live cells

        deltas = (1j, -1j, 1, -1, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j)
        live = target
        for t in range(period):
            visible = {z + d for z in live for d in deltas}
            live = {z for z in visible if sum(z + d in live for d in deltas) in ([2, 3] if z in live else [3])}
            if live == target:
                return t + 1 == period


    @staticmethod
    def sol(period):  # generate random patterns, slow solution
        # def viz(live):
        #     if not live:
        #         return
        #     a, b = min(z.real for z in live), min(z.imag for z in live)
        #     live = {z - (a + b * 1j) for z in live}
        #     m, n = int(max(z.real for z in live)) + 1, int(max(z.imag for z in live)) + 1
        #     for x in range(m):
        #         print("".join("X" if x + y * 1j in live else "," for y in range(n)))

        import random
        rand = random.Random(1)
        # print(f"Looking for {period}:")
        deltas = (1j, -1j, 1, -1, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j)

        completes = [[x + y * 1j for x in range(n) for y in range(n)] for n in range(30)]

        for _attempt in range(10 ** 5):
            n = rand.randrange(3, 10)
            m = rand.randrange(3, n * n)
            live = set(rand.sample(completes[n], m))
            if rand.randrange(2):
                live.update([-z for z in live])
            if rand.randrange(2):
                live.update([z.conjugate() for z in live])
            memory = {}
            for step in range(period * 10):
                key = sum((.123 - .99123j) ** z for z in live) * 10 ** 5
                key = int(key.real), int(key.imag)
                if key in memory:
                    if memory[key] == step - period:
                        # print(period)
                        # viz(live)
                        return [[int(z.real), int(z.imag)] for z in live]
                    break
                memory[key] = step
                visible = {z + d for z in live for d in deltas}
                live = {z for z in visible if sum(z + d in live for d in deltas) in range(3 - (z in live), 4)}

        return None  # failed

    def gen(self, target_num_problems):
        for period in range(1, target_num_problems + 1):
            self.add(dict(period=period), test=(period in {1, 2, 3})) # period 6 takes 30s to test

########################################################################################################################

@register
class Spaceship(Problem):
    """Spaceship (including *unsolved*, open problems)

    Find a [spaceship](https://en.wikipedia.org/wiki/Spaceship_%28cellular_automaton%29) in
    [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
    with a certain period.

    This is an *unsolved* problem for periods 33, 34."""

    @staticmethod
    def sat(init: List[List[int]], period=4):
        live = {x + y * 1j for x, y in init}  # use complex numbers
        init_tot = sum(live)
        target = {z * len(live) - init_tot for z in live}
        deltas = (1j, -1j, 1, -1, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j)

        for t in range(period):
            visible = {z + d for z in live for d in deltas}
            live = {z for z in visible if 3 - (z in live) <= sum(z + d in live for d in deltas) <= 3}
            tot = sum(live)
            if {z * len(live) - tot for z in live} == target:
                return t + 1 == period and tot != init_tot


    # def viz(live):
    #     a, b = min(i for i, j in live), min(j for i, j in live)
    #     live = {(i - a, j - b) for i, j in live}
    #     m, n = max(i for i, j in live), max(j for i, j in live)
    #     for i in range(m + 1):
    #         print("".join("X" if (i, j) in live else "," for j in range(n + 1)))
    #
    #
    # def sol():  # generate random patterns, slow solution
    #     def viz(live):
    #         if not live:
    #             return
    #         a, b = min(z.real for z in live), min(z.imag for z in live)
    #         live = {z - (a + b * 1j) for z in live}
    #         m, n = int(max(z.real for z in live)) + 1, int(max(z.imag for z in live)) + 1
    #         for x in range(m):
    #             print("".join("X" if x + y * 1j in live else "," for y in range(n)))
    #
    #     import random
    #     rand = random.Random(0)
    #     # print(f"Looking for {period}:")
    #     deltas = (1j, -1j, 1, -1, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j)
    #
    #     completes = [[x + y * 1j for x in range(n) for y in range(n)] for n in range(30)]
    #
    #     for _attempt in range(10 ** 4):
    #         n = rand.randrange(3, 10)
    #         m = rand.randrange(3, n * n)
    #         live = set(rand.sample(completes[n], m))
    #         if rand.randrange(2):
    #             live.update([-z for z in live])
    #         if rand.randrange(2):
    #             live.update([z.conjugate() for z in live])
    #         memory = {}
    #         for step in range(period * 10):
    #             if not live:
    #                 break
    #             avg = sum(live)/len(live)
    #             key = sum((.123 - .99123j) ** (z-avg) for z in live) * 10 ** 5
    #
    #             key = int(key.real), int(key.imag)
    #             if key in memory:
    #                 t2, avg2 = memory[key]
    #                 if t2 == step - period and avg2 != avg:
    #                     print(period)
    #                     viz(live)
    #                     return [[int(z.real), int(z.imag)] for z in live]
    #                 break
    #             memory[key] = step, avg
    #             visible = {z + d for z in live for d in deltas}
    #             live = {z for z in visible if sum(z + d in live for d in deltas) in range(3 - (z in live), 4)}
    #
    #     return None  # failed

    def gen(self, target_num_problems):
        for period in range(2, target_num_problems + 2):
            self.add(dict(period=period), test=(period not in (33,34)))


if __name__ == "__main__":
    for problem in get_problems(globals()):
        problem.test()
