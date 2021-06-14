"""
Solve some two-player games
"""

from problems import Problem
from typing import List


# Hint: subclass Problem.Debug for quick testing. Run make_dataset.py to make the dataset
# See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle for more info


class Nim(Problem):
    """Compute optimal play for the classic two-player game [Nim](https://en.wikipedia.org/wiki/Nim)

    In the game of Nim, there are a number of heaps of objects. In each step, a player removes one or more
    objects from a non-empty heap. The player who takes the last object wins. Nim has an elegant theory
    for optimal play based on the xor of the bits.
    """

    timeout = 10  # harder than most problems, get extra time

    @staticmethod
    def sat(cert: List[List[int]], heaps=[5, 9]):  # cert is a sufficient list of desirable states to leave for opponent
        good_leaves = {tuple(h) for h in cert}  # for efficiency, we keep track of h as a tuple of n non-negative ints
        cache = {}

        def is_good_leave(h):
            if h in cache:
                return cache[h]
            next_states = [(*h[:i], k, *h[i + 1:]) for i in range(len(h)) for k in range(h[i])]
            conjecture = (h in good_leaves)
            if conjecture:  # check that it is a good leave
                assert not any(is_good_leave(s) for s in next_states)
            else:  # check that it is a bad leave, only need to check one move
                assert is_good_leave(next(s for s in next_states if s in good_leaves))
            cache[h] = conjecture
            return conjecture

        return is_good_leave(tuple(heaps)) == (tuple(heaps) in good_leaves)

    @staticmethod
    def sol(heaps):
        import itertools

        def val(h):  # return True if h is a good state to leave things in
            xor = 0
            for i in h:
                xor ^= i
            return xor == 0

        return [list(h) for h in itertools.product(*[range(i + 1) for i in heaps]) if val(h)]

    def gen_random(self):
        num_heaps = self.random.randrange(10)
        heaps = [self.random.randrange(10) for _ in range(num_heaps)]
        prod = 1
        for i in heaps:
            prod *= i + 1
        if prod < 10 ** 6:
            self.add(dict(heaps=heaps))


class Mastermind(Problem):
    """Compute a strategy for winning in [mastermind](https://en.wikipedia.org/wiki/Mastermind_%28board_game%29)
     in a given number of guesses.

     Colors are represented by the letters A-F. The representation is as follows.
     A transcript is a string describing the game so far. It consists of rows separated by newlines.
     Each row has 4 letters A-F followed by a space and then two numbers indicating how many are exactly right
     and how many are right but in the wrong location. A sample transcript is as follows:
     ```
     AABB 11
     ABCD 21
     ABDC
     ```
     This is the transcript as the game is in progress. The complete transcript might be:
     ```
     AABB 11
     ABCD 21
     ABDC 30
     ABDE 40
     ```

     A winning strategy is described by a list of transcripts to visit. The next guess can be determined from
     those partial transcripts.
     """

    timeout = 10

    @staticmethod
    def sat(transcripts: List[str], max_moves=10):
        COLORS = "ABCDEF"

        def helper(secret: str, transcript=""):
            if transcript.count("\n") == max_moves:
                return False
            guess = min([t for t in transcripts if t.startswith(transcript)], key=len)[-4:]
            if guess == secret:
                return True
            assert all(g in COLORS for g in guess)
            perfect = {c: sum([g == s == c for g, s in zip(guess, secret)]) for c in COLORS}
            almost = sum(min(guess.count(c), secret.count(c)) - perfect[c] for c in COLORS)
            return helper(secret, transcript + f"{guess} {sum(perfect.values())}{almost}\n")

        return all(helper(r + s + t + u) for r in COLORS for s in COLORS for t in COLORS for u in COLORS)

    @staticmethod
    def sol(max_moves):
        COLORS = "ABCDEF"

        transcripts = []

        ALL = [r + s + t + u for r in COLORS for s in COLORS for t in COLORS for u in COLORS]

        def score(secret, guess):
            perfect = {c: sum([g == s == c for g, s in zip(guess, secret)]) for c in COLORS}
            almost = sum(min(guess.count(c), secret.count(c)) - perfect[c] for c in COLORS)
            return f"{sum(perfect.values())}{almost}"

        def mastermind(transcript="AABB", feasible=ALL):  # mastermind moves
            transcripts.append(transcript)
            assert transcript.count("\n") <= max_moves
            guess = transcript[-4:]
            feasibles = {}
            for secret in feasible:
                scr = score(secret, guess)
                if scr not in feasibles:
                    feasibles[scr] = []
                feasibles[scr].append(secret)
            for scr, secrets in feasibles.items():
                if scr != "40":
                    guesser(transcript + f" {scr}\n", secrets)

        def guesser(transcript, feasible):  # guesser moves
            def max_ambiguity(guess):
                by_score = {}
                for secret2 in feasible:
                    scr = score(secret2, guess)
                    if scr not in by_score:
                        by_score[scr] = 0
                    by_score[scr] += 1
                # for OPTIMAL solution, use return max(by_score.values()) + 0.5 * (guess not in feasible) instead of:
                return max(by_score.values())

            # for optimal solution use guess = min(ALL, key=max_ambiguity) instead of:
            guess = min(feasible, key=max_ambiguity)

            mastermind(transcript + guess, feasible)

        mastermind()

        return transcripts

    def gen(self, target_num_instances):
        for max_moves in [6, 8, 10]:
            self.add(dict(max_moves=max_moves))


class TicTacToeX(Problem):
    """Compute a strategy for X (first player) in tic-tac-toe that guarantees a tie.

    We are looking for a strategy for X that, no matter what the opponent does, X does not lose.

    A board is represented as a 9-char string like an X in the middle would be "....X...." and a
    move is an integer 0-8. The answer is a list of "good boards" that X aims for, so no matter what O does there
    is always good board that X can get to with a single move.
    """

    @staticmethod
    def sat(good_boards: List[str]):
        board_bit_reps = {tuple(sum(1 << i for i in range(9) if b[i] == c) for c in "XO") for b in good_boards}
        win = [any(i & w == w for w in [7, 56, 73, 84, 146, 273, 292, 448]) for i in range(512)]

        def tie(x, o):  # returns True if X has a forced tie/win assuming it's X's turn to move.
            x |= 1 << next(i for i in range(9) if (x | (1 << i), o) in board_bit_reps)
            return not win[o] and (win[x] or all((x | o) & (1 << i) or tie(x, o | (1 << i)) for i in range(9)))

        return tie(0, 0)

    @staticmethod
    def sol():
        win = [any(i & w == w for w in [7, 56, 73, 84, 146, 273, 292, 448]) for i in range(512)]  # 9-bit representation

        good_boards = []

        def x_move(x, o):  # returns True if x wins or ties, x's turn to move
            if win[o]:
                return False
            if x | o == 511:
                return True
            for i in range(9):
                if (x | o) & (1 << i) == 0 and o_move(x | (1 << i), o):
                    good_boards.append("".join(".XO"[((x >> j) & 1) + 2 * ((o >> j) & 1) + (i == j)] for j in range(9)))
                    return True
            return False  # O wins

        def o_move(x, o):  # returns True if x wins or ties, x's turn to move
            if win[x] or x | o == 511:
                return True
            for i in range(9):
                if (x | o) & (1 << i) == 0 and not x_move(x, o | (1 << i)):
                    return False
            return True  # O wins

        res = x_move(0, 0)
        assert res

        return good_boards


class TicTacToeO(Problem):
    """Compute a strategy for O (second player) in tic-tac-toe that guarantees a tie.

    We are looking for a strategy for O that, no matter what the opponent does, O does not lose.

    A board is represented as a 9-char string like an X in the middle would be "....X...." and a
    move is an integer 0-8. The answer is a list of "good boards" that O aims for, so no matter what X does there
    is always good board that O can get to with a single move.
    """

    @staticmethod
    def sat(good_boards: List[str]):
        board_bit_reps = {tuple(sum(1 << i for i in range(9) if b[i] == c) for c in "XO") for b in good_boards}
        win = [any(i & w == w for w in [7, 56, 73, 84, 146, 273, 292, 448]) for i in range(512)]

        def tie(x, o):  # returns True if O has a forced tie/win. It's O's turn to move.
            if o | x != 511:
                o |= 1 << next(i for i in range(9) if (x, o | (1 << i)) in board_bit_reps)
            return not win[x] and (win[o] or all((x | o) & (1 << i) or tie(x | (1 << i), o) for i in range(9)))

        return all(tie(1 << i, 0) for i in range(9))

    @staticmethod
    def sol():
        win = [any(i & w == w for w in [7, 56, 73, 84, 146, 273, 292, 448]) for i in range(512)]  # 9-bit representation

        good_boards = []

        def x_move(x, o):  # returns True if o wins or ties, x's turn to move
            if win[o] or x | o == 511:
                return True
            for i in range(9):
                if (x | o) & (1 << i) == 0 and not o_move(x | (1 << i), o):
                    return False
            return True  # O wins/ties

        def o_move(x, o):  # returns True if o wins or ties, o's turn to move
            if win[x]:
                return False
            if x | o == 511:
                return True
            for i in range(9):
                if (x | o) & (1 << i) == 0 and x_move(x, o | (1 << i)):
                    good_boards.append(
                        "".join(".XO"[((x >> j) & 1) + 2 * ((o >> j) & 1) + 2 * (i == j)] for j in range(9)))
                    return True
            return False  # X wins

        res = x_move(0, 0)
        assert res

        return good_boards


class RockPaperScissors(Problem):
    """Find optimal strategy for Rock-Paper-Scissors zero-sum game

    Find the distribution that guarantees maximum expected value of 0
    """

    @staticmethod
    def sat(probs: List[float]):  # rock prob, paper prob, scissors prob
        assert len(probs) == 3 and abs(sum(probs) - 1) < 1e-6
        return max(probs[(i + 2) % 3] - probs[(i + 1) % 3] for i in range(3)) < 1e-6

    @staticmethod
    def sol():
        return [1 / 3] * 3


if __name__ == "__main__":
    Problem.debug_problems()
