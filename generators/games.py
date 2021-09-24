"""
Some two-player game problems and hard game theory problems
"""

from puzzle_generator import PuzzleGenerator
from typing import List


# See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle to learn about adding puzzles


class Nim(PuzzleGenerator):
    """
    Compute optimal play for the classic two-player game [Nim](https://en.wikipedia.org/wiki/Nim)

    Nim has an elegant theory for optimal play based on the xor of the bits in the heaps.

    Instead of writing a program that plays the game interactively (since interaction is not allowed), we require
    them to determine winning states or beat a certain opponent.
    """

    value_multiplier = 10  # harder than most problems, worth more

    @staticmethod
    def sat(moves: List[List[int]], initial_state=[5, 9, 3, 11, 18, 25, 1, 2, 4, 1]):
        """
        Beat a bot at Nim, a two-player game involving a number of heaps of objects. Players alternate, in each turn
        removing one or more objects from a single non-empty heap. The player who takes the last object wins.
        - initial_state is list of numbers of objects in each heap
        - moves is a list of your moves: [heap, number of objects to take]
        - you play first
        """

        def bot_move():  # bot takes objects from the largest heap to make it match the second largest heap
            vals = sorted(state, reverse=True)
            i_largest = state.index(vals[0])  # largest heap
            state[i_largest] -= max(vals[0] - vals[1], 1)  # must take some, take 1 in case of tie

        state = initial_state[:]  # copy
        for i, n in moves:
            assert 0 < n <= state[i], "Illegal move"
            state[i] -= n
            if set(state) == {0}:
                return True  # you won!
            assert any(state), "You lost!"
            bot_move()

    @staticmethod
    def sol(initial_state):

        state = initial_state[:]
        moves = []

        def bot_move():  # bot takes objects from the largest heap to make it match the second largest heap
            vals = sorted(state, reverse=True)
            i_largest = state.index(vals[0])  # largest heap
            state[i_largest] -= max(vals[0] - vals[1], 1)  # must take some, take 1 in case of tie

        def losing(h):  # return True if h is a losing state
            xor = 0
            for i in h:
                xor ^= i
            return xor == 0

        def optimal_move():
            assert not losing(state)
            for i in range(len(state)):
                for n in range(1, state[i] + 1):
                    state[i] -= n
                    if losing(state):
                        moves.append([i, n])
                        return
                    state[i] += n
            assert False, "Shouldn't reach hear"

        while True:
            optimal_move()
            if max(state) == 0:
                return moves
            bot_move()

    def gen_random(self):
        def losing(h):  # return True if h is a losing state
            xor = 0
            for i in h:
                xor ^= i
            return xor == 0

        num_heaps = self.random.randrange(10)
        initial_state = [self.random.randrange(10) for _ in range(num_heaps)]
        if losing(initial_state):
            return
        prod = 1
        for i in initial_state:
            prod *= i + 1
        if prod < 10 ** 6:
            self.add(dict(initial_state=initial_state))


class Mastermind(PuzzleGenerator):
    """Compute a strategy for winning in [mastermind](https://en.wikipedia.org/wiki/Mastermind_%28board_game%29)
    in a given number of guesses.

    Instead of writing a program that plays the game interactively (since interaction is not allowed), we require
    them to provide a provable winning game tree.
    """

    multiplier = 10  # hard puzzle, takes longer to test

    @staticmethod
    def sat(transcripts: List[str], max_moves=10):
        """
        Come up with a winning strategy for Mastermind in max_moves moves. Colors are represented by the letters A-F.
        The solution representation is as follows.
        A transcript is a string describing the game so far. It consists of rows separated by newlines.
        Each row has 4 letters A-F followed by a space and then two numbers indicating how many are exactly right
        and how many are right but in the wrong location. A sample transcript is as follows:
        AABB 11
        ABCD 21
        ABDC

        This is the transcript as the game is in progress. The complete transcript might be:
        AABB 11
        ABCD 21
        ABDC 30
        ABDE 40

        A winning strategy is described by a list of transcripts to visit. The next guess can be determined from
        those partial transcripts.
        """
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


class TicTacToeX(PuzzleGenerator):
    """Since we don't have interaction, this problem asks for a full tie-guranteeing strategy."""

    @staticmethod
    def sat(good_boards: List[str]):
        """
        Compute a strategy for X (first player) in tic-tac-toe that guarantees a tie. That is a strategy for X that,
        no matter what the opponent does, X does not lose.

        A board is represented as a 9-char string like an X in the middle would be "....X...." and a
        move is an integer 0-8. The answer is a list of "good boards" that X aims for, so no matter what O does there
        is always good board that X can get to with a single move.
        """
        board_bit_reps = {tuple(sum(1 << i for i in range(9) if b[i] == c) for c in "XO") for b in good_boards}
        win = [any(i & w == w for w in [7, 56, 73, 84, 146, 273, 292, 448]) for i in range(512)]

        def tie(x, o):  # returns True if X has a forced tie/win assuming it's X's turn to move.
            x |= 1 << [i for i in range(9) if (x | (1 << i), o) in board_bit_reps][0]
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
            if win[x] or x | o == 511:  # full board
                return True
            for i in range(9):
                if (x | o) & (1 << i) == 0 and not x_move(x, o | (1 << i)):
                    return False
            return True  # O wins

        res = x_move(0, 0)
        assert res

        return good_boards


class TicTacToeO(PuzzleGenerator):
    """Same as above but for 2nd player"""

    @staticmethod
    def sat(good_boards: List[str]):
        """
        Compute a strategy for O (second player) in tic-tac-toe that guarantees a tie. That is a strategy for O that,
        no matter what the opponent does, O does not lose.

        A board is represented as a 9-char string like an X in the middle would be "....X...." and a
        move is an integer 0-8. The answer is a list of "good boards" that O aims for, so no matter what X does there
        is always good board that O can get to with a single move.
        """
        board_bit_reps = {tuple(sum(1 << i for i in range(9) if b[i] == c) for c in "XO") for b in good_boards}
        win = [any(i & w == w for w in [7, 56, 73, 84, 146, 273, 292, 448]) for i in range(512)]

        def tie(x, o):  # returns True if O has a forced tie/win. It's O's turn to move.
            if o | x != 511:  # complete board
                o |= 1 << [i for i in range(9) if (x, o | (1 << i)) in board_bit_reps][0]
            return not win[x] and (win[o] or all((x | o) & (1 << i) or tie(x | (1 << i), o) for i in range(9)))

        return all(tie(1 << i, 0) for i in range(9))

    @staticmethod
    def sol():
        win = [any(i & w == w for w in [7, 56, 73, 84, 146, 273, 292, 448]) for i in range(512)]  # 9-bit representation

        good_boards = []

        def x_move(x, o):  # returns True if o wins or ties, x's turn to move
            if win[o] or x | o == 511:  # full board
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


class RockPaperScissors(PuzzleGenerator):
    @staticmethod
    def sat(probs: List[float]):
        """Find optimal probabilities for playing Rock-Paper-Scissors zero-sum game, with best worst-case guarantee"""
        assert len(probs) == 3 and abs(sum(probs) - 1) < 1e-6
        return max(probs[(i + 2) % 3] - probs[(i + 1) % 3] for i in range(3)) < 1e-6

    @staticmethod
    def sol():
        return [1 / 3] * 3



class Nash(PuzzleGenerator):
    """Computing a [Nash equilibrium](https://en.wikipedia.org/wiki/Nash_equilibrium) for a given
     [bimatrix game](https://en.wikipedia.org/wiki/Bimatrix_game) is known to be
     PPAD-hard in general. However, the challenge is be much easier for an approximate
     [eps-equilibrium](https://en.wikipedia.org/wiki/Epsilon-equilibrium) and of course for small games."""

    multiplier = 10

    @staticmethod
    def sat(strategies: List[List[float]], A = [[1.0, -1.0], [-1.3, 0.8]], B = [[-0.9, 1.1], [0.7, -0.8]], eps=0.01):
        """
        Find an eps-Nash-equilibrium for a given two-player game with payoffs described by matrices A, B.
        For example, for the classic Prisoner dilemma:
           A=[[-1., -3.], [0., -2.]], B=[[-1., 0.], [-3., -2.]], and strategies = [[0, 1], [0, 1]]

        eps is the error tolerance
        """
        m, n = len(A), len(A[0])
        p, q = strategies
        assert len(B) == m and all(len(row) == n for row in A + B), "inputs are a bimatrix game"
        assert len(p) == m and len(q) == n, "solution is a pair of strategies"
        assert sum(p) == sum(q) == 1.0 and min(p + q) >= 0.0, "strategies must be non-negative and sum to 1"
        v = sum(A[i][j] * p[i] * q[j] for i in range(m) for j in range(n))
        w = sum(B[i][j] * p[i] * q[j] for i in range(m) for j in range(n))
        return (all(sum(A[i][j] * q[j] for j in range(n)) <= v + eps for i in range(m)) and
                all(sum(B[i][j] * p[i] for i in range(m)) <= w + eps for j in range(n)))

    @staticmethod
    def sol(A, B, eps):
        NUM_ATTEMPTS = 10 ** 5

        def sat(strategies: List[List[float]], A, B, eps):
            m, n = len(A), len(A[0])
            p, q = strategies
            assert len(B) == m and all(len(row) == n for row in A + B), "inputs are a bimatrix game"
            assert len(p) == m and len(q) == n, "solution is a pair of strategies"
            assert sum(p) == sum(q) == 1.0 and min(p + q) >= 0.0, "strategies must be non-negative and sum to 1"
            v = sum(A[i][j] * p[i] * q[j] for i in range(m) for j in range(n))
            w = sum(B[i][j] * p[i] * q[j] for i in range(m) for j in range(n))
            return (all(sum(A[i][j] * q[j] for j in range(n)) <= v + eps for i in range(m)) and
                    all(sum(B[i][j] * p[i] for i in range(m)) <= w + eps for j in range(n)))

        import random
        r = random.Random(0)
        dims = len(A), len(A[0])
        # possible speedup: remove dominated strategies
        for _attempt in range(NUM_ATTEMPTS):
            strategies = []
            for d in dims:
                s = [max(0.0, r.random() - 0.5) for _ in range(d)]
                tot = sum(s) + 1e-6
                for i in range(d):
                    s[i] = (1.0 - sum(s[:-1])) if i == d - 1 else (s[i] / tot)  # to ensure sum is exactly 1.0
                strategies.append(s)
            if sat(strategies, A, B, eps):
                return strategies

    def gen_random(self):
        m = self.random.randrange(2, 10)
        n = self.random.randrange(2, 10)
        A, B = [[[self.random.random() for _i in range(m)] for _j in range(n)] for _k in range(2)]
        eps = self.random.choice([0.5, 0.1, 0.01])
        solved = self.sol(A, B, eps) is not None
        self.add(dict(A=A, B=B, eps=eps), test=solved)


class ZeroSum(PuzzleGenerator):
    """Compute minimax optimal strategies for a given
     [zero-sum game](https://en.wikipedia.org/wiki/Zero-sum_game). This problem is known to be equivalent to
     Linear Programming. Note that the provided instances are all quite easy---harder solutions could readily
     be made by decreasing the accuracy tolerance `eps` at which point the solution we provided would fail and
     more efficient algorithms would be needed."""

    @staticmethod
    def sat(strategies: List[List[float]], A = [[0., -0.5, 1.], [0.75, 0., -1.], [-1., 0.4, 0.]], eps=0.01):
        """
        Compute minimax optimal strategies for a given zero-sum game up to error tolerance eps.
        For example, rock paper scissors has
        A = [[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]] and strategies = [[0.33, 0.33, 0.34]] * 2
        """
        m, n = len(A), len(A[0])
        p, q = strategies
        assert all(len(row) == n for row in A), "inputs are a matrix"
        assert len(p) == m and len(q) == n, "solution is a pair of strategies"
        assert sum(p) == sum(q) == 1.0 and min(p + q) >= 0.0, "strategies must be non-negative and sum to 1"
        v = sum(A[i][j] * p[i] * q[j] for i in range(m) for j in range(n))
        return (all(sum(A[i][j] * q[j] for j in range(n)) <= v + eps for i in range(m)) and
                all(sum(A[i][j] * p[i] for i in range(m)) >= v - eps for j in range(n)))

    @staticmethod
    def sol(A, eps):
        MAX_ITER = 10**4
        m, n = len(A), len(A[0])
        a = [0 for _i in range(m)]
        b = [0 for _j in range(n)]

        for count in range(1, MAX_ITER):
            i_star = max(range(m), key=lambda i: sum(A[i][j] * b[j] for j in range(n)))
            j_star = min(range(n), key=lambda j: sum(A[i][j] * a[i] for i in range(m)))
            a[i_star] += 1
            b[j_star] += 1
            p = [x / (count + 1e-6) for x in a]
            p[-1] = 1 - sum(p[:-1])  # rounding issues
            q = [x / (count + 1e-6) for x in b]
            q[-1] = 1 - sum(q[:-1])  # rounding issues

            v = sum(A[i][j] * p[i] * q[j] for i in range(m) for j in range(n))
            if (all(sum(A[i][j] * q[j] for j in range(n)) <= v + eps for i in range(m)) and
                    all(sum(A[i][j] * p[i] for i in range(m)) >= v - eps for j in range(n))):
                return [p, q]

    def gen_random(self):
        m = self.random.randrange(2, 10)
        n = self.random.randrange(2, 10)
        A = [[self.random.random() for _i in range(m)] for _j in range(n)]
        eps = self.random.choice([0.5, 0.1, 0.01])
        test = self.sol(A, eps) is not None
        self.add(dict(A=A, eps=eps), test=test)


if __name__ == "__main__":
    PuzzleGenerator.debug_problems()
