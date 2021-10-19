# Summary of Puzzles
This document summarizes the dataset stored in the `puzzles.json` file in this directory. 
These files are generated from the `generators/*.py` files.
The only import for puzzles is `from typing import List` but you should also pass a candidate solution 
through `type_check` from `puzzle_generator.py` before certifying correctness. 

## Puzzles by module: <!-- descriptions come from the module docstring --> 

- [study.py](../generators/study.py), [30 problems](#study): Puzzles used in our user study (the user study didn't have docstrings), see [Programming Puzzles](https://arxiv.org/abs/2106.05784)
- [classic_puzzles.py](../generators/classic_puzzles.py), [23 problems](#classic_puzzles): Classic puzzles
- [human_eval.py](../generators/human_eval.py), [164 problems](#human_eval): Problems inspired by [HumanEval dataset](https://github.com/openai/human-eval) described
in the [codex paper](https://arxiv.org/abs/2107.03374), specifically,
[this](https://github.com/openai/human-eval/blob/fa06031e684fbe1ee429c7433809460c159b66ad/data/HumanEval.jsonl.gz)
version released 7/7/21
- [codeforces.py](../generators/codeforces.py), [47 problems](#codeforces): Problems inspired by the popular programming competition site [codeforces.com](https://codeforces.com)
- [algebra.py](../generators/algebra.py), [4 problems](#algebra): Roots of polynomials
- [basic.py](../generators/basic.py), [23 problems](#basic): Problems testing basic knowledge -- easy to solve if you understand what is being asked
- [chess.py](../generators/chess.py), [5 problems](#chess): Classic chess puzzles
- [compression.py](../generators/compression.py), [2 problems](#compression): Puzzles relating to de/compression
- [conways_game_of_life.py](../generators/conways_game_of_life.py), [3 problems](#conways_game_of_life): Conway's Game of Life problems (see https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
- [games.py](../generators/games.py), [7 problems](#games): Some two-player game problems and hard game theory problems
- [graphs.py](../generators/graphs.py), [12 problems](#graphs): Problems related to graphs such as Conway's 99 problem, finding
[cliques](https://en.wikipedia.org/wiki/Clique_(graph_theory)) of various sizes, shortest path (Dijkstra)
- [ICPC.py](../generators/ICPC.py), [4 problems](#icpc): Problems inspired by the [International Collegiate Programming Contest](https://icpc.global) (ICPC)
- [IMO.py](../generators/IMO.py), [6 problems](#imo): Problems inspired by the
[International Mathematical Olympiad](https://en.wikipedia.org/wiki/International_Mathematical_Olympiad)
[problems](https://www.imo-official.org/problems.aspx)
- [lattices.py](../generators/lattices.py), [2 problems](#lattices): Lattice problems with and without noise
- [number_theory.py](../generators/number_theory.py), [16 problems](#number_theory): Number theory problems
- [probability.py](../generators/probability.py), [5 problems](#probability): Probability problems
- [trivial_inverse.py](../generators/trivial_inverse.py), [39 problems](#trivial_inverse): Trivial problems. Typically for any function, you can construct a trivial example.
For instance, for the len function you can ask for a string of len(s)==100 etc
- [tutorial.py](../generators/tutorial.py), [5 problems](#tutorial): A few example puzzles that were presented with solutions to participants of the study

Total (397 problems, 1,715 instances)


----

## study


Puzzles used in our user study (the user study didn't have docstrings), see [Programming Puzzles](https://arxiv.org/abs/2106.05784).


* <a name="study_1"></a>**Study_1**  (1 instance)
    
    ```python
    def sat(s: str):
        return s.count('o') == 1000 and s.count('oo') == 0
    
    def sol():
        """Find a string with 1000 'o's but no two adjacent 'o's."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return ('h' + 'o') * 1000
    ```
    
    </details>
    
* <a name="study_2"></a>**Study_2**  (1 instance)
    
    ```python
    def sat(s: str):
        return s.count('o') == 1000 and s.count('oo') == 100 and s.count('ho') == 801
    
    def sol():
        """Find a string with 1000 'o's, 100 pairs of adjacent 'o's and 801 copies of 'ho'."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return 'ho' * (800 + 1) + 'o' * (100 * 2 - 1)
    ```
    
    </details>
    
* <a name="study_3"></a>**Study_3**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        return sorted(li) == list(range(999)) and all(li[i] != i for i in range(len(li)))
    
    def sol():
        """Find a permutation of [0, 1, ..., 998] such that the ith element is *not* i, for all i=0, 1, ..., 998."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [((i + 1) % 999) for i in range(999)]
    ```
    
    </details>
    
* <a name="study_4"></a>**Study_4**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        return len(li) == 10 and li.count(li[3]) == 2
    
    def sol():
        """Find a list of length 10 where the fourth element occurs exactly twice."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return list(range(10 // 2)) * 2
    ```
    
    </details>
    
* <a name="study_5"></a>**Study_5**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        return all([li.count(i) == i for i in range(10)])
    
    def sol():
        """Find a list integers such that the integer i occurs i times, for i = 0, 1, 2, ..., 9."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [i for i in range(10) for j in range(i)]
    ```
    
    </details>
    
* <a name="study_6"></a>**Study_6**  (1 instance)
    
    ```python
    def sat(i: int):
        return i % 123 == 4 and i > 10 ** 10
    
    def sol():
        """Find an integer greater than 10^10 which is 4 mod 123."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return 4 + 10 ** 10 + 123 - 10 ** 10 % 123
    ```
    
    </details>
    
* <a name="study_7"></a>**Study_7**  (1 instance)
    
    ```python
    def sat(s: str):
        return str(8 ** 2888).count(s) > 8 and len(s) == 3
    
    def sol():
        """Find a three-digit pattern  that occurs more than 8 times in the decimal representation of 8^2888."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        s = str(8 ** 2888)
        return max({s[i: i + 3] for i in range(len(s) - 2)}, key=lambda t: s.count(t))
    ```
    
    </details>
    
* <a name="study_8"></a>**Study_8**  (1 instance)
    
    ```python
    def sat(ls: List[str]):
        return ls[1234] in ls[1235] and ls[1234] != ls[1235]
    
    def sol():
        """Find a list of more than 1235 strings such that the 1234th string is a proper substring of the 1235th."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [''] * 1235 + ['a']
    ```
    
    </details>
    
* <a name="study_9"></a>**Study_9**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        return ["The quick brown fox jumps over the lazy dog"[i] for i in li] == list(
            "The five boxing wizards jump quickly")
    
    def sol():
        """
        Find a way to rearrange the letters in the pangram "The quick brown fox jumps over the lazy dog" to get
        the pangram "The five boxing wizards jump quickly". The answer should be represented as a list of index
        mappings.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return ['The quick brown fox jumps over the lazy dog'.index(t)
                for t in 'The five boxing wizards jump quickly']
    ```
    
    </details>
    
* <a name="study_10"></a>**Study_10**  (1 instance)
    
    ```python
    def sat(s: str):
        return s in str(8 ** 1818) and s == s[::-1] and len(s) > 11
    
    def sol():
        """Find a palindrome of length greater than 11 in the decimal representation of 8^1818."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        s = str(8 ** 1818)
        return next(s[i: i + le]
                    for le in range(12, len(s) + 1)
                    for i in range(len(s) - le + 1)
                    if s[i: i + le] == s[i: i + le][::-1]
                    )
    ```
    
    </details>
    
* <a name="study_11"></a>**Study_11**  (1 instance)
    
    ```python
    def sat(ls: List[str]):
        return min(ls) == max(ls) == str(len(ls))
    
    def sol():
        """
        Find a list of strings whose length (viewed as a string) is equal to the lexicographically largest element
        and is equal to the lexicographically smallest element.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return ['1']
    ```
    
    </details>
    
* <a name="study_12"></a>**Study_12**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        return all(i + j == 9 for i, j in zip([4] + li, li)) and len(li) == 1000
    
    def sol():
        """Find a list of 1,000 integers where every two adjacent integers sum to 9, and where the first
        integer plus 4 is 9."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [9 - 4, 4] * (1000 // 2)
    ```
    
    </details>
    
* <a name="study_13"></a>**Study_13**  (1 instance)
    
    ```python
    def sat(x: float):
        return str(x - 3.1415).startswith("123.456")
    
    def sol():
        """Find a real number which, when you subtract 3.1415, has a decimal representation starting with 123.456."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return 123.456 + 3.1415
    ```
    
    </details>
    
* <a name="study_14"></a>**Study_14**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        return all([sum(li[:i]) == i for i in range(20)])
    
    def sol():
        """Find a list of integers such that the sum of the first i integers is i, for i=0, 1, 2, ..., 19."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [1] * 20
    ```
    
    </details>
    
* <a name="study_15"></a>**Study_15**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        return all(sum(li[:i]) == 2 ** i - 1 for i in range(20))
    
    def sol():
        """Find a list of integers such that the sum of the first i integers is 2^i -1, for i = 0, 1, 2, ..., 19."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [(2 ** i) for i in range(20)]
    ```
    
    </details>
    
* <a name="study_16"></a>**Study_16**  (1 instance)
    
    ```python
    def sat(s: str):
        return float(s) + len(s) == 4.5
    
    def sol():
        """Find a real number such that when you add the length of its decimal representation to it, you get 4.5.
        Your answer should be the string form of the number in its decimal representation."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return str(4.5 - len(str(4.5)))
    ```
    
    </details>
    
* <a name="study_17"></a>**Study_17**  (1 instance)
    
    ```python
    def sat(i: int):
        return len(str(i + 1000)) > len(str(i + 1001))
    
    def sol():
        """Find a number whose decimal representation is *a longer string* when you add 1,000 to it than when you add 1,001."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return -1001
    ```
    
    </details>
    
* <a name="study_18"></a>**Study_18**  (1 instance)
    
    ```python
    def sat(ls: List[str]):
        return [s + t for s in ls for t in ls if s != t] == 'berlin berger linber linger gerber gerlin'.split()
    
    def sol():
        """
        Find a list of strings that when you combine them in all pairwise combinations gives the six strings:
        'berlin', 'berger', 'linber', 'linger', 'gerber', 'gerlin'
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        seen = set()
        ans = []
        for s in 'berlin berger linber linger gerber gerlin'.split():
            t = s[:3]
            if t not in seen:
                ans.append(t)
                seen.add(t)
        return ans
    ```
    
    </details>
    
* <a name="study_19"></a>**Study_19** 9/15/2021 Updated to take a list rather than a set because it was the only puzzle in the repo with Set argument. (1 instance)
    
    ```python
    def sat(li: List[int]):
        return {i + j for i in li for j in li} == {0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 34}
    
    def sol():
        """
        Find a list of integers whose pairwise sums make the set {0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 34}.
        That is find L such that, { i + j | i, j in L } = {0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 34}.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [0, 1, 2, 3, 17]
    ```
    
    </details>
    
* <a name="study_20"></a>**Study_20** A more interesting version of this puzzle with a length constraint is ShortIntegerPath in graphs.py (1 instance)
    
    ```python
    def sat(li: List[int]):
        return all(j in {i - 1, i + 1, 3 * i} for i, j in zip([0] + li, li + [128]))
    
    def sol():
        """
        Find a list of integers, starting with 0 and ending with 128, such that each integer either differs from
        the previous one by one or is thrice the previous one.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [1, 3, 4, 12, 13, 14, 42, 126, 127]
    ```
    
    </details>
    
* <a name="study_21"></a>**Study_21**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        return all([li[i] != li[i + 1] for i in range(10)]) and len(set(li)) == 3
    
    def sol():
        """
        Find a list integers containing exactly three distinct values, such that no integer repeats
        twice consecutively among the first eleven entries. (So the list needs to have length greater than ten.)
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return list(range(3)) * 10
    ```
    
    </details>
    
* <a name="study_22"></a>**Study_22**  (1 instance)
    
    ```python
    def sat(s: str):
        return s[::2] in s and len(set(s)) == 5
    
    def sol():
        """
        Find a string s containing exactly five distinct characters which also contains as a substring every other
        character of s (e.g., if the string s were 'parrotfish' every other character would be 'profs').
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return """abacadaeaaaaaaaaaa"""
    ```
    
    </details>
    
* <a name="study_23"></a>**Study_23**  (1 instance)
    
    ```python
    def sat(ls: List[str]):
        return tuple(ls) in zip('dee', 'doo', 'dah!')
    
    def sol():
        """
        Find a list of characters which are aligned at the same indices of the three strings 'dee', 'doo', and 'dah!'.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return list(next(zip('dee', 'doo', 'dah!')))
    ```
    
    </details>
    
* <a name="study_24"></a>**Study_24**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        return li.count(17) == 3 and li.count(3) >= 2
    
    def sol():
        """Find a list of integers with exactly three occurrences of seventeen and at least two occurrences of three."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [17] * 3 + [3] * 2
    ```
    
    </details>
    
* <a name="study_25"></a>**Study_25**  (1 instance)
    
    ```python
    def sat(s: str):
        return sorted(s) == sorted('Permute me true') and s == s[::-1]
    
    def sol():
        """Find a permutation of the string 'Permute me true' which is a palindrome."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        s = sorted('Permute me true'[1:])[::2]
        return "".join(s + ['P'] + s[::-1])
    ```
    
    </details>
    
* <a name="study_26"></a>**Study_26**  (1 instance)
    
    ```python
    def sat(ls: List[str]):
        return "".join(ls) == str(8 ** 88) and all(len(s) == 8 for s in ls)
    
    def sol():
        """Divide the decimal representation of 8^88 up into strings of length eight."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [str(8 ** 88)[i:i + 8] for i in range(0, len(str(8 ** 88)), 8)]
    ```
    
    </details>
    
* <a name="study_27"></a>**Study_27**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        return li[li[0]] != li[li[1]] and li[li[li[0]]] == li[li[li[1]]]
    
    def sol():
        """
        Consider a digraph where each node has exactly one outgoing edge. For each edge (u, v), call u the parent and
        v the child. Then find such a digraph where the grandchildren of the first and second nodes differ but they
        share the same great-grandchildren. Represented this digraph by the list of children indices.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [1, 2, 3, 3]
    ```
    
    </details>
    
* <a name="study_28"></a>**Study_28** 9/15/2021: updated to a list since sets were removed from puzzle formats (1 instance)
    
    ```python
    def sat(li: List[int]):
        return all(i in range(1000) and abs(i - j) >= 10 for i in li for j in li if i != j) and len(set(li)) == 100
    
    def sol():
        """Find a list of one hundred integers between 0 and 999 which all differ by at least ten from one another."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return list(range(0, 1000, 10))
    ```
    
    </details>
    
* <a name="study_29"></a>**Study_29** 9/15/2021: updated to a list since sets were removed from puzzle formats (1 instance)
    
    ```python
    def sat(l: List[int]):
        return all(i in range(1000) and abs(i * i - j * j) >= 10 for i in l for j in l if i != j) and len(set(l)) > 995
    
    def sol():
        """
        Find a list of more than 995 distinct integers between 0 and 999, inclusive, such that each pair of integers
        have squares that differ by at least 10.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [0, 4] + list(range(6, 1000))
    ```
    
    </details>
    
* <a name="study_30"></a>**Study_30**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        return all([123 * li[i] % 1000 < 123 * li[i + 1] % 1000 and li[i] in range(1000) for i in range(20)])
    
    def sol():
        """
        Define f(n) to be the residue of 123 times n mod 1000. Find a list of integers such that the first twenty one
        are between 0 and 999, inclusive, and are strictly increasing in terms of f(n).
        """
    ```
    <details><summary>2 hand-written solutions </summary>
    
    ```python
        return sorted(range(1000), key=lambda n: 123 * n % 1000)[:21]
    ```
    
    ```python
        return list(range(1000))[::8][::-1]
    ```
    
    </details>
    
## classic_puzzles

Classic puzzles


* <a name="towersofhanoi"></a>**TowersOfHanoi** [Towers of Hanoi](https://en.wikipedia.org/w/index.php?title=Tower_of_Hanoi)
    
    In this classic version one must move all 8 disks from the first to third peg. (1 instance)
    
    ```python
    def sat(moves: List[List[int]]):
        rods = ([8, 7, 6, 5, 4, 3, 2, 1], [], [])
        for [i, j] in moves:
            rods[j].append(rods[i].pop())
            assert rods[j][-1] == min(rods[j]), "larger disk on top of smaller disk"
        return rods[0] == rods[1] == []
    
    def sol():
        """
        Eight disks of sizes 1-8 are stacked on three towers, with each tower having disks in order of largest to
        smallest. Move [i, j] corresponds to taking the smallest disk off tower i and putting it on tower j, and it
        is legal as long as the towers remain in sorted order. Find a sequence of moves that moves all the disks
        from the first to last towers.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        def helper(m, i, j):
            if m == 0:
                return []
            k = 3 - i - j
            return helper(m - 1, i, k) + [[i, j]] + helper(m - 1, k, j)
    
        return helper(8, 0, 2)
    ```
    
    </details>
    
* <a name="towersofhanoiarbitrary"></a>**TowersOfHanoiArbitrary** [Towers of Hanoi](https://en.wikipedia.org/w/index.php?title=Tower_of_Hanoi)
    
    In this version one must transform a given source state to a target state. (5 instances)
    
    ```python
    def sat(moves: List[List[int]], source=[[0, 7], [4, 5, 6], [1, 2, 3, 8]], target=[[0, 1, 2, 3, 8], [4, 5], [6, 7]]):
        state = [s[:] for s in source]
    
        for [i, j] in moves:
            state[j].append(state[i].pop())
            assert state[j] == sorted(state[j])
    
        return state == target
    
    def sol(source=[[0, 7], [4, 5, 6], [1, 2, 3, 8]], target=[[0, 1, 2, 3, 8], [4, 5], [6, 7]]):
        """
        A state is a partition of the integers 0-8 into three increasing lists. A move is pair of integers i, j in
        {0, 1, 2} corresponding to moving the largest number from the end of list i to list j, while preserving the
        order of list j. Find a sequence of moves that transform the given source to target states.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        state = {d: i for i, tower in enumerate(source) for d in tower}
        final = {d: i for i, tower in enumerate(target) for d in tower}
        disks = set(state)
        assert disks == set(final) and all(isinstance(i, int) for i in state) and len(source) == len(target) >= 3
        ans = []
    
        def move(d, i):  # move disk d to tower i
            if state[d] == i:
                return
            for t in range(3):  # first tower besides i, state[d]
                if t != i and t != state[d]:
                    break
            for d2 in range(d + 1, max(disks) + 1):
                if d2 in disks:
                    move(d2, t)
            ans.append([state[d], i])
            state[d] = i
    
        for d in range(min(disks), max(disks) + 1):
            if d in disks:
                move(d, final[d])
    
        return ans
    ```
    
    </details>
    
* <a name="longestmonotonicsubstring"></a>**LongestMonotonicSubstring** This is a form of the classic
    [Longest increasing subsequence](https://en.wikipedia.org/wiki/Longest_increasing_subsequence) problem
    where the goal is to find a substring with characters in sorted order. (5 instances)
    
    ```python
    def sat(x: List[int], length=13, s="Dynamic programming solves this puzzle!!!"):
        return all(s[x[i]] <= s[x[i + 1]] and x[i + 1] > x[i] >= 0 for i in range(length - 1))
    
    def sol(length=13, s="Dynamic programming solves this puzzle!!!"):
        """
        Remove as few characters as possible from s so that the characters of the remaining string are alphebetical.
        Here x is the list of string indices that have not been deleted.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        # O(N^2) method. Todo: add binary search solution which is O(n log n)
        if s == "":
            return []
        n = len(s)
        dyn = []  # list of (seq length, seq end, prev index)
        for i in range(n):
            try:
                dyn.append(max((length + 1, i, e) for length, e, _ in dyn if s[e] <= s[i]))
            except ValueError:
                dyn.append((1, i, -1))  # sequence ends at i
        _length, i, _ = max(dyn)
        backwards = [i]
        while dyn[i][2] != -1:
            i = dyn[i][2]
            backwards.append(i)
        return backwards[::-1]
    ```
    
    </details>
    
* <a name="longestmonotonicsubstringtricky"></a>**LongestMonotonicSubstringTricky** The same as the above problem, but with a twist! (5 instances)
    
    ```python
    def sat(x: List[int], length=20, s="Dynamic programming solves this classic job-interview puzzle!!!"):
        return all(s[x[i]] <= s[x[i + 1]] and x[i + 1] > x[i] for i in range(length - 1))
    
    def sol(length=20, s="Dynamic programming solves this classic job-interview puzzle!!!"):
        """Find the indices of the longest substring with characters in sorted order"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        # O(N^2) method. Todo: add binary search solution which is O(n log n)
        if s == "":
            return []
        n = len(s)
        dyn = []  # list of (seq length, seq end, prev index)
        for i in range(-n, n):
            try:
                dyn.append(max((length + 1, i, e) for length, e, _ in dyn if s[e] <= s[i]))
            except ValueError:
                dyn.append((1, i, None))  # sequence ends at i
        _length, i, _ = max(dyn)
        backwards = [i]
        while dyn[n + i][2] is not None:
            i = dyn[n + i][2]
            backwards.append(i)
        return backwards[::-1]
    ```
    
    </details>
    
* <a name="quine"></a>**Quine** [Quine](https://en.wikipedia.org/wiki/Quine_%28computing%29) (1 instance)
    
    ```python
    def sat(quine: str):
        return eval(quine) == quine
    
    def sol():
        """Find a string that when evaluated as a Python expression is that string itself."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "(lambda x: f'({x})({chr(34)}{x}{chr(34)})')(\"lambda x: f'({x})({chr(34)}{x}{chr(34)})'\")"
    ```
    
    </details>
    
* <a name="revquine"></a>**RevQuine** Reverse [Quine](https://en.wikipedia.org/wiki/Quine_%28computing%29). The solution we give is from GPT3. (1 instance)
    
    ```python
    def sat(rev_quine: str):
        return eval(rev_quine[::-1]) == rev_quine
    
    def sol():
        """Find a string that, when reversed and evaluated gives you back that same string."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "rev_quine"[::-1]  # thanks GPT-3!
    ```
    
    </details>
    
* <a name="booleanpythagoreantriples"></a>**BooleanPythagoreanTriples** [Boolean Pythagorean Triples Problem](https://en.wikipedia.org/wiki/Boolean_Pythagorean_triples_problem) (4 instances)
    
    ```python
    def sat(colors: List[int], n=100):
        assert set(colors) <= {0, 1} and len(colors) >= n
        squares = {i ** 2: colors[i] for i in range(1, len(colors))}
        return not any(c == d == squares.get(i + j) for i, c in squares.items() for j, d in squares.items())
    
    def sol(n=100):
        """
        Color the first n integers with one of two colors so that there is no monochromatic Pythagorean triple.
        A monochromatic Pythagorean triple is a triple of numbers i, j, k such that i^2 + j^2 = k^2 that
        are all assigned the same color. The input, colors, is a list of 0/1 colors of length >= n.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        sqrt = {i * i: i for i in range(1, n)}
        trips = [(sqrt[i], sqrt[j], sqrt[i + j]) for i in sqrt for j in sqrt if i < j and i + j in sqrt]
        import random
        random.seed(0)
        sol = [random.randrange(2) for _ in range(n)]
        done = False
        while not done:
            done = True
            random.shuffle(trips)
            for i, j, k in trips:
                if sol[i] == sol[j] == sol[k]:
                    done = False
                    sol[random.choice([i, j, k])] = 1 - sol[i]
        return sol
    ```
    
    </details>
    
* <a name="clockangle"></a>**ClockAngle** [Clock Angle Problem](https://en.wikipedia.org/wiki/Clock_angle_problem), easy variant (5 instances)
    
    ```python
    def sat(hands: List[int], target_angle=45):
        h, m = hands
        assert 0 < h <= 12 and 0 <= m < 60
        hour_angle = 30 * h + m / 2
        minute_angle = 6 * m
        return abs(hour_angle - minute_angle) in [target_angle, 360 - target_angle]
    
    def sol(target_angle=45):
        """Find clock hands = [hour, min] such that the angle is target_angle degrees."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        for h in range(1, 13):
            for m in range(60):
                hour_angle = 30 * h + m / 2
                minute_angle = 6 * m
                if abs(hour_angle - minute_angle) % 360 in [target_angle, 360 - target_angle]:
                    return [h, m]
    ```
    
    </details>
    
* <a name="kirkman"></a>**Kirkman** [Kirkman's problem](https://en.wikipedia.org/wiki/Kirkman%27s_schoolgirl_problem) (1 instance)
    
    ```python
    def sat(daygroups: List[List[List[int]]]):
        assert len(daygroups) == 7
        assert all(len(groups) == 5 and {i for g in groups for i in g} == set(range(15)) for groups in daygroups)
        assert all(len(g) == 3 for groups in daygroups for g in groups)
        return len({(i, j) for groups in daygroups for g in groups for i in g for j in g}) == 15 * 15
    
    def sol():
        """
        Arrange 15 people into groups of 3 each day for seven days so that no two people are in the same group twice.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        from itertools import combinations
        import random
        rand = random.Random(0)
        days = [[list(range(15)) for _2 in range(2)] for _ in range(7)]  # each day is pi, inv
        counts = {(i, j): (7 if j in range(k, k + 3) else 0)
                  for k in range(0, 15, 3)
                  for i in range(k, k + 3)
                  for j in range(15) if j != i
                  }
    
        todos = [pair for pair, count in counts.items() if count == 0]
        while True:
            pair = rand.choice(todos)  # choose i and j to make next to each other on some day
            if rand.randrange(2):
                pair = pair[::-1]
    
            a, u = pair
            pi, inv = rand.choice(days)
            assert pi[inv[a]] == a and pi[inv[u]] == u
            bases = [3 * (inv[i] // 3) for i in pair]
            (b, c), (v, w) = [[x for x in pi[b: b + 3] if x != i] for i, b in zip(pair, bases)]
            if rand.randrange(2):
                b, c, = c, b
            # current (a, b, c) (u, v, w). consider swap of u with b to make (a, u, c) (b, v, w)
    
            new_pairs = [(a, u), (c, u), (b, v), (b, w)]
            old_pairs = [(u, v), (u, w), (b, a), (b, c)]
            gained = sum(counts[p] == 0 for p in new_pairs)
            lost = sum(counts[p] == 1 for p in old_pairs)
            if rand.random() <= 100 ** (gained - lost):
                for p in new_pairs:
                    counts[p] += 1
                    counts[p[::-1]] += 1
                for p in old_pairs:
                    counts[p] -= 1
                    counts[p[::-1]] -= 1
                pi[inv[b]], pi[inv[u]], inv[b], inv[u] = u, b, inv[u], inv[b]
                todos = [pair for pair, count in counts.items() if count == 0]
                if len(todos) == 0:
                    return [[pi[k:k + 3] for k in range(0, 15, 3)] for pi, _inv in days]
    ```
    
    </details>
    
* <a name="monkeyandcoconuts"></a>**MonkeyAndCoconuts** [The Monkey and the Coconuts](https://en.wikipedia.org/wiki/The_monkey_and_the_coconuts) (1 instance)
    
    ```python
    def sat(n: int):
        for i in range(5):
            assert n % 5 == 1
            n -= 1 + (n - 1) // 5
        return n > 0 and n % 5 == 1
    
    def sol():
        """
        Find the number of coconuts to solve the following riddle:
            There is a pile of coconuts, owned by five men. One man divides the pile into five equal piles, giving the
            one left over coconut to a passing monkey, and takes away his own share. The second man then repeats the
            procedure, dividing the remaining pile into five and taking away his share, as do the third, fourth, and
            fifth, each of them finding one coconut left over when dividing the pile by five, and giving it to a monkey.
            Finally, the group divide the remaining coconuts into five equal piles: this time no coconuts are left over.
            How many coconuts were there in the original pile?
                                              Quoted from https://en.wikipedia.org/wiki/The_monkey_and_the_coconuts
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        m = 1
        while True:
            n = m
            for i in range(5):
                if n % 5 != 1:
                    break
                n -= 1 + (n - 1) // 5
            if n > 0 and n % 5 == 1:
                return m
            m += 5
    ```
    
    </details>
    
* <a name="no3colinear"></a>**No3Colinear** [No three-in-a-line](https://en.wikipedia.org/wiki/No-three-in-line_problem) (4 instances)
    
    ```python
    def sat(coords: List[List[int]], side=10, num_points=20):
        for i1 in range(len(coords)):
            x1, y1 = coords[i1]
            assert 0 <= x1 < side and 0 <= y1 < side
            for i2 in range(i1):
                x2, y2 = coords[i2]
                for i3 in range(i2):
                    x3, y3 = coords[i3]
                    assert x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) != 0
        return len({(a, b) for a, b in coords}) == len(coords) >= num_points
    
    def sol(side=10, num_points=20):
        """Find num_points points in an side x side grid such that no three points are collinear."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        from itertools import combinations
        assert side <= 5 or side == 10, "Don't know how to solve other sides"
    
        def test(coords):
            return all(p[0] * (q[1] - r[1]) + q[0] * (r[1] - p[1]) + r[0] * (p[1] - q[1])
                       for p, q, r in combinations(coords, 3))
    
        if side <= 5:
            grid = [[i, j] for i in range(side) for j in range(side)]
            return next(list(coords) for coords in combinations(grid, num_points) if test(coords))
    
        if side == 10:
            def mirror(coords):  # rotate to all four corners
                return [[a, b] for x, y in coords for a in [x, side - 1 - x] for b in [y, side - 1 - y]]
    
            grid = [[i, j] for i in range(side // 2) for j in range(side // 2)]
            return next(list(mirror(coords)) for coords in combinations(grid, side // 2) if
                        test(coords) and test(mirror(coords)))
    ```
    
    </details>
    
* <a name="postagestamp"></a>**PostageStamp** [Postage stamp problem](https://en.wikipedia.org/wiki/Postage_stamp_problem) (5 instances)
    
    ```python
    def sat(stamps: List[int], target=80, max_stamps=4, options=[10, 32, 8]):
        for s in stamps:
            assert s in options
        return len(stamps) <= max_stamps and sum(stamps) == target
    
    def sol(target=80, max_stamps=4, options=[10, 32, 8]):
        """Find a selection of at most max_stamps stamps whose total worth is the target value."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        from itertools import combinations_with_replacement
        for n in range(max_stamps + 1):
            for c in combinations_with_replacement(options, n):
                if sum(c) == target:
                    return list(c)
    ```
    
    </details>
    
* <a name="sudoku"></a>**Sudoku** The classic game of [Sudoku](https://en.wikipedia.org/wiki/Sudoku) (5 instances)
    
    ```python
    def sat(x: str, puz="____9_2___7__________1_8_4____2_78____4_____1____69____2_8___5__6__3_7___49______"):
        assert all(c == "_" or c == s for (c, s) in zip(puz, x))
    
        full = set('123456789')
        for i in range(9):
            assert {x[i] for i in range(9 * i, 9 * i + 9)} == full, "invalid row"
            assert {x[i] for i in range(i, i + 81, 9)} == full, "invalid column"
            assert {x[9 * a + b + i + 26 * (i % 3)] for a in range(3) for b in range(3)} == full, "invalid square"
    
        return True
    
    def sol(puz="____9_2___7__________1_8_4____2_78____4_____1____69____2_8___5__6__3_7___49______"):
        """Find the unique valid solution to the Sudoku puzzle"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        """Simple depth-first backtracking solver that branches at the square with fewest possibilities"""
        sets = [{int(c)} if c != '_' else set(range(1, 10)) for c in puz]
    
        groups = []
        for i in range(9):
            groups.append(list(range(9 * i, 9 * i + 9)))
            groups.append(list(range(i, i + 81, 9)))
            groups.append([9 * a + b + i + 26 * (i % 3) for a in range(3) for b in range(3)])
    
        inv = [[] for i in range(81)]
        for g in groups:
            for i in g:
                inv[i].append(g)
    
        def reduce():
            """Reduce possibilities and return False if it's clearly impossible to solve, True otherwise.
            Repeatedly applies two types of logic:
            * When an entry has a single possibility, remove that value from all 20 neighbors
            * When a row/col/square has only one entry with k as a possibility, fill in that possibility
            """
            done = False
            while not done:
                done = True
                for i in range(81):
                    new = sets[i] - {k for g in inv[i] for j in g if j != i and len(sets[j]) == 1 for k in sets[j]}
                    if not new:
                        return False
                    if len(sets[i]) != len(new):
                        sets[i] = new
                        done = False
    
                for g in groups:
                    for k in range(1, 10):
                        possibilities = [i for i in g if k in sets[i]]
                        if not possibilities:
                            return False
                        if len(possibilities) == 1:
                            i = possibilities[0]
                            if len(sets[i]) > 1:
                                done = False
                                sets[i] = {k}
    
            return True
    
        ans = []
    
        counter = 0
    
        def solve_helper():
            nonlocal sets, ans, counter
            counter += 1
            assert len(ans) <= 1, "Sudoku puzzle should have a unique solution"
            old_sets = sets[:]
            if reduce():
                if all(len(s) == 1 for s in sets):
                    ans.append("".join(str(list(s)[0]) for s in sets))
                else:
                    smallest_set = min(range(81), key=lambda i: len(sets[i]) if len(sets[i]) > 1 else 10)
                    for v in sorted(sets[smallest_set]):
                        sets[smallest_set] = {v}
                        solve_helper()
    
            sets = old_sets
    
        solve_helper()
        assert ans, "No solution found"
        return ans[0]
    ```
    
    </details>
    
* <a name="squaringthesquare"></a>**SquaringTheSquare** [Squaring the square](https://en.wikipedia.org/wiki/Squaring_the_square)
    Wikipedia gives a minimal [solution with 21 squares](https://en.wikipedia.org/wiki/Squaring_the_square)
    due to Duijvestijn (1978). (1 instance)
    
    ```python
    def sat(xy_sides: List[List[int]]):
        n = max(x + side for x, y, side in xy_sides)
        assert len({side for x, y, side in xy_sides}) == len(xy_sides) > 1
        for x, y, s in xy_sides:
            assert 0 <= y < y + s <= n and 0 <= x
            for x2, y2, s2 in xy_sides:
                assert s2 <= s or x2 >= x + s or x2 + s2 <= x or y2 >= y + s or y2 + s2 <= y
    
        return sum(side ** 2 for x, y, side in xy_sides) == n ** 2
    
    def sol():
        """
        Partition a square into smaller squares with unique side lengths. A perfect squared path has distinct sides.
        xy_sides is a List of (x, y, side)
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [[0, 0, 50], [0, 50, 29], [0, 79, 33], [29, 50, 25], [29, 75, 4], [33, 75, 37], [50, 0, 35],
                [50, 35, 15], [54, 50, 9], [54, 59, 16], [63, 50, 2], [63, 52, 7], [65, 35, 17], [70, 52, 18],
                [70, 70, 42], [82, 35, 11], [82, 46, 6], [85, 0, 27], [85, 27, 8], [88, 46, 24], [93, 27, 19]]
    ```
    
    </details>
    
* <a name="necklacesplit"></a>**NecklaceSplit** [Necklace Splitting Problem](https://en.wikipedia.org/wiki/Necklace_splitting_problem) (5 instances)
    
    ```python
    def sat(n: int, lace="bbrbrbbbbbbrrrrrrrbrrrrbbbrbrrbbbrbrrrbrrbrrbrbbrrrrrbrbbbrrrbbbrbbrbbbrbrbb"):
        sub = lace[n: n + len(lace) // 2]
        return n >= 0 and lace.count("r") == 2 * sub.count("r") and lace.count("b") == 2 * sub.count("b")
    
    def sol(lace="bbrbrbbbbbbrrrrrrrbrrrrbbbrbrrbbbrbrrrbrrbrrbrbbrrrrrbrbbbrrrbbbrbbrbbbrbrbb"):
        """
        Find a split dividing the given red/blue necklace in half at n so that each piece has an equal number of
        reds and blues.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if lace == "":
            return 0
        return next(n for n in range(len(lace) // 2) if lace[n: n + len(lace) // 2].count("r") == len(lace) // 4)
    ```
    
    </details>
    
* <a name="pandigitalsquare"></a>**PandigitalSquare** [Pandigital](https://en.wikipedia.org/wiki/Pandigital_number) Square (1 instance)
    
    ```python
    def sat(n: int):
        s = str(n * n)
        for i in "0123456789":
            assert s.count(i) == 1
        return True
    
    def sol():
        """Find an integer whose square has all digits 0-9 once."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        for n in range(10 ** 5):
            if sorted([int(s) for s in str(n * n)]) == list(range(10)):
                return n
    ```
    
    </details>
    
* <a name="allpandigitalsquares"></a>**AllPandigitalSquares** All [Pandigital](https://en.wikipedia.org/wiki/Pandigital_number) Squares (1 instance)
    
    ```python
    def sat(nums: List[int]):
        return [sorted([int(s) for s in str(n * n)]) for n in set(nums)] == [list(range(10))] * 174
    
    def sol():
        """Find all 174 integers whose 10-digit square has all digits 0-9 just once."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [i for i in range(-10 ** 5, 10 ** 5) if sorted([int(s) for s in str(i * i)]) == list(range(10))]
    ```
    
    </details>
    
* <a name="cardgame24"></a>**CardGame24** [24 Game](https://en.wikipedia.org/wiki/24_Game)
    
    In this game one is given four numbers from the range 1-13 (Ace-King) and one needs to combine them with
        + - * / (and parentheses)
    to make the number 24.
    The solution to this tricky example is `7 * (3 + 3 / 7)` (5 instances)
    
    ```python
    def sat(expr: str, nums=[3, 7, 3, 7]):
        assert len(nums) == 4 and 1 <= min(nums) and max(nums) <= 13, "hint: nums is a list of four ints in 1..13"
        expr = expr.replace(" ", "")  # ignore whitespace
        digits = ""
        for i in range(len(expr)):
            if i == 0 or expr[i - 1] in "+*-/(":
                assert expr[i] in "123456789(", "Expr cannot contain **, //, or unary -"
            assert expr[i] in "1234567890()+-*/", "Expr can only contain `0123456789()+-*/`"
            digits += expr[i] if expr[i] in "0123456789" else " "
        assert sorted(int(s) for s in digits.split()) == sorted(nums), "Each number must occur exactly once"
        return abs(eval(expr) - 24.0) < 1e-6
    
    def sol(nums=[3, 7, 3, 7]):
        """Find a formula with two 3's and two 7's and + - * / (and parentheses) that evaluates to 24."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        def helper(pairs):
            if len(pairs) == 2:
                (x, s), (y, t) = pairs
                ans = {
                    x + y: f"{s}+{t}",
                    x - y: f"{s}-({t})",
                    y - x: f"{t}-({s})",
                    x * y: f"({s})*({t})"
                }
                if y != 0:
                    ans[x / y] = f"({s})/({t})"
                if x != 0:
                    ans[y / x] = f"({t})/({s})"
                return ans
            ans = {y: t
                   for i in range(len(pairs))
                   for x_s in helper(pairs[:i] + pairs[i + 1:]).items()
                   for y, t in helper([x_s, pairs[i]]).items()}
            if len(pairs) == 3:
                return ans
            ans.update({z: u
                        for i in range(1, 4)
                        for x_s in helper([pairs[0], pairs[i]]).items()
                        for y_t in helper(pairs[1:i] + pairs[i + 1:]).items()
                        for z, u in helper([x_s, y_t]).items()
                        })
            return ans
    
        derivations = helper([(n, str(n)) for n in nums])
        for x in derivations:
            if abs(x - 24.0) < 1e-6:
                return derivations[x]
    ```
    
    </details>
    
* <a name="easy63"></a>**Easy63** An easy puzzle to make 63 using two 8's and one 1's. (1 instance)
    
    ```python
    def sat(s: str):
        return set(s) <= set("18-+*/") and s.count("8") == 2 and s.count("1") == 1 and eval(s) == 63
    
    def sol():
        """Find a formula using two 8s and two 1's and -+*/ that evaluates to 1."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "8*8-1"
    ```
    
    </details>
    
* <a name="harder63"></a>**Harder63** An harder puzzle to make 63 using three 8's and one 1's. (1 instance)
    
    ```python
    def sat(s: str):
        return set(s) <= set("18-+*/") and s.count("8") == 3 and s.count("1") == 1 and eval(s) == 63
    
    def sol():
        """Find an expression using two 8s and two 1's and -+*/ that evaluates to 1."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "8*8-1**8"
    ```
    
    </details>
    
* <a name="waterpouring"></a>**WaterPouring** [Water pouring puzzle](https://en.wikipedia.org/w/index.php?title=Water_pouring_puzzle&oldid=985741928) (5 instances)
    
    ```python
    def sat(moves: List[List[int]], capacities=[8, 5, 3], init=[8, 0, 0], goal=[4, 4, 0]):
        state = init.copy()
    
        for [i, j] in moves:
            assert min(i, j) >= 0, "Indices must be non-negative"
            assert i != j, "Cannot pour from same state to itself"
            n = min(capacities[j], state[i] + state[j])
            state[i], state[j] = state[i] + state[j] - n, n
    
        return state == goal
    
    def sol(capacities=[8, 5, 3], init=[8, 0, 0], goal=[4, 4, 0]):
        """
        Given an initial state of water quantities in jugs and jug capacities, find a sequence of moves (pouring
        one jug into another until it is full or the first is empty) to reaches the given goal state.
        moves is list of [from, to] pairs
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        from collections import deque
        num_jugs = len(capacities)
        start = tuple(init)
        target = tuple(goal)
        trails = {start: ([], start)}
        queue = deque([tuple(init)])
        while target not in trails:
            state = queue.popleft()
            for i in range(num_jugs):
                for j in range(num_jugs):
                    if i != j:
                        n = min(capacities[j], state[i] + state[j])
                        new_state = list(state)
                        new_state[i], new_state[j] = state[i] + state[j] - n, n
                        new_state = tuple(new_state)
                        if new_state not in trails:
                            queue.append(new_state)
                            trails[new_state] = ([i, j], state)
        ans = []
        state = target
        while state != start:
            move, state = trails[state]
            ans.append(move)
        return ans[::-1]
    ```
    
    </details>
    
* <a name="verbalarithmetic"></a>**VerbalArithmetic** Find a substitution of digits for characters to make the numbers add up in a sum like this:
    SEND + MORE = MONEY
    
    The first digit in any number cannot be 0. In this example the solution is `9567 + 1085 = 10652`.
    See [Wikipedia article](https://en.wikipedia.org/wiki/Verbal_arithmetic) (5 instances)
    
    ```python
    def sat(li: List[int], words=['SEND', 'MORE', 'MONEY']):
        assert len(li) == len(words) and all(i > 0 and len(str(i)) == len(w) for i, w in zip(li, words))
        assert len({c for w in words for c in w}) == len({(d, c) for i, w in zip(li, words) for d, c in zip(str(i), w)})
        return sum(li[:-1]) == li[-1]
    
    def sol(words=['SEND', 'MORE', 'MONEY']):
        """
        Find a list of integers corresponding to the given list of strings substituting a different digit for each
        character, so that the last string corresponds to the sum of the previous numbers.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        pi = list(range(10))  # permutation
        letters = []
        order = {}
        steps = []
        tens = 1
        for col in range(1, 1 + max(len(w) for w in words)):
            for w in words:
                is_tot = (w is words[-1])
                if len(w) >= col:
                    c = w[-col]
                    if c in order:
                        if is_tot:
                            kind = "check"
                        else:
                            kind = "seen"
                    else:
                        if is_tot:
                            kind = "derive"
                        else:
                            kind = "add"
                        order[c] = len(letters)
                        letters.append(c)
                    steps.append((kind, order[c], tens))
            tens *= 10
    
        inits = [any(w[0] == c for w in words) for c in letters]
    
        def helper(pos, delta):  # on success, returns True and pi has the correct values
            if pos == len(steps):
                return delta == 0
    
            kind, i, tens = steps[pos]
    
            if kind == "seen":
                return helper(pos + 1, delta + tens * pi[i])
    
            if kind == "add":
                for j in range(i, 10):
                    if pi[j] != 0 or not inits[i]:  # not adding a leading 0
                        pi[i], pi[j] = pi[j], pi[i]
                        if helper(pos + 1, delta + tens * pi[i]):
                            return True
                        pi[i], pi[j] = pi[j], pi[i]
                return False
            if kind == "check":
                delta -= tens * pi[i]
                return (delta % (10 * tens)) == 0 and helper(pos + 1, delta)
    
            assert kind == "derive"
            digit = (delta % (10 * tens)) // tens
            if digit == 0 and inits[i]:
                return False  # would be a leading 0
            j = pi.index(digit)
            if j < i:
                return False  # already used
            pi[i], pi[j] = pi[j], pi[i]
            if helper(pos + 1, delta - tens * digit):
                return True
            pi[i], pi[j] = pi[j], pi[i]
            return False
    
        assert helper(0, 0)
        return [int("".join(str(pi[order[c]]) for c in w)) for w in words]
    ```
    
    </details>
    
* <a name="slidingpuzzle"></a>**SlidingPuzzle** [Sliding puzzle](https://en.wikipedia.org/wiki/15_puzzle)
    The 3-, 8-, and 15-sliding puzzles are classic examples of A* search.
    The problem is NP-hard but the puzzles can all be solved with A* and an efficient representation. (5 instances)
    
    ```python
    def sat(moves: List[int], start=[[5, 0, 2, 3], [1, 9, 6, 7], [4, 14, 8, 11], [12, 13, 10, 15]]):
    
        locs = {i: [x, y] for y, row in enumerate(start) for x, i in enumerate(row)}  # locations, 0 stands for blank
        for i in moves:
            assert abs(locs[0][0] - locs[i][0]) + abs(locs[0][1] - locs[i][1]) == 1
            locs[0], locs[i] = locs[i], locs[0]
        return all(locs[i] == [i % len(start[0]), i // len(start)] for i in locs)
    
    def sol(start=[[5, 0, 2, 3], [1, 9, 6, 7], [4, 14, 8, 11], [12, 13, 10, 15]]):
        """
        In this puzzle, you are given a board like:
        1 2 5
        3 4 0
        6 7 8
    
        and your goal is to transform it to:
        0 1 2
        3 4 5
        6 7 8
    
        by a sequence of swaps with the 0 square (0 indicates blank). The starting configuration is given by a 2d list
        of lists and the answer is represented by a list of integers indicating which number you swap with 0. In the
        above example, an answer would be [1, 2, 5]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        from collections import defaultdict
        import math
        d = len(start)
        N = d * d
        assert all(len(row) == d for row in start)
    
        def get_state(
                li):  # state is an integer with 4 bits for each slot and the last 4 bits indicate where the blank is
            ans = 0
            for i in li[::-1] + [li.index(0)]:
                ans = (ans << 4) + i
            return ans
    
        start = get_state([i for row in start for i in row])
        target = get_state(list(range(N)))
    
        def h(state):  # manhattan distance
            ans = 0
            for i in range(N):
                state = (state >> 4)
                n = state & 15
                if n != 0:
                    ans += abs(i % d - n % d) + abs(i // d - n // d)
            return ans
    
        g = defaultdict(lambda: math.inf)
        g[start] = 0  # shortest p ath lengths
        f = {start: h(start)}  # f[s] = g[s] + h(s)
        backtrack = {}
    
        todo = {start}
        import heapq
        heap = [(f[start], start)]
    
        neighbors = [[i for i in [b - 1, b + 1, b + d, b - d] if i in range(N) and (b // d == i // d or b % d == i % d)]
                     for b in range(N)]
    
        def next_state(s, blank, i):
            assert blank == (s & 15)
            v = (s >> (4 * i + 4)) & 15
            return s + (i - blank) + (v << (4 * blank + 4)) - (v << (4 * i + 4))
    
        while todo:
            (dist, s) = heapq.heappop(heap)
            if f[s] < dist:
                continue
            if s == target:
                # compute path
                ans = []
                while s != start:
                    s, i = backtrack[s]
                    ans.append((s >> (4 * i + 4)) & 15)
                return ans[::-1]
    
            todo.remove(s)
    
            blank = s & 15
            score = g[s] + 1
            for i in neighbors[blank]:
                s2 = next_state(s, blank, i)
    
                if score < g[s2]:
                    # paths[s2] = paths[s] + [s[i]]
                    g[s2] = score
                    backtrack[s2] = (s, i)
                    score2 = score + h(s2)
                    f[s2] = score2
                    todo.add(s2)
                    heapq.heappush(heap, (score2, s2))
    ```
    
    </details>
    
## human_eval

Problems inspired by [HumanEval dataset](https://github.com/openai/human-eval) described
in the [codex paper](https://arxiv.org/abs/2107.03374), specifically,
[this](https://github.com/openai/human-eval/blob/fa06031e684fbe1ee429c7433809460c159b66ad/data/HumanEval.jsonl.gz)
version released 7/7/21.

* <a name="findcloseelements"></a>**FindCloseElements** Inspired by [HumanEval](https://github.com/openai/human-eval) \#0 (5 instances)
    
    ```python
    def sat(pair: List[float], nums=[0.17, 21.3, 5.0, 9.0, 11.0, 4.99, 17.0, 17.0, 12.4, 6.8]):
        a, b = pair
        assert a in nums and b in nums and a != b
        return abs(a - b) == min(x - y for x in nums for y in nums if x > y)
    
    def sol(nums=[0.17, 21.3, 5.0, 9.0, 11.0, 4.99, 17.0, 17.0, 12.4, 6.8]):
        """
        Given a list of numbers, find the two closest distinct numbers in the list.
    
        Sample Input:
        [1.2, 5.23, 0.89, 21.0, 5.28, 1.2]
    
        Sample Output:
        [5.23, 5.28]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        s = sorted(set(nums))
        return min([[a, b] for a, b in zip(s, s[1:])], key=lambda x: x[1] - x[0])
    ```
    
    </details>
    
* <a name="separateparengroups"></a>**SeparateParenGroups** Inspired by [HumanEval](https://github.com/openai/human-eval) \#1 (5 instances)
    
    ```python
    def sat(ls: List[str], combined="() (()) ((() () ())) (() )"):
        for s in ls:
            assert s.count("(") == s.count(")")
            assert all(s[:i].count("(") > s[:i].count(")") for i in range(1, len(s)))  # s is not further divisible
        return ''.join(ls) == combined.replace(' ', '')
    
    def sol(combined="() (()) ((() () ())) (() )"):
        """
        Given a string consisting of whitespace and groups of matched parentheses, split it
        into groups of perfectly matched parentheses without any whitespace.
    
        Sample Input:
        '( ()) ((()()())) (()) ()'
    
        Sample Output:
        ['(())', '((()()()))', '(())', '()']
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        cur = ''
        ans = []
        depth = 0
        for c in combined.replace(' ', ''):
            cur += c
            if c == '(':
                depth += 1
            else:
                assert c == ')'
                depth -= 1
                if depth == 0:
                    ans.append(cur)
                    cur = ''
        return ans
    ```
    
    </details>
    
* <a name="frac"></a>**Frac** Inspired by [HumanEval](https://github.com/openai/human-eval) \#2 (5 instances)
    
    ```python
    def sat(x: float, v=523.12892):
        return 0 <= x < 1 and (v - x).is_integer()
    
    def sol(v=523.12892):
        """
        Given a floating point number, find its fractional part.
    
        Sample Input:
        4.175
    
        Sample Output:
        0.175
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return v % 1.0
    ```
    
    </details>
    
* <a name="firstnegcumulative"></a>**FirstNegCumulative** Inspired by [HumanEval](https://github.com/openai/human-eval) \#3 (5 instances)
    
    ```python
    def sat(firsts: List[int], balances=[[2, 7, -2, 4, 3, -15, 10, -45, 3], [3, 4, -17, -1], [100, -100, -101], [-1]]):
        for i, bals in enumerate(balances):
            total = 0
            for b in bals:
                total += b
                if total < 0:
                    assert total == firsts[i]
                    break
        return True
    
    def sol(balances=[[2, 7, -2, 4, 3, -15, 10, -45, 3], [3, 4, -17, -1], [100, -100, -101], [-1]]):
        """
        Given a list of numbers which represent bank deposits and withdrawals, find the *first* negative balance.
    
        Sample Input:
        [[12, -5, 3, -99, 14, 88, -99], [-1, 2, 5]]
    
        Sample Output:
        [-89, -1]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        firsts = []
        for bals in balances:
            total = 0
            for b in bals:
                total += b
                if total < 0:
                    firsts.append(total)
                    break
        return firsts
    ```
    
    </details>
    
* <a name="minsquareddeviation"></a>**MinSquaredDeviation** Loosely inspired by [HumanEval](https://github.com/openai/human-eval) \#4
    
    The HumanEval problem was simply to compute the mean absolute deviation. This problem is more interesting.
    It requires minimizing the sum of squared deviations, which turns out to be the mean `mu`. Moreover, if
    `mu` is the mean of the numbers then a simple calculation shows that:
    
    `sum((mu - n) ** 2 for n in nums) == sum((m - n) ** 2 for m in nums for n in nums) / (2 * len(nums))`
    
    We use 0.501 rather than 1/2 to deal with rounding errors. (5 instances)
    
    ```python
    def sat(x: float, nums=[12, -2, 14, 3, -15, 10, -45, 3, 30]):
        return sum((n - x) ** 2 for n in nums) * len(nums) <= sum((m - n) ** 2 for m in nums for n in nums) * .5 + 1e-4
    
    def sol(nums=[12, -2, 14, 3, -15, 10, -45, 3, 30]):
        """
        Given a list of numbers, find x that minimizes mean squared deviation.
    
        Sample Input:
        [4, -5, 17, -9, 14, 108, -9]
    
        Sample Output:
        17.14285
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sum(nums) / len(nums)  # mean minimizes mean squared deviation
    ```
    
    </details>
    
* <a name="intersperse"></a>**Intersperse** Inspired by [HumanEval](https://github.com/openai/human-eval) \#5 (5 instances)
    
    ```python
    def sat(li: List[int], nums=[12, 23, -2, 5, 0], sep=4):
        return li[::2] == nums and li[1::2] == [sep] * (len(nums) - 1)
    
    def sol(nums=[12, 23, -2, 5, 0], sep=4):
        """
        Given a list of numbers and a number to inject, create a list containing that number in between each pair of
        adjacent numbers.
    
        Sample Input:
        [8, 14, 21, 17, 9, -5], 3
    
        Sample Output:
        [8, 3, 14, 3, 21, 3, 17, 3, 9, 3, -5]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        ans = [sep] * (2 * len(nums) - 1)
        ans[::2] = nums
        return ans
    ```
    
    </details>
    
* <a name="deepestparens"></a>**DeepestParens** Inspired by [HumanEval](https://github.com/openai/human-eval) \#6 (5 instances)
    
    ```python
    def sat(depths: List[int], parens="() (()) ((()()())) (((((((())))))))"):
        groups = parens.split()
        for depth, group in zip(depths, groups):
            budget = depth
            success = False
            for c in group:
                if c == '(':
                    budget -= 1
                    if budget == 0:
                        success = True
                    assert budget >= 0
                else:
                    assert c == ')'
                    budget += 1
            assert success
    
        return len(groups) == len(depths)
    
    def sol(parens="() (()) ((()()())) (((((((())))))))"):
        """
        Given a string consisting of groups of matched nested parentheses separated by parentheses,
        compute the depth of each group.
    
        Sample Input:
        '(()) ((()()())) (()) ()'
    
        Sample Output:
        [2, 3, 2, 1]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        def max_depth(s):
            m = 0
            depth = 0
            for c in s:
                if c == '(':
                    depth += 1
                    m = max(m, depth)
                else:
                    assert c == ')'
                    depth -= 1
            assert depth == 0
            return m
    
        return [max_depth(s) for s in parens.split()]
    ```
    
    </details>
    
* <a name="findcontainers"></a>**FindContainers** Inspired by [HumanEval](https://github.com/openai/human-eval) \#7 (5 instances)
    
    ```python
    def sat(containers: List[str], strings=['cat', 'dog', 'shatter', 'bear', 'at', 'ta'], substring="at"):
        i = 0
        for s in strings:
            if substring in s:
                assert containers[i] == s
                i += 1
        return i == len(containers)
    
    def sol(strings=['cat', 'dog', 'shatter', 'bear', 'at', 'ta'], substring="at"):
        """
        Find the strings in a list containing a given substring
    
        Sample Input:
        ['cat', 'dog', 'bear'], 'a'
    
        Sample Output:
        ['cat', 'bear']
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [s for s in strings if substring in s]
    ```
    
    </details>
    
* <a name="sumproduct"></a>**SumProduct** Inspired by [HumanEval](https://github.com/openai/human-eval) \#8 (5 instances)
    
    ```python
    def sat(nums: List[int], tot=14, prod=99):
        assert sum(nums) == tot
        p = 1
        for n in nums:
            p *= n
        return p == prod
    
    def sol(tot=14, prod=99):
        """
        Find a list of numbers with a given sum and a given product.
    
        Sample Input:
        12, 32
    
        Sample Output:
        [2, 8, 2]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        ans = [prod]
        while sum(ans) > tot:
            ans += [-1, -1]
        ans += [1] * (tot - sum(ans))
        return ans
    ```
    
    </details>
    
* <a name="rollingmax"></a>**RollingMax** Inspired by [HumanEval](https://github.com/openai/human-eval) \#9 (5 instances)
    
    ```python
    def sat(maxes: List[int], nums=[1, 4, 3, -6, 19]):
        assert len(maxes) == len(nums)
        for i in range(len(nums)):
            if i > 0:
                assert maxes[i] == max(maxes[i - 1], nums[i])
            else:
                assert maxes[0] == nums[0]
        return True
    
    def sol(nums=[1, 4, 3, -6, 19]):
        """
        Find a list whose ith element is the maximum of the first i elements of the input list.
    
        Sample Input:
        [2, 8, 2]
    
        Sample Output:
        [2, 8, 8]
        """
    ```
    <details><summary>2 hand-written solutions </summary>
    
    ```python
        return [max(nums[:i]) for i in range(1, len(nums) + 1)]
    ```
    
    ```python
        ans = []
        if nums:
            m = nums[0]
            for n in nums:
                m = max(n, m)
                ans.append(m)
        return ans
    ```
    
    </details>
    
* <a name="palindromecontaining"></a>**PalindromeContaining** Inspired by [HumanEval](https://github.com/openai/human-eval) \#10 (5 instances)
    
    ```python
    def sat(ans: str, s="so easy", length=20):
        return ans == ans[::-1] and len(ans) == length and s in ans
    
    def sol(s="so easy", length=20):
        """
        Find a palindrome of a given length containing a given string.
    
        Sample Input:
        "abba", 6
    
        Sample Output:
        "cabbac"
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        ls = list(s)
        for i in range(length - len(s) + 1):
            arr = ['x'] * length
            arr[i:i + len(s)] = ls
            a = length - i - 1
            b = length - (i + len(s)) - 1
            if b == -1:
                b = None
            arr[a:b:-1] = ls
            if arr == arr[::-1]:
                ans = "".join(arr)
                if s in ans:
                    return ans
        assert False, "shouldn't reach here"
    ```
    
    </details>
    
* <a name="binarystrxor"></a>**BinaryStrXOR** Inspired by [HumanEval](https://github.com/openai/human-eval) \#11 (5 instances)
    
    ```python
    def sat(str_num: str, nums=['100011101100001', '100101100101110']):
        a, b = nums
        return int(str_num, 2) == int(a, 2) ^ int(b, 2)
    
    def sol(nums=['100011101100001', '100101100101110']):
        """
        Find a the XOR of two given strings interpreted as binary numbers.
    
        Sample Input:
        "0001", "1011"
    
        Sample Output:
        "1010"
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        a, b = nums
        ans = int(a, 2) ^ int(b, 2)
        return format(ans, "b")
    ```
    
    </details>
    
* <a name="longeststr"></a>**LongestStr** Inspired by [HumanEval](https://github.com/openai/human-eval) \#12 (5 instances)
    
    ```python
    def sat(ans: str, words=['these', 'are', 'some', 'pretty', 'long', 'words']):
        return ans in words and all(len(ans) >= len(w) for w in words)
    
    def sol(words=['these', 'are', 'some', 'pretty', 'long', 'words']):
        """
        Find the longest of a list of strings
    
        Sample Input:
        ["cat", "dog", "sheep", "chimp"]
    
        Sample Output:
        "sheep"
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return max(words, key=len)
    ```
    
    </details>
    
* <a name="certifiedgcd"></a>**CertifiedGCD** Inspired by [HumanEval](https://github.com/openai/human-eval) \#13 (5 instances)
    
    ```python
    def sat(ans: List[int], m=200004931, n=66679984):
        gcd, a, b = ans
        return m % gcd == n % gcd == 0 and a * m + b * n == gcd and gcd > 0
    
    def sol(m=200004931, n=66679984):
        """
        Find the greatest common divisor of two integers m, n and a certificate a, b such that m*a + n*b = gcd
    
        Sample Input:
        20, 30
    
        Sample Output:
        10, -1, 1
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        """
        Derivation of solution below
        Recursive solution guarantees a * (big % small) + b * small == gcd
        Let d = big // small so (big % small) == big - small * d
        gives a * (big - small * d) + b * small == gcd
        or equivalently (b - a * d) * small + a * big == gcd
        """
    
        def gcd_cert(small, big):
            """Returns gcd, a, b, such that small * a + big * b == gcd"""
            assert 0 < small <= big
            if big % small == 0:
                return [small, 1, 0]
            gcd, a, b = gcd_cert(big % small, small)
            return [gcd, b - a * (big // small), a]
    
        if m < n:
            return gcd_cert(m, n)
        gcd, a, b = gcd_cert(n, m)
        return [gcd, b, a]
    ```
    
    </details>
    
* <a name="allprefixes"></a>**AllPrefixes** Inspired by [HumanEval](https://github.com/openai/human-eval) \#14 (5 instances)
    
    ```python
    def sat(prefixes: List[str], s="donesezichethofalij"):
        return all(s.startswith(p) for p in prefixes) and len(set(prefixes)) > len(s)
    
    def sol(s="donesezichethofalij"):
        """
        Find all prefixes of a given string
    
        Sample Input:
        "aabcd"
    
        Sample Output:
        ["", "a", "aa", "aab", "aabc", "aabcd"]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [s[:i] for i in range(len(s) + 1)]
    ```
    
    </details>
    
* <a name="spaceyrange"></a>**SpaceyRange** Inspired by [HumanEval](https://github.com/openai/human-eval) \#15 (5 instances)
    
    ```python
    def sat(ans: str, n=15):
        return [int(i) for i in ans.split(' ')] == list(range(n + 1))
    
    def sol(n=15):
        """
        Find a string consisting of the non-negative integers up to n inclusive
    
        Sample Input:
        4
    
        Sample Output:
        '0 1 2 3 4'
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return ' '.join(str(i) for i in range(n + 1))
    ```
    
    </details>
    
* <a name="distinctchars"></a>**DistinctChars** Inspired by [HumanEval](https://github.com/openai/human-eval) \#16 (5 instances)
    
    ```python
    def sat(ans: List[str], s="The quick brown fox jumps over the lazy dog!", n=28):
        assert all(ans.count(c.lower()) == 1 for c in s)
        assert all(c == c.lower() for c in ans)
        assert all(c in s.lower() for c in ans)
        return True
    
    def sol(s="The quick brown fox jumps over the lazy dog!", n=28):
        """
        Find the set of distinct characters in a string, ignoring case
    
        Sample Input:
        'HELlo', 4
    
        Sample Output:
        ['h', 'e', 'l', 'o']
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return list(set(s.lower()))
    ```
    
    </details>
    
* <a name="parsemusic"></a>**ParseMusic** Inspired by [HumanEval](https://github.com/openai/human-eval) \#17 (5 instances)
    
    ```python
    def sat(beats: List[int], score="o o o| o| .| .| .| o| o| o o o| .|"):
        return " ".join({1: '.|', 2: 'o|', 4: 'o'}[b] for b in beats) == score
    
    def sol(score="o o o| o| .| .| .| o| o| o o o| .|"):
        """
        Parse a string of notes to beats, 'o'=4, 'o|'=2, '.|'=1
    
        Example input:
        'o o .| o|'
    
        Example output:
        [4, 4, 1, 2]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        mapping = {'.|': 1, 'o|': 2, 'o': 4}
        return [mapping[note] for note in score.split()]
    ```
    
    </details>
    
* <a name="overlappingcount"></a>**OverlappingCount** Inspired by [HumanEval](https://github.com/openai/human-eval) \#18 (5 instances)
    
    ```python
    def sat(ans: List[int], s="Bananannanaannanaanananananana", sub="anan", count=7):
        return all(sub == s[i:i + len(sub)] and i >= 0 for i in ans) and len(set(ans)) >= count
    
    def sol(s="Bananannanaannanaanananananana", sub="anan", count=7):
        """
        Find occurrences of a substring in a parent string *including overlaps*
    
        Sample Input:
        'helllo', 'll'
    
        Sample Output:
        [2, 3]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        ans = []
        for i in range(len(s) + 1):
            if s[i:i + len(sub)] == sub:
                ans.append(i)
        return ans
    ```
    
    </details>
    
* <a name="sortnumbers"></a>**SortNumbers** Inspired by [HumanEval](https://github.com/openai/human-eval) \#19 (5 instances)
    
    ```python
    def sat(ans: str, s="six one four three two nine eight"):
        nums = 'zero one two three four five six seven eight nine'.split()
        return [nums.index(x) for x in ans.split(" ")] == sorted([nums.index(x) for x in s.split(" ")])
    
    def sol(s="six one four three two nine eight"):
        """
        Sort numbers based on strings
    
        Sample input
        ---
        "six one four"
    
        Sample output
        ---
        "one four six"
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        nums = 'zero one two three four five six seven eight nine'.split()
        arr = [nums.index(x) for x in s.split()]
        arr.sort()
        ans = " ".join([nums[i] for i in arr])
        return ans
    ```
    
    </details>
    
* <a name="findclosepair"></a>**FindClosePair** Inspired by [HumanEval](https://github.com/openai/human-eval) \#20 (5 instances)
    
    ```python
    def sat(inds: List[int], nums=[0.31, 21.3, 5.0, 9.0, 11.0, 5.01, 17.2]):
        a, b = inds
        assert a != b and a >= 0 and b >= 0
        for i in range(len(nums)):
            for j in range(i):
                assert abs(nums[i] - nums[j]) >= abs(nums[b] - nums[a])
        return True
    
    def sol(nums=[0.31, 21.3, 5.0, 9.0, 11.0, 5.01, 17.2]):
        """
        Given a list of numbers, find the indices of the closest pair.
    
        Sample Input:
        [1.2, 5.25, 0.89, 21.0, 5.23]
    
        Sample Output:
        [4, 1]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        best = [0, 1]
        best_score = abs(nums[1] - nums[0])
        for i in range(len(nums)):
            for j in range(i):
                score = abs(nums[i] - nums[j])
                if score < best_score:
                    best_score = score
                    best = [i, j]
        return best
    ```
    
    </details>
    
* <a name="rescale"></a>**Rescale** Inspired by [HumanEval](https://github.com/openai/human-eval) \#21 (5 instances)
    
    ```python
    def sat(ans: List[float], nums=[13.0, 17.0, 17.0, 15.5, 2.94]):
        assert min(ans) == 0.0 and max(ans) == 1.0
        a = min(nums)
        b = max(nums)
        for i in range(len(nums)):
            x = a + (b - a) * ans[i]
            assert abs(nums[i] - x) < 1e-6
        return True
    
    def sol(nums=[13.0, 17.0, 17.0, 15.5, 2.94]):
        """
        Rescale and shift numbers so that they cover the range [0, 1]
    
        Sample input
        ---
        [18.5, 17.0, 18.0, 19.0, 18.0]
    
        Sample output
        ---
        [0.75, 0.0, 0.5, 1.0, 0.5]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        nums = nums.copy()
    
        a = min(nums)
        b = max(nums)
        if b - a == 0:
            return [0.0] + [1.0] * (len(nums) - 1)
        for i in range(len(nums)):
            nums[i] = (nums[i] - a) / (b - a)
        return nums
    ```
    
    </details>
    
* <a name="filterints"></a>**FilterInts** Inspired by [HumanEval](https://github.com/openai/human-eval) \#22 (5 instances)
    
    ```python
    def sat(candidates: List[str], int_indices=[2, 4, 7, 9, 101]):
        for i in int_indices:
            int(candidates[i])
        for i, s in enumerate(candidates):
            if i not in int_indices:
                try:
                    int(s)
                    return False
                except ValueError:
                    pass
        return True
    
    def sol(int_indices=[2, 4, 7, 9, 101]):
        """
        Find a list of strings where the only valid integers are at the given indices
    
        Sample input
        ---
        [2, 4, 5]
    
        Sample output
        ---
        ["cat", "2.7", "2", "", "3", "-17", "free"]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if not int_indices:
            return []
        ans = [""] * (1 + max(abs(i) for i in int_indices))
        for i in int_indices:
            ans[i] = "17"
        return ans
    ```
    
    </details>
    
* <a name="strlength"></a>**StrLength** Inspired by [HumanEval](https://github.com/openai/human-eval) \#23 (5 instances)
    
    ```python
    def sat(lengths: List[int], strs=['pneumonoultramicroscopicsilicovolcanoconiosis', ' ', 'foo', '2.5']):
        for length, s in zip(lengths, strs):
            try:
                s[length]
                return False
            except IndexError:
                s[length - 1]
        return len(lengths) == len(strs)
    
    def sol(strs=['pneumonoultramicroscopicsilicovolcanoconiosis', ' ', 'foo', '2.5']):
        """
        Find the lengths of a list of non-empty strings
    
        Sample input
        ---
        ["foo", "bars"]
    
        Sample output
        ---
        [3, 4]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [len(s) for s in strs]
    ```
    
    </details>
    
* <a name="largestdivisor"></a>**LargestDivisor** Inspired by [HumanEval](https://github.com/openai/human-eval) \#24 (5 instances)
    
    ```python
    def sat(d: int, n=123456):
        return n % d == 0 and d < n and all(n % e for e in range(d + 1, n))
    
    def sol(n=123456):
        """
        Find the largest integer divisor of a number n that is less than n
    
        Sample input
        ---
        1000
    
        Sample output
        ---
        500
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return next(d for d in range(n - 1, 0, -1) if n % d == 0)
    ```
    
    </details>
    
* <a name="primefactorization"></a>**PrimeFactorization** Inspired by [HumanEval](https://github.com/openai/human-eval) \#25 (5 instances)
    
    ```python
    def sat(factors: List[int], n=123456, num_factors=8):
        assert len(factors) == num_factors
        prod = 1
        for d in factors:
            prod *= d
            assert d > 1
        return prod == n
    
    def sol(n=123456, num_factors=8):
        """
        Factor number n into a given number of non-trivial factors
    
        Sample input
        ---
        1000, 6
    
        Sample output
        ---
        [2, 2, 2, 5, 5, 5]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if num_factors == 0:
            return []
        if num_factors == 1:
            return [n]
        ans = []
        for d in range(2, n):
            while n % d == 0:
                n //= d
                ans.append(d)
                if len(ans) == num_factors - 1:
                    ans.append(n)
                    return ans
        assert False
    ```
    
    </details>
    
* <a name="dedup"></a>**Dedup** Inspired by [HumanEval](https://github.com/openai/human-eval) \#26 (5 instances)
    
    ```python
    def sat(ans: List[int], li=[2, 19, 2, 53, 1, 1, 2, 44, 17, 0, 19, 31]):
        return set(ans) == set(li) and all(li.index(ans[i]) < li.index(ans[i + 1]) for i in range(len(ans) - 1))
    
    def sol(li=[2, 19, 2, 53, 1, 1, 2, 44, 17, 0, 19, 31]):
        """
        Remove duplicates from a list of integers, preserving order
    
        Sample input
        ---
        [1, 3, 2, 9, 2, 1, 55]
    
        Sample output
        ---
        [1, 3, 2, 9, 55]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        seen = set()
        ans = []
        for n in li:
            if n not in seen:
                ans.append(n)
                seen.add(n)
        return ans
    ```
    
    </details>
    
* <a name="flipcase"></a>**FlipCase** Inspired by [HumanEval](https://github.com/openai/human-eval) \#27 (5 instances)
    
    ```python
    def sat(ans: str, s="FlIp ME!"):
        return len(ans) == len(s) and all({c, d} == {d.upper(), d.lower()} for c, d in zip(ans, s))
    
    def sol(s="FlIp ME!"):
        """
        Flip case
    
        Sample input
        ---
        'cAt'
    
        Sample output
        ---
        'CaT'
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "".join(c.lower() if c.upper() == c else c.upper() for c in s)
    ```
    
    </details>
    
* <a name="catstrings"></a>**CatStrings** Inspired by [HumanEval](https://github.com/openai/human-eval) \#28 (5 instances)
    
    ```python
    def sat(cat: str, strings=['Will', 'i', 'am', 'Now', 'here']):
        i = 0
        for s in strings:
            for c in s:
                assert cat[i] == c
                i += 1
        return i == len(cat)
    
    def sol(strings=['Will', 'i', 'am', 'Now', 'here']):
        """
        Concatenate a list of strings
    
        Sample input
        ---
        ['cat', 'dog', 'bird']
    
        Sample output
        ---
        'catdogbird'
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "".join(strings)
    ```
    
    </details>
    
* <a name="findextensions"></a>**FindExtensions** Inspired by [HumanEval](https://github.com/openai/human-eval) \#29 (5 instances)
    
    ```python
    def sat(extensions: List[str], strings=['cat', 'dog', 'shatter', 'donut', 'at', 'todo'], prefix="do"):
        i = 0
        for s in strings:
            if s.startswith(prefix):
                assert extensions[i] == s
                i += 1
        return i == len(extensions)
    
    def sol(strings=['cat', 'dog', 'shatter', 'donut', 'at', 'todo'], prefix="do"):
        """
        Find the strings in a list starting with a given prefix
    
        Sample Input:
        ['cat', 'car', 'fear', 'center'], 'ca'
    
        Sample Output:
        ['cat', 'car']
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [s for s in strings if s.startswith(prefix)]
    ```
    
    </details>
    
* <a name="findpositives"></a>**FindPositives** Inspired by [HumanEval](https://github.com/openai/human-eval) \#30 (5 instances)
    
    ```python
    def sat(positives: List[int], nums=[2, 2342, -2, 32, -8, -5, 2342, 0, -9, 44, 11]):
        stack = positives[::-1]
        for n in nums:
            assert n <= 0 or n == stack.pop()
        return stack == []
    
    def sol(nums=[2, 2342, -2, 32, -8, -5, 2342, 0, -9, 44, 11]):
        """
        Find the positive integers in a list
    
        Sample Input:
        [-1, 3, 19, -2, 0, 44, 0, 44, 11]
    
        Sample Output:
        [3, 19, 44, 44, 11]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [i for i in nums if i > 0]
    ```
    
    </details>
    
* <a name="fermatcomposites"></a>**FermatComposites** Inspired by [HumanEval](https://github.com/openai/human-eval) \#31 (5 instances)
    
    ```python
    def sat(certificates: List[int], nums=[1449, 14, 21, 105, 217]):
        return all(pow(cert, n - 1, n) > 1 for cert, n in zip(certificates, nums)) and len(certificates) == len(nums)
    
    def sol(nums=[1449, 14, 21, 105, 217]):
        """
        Find Fermat composite certificates for a list of numbers > 1
    
        Sample Input:
        [1469]
    
        Sample Output:
        [3]  # because (3 ** 1468) % 1469 != 1
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [next(i for i in range(2, n) if pow(i, n - 1, n) > 1) for n in nums]
    ```
    
    </details>
    
* <a name="odddegreepolynomialroot"></a>**OddDegreePolynomialRoot** Polynomials of odd degree always have a real solution.
    
    Inspired by [HumanEval](https://github.com/openai/human-eval) \#32 (5 instances)
    
    ```python
    def sat(root: float, coeffs=[1, 2, 3, 17]):
        return abs(sum(coeff * (root ** i) for i, coeff in enumerate(coeffs))) < 1e-4
    
    def sol(coeffs=[1, 2, 3, 17]):
        """
        Find a real root of an odd degree polynomial from its coefficients
    
        Sample Input:
        [1, 0, 8]
    
        Sample Output:
        -2.0  # 1*(-2.0)^3 + 8 == 0
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        def p(x):
            return sum(coeff * (x ** i) for i, coeff in enumerate(coeffs))
    
        for attempt in range(100):
            a, b = -(10 ** attempt), (10 ** attempt)
            p_a, p_b = p(a), p(b)
            while p_a * p_b <= 0:
                mid = (a + b) / 2
                p_mid = p(mid)
                if abs(p_mid) < 1e-4:
                    return mid
                assert mid not in [a, b]
                if p_mid * p_a > 0:
                    a, p_a = mid, p_mid
                else:
                    b, p_b = mid, p_mid
    
        assert False, "Root finder failed on 100 attempts"
    ```
    
    </details>
    
* <a name="twothirdssorted"></a>**TwoThirdsSorted** Inspired by [HumanEval](https://github.com/openai/human-eval) \#33 (5 instances)
    
    ```python
    def sat(li: List[int], orig=[1, -2, 3, 17, 8, 4, 12, 3, 18, 5, -29, 0, 0]):
        assert orig[::3] == li[::3], "Keep every third entry fixed"
        assert sorted(li) == sorted(orig), "Not even a permutation"
        assert all(li[i] <= li[i + 1] for i in range(1, len(li) - 1, 3))
        assert all(li[i] <= li[i + 2] for i in range(2, len(li) - 2, 3))
        return True
    
    def sol(orig=[1, -2, 3, 17, 8, 4, 12, 3, 18, 5, -29, 0, 0]):
        """
        Start with a list of integers, keep every third element in place and otherwise sort the list
    
        Sample Input:
        [8, 0, 7, 2, 9, 4, 1, 2, 8, 3]
    
        Sample Output:
        [8, 0, 2, 2, 4, 8, 1, 8, 9, 3]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        n = len(orig)
        your_list = orig[::3]
        sub = orig[:]
        for i in range(int((len(sub) + 2) / 3)):
            sub.pop((2 * i))
        sub = sorted(sub)
        answ = []
        for i in range(int(n / 3)):
            answ.append(your_list[i])
            answ.append(sub[i * 2])
            answ.append(sub[i * 2 + 1])
        if n % 3 == 1:
            answ.append(your_list[-1])
        if n % 3 == 2:
            answ.append(your_list[-1])
            answ.append(sub[-1])
        return answ
    ```
    
    </details>
    
* <a name="uniquesorted"></a>**UniqueSorted** Inspired by [HumanEval](https://github.com/openai/human-eval) \#34 (5 instances)
    
    ```python
    def sat(li: List[int], orig=[1, 1, 3, 2, 0, 8, 32, -4, 0]):
        for i in range(len(li) - 1):
            assert li[i] < li[i + 1]
            assert li[i] in orig
        for n in orig:
            assert n in li
        return True
    
    def sol(orig=[1, 1, 3, 2, 0, 8, 32, -4, 0]):
        """
        Find an increasing sequence consisting of the elements of the original list.
    
        Sample Input:
        [8, 0, 7, 2, 9, 4, 4, -2, 8, 3]
    
        Sample Output:
        [-2, 0, 2, 3, 4, 7, 8, 9]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        my_list = sorted(set(orig))
        return my_list
    ```
    
    </details>
    
* <a name="maxint"></a>**MaxInt** Inspired by [HumanEval](https://github.com/openai/human-eval) \#35 (5 instances)
    
    ```python
    def sat(m: int, hello=[1, 31, 3, 2, 0, 18, 32, -4, 2, -1000, 3502145, 3502145, 21, 18, 2, 60]):
        return m in hello and not any(m < i for i in hello)
    
    def sol(hello=[1, 31, 3, 2, 0, 18, 32, -4, 2, -1000, 3502145, 3502145, 21, 18, 2, 60]):
        """
        Find the largest integer in a sequence
    
        Sample Input:
        [8, 0, 1, 4, 9, 3, 4, -2, 8, 3]
    
        Sample Output:
        9
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return max(hello)
    ```
    
    </details>
    
* <a name="seveneleventhirteen"></a>**SevenElevenThirteen** Inspired by [HumanEval](https://github.com/openai/human-eval) \#36 (4 instances)
    
    ```python
    def sat(li: List[List[int]], n=19723, lower=1000):
        assert len({(i, j) for i, j in li}) >= lower, "not enough 7's (ignoring duplicates)"
        return all(str(i)[j] == '7' and (i % 11 == 0 or i % 13 == 0) and 0 <= i < n and 0 <= j for i, j in li)
    
    def sol(n=19723, lower=1000):
        """
        Find all 7's in integers less than n that are divisible by 11 or 13
    
        Sample Input:
        79, 3
    
        Sample Output:
        [[77, 0], [77, 1], [78, 0]]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [[i, j] for i in range(n) if (i % 11 == 0 or i % 13 == 0) for j, c in enumerate(str(i)) if c == '7']
    ```
    
    </details>
    
* <a name="halfsorted"></a>**HalfSorted** Inspired by [HumanEval](https://github.com/openai/human-eval) \#37 (5 instances)
    
    ```python
    def sat(li: List[int], orig=[1, 6, 3, 41, 19, 4, 12, 3, 18, 5, -29, 0, 19521]):
        return orig[1::2] == li[1::2] and li[::2] == sorted(orig[::2])
    
    def sol(orig=[1, 6, 3, 41, 19, 4, 12, 3, 18, 5, -29, 0, 19521]):
        """
        Start with a list of integers, keep every other element in place and otherwise sort the list
    
        Sample Input:
        [8, 0, 7, 2, 9, 4, 1, 2, 8, 3]
    
        Sample Output:
        [1, 0, 2, 2, 4, 8, 8, 8, 9, 3]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        n = len(orig)
        odds = orig[1::2]
        evens = sorted(orig[::2])
        ans = []
        for i in range(len(evens)):
            ans.append(evens[i])
            if i < len(odds):
                ans.append(odds[i])
        return ans
    ```
    
    </details>
    
* <a name="threecycle"></a>**ThreeCycle** Inspired by [HumanEval](https://github.com/openai/human-eval) \#38 (5 instances)
    
    ```python
    def sat(s: str, target="Hello world"):
    
        def cycle3(trip):
            return trip if len(trip) != 3 else trip[2] + trip[:2]
    
        return target == "".join(cycle3(s[i: i + 3]) for i in range(0, len(s), 3))
    
    def sol(target="Hello world"):
        """
        Given a target string, find a string s such that when each group of three consecutive characters is cycled
        forward one character, you achieve the target string.
    
        Sample Input:
        "This is a test"
    
        Sample Output:
        'hiT is aste st'
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        def un_cycle3(trip):
            return trip if len(trip) != 3 else trip[1:3] + trip[0]
    
        return "".join(un_cycle3(target[i: i + 3]) for i in range(0, len(target), 3))
    ```
    
    </details>
    
* <a name="primefib"></a>**PrimeFib** Inspired by [HumanEval](https://github.com/openai/human-eval) \#39
    
    Ira Gessel observed that n is a Fibonacci number if and if either 5 n^2 - 4 or 5 n^2 + 4 is a perfect square (5 instances)
    
    ```python
    def sat(n: int, lower=123456):
        assert any((i ** 0.5).is_integer() for i in [5 * n * n - 4, 5 * n * n + 4]), "n must be a Fibonacci number"
        assert all(n % i for i in range(2, int(n ** 0.5) + 1)), "n must be prime"
        return n > lower
    
    def sol(lower=123456):
        """
        Find a prime Fibonacci number bigger than a certain threshold, using Ira Gessel's test for Fibonacci numbers.
    
        Sample Input:
        10
    
        Sample Output:
        11
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        m, n = 2, 3
        while True:
            m, n = n, (m + n)
            if n > lower and all(n % i for i in range(2, int(n ** 0.5) + 1)):
                return n
    ```
    
    </details>
    
* <a name="triplezerosum"></a>**TripleZeroSum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#40
        
    Similar to but harder than PairZeroSum \#43.
        
    This is a version of the classic [3SUM](https://en.wikipedia.org/wiki/3SUM) problem. (5 instances)
    
    ```python
    def sat(inds: List[int], nums=[12, 6, 41, 15, -10452, 18242, 10440, 6, 6, 6, 6]):
        return len(inds) == 3 and sum(nums[i] for i in inds) == 0
    
    def sol(nums=[12, 6, 41, 15, -10452, 18242, 10440, 6, 6, 6, 6]):
        """
        Find the indices of three numbers that sum to 0 in a list.
    
        --- Example input ---
        [1, 2, 4, -3, 5]
    
        --- Example output ---
        [0, 1, 3]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        # \tilde{O}(n^2) algorithm
        inv = {n: i for i, n in enumerate(nums)}  # note that later duplicates will override earlier entries
        for i, n in enumerate(nums):
            if inv[n] == i:
                del inv[n]
            if any((-m - n) in inv for m in nums[:i]):  # found solution!
                j, m = next((j, m) for j, m in enumerate(nums) if (-m - n) in inv)
                k = inv[-m - n]
                return sorted([i, j, k])
    ```
    
    </details>
    
* <a name="numpasses"></a>**NumPasses** Inspired by [HumanEval](https://github.com/openai/human-eval) \#41 (5 instances)
    
    ```python
    def sat(count: int, n=981):
        for i in range(n):
            for j in range(n):
                count -= 1
        return count == 0
    
    def sol(n=981):
        """
        Given n cars traveling East and n cars traveling West on a road, how many passings will there be?
        A passing is when one car passes another. The East-bound cars all begin further West than the West-bound cars.
    
        --Sample input--
        2
    
        --Sample output--
        4
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return n ** 2
    ```
    
    </details>
    
* <a name="listinc"></a>**ListInc** Increment each element of a list by 1
    
    Inspired by [HumanEval](https://github.com/openai/human-eval) \#42 (5 instances)
    
    ```python
    def sat(new_list: List[int], old_list=[321, 12, 532, 129, 9, -12, 4, 56, 90, 0]):
        return [i - 1 for i in new_list] == old_list
    
    def sol(old_list=[321, 12, 532, 129, 9, -12, 4, 56, 90, 0]):
        """
        Decrement each element of new_list by 1 and check that it's old_list
    
        Sample Input:
        [17, 15, 99]
    
        Sample Output:
        [18, 16, 100]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [i + 1 for i in old_list]
    ```
    
    </details>
    
* <a name="pairzerosum"></a>**PairZeroSum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#43
    
    Similar to TripleZeroSum \#40 (5 instances)
    
    ```python
    def sat(inds: List[int], nums=[12, -10452, 18242, 10440, 81, 241, 525, -18242, 91, 20]):
        a, b = inds
        return nums[a] + nums[b] == 0 and a >= 0 and b >= 0
    
    def sol(nums=[12, -10452, 18242, 10440, 81, 241, 525, -18242, 91, 20]):
        """
        Find the indices of two numbers that sum to 0 in a list.
    
        Sample Input:
        [1, -4, -4, 7, -3]
    
        Sample Output:
        [1, 2]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        s = set(nums)
        for i in s:
            if -i in s:
                return [nums.index(i), nums.index(-i)]
    ```
    
    </details>
    
* <a name="changebase"></a>**ChangeBase** Inspired by [HumanEval](https://github.com/openai/human-eval) \#44 (5 instances)
    
    ```python
    def sat(s: str, n=142, base=7):
        return int(s, base) == n
    
    def sol(n=142, base=7):
        """
        Write n in the given base as a string
    
        Sample Input:
        n=23, base=12
    
        Sample Output:
        '1A'
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        assert 2 <= base <= 10
        ans = ""
        while n:
            ans = str(n % base) + ans
            n //= base
        return ans or "0"
    ```
    
    </details>
    
* <a name="trianglearea"></a>**TriangleArea** Inspired by [HumanEval](https://github.com/openai/human-eval) \#45 (5 instances)
    
    ```python
    def sat(height: int, area=1319098728582, base=45126):
        return base * height == 2 * area
    
    def sol(area=1319098728582, base=45126):
        """
        Find the height of a triangle given the area and base. It is guaranteed that the answer is an integer.
    
        Sample Input:
        area = 6, base = 3
    
        Sample Output:
        4
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return (2 * area) // base
    ```
    
    </details>
    
* <a name="fib4"></a>**Fib4** Inspired by [HumanEval](https://github.com/openai/human-eval) \#46
    
    Almost identical to problem 63 (5 instances)
    
    ```python
    def sat(init: List[int], target=2021):
        a, b, c, d = init
        for i in range(99):
            a, b, c, d = b, c, d, (a + b + c + d)
        return a == target
    
    def sol(target=2021):
        """
        Define a four-wise Fibonacci sequence to be a sequence such that each number is the sum of the previous
        four. Given a target number, find an initial four numbers such that the 100th number in the sequence is the
        given target number.
    
        Sample Input:
        0
    
        Sample Output:
        [0, 0, 0, 0]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        nums = [target, 0, 0, 0]
        for i in range(99):
            x = nums[3] - sum(nums[:3])  # x is such that x + nums[:3] == nums[3]
            nums = [x] + nums[:3]
        return nums
    ```
    
    </details>
    
* <a name="median"></a>**Median** One definition of the median is a number that minimizes the sum of absolute deviations. When there are an
    even number of items, there is an interval of valid solutions.
    
    Inspired by [HumanEval](https://github.com/openai/human-eval) \#47 (5 instances)
    
    ```python
    def sat(x: int, nums=[132666041, 237412, 28141, -12, 11939, 912414, 17], upper=133658965):
        dev = sum(n - x for n in nums)
        return dev <= upper
    
    def sol(nums=[132666041, 237412, 28141, -12, 11939, 912414, 17], upper=133658965):
        """
        Find an integer that minimizes the sum of absolute deviations with respect to the given numbers.
    
        Sample Input:
        [3, 6, 1, 2, 5, 4, 100], upper=105
    
        Sample Output:
        4
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sorted(nums)[len(nums) // 2] if nums else 0
    ```
    
    </details>
    
* <a name="palindrome"></a>**Palindrome** Inspired by [HumanEval](https://github.com/openai/human-eval) \#48 (5 instances)
    
    ```python
    def sat(pals: List[bool], strs=['palindrome', 'madamimadam', '', 'foo', 'eyes', '(-:-)']):
        return all(pals[i] == (s == s[::-1]) for i, s in enumerate(strs))
    
    def sol(strs=['palindrome', 'madamimadam', '', 'foo', 'eyes', '(-:-)']):
        """
        Test whether the given strings are palindromes
    
        Sample Input:
        ["aba", "no"]
    
        Sample Output:
        [True, False]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [s == s[::-1] for s in strs]
    ```
    
    </details>
    
* <a name="littlefermat"></a>**LittleFermat** Harder but loosely inspired by [HumanEval](https://github.com/openai/human-eval) \#49 (1 instance)
    
    ```python
    def sat(exp_poly: List[int], d=74152093423, poly=[1, 6, 3, 1, 0, 4, 4]):
        p = len(poly)
        assert p > 2 and all(p % i for i in range(2, p)), "Hint: p is a prime > 2"
    
        def val(coeffs, n):  # evaluate polynomial mod p
            return sum(c * pow(n, i, p) for i, c in enumerate(coeffs)) % p
    
        return all(val(exp_poly, n) == pow(val(poly, n), d, p) for n in range(p))
    
    def sol(d=74152093423, poly=[1, 6, 3, 1, 0, 4, 4]):
        """
        Fermat's little theorem implies that any polynomial can be written equivalently as a degree p-1
        polynomial (mod p).
        Given the p coefficients of a polynomial poly, compute a polynomial equivalent to poly^d (mod p).
    
        Sample Input:
        d=2, poly=[1, 0, 0, 1, 0]  # 1 + x^3
    
        Sample Output:
        [1, 0, 1, 2, 0]  # 1+ x^2 + 2x^3 because (1 + x^3)^2 = 1 + 2x^3 + x^6 and x^6 = x^2 (mod 5)
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        """
        Use repeated squaring to exponentiate polynomial
        """
        p = len(poly)
    
        def prod(poly1, poly2):  # multiply two polynomials mod p
            ans = [0] * p
            for i, a in enumerate(poly1):
                for j, b in enumerate(poly2):
                    e = (i + j) % (p - 1)
                    if e == 0 and i + j > 1:
                        e = p - 1
                    ans[e] = (ans[e] + a * b) % p
            return ans
    
        ans = [1] + [0] * (p - 1)
        while d:
            if d % 2:
                ans = prod(ans, poly)
            poly = prod(poly, poly)
            d //= 2
        # for i in range(d):
        #     ans = prod(ans, poly)
        return ans
    ```
    
    </details>
    
* <a name="shiftchars"></a>**ShiftChars** Inspired by [HumanEval](https://github.com/openai/human-eval) \#50 (5 instances)
    
    ```python
    def sat(orig: str, result="Hello, world!", shift=7):
        n = len(result)
        assert len(orig) == n
        return all(ord(orig[i]) + shift == ord(result[i]) for i in range(n))
    
    def sol(result="Hello, world!", shift=7):
        """
        Find a string which, when each character is shifted (ascii incremented) by shift, gives the result.
    
        Sample Input:
        result='very good', shift=-1
    
        Sample Output:
        'wfsz!hppe'
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "".join(chr(ord(c) - shift) for c in result)
    ```
    
    </details>
    
* <a name="removevowels"></a>**RemoveVowels** Inspired by [HumanEval](https://github.com/openai/human-eval) \#51
    
    Related to FindVowels \#54 (5 instances)
    
    ```python
    def sat(txt: str, text="Hello, world!"):
        n = 0
        for c in text:
            if c.lower() not in "aeiou":
                assert txt[n] == c
                n += 1
        assert n == len(txt)
        return True
    
    def sol(text="Hello, world!"):
        """
        Remove the vowels from the original string.
    
        Sample Input:
        "very good"
    
        Sample Output:
        'vry gd'
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "".join(c for c in text if c.lower() not in "aeiou")
    ```
    
    </details>
    
* <a name="belowthreshold"></a>**BelowThreshold** Inspired by [HumanEval](https://github.com/openai/human-eval) \#52 (5 instances)
    
    ```python
    def sat(indexes: List[int], nums=[0, 2, 17, 4, 4213, 322, 102, 29, 15, 39, 55], thresh=100):
        j = 0
        for i, n in enumerate(nums):
            if n < thresh:
                assert indexes[j] == i
                j += 1
        assert j == len(indexes)
        return True
    
    def sol(nums=[0, 2, 17, 4, 4213, 322, 102, 29, 15, 39, 55], thresh=100):
        """
        Find the indexes of numbers below a given threshold
    
        Sample Input:
        nums=[4, 7, 11, 5], threshold=10
    
        Sample Output:
        [0, 1, 3]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [i for i, n in enumerate(nums) if n < thresh]
    ```
    
    </details>
    
* <a name="listtotal"></a>**ListTotal** Inspired by [HumanEval](https://github.com/openai/human-eval) \#53 (5 instances)
    
    ```python
    def sat(n: int, nums=[10, 42, 17, 9, 1315182, 184, 102, 29, 15, 39, 755]):
        return sum(nums + [-n]) == 0
    
    def sol(nums=[10, 42, 17, 9, 1315182, 184, 102, 29, 15, 39, 755]):
        """
        Find the number which when appended to the list makes the total 0
    
        Sample Input:
        [1, 2, 3]
    
        Sample Output:
        -6
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sum(nums)
    ```
    
    </details>
    
* <a name="diffchars"></a>**DiffChars** Inspired by [HumanEval](https://github.com/openai/human-eval) \#54 (5 instances)
    
    ```python
    def sat(c: str, a="the quick brown fox jumped over the lazy dog", b="how vexingly quick daft zebras jump"):
        return (c in a) != (c in b)
    
    def sol(a="the quick brown fox jumped over the lazy dog", b="how vexingly quick daft zebras jump"):
        """
        Find a character in one string that is not in the other.
    
        Sample Input:
        'Do you like green eggs and ham?', 'I do not like green eggs and ham.'
    
        Sample Output:
        't'  # or .?yI
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sorted(set(a).symmetric_difference(b))[0]
    ```
    
    </details>
    
* <a name="fibonacci"></a>**Fibonacci** Inspired by [HumanEval](https://github.com/openai/human-eval) \#55 (5 instances)
    
    ```python
    def sat(nums: List[int], n=1402):
        return nums[0] == nums[1] == 1 and all(nums[i + 2] == nums[i + 1] + nums[i] for i in range(n - 2))
    
    def sol(n=1402):
        """
        Find the first n Fibonacci numbers
    
        Sample Input:
        4
    
        Sample Output:
        [1, 1, 2, 3]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        ans = [1, 1]
        while len(ans) < n:
            ans.append(ans[-1] + ans[-2])
        return ans
    ```
    
    </details>
    
* <a name="matchbrackets"></a>**MatchBrackets** Inspired by [HumanEval](https://github.com/openai/human-eval) \#56 (5 instances)
    
    ```python
    def sat(matches: List[int], brackets="<<>><<<><>><<>>>"):
        for i in range(len(brackets)):
            j = matches[i]
            c = brackets[i]
            assert brackets[j] != c and matches[j] == i and all(i < matches[k] < j for k in range(i + 1, j))
        return len(matches) == len(brackets)
    
    def sol(brackets="<<>><<<><>><<>>>"):
        """
        Find the index of the matching brackets for each character in the string
    
        Sample Input:
        "<><>"
    
        Sample Output:
        [1, 0, 3, 2]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        matches = [-1] * len(brackets)
        opens = []
        for i, c in enumerate(brackets):
            if c == "<":
                opens.append(i)
            else:
                assert c == ">"
                j = opens.pop()
                matches[i] = j
                matches[j] = i
        return matches
    ```
    
    </details>
    
* <a name="monotonic"></a>**Monotonic** Inspired by [HumanEval](https://github.com/openai/human-eval) \#57 (5 instances)
    
    ```python
    def sat(direction: str, nums=[2, 4, 17, 29, 31, 1000, 416629]):
        if direction == "increasing":
            return all(nums[i] < nums[i + 1] for i in range(len(nums) - 1))
        if direction == "decreasing":
            return all(nums[i + 1] < nums[i] for i in range(len(nums) - 1))
    
    def sol(nums=[2, 4, 17, 29, 31, 1000, 416629]):
        """
        Determine the direction ('increasing' or 'decreasing') of monotonic sequence nums
    
        Sample Input:
        [1, 2, 5]
    
        Sample Output:
        "increasing"
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "increasing" if len(nums) > 1 and nums[1] > nums[0] else "decreasing"
    ```
    
    </details>
    
* <a name="commonnumbers"></a>**CommonNumbers** Inspired by [HumanEval](https://github.com/openai/human-eval) \#58 (5 instances)
    
    ```python
    def sat(common: List[int], a=[2, 416629, 2, 4, 17, 29, 31, 1000], b=[31, 2, 4, 17, 29, 41205]):
        return all((i in common) == (i in a and i in b) for i in a + b + common)
    
    def sol(a=[2, 416629, 2, 4, 17, 29, 31, 1000], b=[31, 2, 4, 17, 29, 41205]):
        """
        Find numbers common to a and b
    
        Sample Input:
        [1, 2, 3], [3, 4, 5]
    
        Sample Output:
        [3]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sorted(set(a).intersection(set(b)))
    ```
    
    </details>
    
* <a name="largestprimefactor"></a>**LargestPrimeFactor** Inspired by [HumanEval](https://github.com/openai/human-eval) \#59 (5 instances)
    
    ```python
    def sat(p: int, n=101076):
    
        def is_prime(m):
            return all(m % i for i in range(2, m - 1))
    
        return is_prime(p) and n % p == 0 and p > 0 and all(n % i or not is_prime(i) for i in range(p + 1, n))
    
    def sol(n=101076):
        """
        Find the largest prime factor of n.
    
        Sample Input:
        125
    
        Sample Output:
        5
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        def is_prime(m):
            return all(m % i for i in range(2, m - 1))
    
        return next(n // i for i in range(1, n) if n % i == 0 and is_prime(n // i))
    ```
    
    </details>
    
* <a name="cumulativesums"></a>**CumulativeSums** Inspired by [HumanEval](https://github.com/openai/human-eval) \#60 (5 instances)
    
    ```python
    def sat(sums: List[int], n=104):
        return all(sums[i + 1] - sums[i] == i for i in range(n)) and sums[0] == 0
    
    def sol(n=104):
        """
        Find the sums of the integers from 1 to n
    
        Sample Input:
        3
    
        Sample Output:
        [0, 1, 3, 6]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        ans = [0]
        for i in range(n):
            ans.append(ans[-1] + i)
        return ans
    ```
    
    </details>
    
* <a name="parendepth"></a>**ParenDepth** Inspired by [HumanEval](https://github.com/openai/human-eval) \#61
    
    Note that problems 61 and 56 are essentially the same (5 instances)
    
    ```python
    def sat(matches: List[int], parens="((())()(()()))(())"):
        for i, (j, c) in enumerate(zip(matches, parens)):
            assert parens[j] != c and matches[j] == i and all(i < matches[k] < j for k in range(i + 1, j))
        return len(matches) == len(parens)
    
    def sol(parens="((())()(()()))(())"):
        """
        Find the index of the matching parentheses for each character in the string
    
        Sample Input:
        "()((()))"
    
        Sample Output:
        [1, 0, 7, 6, 5, 4, 3, 2]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        matches = [-1] * len(parens)
        opens = []
        for i, c in enumerate(parens):
            if c == "(":
                opens.append(i)
            else:
                assert c == ")"
                j = opens.pop()
                matches[i] = j
                matches[j] = i
        return matches
    ```
    
    </details>
    
* <a name="derivative"></a>**Derivative** Inspired by [HumanEval](https://github.com/openai/human-eval) \#62
    
    This puzzle gives the raw definition of a derivative in terms of small changes in x. (5 instances)
    
    ```python
    def sat(derivative: List[int], poly=[2, 1, 0, 4, 19, 231, 0, 5]):
    
        def val(poly, x):
            return sum(coeff * (x ** i) for i, coeff in enumerate(poly))
    
        return all(abs(val(poly, x + 1e-8) - val(poly, x) - 1e-8 * val(derivative, x)) < 1e-4 for x in range(len(poly)))
    
    def sol(poly=[2, 1, 0, 4, 19, 231, 0, 5]):
        """
        Find the derivative of the given polynomial, with coefficients in order of increasing degree
    
        Sample Input:
        [3, 4, 1] # 3 + 4x + x^2
    
        Sample Output:
        [2, 4]   # 4 + 2x^2
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [i * poly[i] for i in range(1, len(poly))]
    ```
    
    </details>
    
* <a name="fib3"></a>**Fib3** Inspired by [HumanEval](https://github.com/openai/human-eval) \#63
    
    Almost identical to problem 46 (5 instances)
    
    ```python
    def sat(init: List[int], target=124156):
        a, b, c = init
        for i in range(16):
            a, b, c = b, c, (a + b + c)
        return a == target
    
    def sol(target=124156):
        """
        Define a triple-Fibonacci sequence to be a sequence such that each number is the sum of the previous
        three. Given a target number, find an initial triple such that the 17th number in the sequence is the
        given target number.
    
        Sample Input:
        0
    
        Sample Output:
        [0, 0, 0]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        nums = [target, 0, 0]
        for i in range(16):
            x = nums[-1] - sum(nums[:-1])  # x is such that x + nums[:3] == nums[3]
            nums = [x] + nums[:-1]
        return nums
    ```
    
    </details>
    
* <a name="findvowels"></a>**FindVowels** Inspired by [HumanEval](https://github.com/openai/human-eval) \#64
    
    Very similar to RemoveVowels \#51 (5 instances)
    
    ```python
    def sat(vowels: List[str], texts=['Hello, world!', 'Goodbye, world!']):
        for v, t in zip(vowels, texts):
            i = 0
            for j, c in enumerate(t):
                if c.lower() in "aeiou" or c.lower() == 'y' and j == len(t) - 1:
                    assert v[i] == c
                    i += 1
            assert i == len(v)
        return len(vowels) == len(texts)
    
    def sol(texts=['Hello, world!', 'Goodbye, world!']):
        """
        Find the vowels from each of the original texts (y counts as a vowel at the end of the word)
    
        Sample Input:
        ["You can do it!", "CAT"]
    
        Sample Output:
        ["ouaoi", "A"]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return ["".join(c for c in text if c.lower() in "aeiou") + (text[-1] if text[-1].lower() == "y" else "")
                for text in texts]
    ```
    
    </details>
    
* <a name="circularshiftnum"></a>**CircularShiftNum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#65 (5 instances)
    
    ```python
    def sat(shifted: str, n=124582369835, shift=3):
        if shift > len(str(n)):
            return n == int(shifted[::-1])
        return n == int(shifted[-shift:] + shifted[:-shift])
    
    def sol(n=124582369835, shift=3):
        """
        Shift the decimal digits n places to the left, wrapping the extra digits around. If shift > the number of
        digits of n, reverse the string.
    
        n=12345 shift=2 => '34512'
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        s = str(n)
        if shift > len(s):
            return s[::-1]
        return s[shift:] + s[:shift]
    ```
    
    </details>
    
* <a name="charsum"></a>**CharSum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#66 (5 instances)
    
    ```python
    def sat(tot: int, s="Add ME uP AND YOU WILL GET A BIG NUMBER!"):
        for c in s:
            if c.isupper():
                tot -= ord(c)
        return tot == 0
    
    def sol(s="Add ME uP AND YOU WILL GET A BIG NUMBER!"):
        """
        Compute the sum of the ASCII values of the upper-case characters in the string.
    
        Sample Input:
        ARt
    
        Sample Output:
        147 # = 65 + 82
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sum(ord(c) for c in s if c.isupper())
    ```
    
    </details>
    
* <a name="missingbananas"></a>**MissingBananas** Inspired by [HumanEval](https://github.com/openai/human-eval) \#67 (5 instances)
    
    ```python
    def sat(bananas: int, bowl="5024 apples and 12189 oranges", total=12491241):
        bowl += f" and {bananas} bananas"
        return sum([int(s) for s in bowl.split() if s.isdigit()]) == total
    
    def sol(bowl="5024 apples and 12189 oranges", total=12491241):
        """
        Determine how many bananas are necessary to reach a certain total amount of fruit
    
        bowl="3 apples and 4 oranges", total=12 => 5
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        apples, oranges = [int(s) for s in bowl.split() if s.isdigit()]
        return total - apples - oranges
    ```
    
    </details>
    
* <a name="smallesteven"></a>**SmallestEven** Inspired by [HumanEval](https://github.com/openai/human-eval) \#68 (5 instances)
    
    ```python
    def sat(val_index: List[int], nums=[125123, 422323, 141, 5325, 812152, 9, 42145, 5313, 421, 812152]):
        if val_index == []:
            return all(n % 2 == 1 for n in nums)
        v, i = val_index
        assert v % 2 == 0 and nums[i] == v
        return all(n > v or n % 2 == 1 for n in nums[:i]) and all(n >= v or n % 2 == 1 for n in nums[i:])
    
    def sol(nums=[125123, 422323, 141, 5325, 812152, 9, 42145, 5313, 421, 812152]):
        """
        Given an array of nums representing a branch on a binary tree, find the minimum even value and its index.
        In the case of a tie, return the smallest index. If there are no even numbers, the answer is [].
    
        Sample Input:
        [1, 7, 4, 6, 10, 11, 14]
    
        Sample Output:
        [4, 2]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if any(n % 2 == 0 for n in nums):
            return min([v, i] for i, v in enumerate(nums) if v % 2 == 0)
        else:
            return []
    ```
    
    </details>
    
* <a name="greatesthindex"></a>**GreatestHIndex** Inspired by [HumanEval](https://github.com/openai/human-eval) \#69 (5 instances)
    
    ```python
    def sat(h: int, seq=[3, 1, 4, 17, 5, 17, 2, 1, 41, 32, 2, 5, 5, 5, 5]):
        for i in seq:
            assert not (i > 0 and i > h and seq.count(i) >= i)
        return h == -1 or seq.count(h) >= h > 0
    
    def sol(seq=[3, 1, 4, 17, 5, 17, 2, 1, 41, 32, 2, 5, 5, 5, 5]):
        """
        Find the h-index, the largest positive number h such that that h occurs in the sequence at least h times.
        h = -1 if there is no such positive number.
    
        Sample Input:
        [1, 2, 2, 3, 3, 3, 4, 4]
    
        Sample Output:
        3
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return max([-1] + [i for i in seq if i > 0 and seq.count(i) >= i])
    ```
    
    </details>
    
* <a name="wildsort"></a>**WildSort** Inspired by [HumanEval](https://github.com/openai/human-eval) \#70 (5 instances)
    
    ```python
    def sat(strange: List[int], li=[30, 12, 42, 717, 45, 317, 200, -1, 491, 32, 15]):
        assert sorted(strange) == sorted(li), "Must be a permutation"
        return all(n == (min, max)[i % 2](strange[i:]) for i, n in enumerate(strange))
    
    def sol(li=[30, 12, 42, 717, 45, 317, 200, -1, 491, 32, 15]):
        """
        Find the following strange sort of li: the first element is the smallest, the second is the largest of the
        remaining, the third is the smallest of the remaining, the fourth is the smallest of the remaining, etc.
    
        Sample Input:
        [1, 2, 7, 3, 4, 5, 6]
    
        Sample Output:
        [1, 7, 2, 6, 3, 5, 4]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        s = sorted(li)
        i = 0
        j = len(li) - 1
        ans = []
        while i <= j:
            if len(ans) % 2:
                ans.append(s[j])
                j -= 1
            else:
                ans.append(s[i])
                i += 1
        return ans
    ```
    
    </details>
    
* <a name="herontriangle"></a>**HeronTriangle** Inspired by [HumanEval](https://github.com/openai/human-eval) \#71
    
    That problem essentially asks for Heron's formula for the area of a triangle in terms of its three sides.
    In our version, we consider the related problem (also solved by Heron's formula) of finding 2d coordinates
    of a triangle with the given sides. If one knows the area, this is a straightforward calculation. (5 instances)
    
    ```python
    def sat(coords: List[List[float]], sides=[8.9, 10.8, 17.0]):
        assert len(coords) == 3
        sides2 = [((x - x2) ** 2 + (y - y2) ** 2) ** 0.5 for i, (x, y) in enumerate(coords) for x2, y2 in coords[:i]]
        return all(abs(a - b) < 1e-6 for a, b in zip(sorted(sides), sorted(sides2)))
    
    def sol(sides=[8.9, 10.8, 17.0]):
        """
        Find the coordinates of a triangle with the given side lengths
    
        Sample Input:
        [3.0, 4.0, 5.0
    
        Sample Output:
        [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        a, b, c = sorted(sides)
    
        s = sum(sides) / 2  # semi-perimeter
        area = (s * (s - a) * (s - b) * (s - c)) ** 0.5  # Heron's formula
    
        y = 2 * area / a  # height
        x = (c ** 2 - y ** 2) ** 0.5
        return [[0.0, 0.0], [a, 0.0], [x, y]]
    ```
    
    </details>
    
* <a name="investigatecrash"></a>**InvestigateCrash** Inspired by [HumanEval](https://github.com/openai/human-eval) \#72 (5 instances)
    
    ```python
    def sat(problem: int, weights=[1, 2, 5, 2, 1, 17], max_weight=100):
        if problem == -1:
            return sum(weights) > max_weight
        return weights[problem] != weights[- 1 - problem]
    
    def sol(weights=[1, 2, 5, 2, 1, 17], max_weight=100):
        """
        An object will "fly" if its weights are a palindrome and sum to <= max_weight. The given object won't fly.
        You have to determine why. Find index where the weights aren't a palindrome or -1 if weights are too big.
    
        weights=[77, 40], max_weight=100 => -1
    
        weights=[1,2,3], max_weight=50   => 0 # because 1 != 3
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if sum(weights) > max_weight:
            return -1
        return next(i for i, w in enumerate(weights) if weights[-i - 1] != weights[i])
    ```
    
    </details>
    
* <a name="closestpalindrome"></a>**ClosestPalindrome** Inspired by [HumanEval](https://github.com/openai/human-eval) \#73 (5 instances)
    
    ```python
    def sat(pal: str, s="palindromordinals"):
        assert pal == pal[::-1] and len(pal) == len(s)
        return sum(a != b for a, b in zip(pal, s)) == sum(a != b for a, b in zip(s, s[::-1])) // 2
    
    def sol(s="palindromordinals"):
        """
        Find the closest palindrome
    
        Sample Input:
        "cat"
    
        Sample Output:
        "tat"
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        n = len(s)
        return s[:(n + 1) // 2] + s[:n // 2][::-1]
    ```
    
    </details>
    
* <a name="narrowerlist"></a>**NarrowerList** Inspired by [HumanEval](https://github.com/openai/human-eval) \#74 (5 instances)
    
    ```python
    def sat(li: List[str], lists=[['this', 'list', 'is', 'narrow'], ['I', 'am', 'shorter but wider']]):
        width = sum(len(s) for s in li)
        for li2 in lists:
            assert width <= sum(len(s) for s in li2)
        return li in lists
    
    def sol(lists=[['this', 'list', 'is', 'narrow'], ['I', 'am', 'shorter but wider']]):
        """
        Find the list that has fewer total characters (including repetitions)
    
        Sample Input:
        [["sh", "ort"], ["longest"]]
    
        Sample Output:
        [["sh", "ort"]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return min(lists, key=lambda x: sum(len(i) for i in x))
    ```
    
    </details>
    
* <a name="threeprimes"></a>**ThreePrimes** Inspired by [HumanEval](https://github.com/openai/human-eval) \#75 (1 instance)
    
    ```python
    def sat(factors: List[List[int]]):
        primes = set(range(2, 1000))
        for n in range(2, 1000):
            if n in primes:
                primes.difference_update(range(2 * n, 1000, n))
        assert all(p in primes for f in factors for p in f), "all factors must be prime"
        nums = {p * q * r for p, q, r in factors}
        return max(nums) < 1000 and len(nums) == 247
    
    def sol():
        """
        Find all 247 integers <= 1000 that are the product of exactly three primes.
        Each integer should represented as the list of its three prime factors.
        [[2, 2, 2], [2, 2, 3],  [2, 2, 5], ...
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        primes = set(range(2, 1000))
        for n in range(2, 1000):
            if n in primes:
                primes.difference_update(range(2 * n, 1000, n))
        return [[p, q, r] for p in primes for q in primes if p <= q for r in primes if q <= r and p * q * r < 1000]
    ```
    
    </details>
    
* <a name="integerlog"></a>**IntegerLog** Inspired by [HumanEval](https://github.com/openai/human-eval) \#76 (5 instances)
    
    ```python
    def sat(x: int, a=3, n=1290070078170102666248196035845070394933441741644993085810116441344597492642263849):
        return a ** x == n
    
    def sol(a=3, n=1290070078170102666248196035845070394933441741644993085810116441344597492642263849):
        """Find an integer exponent x such that a^x = n
        Sample Input:
        a=2, n=1024
    
        Sample Output:
        x = 10
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        m = 1
        x = 0
        while m != n:
            x += 1
            m *= a
        return x
    ```
    
    </details>
    
* <a name="cuberoot"></a>**CubeRoot** Inspired by [HumanEval](https://github.com/openai/human-eval) \#77
    
    We made it harder by giving very large n for which `round(n ** (1/3))` (5 instances)
    
    ```python
    def sat(x: int, n=42714774173606970182754018064350848294149432972747296768):
        return x ** 3 == n
    
    def sol(n=42714774173606970182754018064350848294149432972747296768):
        """Find an integer that when cubed is n
    
        Sample Input:
        21
    
        Sample Output:
        3
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        # Using Newton's method
        m = abs(n)
        x = round(abs(n) ** (1 / 3))
        while x ** 3 != m:
            x += (m - x ** 3) // (3 * x ** 2)
        return -x if n < 0 else x
    ```
    
    </details>
    
* <a name="hexprimes"></a>**HexPrimes** Inspired by [HumanEval](https://github.com/openai/human-eval) \#78 (5 instances)
    
    ```python
    def sat(primes: List[bool], n="A4D4455214122CE192CCBE3"):
        return all(primes[i] == (c in "2357BD") for i, c in enumerate(n))
    
    def sol(n="A4D4455214122CE192CCBE3"):
        """Determine which characters of a hexidecimal correspond to prime numbers
    
        Sample Input:
        "123ABCD"
    
        Sample Output:
        [False, True, True, False, True, False True]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [c in "2357BD" for c in n]
    ```
    
    </details>
    
* <a name="binarize"></a>**Binarize** Inspired by [HumanEval](https://github.com/openai/human-eval) \#79 (5 instances)
    
    ```python
    def sat(b: str, n=5324680297138495285):
        assert b[:4] == b[-4:] == 'bits'
        inside = b[4:-4]
        assert all(c in "01" for c in inside)
        assert inside[0] == "1" or len(inside) == 1
        m = 0
        for c in inside:
            m = 2 * m + int(c)
        return m == n
    
    def sol(n=5324680297138495285):
        """Write n base 2 followed and preceded by 'bits'
        Sample Input:
        2
    
        Sample Output:
        bits10bits
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        s = bin(n)[2:]
        return f'bits{s}bits'
    ```
    
    </details>
    
* <a name="nearbyduplicates"></a>**NearbyDuplicates** Inspired by [HumanEval](https://github.com/openai/human-eval) \#80 (5 instances)
    
    ```python
    def sat(indices: List[int], s="I am an unhappy string!"):
        i, j = indices
        return s[i] == s[j] and 0 <= i < j < i + 3
    
    def sol(s="I am an unhappy string!"):
        """A string is happy if every three consecutive characters are distinct. Find two indices making s unhappy.
        Sample Input:
        "street"
    
        Sample Output:
        [3, 4]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        for i in range(len(s) - 2):
            if s[i] == s[i + 1]:
                return [i, i + 1]
            if s[i] == s[i + 2]:
                return [i, i + 2]
    ```
    
    </details>
    
* <a name="grader"></a>**Grader** Inspired by [HumanEval](https://github.com/openai/human-eval) \#81 (5 instances)
    
    ```python
    def sat(grades: List[str], gpas=[2.8, 3.1, 4.0, 2.2, 3.1, 2.5, 0.9]):
        assert len(grades) == len(gpas)
        letters = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'F']
        scores = [4.0, 3.7, 3.4, 3.0, 2.7, 2.4, 2.0, 1.7, 1.4, 0.0]
        for grade, gpa in zip(grades, gpas):
            i = letters.index(grade)
            assert gpa >= scores[i]
            assert i == 0 or gpa <= scores[i - 1]
        return True
    
    def sol(gpas=[2.8, 3.1, 4.0, 2.2, 3.1, 2.5, 0.9]):
        """
        Convert GPAs to letter grades according to the following table:
        4.0: A+
        3.7: A
        3.4: A-
        3.0: B+
        2.7: B
        2.4: B-
        2.0: C+
        1.7: C
        1.4: C-
        below: F
    
        Sample input: [4.0, 3.5, 3.8]
        Sample output: ['A+', 'A-', 'A']
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        letters = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'F']
        scores = [4.0, 3.7, 3.4, 3.0, 2.7, 2.4, 2.0, 1.7, 1.4, 0.0]
        ans = []
        for gpa in gpas:
            i = 0
            while gpa < scores[i]:
                i += 1
            ans.append(letters[i])
        return ans
    ```
    
    </details>
    
* <a name="factorstring"></a>**FactorString** Inspired by [HumanEval](https://github.com/openai/human-eval) \#82 (5 instances)
    
    ```python
    def sat(factor: str, s="catscatcatscatcatscat"):
        return len(factor) < len(s) and s == factor * (len(s) // len(factor))
    
    def sol(s="catscatcatscatcatscat"):
        """Find a string which when repeated more than once gives s
        Sample Input:
        "haha"
    
        Sample Output:
        "ha"
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        n = len(s)
        return next(s[:i] for i in range(1, len(s)) if s == s[:i] * (n // i))
    ```
    
    </details>
    
* <a name="oneended"></a>**OneEnded** Inspired by [HumanEval](https://github.com/openai/human-eval) \#83 (1 instance)
    
    ```python
    def sat(nums: List[int], n=5):
        count = 18 * (10 ** (n - 2)) if n > 1 else 1
        strs = {str(n) for n in nums}
        return len(strs) == count and all(s.startswith("1") or s.endswith("1") and len(s) == n for s in strs)
    
    def sol(n=5):
        """Find all n-digit integers that start or end with 1
    
        1 => [1]"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        ans = []
        for i in range(10 ** (n - 1), 10 ** n):
            assert len(str(i)) == n
            if str(i).startswith("1") or str(i).endswith("1"):
                ans.append(i)
        return ans
    ```
    
    </details>
    
* <a name="bitsum"></a>**BitSum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#84 (5 instances)
    
    ```python
    def sat(n: int, b=107, s=25):
        n_str = bin(n)[2:]  # n in binary
        return len(n_str) == b and sum(int(i) for i in n_str) == s
    
    def sol(b=107, s=25):
        """Find an b-bit integer with a bit-sum of s
    
        b=3, s=2 => 5 # 5 is 101 in binary
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return int("1" * s + "0" * (b - s), 2)
    ```
    
    </details>
    
* <a name="evenoddsum"></a>**EvenOddSum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#85
    
    Very similar to OddEvenSum \#121 (5 instances)
    
    ```python
    def sat(even_odd_sum: int, nums=[2341, 125146894, 12521, -12451293476325, 535284623934, 132974693614350]):
        for i in nums[1::2]:
            if i % 2 == 0:
                even_odd_sum -= i
        return even_odd_sum == 0
    
    def sol(nums=[2341, 125146894, 12521, -12451293476325, 535284623934, 132974693614350]):
        """Find the sum of the even elements that are at odd indices
    
        [1, 2, 8, 3, 9, 4] => 6
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sum(i for i in nums[1::2] if i % 2 == 0)
    ```
    
    </details>
    
* <a name="antishuffle"></a>**AntiShuffle** Inspired by [HumanEval](https://github.com/openai/human-eval) \#86 (5 instances)
    
    ```python
    def sat(s: str, orig="Hello world!!!"):
        for a, b in zip(s.split(' '), orig.split(' ')):
            for i in range(len(a) - 1):
                assert a[i] <= a[i + 1], "characters must s-words be in increasing order"
            assert len(a) == len(b) and all(a.count(c) == b.count(c) for c in b), "must have same chars"
        return len(s) == len(orig)
    
    def sol(orig="Hello world!!!"):
        """Create a new string by taking s, and word by word rearranging its characters in ascii order
        Sample input:
        'maltos wow'
    
        Sample output:
        'almost oww'
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return " ".join("".join(sorted(w)) for w in orig.split(' '))
    ```
    
    </details>
    
* <a name="unevenfind"></a>**UnevenFind** Inspired by [HumanEval](https://github.com/openai/human-eval) \#87 (5 instances)
    
    ```python
    def sat(indices: List[List[int]], uneven=[[1, 3, 2, 32, 17], [17, 2, 48, 17], [], [9, 35, 4], [3, 17]], target=17):
        for i, j in indices:
            assert uneven[i][j] == target
        for i, row in enumerate(uneven):
            for j, n in enumerate(row):
                assert n != target or [i, j] in indices
        return True
    
    def sol(uneven=[[1, 3, 2, 32, 17], [17, 2, 48, 17], [], [9, 35, 4], [3, 17]], target=17):
        """Find the indices of all occurrences of target in the uneven matrix
        Sample input:
        uneven=[[2, 3, 2], [], [9, 2]], target=2
    
        Sample output:
        [[0, 0], [0, 2], [2, 1]]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [[i, j] for i, row in enumerate(uneven) for j, n in enumerate(row) if n == target]
    ```
    
    </details>
    
* <a name="updownsort"></a>**UpDownSort** Inspired by [HumanEval](https://github.com/openai/human-eval) \#88 (1 instance)
    
    ```python
    def sat(up_down: List[int], nums=[17, 2, 3, 523, 18, -2, 0, 2, -1]):
        assert all(up_down.count(i) == nums.count(i) for i in set(up_down + nums)), "not a reordering"
        increasing_sign = 1 if ((nums[0] + nums[-1]) % 2 == 1) else -1
        return all((up_down[i + 1] - up_down[i]) * increasing_sign >= 0 for i in range(len(up_down) - 1))
    
    def sol(nums=[17, 2, 3, 523, 18, -2, 0, 2, -1]):
        """Reorder nums in increasing/decreasing order based on whether the first plus last element is even/odd
    
        Sample input:
        [1, 7, 4]
    
        Sample output:
        [1, 4, 7] # because 1 + 4 is odd
    
        Sample input:
        [1, 7, 5]
    
        Sample output:
        [8, 5, 1] # because 1 + 5 is even
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sorted(nums, reverse=(False if (nums[0] + nums[-1]) % 2 else True))
    ```
    
    </details>
    
* <a name="substitutioncypher"></a>**SubstitutionCypher** Inspired by [HumanEval](https://github.com/openai/human-eval) \#89 (5 instances)
    
    ```python
    def sat(encrypted: str, orig="Hello, world!"):
        assert len(encrypted) == len(orig)
        return all(chr(ord(a) - 2 * 2) == b for a, b in zip(encrypted, orig))
    
    def sol(orig="Hello, world!"):
        """Apply a substitution cypher in which each character is advanced by two multiplied by two places.
    
        'substitution cypher' => 'wyfwxmxyxmsr$g}tliv'
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "".join(chr(ord(b) + 2 * 2) for b in orig)
    ```
    
    </details>
    
* <a name="secondsmallestunique"></a>**SecondSmallestUnique** Inspired by [HumanEval](https://github.com/openai/human-eval) \#90 (5 instances)
    
    ```python
    def sat(n: int, nums=[17, -1023589211, -293485382500, 31, -293485382500, 105762, 94328103589]):
        assert n in nums
        return len({i for i in nums if i <= n}) == 2
    
    def sol(nums=[17, -1023589211, -293485382500, 31, -293485382500, 105762, 94328103589]):
        """Find the second smallest unique number in the list nums.
    
        Sample input:
        [2, 5, 2, 7, 9]
    
        Sample output:
        5
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sorted(set(nums))[1]
    ```
    
    </details>
    
* <a name="findbored"></a>**FindBored** Inspired by [HumanEval](https://github.com/openai/human-eval) \#91 (5 instances)
    
    ```python
    def sat(boring: List[str], text="This is not boring. I am boring! I am sooo tired."):
        sentences = text.replace("!", ".").replace("?", ".").split(".")
        boring_and_exciting = boring + [s for s in sentences if s.split()[:1] != ["I"]]
        return sorted(boring_and_exciting) == sorted(sentences)
    
    def sol(text="This is not boring. I am boring! I am sooo tired."):
        """A bored sentence starts with the word "I". Find all bored sentences in s. Sentence delimiters are '.!?'
    
        --- Example input ---
        'I wrote this. You read it? I think I am so cool. In another time, I would be lame.'
    
        --- Example output ---
        ['I wrote this', ' I think I am so cool']
    
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.split()[:1] == ["I"]]
    ```
    
    </details>
    
* <a name="identifyzerotrips"></a>**IdentifyZeroTrips** Inspired by [HumanEval](https://github.com/openai/human-eval) \#92 (5 instances)
    
    ```python
    def sat(zero_sums: List[bool], trips=[[1253532, -3920635, 332], [-24, 18, 6], [0, 5, -5], [1, 1, 1], [-20, 17, 4]]):
        return len(zero_sums) == len(trips) and all(z == ((a + b + c) == 0) for z, (a, b, c) in zip(zero_sums, trips))
    
    def sol(trips=[[1253532, -3920635, 332], [-24, 18, 6], [0, 5, -5], [1, 1, 1], [-20, 17, 4]]):
        """Determine which triples sum to zero
    
        --- Example input ---
        [1, 2, 4, -3, 5]
    
        --- Example output ---
        [0, 1, 3]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [sum(t) == 0 for t in trips]
    ```
    
    </details>
    
* <a name="weirddecodevowels"></a>**WeirdDecodeVowels** Inspired by [HumanEval](https://github.com/openai/human-eval) \#93 (5 instances)
    
    ```python
    def sat(s: str, target="Hello, world!"):
        subs = {ord(c): ord(c) + 2 for c in "aeiouAEIOU"}
        return s.swapcase() == target.translate(subs)
    
    def sol(target="Hello, world!"):
        """Find string s that, when case is flipped gives target where vowels are replaced by chars two later.
        --- Example input ---
        'THIS is a TEST'
    
        --- Example output ---
        'thks KS C tgst'
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        subs = {ord(c): ord(c) + 2 for c in "aeiouAEIOU"}
        return target.translate(subs).swapcase()
    ```
    
    </details>
    
* <a name="largestprimedigitsum"></a>**LargestPrimeDigitSum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#94 (5 instances)
    
    ```python
    def sat(ans: List[int], nums=[23, 17, 201, 14, 10473, 43225, 421, 423, 11, 10, 2022, 342157]):
        i, digit_sum = ans
        n = nums[i]
    
        def is_prime(n):
            return n > 1 and all(n % j for j in range(2, int(n ** 0.5) + 1))
    
        return is_prime(n) and all(m <= n for m in nums if is_prime(m)) and digit_sum == sum(int(c) for c in str(n))
    
    def sol(nums=[23, 17, 201, 14, 10473, 43225, 421, 423, 11, 10, 2022, 342157]):
        """Find the index of the largest prime in the list and the sum of its digits
    
        --- Example input ---
        [2, 4, 7, 19, 21]
    
        --- Example output ---
        [3, 10]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        def is_prime(n):
            return n > 1 and all(n % j for j in range(2, int(n ** 0.5) + 1))
    
        n, i = max((n, i) for i, n in enumerate(nums) if is_prime(n))
        return [i, sum(int(c) for c in str(n))]
    ```
    
    </details>
    
* <a name="oddcase"></a>**OddCase** Inspired by [HumanEval](https://github.com/openai/human-eval) \#95 (5 instances)
    
    ```python
    def sat(different: str, d={'cat': 'CAT', 'tree': 'T', 'pick me': 'not', 'OK': 'red', 'blah': 'blah', 'z': 'Z'}):
        return different in d and all(k.islower() != different.islower() for k in d if k != different)
    
    def sol(d={'cat': 'CAT', 'tree': 'T', 'pick me': 'not', 'OK': 'red', 'blah': 'blah', 'z': 'Z'}):
        """Find the dictionary key whose case is different than all other keys
    
        --- Example input ---
        {"red": "", "GREEN": "", "blue": "orange"}
    
        --- Example output ---
        "GREEN"
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        for different in d:
            if all(k.islower() != different.islower() for k in d if k != different):
                return different
    ```
    
    </details>
    
* <a name="primesupto"></a>**PrimesUpTo** Inspired by [HumanEval](https://github.com/openai/human-eval) \#96 (5 instances)
    
    ```python
    def sat(primes: List[int], n=1234):
        assert all(1 < p for p in primes) and all(p % q for p in primes for q in primes if q < p)
        return len({i for p in primes for i in range(p, n, p)}) == max(n - 2, 0)
    
    def sol(n=1234):
        """Find all primes up to n
    
        --- Example input ---
        9
    
        --- Example output ---
        [2, 3, 5, 7]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        primes = []
        candidates = set(range(2, n))
        for i in range(2, n):
            if i in candidates:
                primes.append(i)
                candidates.difference_update(range(i, n, i))
        return primes
    ```
    
    </details>
    
* <a name="unitsproduct"></a>**UnitsProduct** Inspired by [HumanEval](https://github.com/openai/human-eval) \#97 (5 instances)
    
    ```python
    def sat(prod: int, nums=[17, 24, 39, 15, 11, 201, 97, 65, 18]):
        if not all(nums):
            return prod == 0
        for n in nums:
            k = abs(n % 10)
            if k == 0:
                return prod == 0
            assert prod % k == 0
            prod //= k
        return prod == 1
    
    def sol(nums=[17, 24, 39, 15, 11, 201, 97, 65, 18]):
        """Find the product of the units digits in the numbers
    
        [12, 34] => 8
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        prod = 1
        for n in nums:
            prod *= abs(n % 10)
        return prod
    ```
    
    </details>
    
* <a name="uppercaseeven"></a>**UppercaseEven** Inspired by [HumanEval](https://github.com/openai/human-eval) \#98 (5 instances)
    
    ```python
    def sat(positions: List[int], s="ThIs is A tEsT, Or *IS* iT?"):
        assert all(s[i] in "AEIOU" for i in positions)
        return all(i in positions or c not in "AEIOU" or i % 2 == 1 for i, c in enumerate(s))
    
    def sol(s="ThIs is A tEsT, Or *IS* iT?"):
        """Find the positions of all uppercase vowels (not counting Y) in even indices
    
        "EAT here NOW" => [0, 10]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [i for i, c in enumerate(s) if i % 2 == 0 and c in "AEIOU"]
    ```
    
    </details>
    
* <a name="closestinteger"></a>**ClosestInteger** Inspired by [HumanEval](https://github.com/openai/human-eval) \#99
    
    Since we can tolerate more than one answer per puzzle, we do not need to specify a tie-breaking rule. (5 instances)
    
    ```python
    def sat(n: int, x=329437923.5):
        return abs(n - x) <= 0.5
    
    def sol(x=329437923.5):
        """Round to nearest integer
    
        --- input ---
        3.7
    
        --- output ---
        4
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return round(x)
    ```
    
    </details>
    
* <a name="stonepiles"></a>**StonePiles** Inspired by [HumanEval](https://github.com/openai/human-eval) \#100 (5 instances)
    
    ```python
    def sat(li: List[int], n=909):
        return li[0] == n and len(li) == n and all(b - a == 2 for a, b in zip(li, li[1:]))
    
    def sol(n=909):
        """We are making n stone piles! The first pile has n stones. If n is even, then all piles have an even
        number of stones. If n is odd, all piles have an odd number of stones. Each pile must more stones
        than the previous pile but as few as possible. Return the number of stones in each pile.
    
        2 => [2, 4]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [n + 2 * i for i in range(n)]
    ```
    
    </details>
    
* <a name="completesplit"></a>**CompleteSplit** Inspired by [HumanEval](https://github.com/openai/human-eval) \#101 (5 instances)
    
    ```python
    def sat(splits: List[List[str]], string="Hello, world!  You look like you're on turtles."):
        words, separators = splits
        assert len(words) == len(separators) + 1
        merged = []
        for w, s in zip(words, separators + [" "]):
            assert s.count(" ") + s.count(",") == len(s) > 0
            assert w.count(" ") + w.count(",") == 0
            merged += [w, s]
        return "".join(merged[:-1]) == string
    
    def sol(string="Hello, world!  You look like you're on turtles."):
        """
        Split a string of words separated by commas and spaces into 2 lists: words and separators
    
        Sample input: "Hi there, Anna"
        Sample output: [["Hi", "there", "Anna"], [" ", ", "]]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        import re
        merged = re.split(r"([ ,]+)", string)
        return [merged[::2], merged[1::2]]
    ```
    
    </details>
    
* <a name="biggesteven"></a>**BiggestEven** Inspired by [HumanEval](https://github.com/openai/human-eval) \#102 (5 instances)
    
    ```python
    def sat(x: int, a=145, b=24126846790974):
        if x == -1:
            return all(i % 2 == 1 for i in range(a, b + 1))
        return a <= x <= b and all(i % 2 == 1 for i in range(x + 1, b + 1))
    
    def sol(a=145, b=24126846790974):
        """Return the biggest even number between a and b inclusive, or -1 if there is no such number
    
        Example input:
        a=20, b=99
    
        Example output:
        98
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if a > b or (a == b and a % 2 == 1):
            return -1
        return b if b % 2 == 0 else b - 1
    ```
    
    </details>
    
* <a name="binaryaverage"></a>**BinaryAverage** Inspired by [HumanEval](https://github.com/openai/human-eval) \#103 (5 instances)
    
    ```python
    def sat(s: str, a=-103252, b=10657):
        n = int(s, 2)
        r = range(a, b)
        if len(r) == 0:
            return n == -1
        mu = sum(r) / len(r)
        return abs(mu - n) <= min(abs(mu - n - 1), abs(mu - n + 1))
    
    def sol(a=-103252, b=10657):
        """Return the average of the numbers a through b rounded to nearest integer, in binary
        (or -1 if there are no such numbers)
    
        a=4, b=7 => '110' because the mean of 4, 5, 6 is 5 which is 110 in binary
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        r = range(a, b)
        if len(r) == 0:
            return "-1"
        return bin(round(sum(r) / len(r)))
    ```
    
    </details>
    
* <a name="sortedodds"></a>**SortedOdds** Inspired by [HumanEval](https://github.com/openai/human-eval) \#104 (5 instances)
    
    ```python
    def sat(sub: List[int], nums=[17, 20, -100, 101, 423258, 19949, 0, 20174, 9351773, -11]):
        for i in range(len(sub)):
            n = sub[i]
            assert n == min(sub[i:])
            assert all(int(c) % 2 for c in str(abs(n)))  # all odd digits
            assert sub.count(n) == nums.count(n)
    
        for n in nums:
            if n not in sub:
                assert any(int(c) % 2 == 0 for c in str(abs(n)))
    
        return True
    
    def sol(nums=[17, 20, -100, 101, 423258, 19949, 0, 20174, 9351773, -11]):
        """Find the sublist of numbers with only odd digits in increasing order
    
        [17, 21, 18, 1, 4] => [1, 17, 21]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sorted(n for n in nums if all(int(c) % 2 for c in str(abs(n))))
    ```
    
    </details>
    
* <a name="backwardsdigits"></a>**BackwardsDigits** Inspired by [HumanEval](https://github.com/openai/human-eval) \#105 (5 instances)
    
    ```python
    def sat(backwards_digits: List[str], nums=[0, 2, 14, -2, 3, 8, 4, 5, 5, 7, 21, 101, 41, 2, 9, 6]):
        digits = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
        li = [digits[s] for s in backwards_digits]
        for i, n in enumerate(li):
            assert n == max(li[i: i + 2])
            assert nums.count(n) == li.count(n)
    
        return all(n not in range(1, 10) or n in li for n in nums)
    
    def sol(nums=[0, 2, 14, -2, 3, 8, 4, 5, 5, 7, 21, 101, 41, 2, 9, 6]):
        """Return the single digits in nums sorted backwards and converted to English words
    
        [2, 3, 4, 5, 17] => ['five', 'four', 'three', 'two']
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        digits = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
        return [digits[n] for n in sorted(nums, reverse=True) if n in digits]
    ```
    
    </details>
    
* <a name="alternatingfactorials"></a>**AlternatingFactorials** Inspired by [HumanEval](https://github.com/openai/human-eval) \#106 (5 instances)
    
    ```python
    def sat(li: List[int], n=100):
        assert len(li) == n
        for i, m in enumerate(li):
            if i < 2:
                assert m == i + 1
            elif i % 2 == 1:
                assert m == li[i - 2] + i + (i + 1)
            else:
                assert m == li[i - 2] * i * (i + 1)
        return True
    
    def sol(n=100):
        """Output a list of n integers, where the mth entry is m! if m is even or else (1+2+...+m)
    
        5 => [1, 2, 6, 9, 120]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        ans = []
        for i in range(n):
            if i < 2:
                m = i + 1
            elif i % 2 == 1:
                m = ans[i - 2] + i + (i + 1)
            else:
                m = ans[i - 2] * i * (i + 1)
            ans.append(m)
    
        return ans
    ```
    
    </details>
    
* <a name="evenpalindromenumbers"></a>**EvenPalindromeNumbers** Inspired by [HumanEval](https://github.com/openai/human-eval) \#107 (5 instances)
    
    ```python
    def sat(pals: List[int], n=1099, count=49):
        return all(0 <= i <= n and str(i) == str(i)[::-1] and i % 2 == 0 for i in pals) and len(set(pals)) >= count
    
    def sol(n=1099, count=49):
        """Find all even palindromes up to n
    
        3 => [0, 2]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [i for i in range(0, n + 1, 2) if str(i) == str(i)[::-1]]
    ```
    
    </details>
    
* <a name="positivedigitsums"></a>**PositiveDigitSums** Inspired by [HumanEval](https://github.com/openai/human-eval) \#108 (5 instances)
    
    ```python
    def sat(pos: List[int], nums=[-804, 9124, -945, 2410, 0, 21, -123]):
        for n in pos + nums:
            s = str(n)
            if int(s[:2]) + sum(int(c) for c in s[2:]) <= 0:
                assert n not in pos
            else:
                assert pos.count(n) == nums.count(n)
        return True
    
    def sol(nums=[-804, 9124, -945, 2410, 0, 21, -123]):
        """Filter for the numbers in nums whose sum of digits is > 0, where the first digit can be negative.
    
        [12, -7, -102, -100] => [12, -102]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        def bad(n):
            s = str(n)
            return int(s[:2]) + sum(int(c) for c in s[2:]) <= 0
    
        return [n for n in nums if not bad(n)]
    ```
    
    </details>
    
* <a name="rotatesort"></a>**RotateSort** Inspired by [HumanEval](https://github.com/openai/human-eval) \#109
    
    This puzzle (and RotateString from #154) use the fact that a string is a rotation of r if it is a substring of r+r (5 instances)
    
    ```python
    def sat(original: List[int], arr=[2, 3, -1, -1, 0, 1, 1]):
        assert str(original)[1:-1] in str(sorted(original) * 2), "Not ring sorted"
        return any(original == arr[:i] + arr[i + 1:] for i in range(len(arr) + 1))
    
    def sol(arr=[2, 3, -1, -1, 0, 1, 1]):
        """
        An array is ring-sorted if it is a "rotation" of a non-decreasing list.
        Remove at most one element from arr to make it ring-sorted.
    
        [1, 2, 3, -1, 6, 0] => [1, 2, 3, -1, 0]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        def sat(near):
            order_violations = 0
            erasures = 0
            for i, n in enumerate(near):
                if n < near[i - 1]:  # -1 when i =0 gives last element
                    order_violations += 1
                while n != arr[i + erasures]:
                    erasures += 1
            return order_violations <= 1 and erasures <= 1
    
        candidates = [arr] + [arr[:i] + arr[i + 1:] for i in range(len(arr))]
        return next(near for near in candidates if sat(near))
    ```
    
    </details>
    
* <a name="parityexchange"></a>**ParityExchange** Inspired by [HumanEval](https://github.com/openai/human-eval) \#110 (5 instances)
    
    ```python
    def sat(swaps: List[List[int]], nums1=[1, 3, 2, 4, 5, 8, 7, 11], nums2=[0, 7, 0, 8, 19, 4, 41, 43, 42]):
        copy1 = nums1[:]
        copy2 = nums2[:]
        for i, j in swaps:
            copy1[i], copy2[j] = copy2[j], copy1[i]
        return all(n % 2 == 0 for n in copy1)
    
    def sol(nums1=[1, 3, 2, 4, 5, 8, 7, 11], nums2=[0, 7, 0, 8, 19, 4, 41, 43, 42]):
        """
        Find a sequence of swaps (indices into two lists) such that, after making those swaps, all numbers in the
        first list are even
    
        [1, 3, 4] [2, 4, 5] => [0, 1]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        odds = [i for i, n in enumerate(nums1) if n % 2 == 1]
        evens = [i for i, n in enumerate(nums2) if n % 2 == 0]
        return [[i, j] for i, j in zip(odds, evens)]
    ```
    
    </details>
    
* <a name="charcounts"></a>**CharCounts** Inspired by [HumanEval](https://github.com/openai/human-eval) \#111 (5 instances)
    
    ```python
    def sat(s: str, counts={'a': 4, 'b': 17, 'd': 101, 'e': 0, 'f': 12}):
        chars = s.split()
        for c in chars:
            assert chars.count(c) == counts[c]
        return len(chars) == sum(counts.values())
    
    def sol(counts={'a': 4, 'b': 17, 'd': 101, 'e': 0, 'f': 12}):
        """Find a string consisting of space-separated characters with given counts
    
        {"f": 1, "o": 2} => "oof"
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return " ".join(c for c, i in counts.items() for _ in range(i))
    ```
    
    </details>
    
* <a name="delpalindrome"></a>**DelPalindrome** Inspired by [HumanEval](https://github.com/openai/human-eval) \#112 (5 instances)
    
    ```python
    def sat(strings: List[str], a="this is a test", b="cat"):
        s, is_palindrome = strings
        i = 0
        for c in a:
            if c not in b:
                assert s[i] == c
                i += 1
        assert i == len(s)
        return is_palindrome == str(s == s[::-1])
    
    def sol(a="this is a test", b="cat"):
        """
        Return a pair of a strings where the first string is the same as a with all the characters of b removed,
        and the second string is 'True' if this string is a palindrome otherwise 'False'.
    
        a="madam, I'm adam." b = "Yes, we're here." => ['madamImadam', 'True']
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        s = "".join(c for c in a if c not in b)
        return [s, str(s == s[::-1])]
    ```
    
    </details>
    
* <a name="replaceme"></a>**ReplaceMe** Inspired by [HumanEval](https://github.com/openai/human-eval) \#113 (5 instances)
    
    ```python
    def sat(answers: List[str], lst=['234515', '21503', '2506236943']):
        if len(answers) != len(lst):
            return False
        for a, s in zip(answers, lst):
            if "t" in a:
                return False
            num_odds = sum(int(i) % 2 for i in s)
            if a.replace(str(num_odds), "t") != "this is a test":
                return False
        return True
    
    def sol(lst=['234515', '21503', '2506236943']):
        """For each string in lst, count the number of odd digits. Find a string with no t's such that replacing
        this number by t gives the string 'this is a test'
    
        ["123", "2"] => ["2his is a 2es2", "0his a 0es0"]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return ["this is a test".replace("t", str(sum(c in "13579" for c in s))) for s in lst]
    ```
    
    </details>
    
* <a name="minsubarraysum"></a>**MinSubArraySum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#114
    
    This is harder than \#1114. The arrays here are chosen to be long enough that the brute-force n^2 algorithm takes
    while the O(n) algorithm takes milliseconds. (5 instances)
    
    ```python
    def sat(start_end: List[int], base=7, p=50741, upper=-4897754):
        start, end = start_end
        return sum(pow(base, i, p) - p // 2 for i in range(start, end)) <= upper
    
    def sol(base=7, p=50741, upper=-4897754):
        """Find the start and end of the smallest-sum subarray of [(base^i mod p) - p/2 for i=start,..., end]
    
        base=3, p=7, upper =-3 => [0, 3]
        # because -3 is the sum of the elements [0:3] of [-2, 0, -1, 3, 1, 2, -2, 0, -1, 3 ...
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        tot = 0
        best_tot = 0
        best_end = 0
        best_start = 0
        largest_cumulative_sum = 0
        largest_cumulative_sum_index = 0
    
        n = 1
    
        for i in range(p + 1):
            if tot > largest_cumulative_sum:
                largest_cumulative_sum = tot
                largest_cumulative_sum_index = i
            if tot - largest_cumulative_sum < best_tot:
                best_tot = tot - largest_cumulative_sum
                best_start = largest_cumulative_sum_index
                best_end = i
    
            tot += (n - p // 2)
            n = (n * base) % p
    
        return [best_start, best_end]
    ```
    
    </details>
    
* <a name="buckets"></a>**Buckets** Inspired by [HumanEval](https://github.com/openai/human-eval) \#115 (5 instances)
    
    ```python
    def sat(wells: List[List[List[int]]], grid=[[1, 1, 0, 1, 1], [0, 0, 0, 0, 0], [1, 1, 0, 0, 1]], capacity=2):
        grid2 = [[0 for _ in row] for row in grid]
        for group in wells:
            assert len(group) <= capacity
            for i, j in group:
                assert grid2[i][j] == 0
                grid2[i][j] = 1
        assert sum(len(group) != capacity for group in wells) <= 1  # at most one under-capacity group
        return grid2 == grid
    
    def sol(grid=[[1, 1, 0, 1, 1], [0, 0, 0, 0, 0], [1, 1, 0, 0, 1]], capacity=2):
        """Given a grid, partition the 1's into groups of capacity [x, y] pairs, with at most one incomplete group"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        ans = []
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                if val == 1:
                    if not ans or len(ans[-1]) == capacity:
                        ans.append([])
                    ans[-1].append([i, j])
        return ans
    ```
    
    </details>
    
* <a name="binarysort"></a>**BinarySort** Inspired by [HumanEval](https://github.com/openai/human-eval) \#116 (5 instances)
    
    ```python
    def sat(ordered: List[int], arr=[4, 2, 3, -1, 15, 2, 6, 9, 5, 16, 1048576]):
        if sorted(ordered) != sorted(arr):
            return False  # not even a permutation
        return all(bin(a).count("1") <= bin(b).count("1") for a, b in zip(ordered, ordered[1:]))
    
    def sol(arr=[4, 2, 3, -1, 15, 2, 6, 9, 5, 16, 1048576]):
        """Sort the numbers in arr based on the number of 1's in their binary representation.
    
        [1, 2, 3, 4, 6] => [1, 2, 4, 3, 6]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sorted(arr, key=lambda n: bin(n).count("1"))
    ```
    
    </details>
    
* <a name="consonantfilter"></a>**ConsonantFilter** Inspired by [HumanEval](https://github.com/openai/human-eval) \#117 (5 instances)
    
    ```python
    def sat(words: List[str], s="This is not a very hard puzzle", n=3):
        i = 0
        for w in s.split():
            num_consonants = 0
            for c in w.lower():
                if c not in "aeiou":
                    num_consonants += 1
            if num_consonants == n:
                if words[i] != w:
                    return False
                i += 1
        return i == len(words)
    
    def sol(s="This is not a very hard puzzle", n=3):
        """Find all words in the string with n consonants
    
        Sample input:
        s="An eye for an I", n=1
        Sample output:
        ["An", "eye", "an"]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [w for w in s.split() if sum(c.lower() not in "aeiou" for c in w) == n]
    ```
    
    </details>
    
* <a name="vowelsandwich"></a>**VowelSandwich** Inspired by [HumanEval](https://github.com/openai/human-eval) \#118 (5 instances)
    
    ```python
    def sat(ham: str, s="Any vowel is OK"):
        vows = "aeiou"
        cons = "bcdfghjklmnpqrstvwxz"
        return ham in s and ham[0].lower() in cons and ham[1].lower() in vows and ham[2].lower() in cons
    
    def sol(s="Any vowel is OK"):
        """Find any vowel sandwich, a string consisting of a vowel between two consonants, contained in s
    
        "sandwhich" => "hic"
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        vows = "aeiou"
        cons = "bcdfghjklmnpqrstvwxz"
        return next(s[i - 1:i + 2] for i in range(1, len(s) - 1)
                    if s[i].lower() in vows and s[i - 1].lower() in cons and s[i + 1].lower() in cons)
    ```
    
    </details>
    
* <a name="parenthesespermutation"></a>**ParenthesesPermutation** Inspired by [HumanEval](https://github.com/openai/human-eval) \#119
        
    This is harder version in which you need to find a permutation of many substrings. Brute force is too slow. (5 instances)
    
    ```python
    def sat(perm: str, s="))(  )()()() )))(( ))))((( )))))(((( ))))))))((((((( ))))))((((( )))))))(((((( )))))))))(((((((  (((((((((("):
        assert sorted(perm.split()) == sorted(s.split()), "Must be a permutation of the space-delimited 'groups'"
        return all(perm[:i].count("(") >= perm[:i].count(")") for i in range(len(perm)))
    
    def sol(s="))(  )()()() )))(( ))))((( )))))(((( ))))))))((((((( ))))))((((( )))))))(((((( )))))))))(((((((  (((((((((("):
        """The string s consists of groups of parentheses separated by spaces.
        Permute the groups such that the parentheses match.
    
        "( ) )(" => "( )( )"
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        assert all(c in "( )" for c in s)
        parts = s.split()
    
        def min_depth(part):
            """Returns the lowest depth <= 0"""
            ans = 0
            depth = 0
            for c in part:
                if c == ")":
                    depth -= 1
                    ans = min(ans, depth)
                else:
                    depth += 1
            return ans
    
        def greedy_reorder(subs):
            """Reorder a bunch of parentheses substrings so as to maintain # ('s > # )'s """
            queue = subs[:]
            subs[:] = []
            height = 0
            while queue:
                best = max([s for s in queue if min_depth(s) + height >= 0], key=lambda s: s.count("(") - s.count(")"))
                height += best.count("(") - best.count(")")
                subs.append(best)
                queue.remove(best)
    
        lefts = [s for s in parts if s.count("(") >= s.count(")")]
    
        greedy_reorder(lefts)
    
        def mirror(sub):
            return "".join(")" if c == "(" else "(" for c in sub[::-1])
    
        rights = [mirror(s) for s in parts if s.count("(") < s.count(")")]  # mirror temporarily for reordering
    
        greedy_reorder(rights)
        return " ".join(lefts + [mirror(s) for s in rights[::-1]])
    ```
    
    </details>
    
* <a name="biggestk"></a>**BiggestK** Inspired by [HumanEval](https://github.com/openai/human-eval) \#120 (5 instances)
    
    ```python
    def sat(biggest: List[int], k=7, nums=[31, 1, 2, -10, -2, 4, 17, 18, 20, 14, 20, 21, 18, 0]):
        if len(biggest) != k:
            return False
        smallest = nums[:]
        for n in biggest:
            smallest.remove(n)
        return k == 0 or k == len(nums) or max(smallest) <= min(biggest)
    
    def sol(k=7, nums=[31, 1, 2, -10, -2, 4, 17, 18, 20, 14, 20, 21, 18, 0]):
        """Find the largest k numbers
    
        k=2, [1, 2, 3, 4, 5, 5, 3, 5, 2] => [5, 5]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sorted(nums, reverse=True)[:k]
    ```
    
    </details>
    
* <a name="oddevensum"></a>**OddEvenSum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#121
    
    Very similar to EvenOddSum from \#85 (5 instances)
    
    ```python
    def sat(tot: int, nums=[18, 42152, 125023521, -1221873620123, 17, 19]):
        for i in nums[::2]:
            if i % 2 == 1:
                tot -= i
        return tot == 0
    
    def sol(nums=[18, 42152, 125023521, -1221873620123, 17, 19]):
        """Find the sum of the odd elements that are at even indices
    
        [0, 1, 2, 3, 5, 6] => 5
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sum(i for i in nums[::2] if i % 2 == 1)
    ```
    
    </details>
    
* <a name="longearlysum"></a>**LongEarlySum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#122
        
    Changed slightly to make the answer not be a small integer. (5 instances)
    
    ```python
    def sat(tot: int, k=5, nums=[1252, 125273523, 0, 42, 100, 214532, 2, 0, 11, 14]):
        for n in nums[:k]:
            if len(str(abs(n))) > 2:
                tot -= n
        return tot == 0
    
    def sol(k=5, nums=[1252, 125273523, 0, 42, 100, 214532, 2, 0, 11, 14]):
        """Find the sum of the numbers among the first k with more than 2 digits
    
        k=3, nums=[2, 102, 12, 1000] => 102
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sum(n for n in nums[:k] if len(str(abs(n))) > 2)
    ```
    
    </details>
    
* <a name="oddcollatz"></a>**OddCollatz** Inspired by [HumanEval](https://github.com/openai/human-eval) \#123 (5 instances)
    
    ```python
    def sat(odds: List[int], n=1243272912731):
        num_odds = 0
        while True:
            if n % 2 == 1:
                num_odds += 1
                if n not in odds:
                    return False
            if n <= 1:
                return num_odds == len(odds)
            n = (3 * n + 1) if n % 2 == 1 else n // 2
    
    def sol(n=1243272912731):
        """Find the odd numbers in the collatz sequence starting at n
    
        3 => [3, 5, 1]  # because the Collatz sequence starting with 3 is [3, 10, 5, 16, 8, 4, 2, 1]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        ans = []
        while True:
            if n % 2 == 1:
                ans.append(n)
            if n <= 1:
                return ans
            n = (3 * n + 1) if n % 2 == 1 else n // 2
    ```
    
    </details>
    
* <a name="datediff"></a>**DateDiff** Inspired by [HumanEval](https://github.com/openai/human-eval) \#124 (5 instances)
    
    ```python
    def sat(s: str, target=-2075):
        assert all(c in "0123457689-" for c in s) and s[2] == s[5] == "-"
        m, d, y = [int(n) for n in s.split("-")]
        assert m in range(1, 13)
        assert d in range(1, 32)
        if m in [4, 6, 9, 11]:
            assert d <= 30
        if m == 2:
            assert d <= 29
        return m - d - y == target
    
    def sol(target=-2075):
        """Find a valid date mm-dd-yyyy such that the date, viewed as a mathematical expression, evaluates to target
    
        -2029 => "10-18-2021" # because 10-18-2021 == -2029
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if target >= -30:
            return "12-01-" + str(11 - target).zfill(4)
        return "01-31-" + str(-30 - target).zfill(4)
    ```
    
    </details>
    
* <a name="strangesplit"></a>**StrangeSplit** Inspired by [HumanEval](https://github.com/openai/human-eval) \#125 (5 instances)
    
    ```python
    def sat(lst: List[str], s="Hello, world!"):
        if " " in s:
            return " ".join(lst) == s
        if "," in s:
            return ",".join(lst) == s
        return "".join(lst) == "".join(c for c in s if c.islower() and ord(c) % 2 == 0)
    
    def sol(s="Hello, world!"):
        """Split s into strings if there is a space in s, otherwise split on commas if there is a comma, otherwise
        return the list of lowercase letters with odd order (order of a = 0, b = 1, etc.)
    
        "a b c" => ["a", "b", "c"]
        "a,b" => ["a", "b"]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if " " in s:
            return s.split(" ")
        if "," in s:
            return s.split(",")
        return [c for c in s if c.islower() and ord(c) % 2 == 0]
    ```
    
    </details>
    
* <a name="increasingviolation"></a>**IncreasingViolation** Inspired by [HumanEval](https://github.com/openai/human-eval) \#126 (5 instances)
    
    ```python
    def sat(violation: List[int], nums=[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 17, 17, 18, 19, 20, 22, 24]):
        if not violation:
            return all(nums[i] < nums[i + 1] for i in range(len(nums) - 1))
        i, j = violation
        return 0 <= i < j and nums[i] >= nums[j]
    
    def sol(nums=[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 17, 17, 18, 19, 20, 22, 24]):
        """
        Find the indices of two entries that show that the list is not in increasing order.
        If there are no violations (they are increasing), return an empty list.
    
        [1,2,3,0,4,5,6] => [1, 3]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        for i in range(len(nums) - 1):
            if nums[i] >= nums[i + 1]:
                return [i, i + 1]
        return []
    ```
    
    </details>
    
* <a name="primeintervalintersection"></a>**PrimeIntervalIntersection** Inspired by [HumanEval](https://github.com/openai/human-eval) \#127 (5 instances)
    
    ```python
    def sat(interval2: List[int], interval1=[32157, 93210127]):
        intersection_width = min(interval1[1], interval2[1]) - max(interval1[0], interval2[0])
        return intersection_width > 1 and all(intersection_width % i for i in range(2, intersection_width))
    
    def sol(interval1=[32157, 93210127]):
        """Find an interval whose intersection with a given interval has a width that is a prime integer.
    
        [7, 100] => [0, 10]  # because 10-7=3 is prime
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        a, b = interval1
        assert b - a >= 2
        return [a, a + 2]
    ```
    
    </details>
    
* <a name="productsigns"></a>**ProductSigns** Inspired by [HumanEval](https://github.com/openai/human-eval) \#128
        
    Easy puzzle since the answer is computed in the puzzle, but it is okay to have a few trivial puzzles. (5 instances)
    
    ```python
    def sat(n: int, arr=[1, 7, -20052, 14, -3, -11, 1025235, 14]):
        tot = 0
    
        for i in arr:
            if tot >= 0:
                tot += abs(i)
            else:
                tot -= abs(i)
            if i < 0:
                tot = -tot
            elif i == 0:
                tot = 0
                break
    
        return n == tot
    
    def sol(arr=[1, 7, -20052, 14, -3, -11, 1025235, 14]):
        """Find the sum of the magnitudes of the elements in the array with a sign that is equal to the product of
        the signs of the entries.
    
        [1, -2, 3] => -6  # negative because there is one negative
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        tot = sum(abs(i) for i in arr)
        if all(arr):
            return tot if sum(i < 0 for i in arr) % 2 == 0 else -tot
        return 0
    ```
    
    </details>
    
* <a name="lexpath"></a>**LexPath** Inspired by [HumanEval](https://github.com/openai/human-eval) \#129 (5 instances)
    
    ```python
    def sat(path: List[int], k=10, edges=[[2, 4], [3], [4, 1], [4], [0]]):
    
        def check(prefix):
            for i, j in zip(path, prefix):
                if i != j:
                    return i < j
            return len(prefix) >= k or all(check(prefix + [i]) for i in edges[prefix[-1]])
    
        return all(path[i] in edges[path[i - 1]] for i in range(1, k)) and all(check([i]) for i in range(len(edges)))
    
    def sol(k=10, edges=[[2, 4], [3], [4, 1], [4], [0]]):
        """Find the lexicographically smallest path of length k in graph with given edge matrix (and no dead ends)
    
        k=3, edges=[[1,3], [0, 3], [2], [3]] => [0, 1, 0] # because 0-1 and 1-0 are edges
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        path = []
        while len(path) < k:
            path.append(min(edges[path[-1]]) if path else 0)
        return path
    ```
    
    </details>
    
* <a name="tribonacci"></a>**Tribonacci** Inspired by [HumanEval](https://github.com/openai/human-eval) \#130
    
    This puzzle is a bit harder because the definition is slightly different at seq[1]. (5 instances)
    
    ```python
    def sat(seq: List[int], length=181):
        return all(seq[n] == (seq[n - 1] + seq[n - 2] + seq[n + 1] if n % 2 else 1 + n // 2) for n in range(length))
    
    def sol(length=181):
        """Find a sequence where seq[n] == 1 + n / 2 for even n, and
        seq[n] == seq[n - 1] + seq[n - 2] + seq[n + 1] for odd n < length."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        seq = []
        while len(seq) <= length:
            n = len(seq)
            if n % 2 == 0:
                seq.append(1 + n // 2)
            else:
                seq.append(sum(seq[-2:]) + (1 + (n + 1) // 2))
        return seq + [0]  # appending 0 at the end makes it easier so that seq[n-2] == 0 for n == 1
    ```
    
    </details>
    
* <a name="oddproduct"></a>**OddProduct** Inspired by [HumanEval](https://github.com/openai/human-eval) \#131 (5 instances)
    
    ```python
    def sat(prod: int, n=14235764939971075543215213):
    
        for c in str(n):
            i = int(c)
            if i % 2 == 1:
                assert prod % i == 0
                prod //= i
        return prod == any(int(c) % 2 for c in str(n))
    
    def sol(n=14235764939971075543215213):
        """Return the product of the odd digits in n, or 0 if there aren't any
    
        12345 => 15
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if any(int(c) % 2 for c in str(n)):
            prod = 1
            for c in str(n):
                if int(c) % 2 == 1:
                    prod *= int(c)
            return prod
        return 0
    ```
    
    </details>
    
* <a name="validbracketsubsequence"></a>**ValidBracketSubsequence** Inspired by [HumanEval](https://github.com/openai/human-eval) \#132 (5 instances)
    
    ```python
    def sat(valid: str, s="]]]]]]]]]]]]]]]]][][][][]]]]]]]]]]][[[][[][[[[[][][][]][[[[[[[[[[[[[[[[[["):
        assert valid in s
        depths = [0]
        for c in valid:
            if c == "[":
                depths.append(depths[-1] + 1)
            elif c == "]":
                depths.append(depths[-1] - 1)
        return depths[-1] == 0 and min(depths) == 0 and max(depths) > 1
    
    def sol(s="]]]]]]]]]]]]]]]]][][][][]]]]]]]]]]][[[][[][[[[[][][][]][[[[[[[[[[[[[[[[[["):
        """Find a valid substring of s that contains matching brackets, at least one of which is nested
    
        "]][][[]]]" => "[][[]]"
        """
    ```
    <details><summary>2 hand-written solutions </summary>
    
    ```python
        left = []
        nested = False
        for i, c in enumerate(s):
            if c == "[":
                if len(left) == 2:
                    left = [left[1], i]
                    nested = False
                else:
                    left.append(i)
            elif c == "]":
                if not left:
                    continue
                if len(left) == 1 and nested:
                    return s[left[0]:i + 1]
                elif len(left) == 2:
                    nested = True
                left.pop()
        assert False
    ```
    
    ```python
        import re
        return re.search(r"\[(\[\])+\]", s).group(0)
    ```
    
    </details>
    
* <a name="ceilingsquares"></a>**CeilingSquares** Inspired by [HumanEval](https://github.com/openai/human-eval) \#133 (5 instances)
    
    ```python
    def sat(running_squares: List[int], x=[201.1, 301.4, -18.1, 1244122.0, 10101.0101, 10000000.0]):
        for i, v in enumerate(x):
            ceiling = int(v) + (v > 0 and not v.is_integer())
            square = ceiling ** 2
            if running_squares[i] != square + (i > 0 and running_squares[i - 1]):
                return False
    
        return len(running_squares) == len(x)
    
    def sol(x=[201.1, 301.4, -18.1, 1244122.0, 10101.0101, 10000000.0]):
        """Round each float in x up to the next integer and return the running total of the integer squares
    
        [2.4, 3.7, 0.1] => [9, 25, 26]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        from math import ceil
        running_squares = []
        tot = 0
        for v in x:
            tot += ceil(v) ** 2
            running_squares.append(tot)
        return running_squares
    ```
    
    </details>
    
* <a name="lastletters"></a>**LastLetters** Inspired by [HumanEval](https://github.com/openai/human-eval) \#134 (5 instances)
    
    ```python
    def sat(y: List[bool], x=['Hello, world!', 'cat', '', 'a test', 'test a', 'i e', 'o', 'I O U', 'You and I']):
        assert len(x) == len(y)
        for s, b in zip(x, y):
            if len(s.split(" ")[-1]) == 1:
                assert b == s[-1].isalpha()
            else:
                assert not b
        return True
    
    def sol(x=['Hello, world!', 'cat', '', 'a test', 'test a', 'i e', 'o', 'I O U', 'You and I']):
        """Determine, for each string in x, whether the last character is an isolated letter
    
        ["a b c", "abc"] => [True, False]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [len(s.split(" ")[-1]) == 1 and s[-1].isalpha() for s in x]
    ```
    
    </details>
    
* <a name="drops"></a>**Drops** Inspired by [HumanEval](https://github.com/openai/human-eval) \#135 (1 instance)
    
    ```python
    def sat(drop_indexes: List[int], nums=[2, -1, 14, 8, 9, 9, 8, 4, 2, 4, 3, -100, 1000, 18, 4, -2, -3, -3, 1, 0]):
        d = 0
        for i in range(1, len(nums)):
            if nums[i] < nums[i - 1]:
                assert drop_indexes[d] == i
                d += 1
        return d == len(drop_indexes)
    
    def sol(nums=[2, -1, 14, 8, 9, 9, 8, 4, 2, 4, 3, -100, 1000, 18, 4, -2, -3, -3, 1, 0]):
        """Find the indices for which the nums array drops.
    
        [1,2,3,0,2,4,1] => [3,6]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [i for i in range(1, len(nums)) if nums[i] < nums[i - 1]]
    ```
    
    </details>
    
* <a name="largestnegsmallestpos"></a>**LargestNegSmallestPos** Inspired by [HumanEval](https://github.com/openai/human-eval) \#136 (5 instances)
    
    ```python
    def sat(extremes: List[int], nums=[-10, -4, 100, -40, 2, 2, 3, 17, -50, -25, 18, 41, 9, 11, 15]):
        neg, pos = extremes
        if neg == 0:
            assert nums == [] or min(nums) >= 0
        else:
            assert neg < 0 and neg in nums and all(n >= 0 or n <= neg for n in nums)
        if pos == 0:
            assert nums == [] or max(nums) <= 0
        else:
            assert pos > 0 and pos in nums and all(n <= 0 or n >= pos for n in nums)
        return True
    
    def sol(nums=[-10, -4, 100, -40, 2, 2, 3, 17, -50, -25, 18, 41, 9, 11, 15]):
        """Find the largest negative ans smallest positive numbers (or 0 if none)
    
        [-2, -4, 14, 50] => [-2, 14]
        [3, 22] => [0, 3]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        pos = [n for n in nums if n > 0]
        neg = [n for n in nums if n < 0]
        return [max(neg) if neg else 0, min(pos) if pos else 0]
    ```
    
    </details>
    
* <a name="largeststringnum"></a>**LargestStringNum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#137 (5 instances)
    
    ```python
    def sat(x: float, str_nums=['1,3', '-11', '17.5', '-11', '2', '2.2', '2,2', '4', '-18,18', '99.09']):
        found = False
        for s in str_nums:
            y = float(s.replace(",", "."))
            assert y <= x
            if y == x:
                found = True
        return found
    
    def sol(str_nums=['1,3', '-11', '17.5', '-11', '2', '2.2', '2,2', '4', '-18,18', '99.09']):
        """Find the largest number where commas or periods are decimal points
    
        ["99,9", "100"] => 100.0
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return max(float(s.replace(",", ".")) for s in str_nums)
    ```
    
    </details>
    
* <a name="even4sum"></a>**Even4Sum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#138 (5 instances)
    
    ```python
    def sat(summands: List[int], n=1234567890):
        return sum(summands) == n and min(summands) > 0 and len(summands) == 4 and all(s % 2 == 0 for s in summands)
    
    def sol(n=1234567890):
        """Find four positive even integers whose sum is n
    
        100 => [22, 24, 26, 28]"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [2] * 3 + [n - 6]
    ```
    
    </details>
    
* <a name="inversesuperfactorial"></a>**InverseSuperFactorial** Inspired by [HumanEval](https://github.com/openai/human-eval) \#139 (5 instances)
    
    ```python
    def sat(nums: List[int], super_factorials=[1, 2, 1]):
        for i, sf in enumerate(super_factorials):
            n = nums[i]
            for j in range(n, 0, -1):
                k = j ** (n - j + 1)
                assert sf % k == 0, f"{i} {sf} {j} {n}"
                sf //= k
            assert sf == 1
        return True
    
    def sol(super_factorials=[1, 2, 1]):
        """The super-factorial of n is n! (n-1)! (n-2)! ... 1!. Invert a given list of super-factorials.
    
        [1, 2, 2, 12] => [1, 2, 2, 3]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        queue = set(super_factorials)
        cache = {}
        n = 1
        fact = 1
        s_fact = 1
        while queue:
            fact *= n
            s_fact *= fact
            if s_fact in queue:
                queue.remove(s_fact)
                cache[s_fact] = n
            n += 1
        return [cache[sf] for sf in super_factorials]
    ```
    
    </details>
    
* <a name="expandspaces"></a>**ExpandSpaces** Inspired by [HumanEval](https://github.com/openai/human-eval) \#140 (5 instances)
    
    ```python
    def sat(orig: str, target="-Hello,_world!__This_is-so-easy!-"):
        assert "_" not in orig and "-" not in orig
        new = ""
        space_count = 0
        for c in orig:
            if c == " ":
                space_count += 1
            else:
                new += ("-" if space_count > 2 else "_" * space_count)
                new += c
                space_count = 0
        new += ("-" if space_count > 2 else "_" * space_count)
        return new == target
    
    def sol(target="-Hello,_world!__This_is-so-easy!-"):
        """Find a string such that, when three or more spaces are compacted to a '-' and one or two spaces are
        replaced by underscores, leads to the target.
    
        "_o-k__?-" => "  o        k  ?     "
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return target.replace("-", " " * 3).replace("_", " ")
    ```
    
    </details>
    
* <a name="filenameok"></a>**FilenameOK** Inspired by [HumanEval](https://github.com/openai/human-eval) \#141 (5 instances)
    
    ```python
    def sat(valids: List[str], filenames=['cat.txt', '!jog.dll', '31F9.html', 'Is this okay?.txt', '.exe', '']):
        assert len(valids) == len(filenames)
        for v, f in zip(valids, filenames):
            n_digits = sum(c.isdigit() for c in f)
            if v == "Yes":
                prefix, ext = f.split(".")
                assert ext in ["txt", "dll", "exe"] and prefix[0].isalpha() and n_digits < 4
            else:
                assert v == "No"
                assert f.split(".")[1:] not in [['txt'], ['dll'], ['exe']] or not f[0].isalpha() or n_digits > 3
        return True
    
    def sol(filenames=['cat.txt', '!jog.dll', '31F9.html', 'Is this okay?.txt', '.exe', '']):
        """Return a list of Yes/No strings that determine whether candidate filename is valid. A valid filename
        should end in .txt, .exe, or .dll, and should have at most three digits, no additional periods
    
        ["train.jpg", "doc10234.txt", "3eadme.txt"] = ["No", "No", "Yes"]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return ["Yes" if
                f.split(".")[1:] in [['txt'], ['dll'], ['exe']] and f[0].isalpha() and sum(c.isdigit() for c in f) < 4
                else "No"
                for f in filenames]
    ```
    
    </details>
    
* <a name="findstrangesum"></a>**FindStrangeSum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#142 (5 instances)
    
    ```python
    def sat(lst: List[int], tot=1125181293221):
        return sum(n ** 2 if n % 3 == 0 else n ** 3 if n % 4 == 0 else n for n in lst) == tot
    
    def sol(tot=1125181293221):
        """Find a list of integers such that tot is the sum of (n^2 if 3 | n, else n^3 if 4 | n, else n)"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        residue = (tot - 1) % 12
        return [1] * residue + [tot - residue]
    ```
    
    </details>
    
* <a name="primewords"></a>**PrimeWords** Inspired by [HumanEval](https://github.com/openai/human-eval) \#143 (5 instances)
    
    ```python
    def sat(primes: str, s="This is a test of whether you would want to do such strange puzzles"):
    
        def is_prime(n):
            return n > 1 and all(n % j for j in range(2, int(n ** 0.5) + 1))
    
        prime_words = primes.split()
        i = 0
        for word in s.split():
            if is_prime(len(word)):
                assert prime_words[i] == word
                i += 1
    
        return i == len(prime_words)
    
    def sol(s="This is a test of whether you would want to do such strange puzzles"):
        """Find the string consisting of all the words whose lengths are prime numbers
    
        "A bird in the hand is worth two in the bush" => "in the is worth two in the"
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        def is_prime(n):
            return n > 1 and all(n % j for j in range(2, int(n ** 0.5) + 1))
    
        return " ".join(w for w in s.split() if is_prime(len(w)))
    ```
    
    </details>
    
* <a name="simplifyproductfraction"></a>**SimplifyProductFraction** Inspired by [HumanEval](https://github.com/openai/human-eval) \#144 (5 instances)
    
    ```python
    def sat(z: str, x="-8142432/763083", y="66/-13474", max_len=18):
        [[a, b], [c, d], [u, v]] = [[int(n) for n in s.split("/")] for s in [x, y, z]]
        return a * c * v == b * d * u and len(z) <= max_len
    
    def sol(x="-8142432/763083", y="66/-13474", max_len=18):
        """Write x * y as the shortest equivalent fraction using at most max_len chars
    
        x="-2/3", y="-3/8", max_len=3 => "1/4"
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        [[a, b], [c, d]] = [[int(n) for n in s.split("/")] for s in [x, y]]
        num, den = a * c, b * d
        if num < 0 and den < 0:
            num, den = -num, -den
        if num == 0:
            return "0/1"
    
        def gcd(a, b):
            a, b = min(a, b), max(a, b)
            if b % a == 0:
                return a
            return gcd(b % a, a)
    
        d = gcd(abs(num), abs(den))
        return f'{num // d}/{den // d}'
    ```
    
    </details>
    
* <a name="sortbydigitsum"></a>**SortByDigitSum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#145 (5 instances)
    
    ```python
    def sat(ordered: List[int], nums=[1, 0, -1, -100, 10, 14, 235251, 11, 10000, 2000001, -155]):
        digit_sums = [sum(int(c) for c in str(n) if c != "-") for n in ordered]
        return sorted(ordered) == sorted(nums) and digit_sums == sorted(digit_sums)
    
    def sol(nums=[1, 0, -1, -100, 10, 14, 235251, 11, 10000, 2000001, -155]):
        """Sort the numbers by the sum of their digits
    
        [17, 21, 0] => [0, 17, 21]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sorted(nums, key=lambda n: sum(int(c) for c in str(n) if c != "-"))
    ```
    
    </details>
    
* <a name="bigodds"></a>**BigOdds** Inspired by [HumanEval](https://github.com/openai/human-eval) \#146 (5 instances)
    
    ```python
    def sat(odds: List[int], nums=[204, 109, 203, 17, 45, 11, 21, 99, 909, 16, -33, 3, 17]):
        assert all(o > 10 and odds.count(o) == nums.count(o) and int(str(o)[i]) % 2 for o in odds for i in [-1, 0])
        return all(n in odds or n <= 10 or int(str(n)[0]) % 2 == 0 or int(str(n)[-1]) % 2 == 0 for n in nums)
    
    def sol(nums=[204, 109, 203, 17, 45, 11, 21, 99, 909, 16, -33, 3, 17]):
        """Find the numbers that are greater than 10 and have odd first and last digits
    
        [73, 4, 72] => [73]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [n for n in nums if n > 10 and (int(str(n)[0]) * int(str(n)[-1])) % 2]
    ```
    
    </details>
    
* <a name="threeples"></a>**Threeples** Inspired by [HumanEval](https://github.com/openai/human-eval) \#147 (5 instances)
    
    ```python
    def sat(trips: List[List[int]], a=[1, 0, -17, 42, 321, 36, 429, 35, 10, 923, 35, 18, 0, 17, 24, 32, 8], count=221):
        assert len({tuple(t) for t in trips}) >= count
        return all(0 <= i < j < k and (a[i] + a[j] + a[k]) % 3 == 0 for i, j, k in trips)
    
    def sol(a=[1, 0, -17, 42, 321, 36, 429, 35, 10, 923, 35, 18, 0, 17, 24, 32, 8], count=221):
        """Find all triples of increasing indices where the sum of the numbers is divisible by three
    
        a=[1, 2, 4, 8, 14, 10], count=2 => [[0, 2, 5], [1, 3, 4]] = > because 1 + 4 + 10, 2 + 8 + 14 are divisible by 3
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        n = len(a)
        return [[i, j, k] for k in range(2, n) for j in range(k) for i in range(j) if (a[i] + a[j] + a[k]) % 3 == 0]
    ```
    
    </details>
    
* <a name="planetrange"></a>**PlanetRange** Inspired by [HumanEval](https://github.com/openai/human-eval) \#148 (5 instances)
    
    ```python
    def sat(planets_between: List[str], a="Mars", b="Neptune"):
        assert " " not in "".join(planets_between)
        return " ".join([a] + planets_between + [b]) in "Venus Earth Mars Jupiter Saturn Uranus Neptune Pluto"
    
    def sol(a="Mars", b="Neptune"):
        """Find all planets between the two given planets
    
        a="Jupiter", b="Pluto" => ["Saturn" "Uranus" "Neptune"]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        planets = "Venus Earth Mars Jupiter Saturn Uranus Neptune Pluto".split()
        return planets[planets.index(a) + 1:planets.index(b)]
    ```
    
    </details>
    
* <a name="evenwords"></a>**EvenWords** Inspired by [HumanEval](https://github.com/openai/human-eval) \#149 (5 instances)
    
    ```python
    def sat(evens: List[str], words=['The', 'worm', 'ate', 'a', 'bird', 'imagine', 'that', '!', 'Absurd', '!!']):
        lens = [len(w) for w in evens]
        assert all(lens[i] % 2 == 0 and lens[i] == max(lens[:i + 1]) and w in words for i, w in enumerate(evens))
        return all((len(w) % 2 == 1 or w in evens) for w in words)
    
    def sol(words=['The', 'worm', 'ate', 'a', 'bird', 'imagine', 'that', '!', 'Absurd', '!!']):
        """Find the even-length words and sort them by length.
    
        ["soup", "not", "splendid"] => ["soup", "splendid"]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sorted([w for w in words if len(w) % 2 == 0], key=lambda w: (len(w), w))
    ```
    
    </details>
    
* <a name="primesel"></a>**PrimeSel** Inspired by [HumanEval](https://github.com/openai/human-eval) \#150 (5 instances)
    
    ```python
    def sat(neighbors: List[int], nums=[14, 7, 11, 13, 7, 4, 19, 2, 55, 13, 31, 14, 2, 9, -7, 0, 88, 13, 13]):
    
        def prime(m):
            return all(m % i for i in range(2, m - 1))
    
        goods = set()
        for i, n in enumerate(nums):
            if (i > 0 and prime(nums[i - 1])) or (i < len(nums) - 1 and prime(nums[i + 1])):
                goods.add(n)
    
        return set(neighbors) == goods and all(n == min(neighbors[i:]) for i, n in enumerate(neighbors))
    
    def sol(nums=[14, 7, 11, 13, 7, 4, 19, 2, 55, 13, 31, 14, 2, 9, -7, 0, 88, 13, 13]):
        """Find a list of all numbers that are adjacent to a prime number in the list, sorted without duplicates
    
        [2, 17, 16, 0, 6, 4, 5] => [2, 4, 16, 17]"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        def prime(m):
            return all(m % i for i in range(2, m - 1))
    
        return sorted({
            n for i, n in enumerate(nums)
            if (i > 0 and prime(nums[i - 1])) or (i < len(nums) - 1 and prime(nums[i + 1]))
        })
    ```
    
    </details>
    
* <a name="evensqure"></a>**EvenSqure** Inspired by [HumanEval](https://github.com/openai/human-eval) \#151 (5 instances)
    
    ```python
    def sat(tot: int, xs=[123.0, 872322.0, 542.2, -127.5, 18214.0, 3732.4, 12832.4, 23523800.0]):
        for x in xs:
            if x.is_integer() and x > 0 and x % 2 == 0:
                tot -= int(x) ** 2
    
        return tot == 0
    
    def sol(xs=[123.0, 872322.0, 542.2, -127.5, 18214.0, 3732.4, 12832.4, 23523800.0]):
        """Find the sum of the squares of the positive even integers
    
        [2.0, 3.0, 2.5, 4.0] => 20
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sum(int(x) ** 2 for x in xs if x.is_integer() and x > 0 and x % 2 == 0)
    ```
    
    </details>
    
* <a name="arraydiff"></a>**ArrayDiff** Inspired by [HumanEval](https://github.com/openai/human-eval) \#152 (5 instances)
    
    ```python
    def sat(b: List[int], a=[1, 2, 3, 0, 4, 17, 2, 4, 5, 9, 8, 4], c=[1, 2, 3, 4, 0, 16, 2, 3, 5, 9, 8, 4]):
        return len(b) == len(a) and all(i + j == k for i, j, k in zip(a, b, c))
    
    def sol(a=[1, 2, 3, 0, 4, 17, 2, 4, 5, 9, 8, 4], c=[1, 2, 3, 4, 0, 16, 2, 3, 5, 9, 8, 4]):
        """Find an array that when added to vector a gives array vector c
    
        [1, 2, 3], [4, 17, 5] => [3, 15, 2]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [k - i for i, k in zip(a, c)]
    ```
    
    </details>
    
* <a name="strongestextension"></a>**StrongestExtension** Inspired by [HumanEval](https://github.com/openai/human-eval) \#153 (5 instances)
    
    ```python
    def sat(s: str, class_name="TestClass", extensions=['extEnd', 'LOL', 'SuPeRbLy', 'v9ACLQWTEW', 'PickMe', 'AI']):
        assert s.startswith(class_name + ".")
        ext = s[len(class_name) + 1:]
    
        def case_delta(x: str):
            tot = 0
            for c in x:
                if c.isupper():
                    tot += 1
                elif c.islower():
                    tot -= 1
            return tot
    
        return ext in extensions and case_delta(ext) == max([case_delta(x) for x in extensions])
    
    def sol(class_name="TestClass", extensions=['extEnd', 'LOL', 'SuPeRbLy', 'v9ACLQWTEW', 'PickMe', 'AI']):
        """Find the class_name.extension for the extension that has the largest #capitals - #lowercase letters"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        def case_delta(x: str):
            tot = 0
            for c in x:
                if c.isupper():
                    tot += 1
                elif c.islower():
                    tot -= 1
            return tot
    
        return class_name + "." + max(extensions, key=case_delta)
    ```
    
    </details>
    
* <a name="rotatestring"></a>**RotateString** Inspired by [HumanEval](https://github.com/openai/human-eval) \#154
    
    This puzzle (and RotateSort from #109) use the fact that a string is a rotation of r if it is a substring of r+r (5 instances)
    
    ```python
    def sat(r: str, s="light star", t="I love to look at the starlight!"):
        return r in t and len(r) == len(s) and r in s + s
    
    def sol(s="light star", t="I love to look at the starlight!"):
        """Find a rotation of string s that is a substring of t
    
        Input Example:
        s="test", t="I love lattes"
    
        Output Example:
        "ttes"
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return next(s[i:] + s[:i] for i in range(len(s)) if s[i:] + s[:i] in t)
    ```
    
    </details>
    
* <a name="evenodddigits"></a>**EvenOddDigits** Inspired by [HumanEval](https://github.com/openai/human-eval) \#155 (5 instances)
    
    ```python
    def sat(n: int, evens=17, odds=3):
        for c in str(n):
            if int(c) % 2 == 0:
                evens -= 1
            else:
                odds -= 1
        return evens == 0 and odds == 0
    
    def sol(evens=17, odds=3):
        """Find an integer n >= 0 with the given number of even and odd digits.
    
        evens=3, odds=4 => 2381695"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return int("2" * evens + "1" * odds)
    ```
    
    </details>
    
* <a name="romannumerals"></a>**RomanNumerals** Inspired by [HumanEval](https://github.com/openai/human-eval) \#156
        
    Do not add a reverse puzzle converting roman numerals to arabic numbers as it would give away the solution. (5 instances)
    
    ```python
    def sat(roman: str, n=2414):
        key = {1000: 'm', 900: 'cm', 500: 'd', 400: 'cd',
               100: 'c', 90: 'xc', 50: 'l', 40: 'xl',
               10: 'x', 9: 'ix', 5: 'v', 4: 'iv',
               1: 'i'}
        m = 0
        for base in [1000, 100, 10, 1]:
            for mul in [9, 4, 5, 1, 1, 1]:  # up to three 1's, move on after 9 or 4
                val = base * mul
                if val in key and roman.startswith(key[val]):
                    m += val
                    roman = roman[len(key[val]):]
                    if mul == 9 or mul == 4:  # 9 or 4 can't be followed by anything else
                        break
        return m == n
    
    def sol(n=2414):
        """Convert integer 0 < n < 4000 to roman numerals, and make it lowercase
    
        11 => "xi"
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        units = dict(m=1000, cm=900, d=500, cd=400, c=100, xc=90, l=50, xl=40, x=10, ix=9, v=5, iv=4, i=1)
        roman = ""
        for s, i in units.items():
            while n >= i:
                roman += s
                n -= i
        return roman
    ```
    
    </details>
    
* <a name="pythagoreantriples"></a>**PythagoreanTriples** Inspired by [HumanEval](https://github.com/openai/human-eval) \#157 (5 instances)
    
    ```python
    def sat(triples: List[List[int]], n=920, m=799):
        for a, b, c in triples:
            if not (a * a + b * b == c * c and 0 < a < b < c <= n):
                return False
        return triples == sorted(triples) and len(triples) >= m
    
    def sol(n=920, m=799):
        """Find m Pythagorean triples a^2 + b^2 == c^2 for integers 0 < a < b < c <= n, in sorted order
    
        (n=6, m=1) => [[3, 4, 5]]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [[a, b, int((a * a + b * b) ** 0.5)]
                for a in range(3, int(n / (2 ** 0.5)))
                for b in range(a + 1, int((n * n - a * a) ** 0.5) + 1)
                if ((a * a + b * b) ** 0.5).is_integer()]
    ```
    
    </details>
    
* <a name="mostunique"></a>**MostUnique** Inspired by [HumanEval](https://github.com/openai/human-eval) \#158 (5 instances)
    
    ```python
    def sat(s: str, pool=['cat', 'catatatatctsa', 'abcdefhijklmnop', '124259239185125', '', 'foo', 'unique']):
        assert s in pool
        n = len(set(s))
        for p in pool:
            assert len(set(p)) <= n
        return True
    
    def sol(pool=['cat', 'catatatatctsa', 'abcdefhijklmnop', '124259239185125', '', 'foo', 'unique']):
        """Select a string from the pool with the most unique characters
    
        ["woooow", "cow"] => "cow"
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return max(pool, key=lambda x: len(set(x)))
    ```
    
    </details>
    
* <a name="hungryrabbits"></a>**HungryRabbits** Inspired by [HumanEval](https://github.com/openai/human-eval) \#159 (5 instances)
    
    ```python
    def sat(results: List[List[int]], stats=[[2, 3, 18], [4, 9, 2], [2, 5, 7], [3, 8, 12], [4, 9, 106]]):
        assert len(results) == len(stats)
        for (tot, remaining), (eaten, need, stock) in zip(results, stats):
            assert tot - eaten == min(need, stock)
            assert stock < need and remaining == 0 or stock >= need and remaining + need == stock
        return True
    
    def sol(stats=[[2, 3, 18], [4, 9, 2], [2, 5, 7], [3, 8, 12], [4, 9, 106]]):
        """For each triple of eaten, need, stock return a pair of total appetite and remaining
    
        [[2, 5, 6], [3, 9, 22]] => [[7, 1], [12, 13]]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        results = []
        for (eaten, need, stock) in stats:
            results.append([eaten + min(need, stock), max(0, stock - need)])
        return results
    ```
    
    </details>
    
* <a name="evaluateoperators"></a>**EvaluateOperators** Inspired by [HumanEval](https://github.com/openai/human-eval) \#160 (5 instances)
    
    ```python
    def sat(ops: List[str], target=2021, nums=[4, 6, 2, 1, 1, 3, 9]):
        assert len(ops) == len(set(ops)) and set(ops) == {"**", "*", "+", "-", "//", "%"}
        expr = str(nums[0])
        for n, op in zip(nums[1:], ops):
            expr += op + str(n)
        return eval(expr) == target
    
    def sol(target=2021, nums=[4, 6, 2, 1, 1, 3, 9]):
        """Find a permutation of the operators +-*/^% which when inserted between nums evaluates to target
    
        target=3, nums=[7, 2, 3, 4, 5, 1, 6] => ["+", "*", "**", "%", "//", "-"]
                                                # because 7 + 2 * 3 ** 4 % 5 // 1 - 6 == 3
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        from itertools import permutations
        for ops in permutations(["**", "*", "+", "-", "//", "%"]):
            expr = str(nums[0])
            for n, op in zip(nums[1:], ops):
                expr += op + str(n)
            try:
                if eval(expr) == target:
                    return list(ops)
            except (ZeroDivisionError, SyntaxError):
                pass
        assert False
    ```
    
    </details>
    
* <a name="reversecase"></a>**ReverseCase** Inspired by [HumanEval](https://github.com/openai/human-eval) \#161 (5 instances)
    
    ```python
    def sat(rev: List[str], strs=['cat', 'u8u', '12532', '', '191', '4tUn8', 'ewrWQTEW', 'i', 'IoU']):
        assert len(rev) == len(strs)
        return all(r.swapcase() == s != r or r[::-1] == s == s.swapcase() for r, s in zip(rev, strs))
    
    def sol(strs=['cat', 'u8u', '12532', '', '191', '4tUn8', 'ewrWQTEW', 'i', 'IoU']):
        """Reverse the case of all strings. For those strings which contain no letters, reverse the strings.
    
        ["Test", "!@#"] => ["tEST", "#@!"]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [s.swapcase() if s.swapcase() != s else s[::-1] for s in strs]
    ```
    
    </details>
    
* <a name="zobristcollision"></a>**ZobristCollision** Inspired by [HumanEval](https://github.com/openai/human-eval) \#162
    
    The original problem was to compute an MD5 hash. This puzzle is a problem in the space of hashing, but of a
    different nature. (1 instance)
    
    ```python
    def sat(positions: List[List[int]]):
    
        table = [[(i * 429436219 + j * 100239120) % 63491564 for j in range(13)] for i in range(64)]
    
        def zobrist(pos):
            h = 0
            for i in range(64):
                if pos[i]:
                    h ^= table[i][pos[i]]
            return h
    
        a, b = positions
        return zobrist(a) == zobrist(b) and a != b
    
    def sol():
        """Find a collision for the given Zobrist chess board hash: https://en.wikipedia.org/wiki/Zobrist_hashing
    
        Each of the two positions should be encoded as a list of 64 integers 0-12"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        hashes = {}
        table = [[(i * 429436219 + j * 100239120) % 63491564 for j in range(13)] for i in range(64)]
    
        def zobrist(pos):
            h = 0
            for i in range(64):
                if pos[i]:
                    h ^= table[i][pos[i]]
            return h
    
        for i in range(1, 100000000):
            pos = [(i * 42 + ((i + 1) * j * 12589) % 54321) % 13 for j in range(64)]  # pseudo-random board
            h = zobrist(pos)
            if h in hashes:
                return [pos, hashes[h]]
            else:
                hashes[h] = pos
    ```
    
    </details>
    
* <a name="evenbetween"></a>**EvenBetween** Inspired by [HumanEval](https://github.com/openai/human-eval) \#163
    
    The original problem was trivial to list the even single-digit numbers between two numbers:
    `a=2, b=12` => `[4, 6, 8]`. In this puzzle, we consider the string of even numbers formed when counting from
    `a` to `b`, e.g., `"1618202224262830"` when counting from `15` to `30`. The puzzle is, given such a string,
    find `a` and `b`. (5 instances)
    
    ```python
    def sat(ab: List[int], s="3298832990329923299432996329983300033002"):
        return abs(ab[0] - ab[1]) > 4 and s == "".join(str(i) for i in range(min(ab), max(ab) + 1) if i % 2 == 0)
    
    def sol(s="3298832990329923299432996329983300033002"):
        """Find integers [a, b] that are at least 5 apart and such that concatenating the even numbers
        between them gives the string s
    
        "32343638" => [31, 38]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        for i in range(1, len(s)):
            n = int(s[:i])
            n -= (n + 1) % 2  # make n odd
            m = n + 1  # next even
            t = ""
            while len(t) < len(s):
                t += str(m)
                m += 2
            if s == t:
                return [n, m - 1]
    
        assert False
    ```
    
    </details>
    
## codeforces

Problems inspired by the popular programming competition site [codeforces.com](https://codeforces.com)

* <a name="iseven"></a>**IsEven** Inspired by [Codeforces Problem 4 A](https://codeforces.com/problemset/problem/4/A) (5 instances)
    
    ```python
    def sat(b: bool, n=10):
        i = 0
        while i <= n:
            if i + i == n:
                return b == True
            i += 1
        return b == False
    
    def sol(n=10):
        """Determine if n can be evenly divided into two equal numbers. (Easy)"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return n % 2 == 0
    ```
    
    </details>
    
* <a name="abbreviate"></a>**Abbreviate** Inspired by [Codeforces Problem 71 A](https://codeforces.com/problemset/problem/71/A) (5 instances)
    
    ```python
    def sat(s: str, word="antidisestablishmentarianism", max_len=10):
        if len(word) <= max_len:
            return word == s
        return int(s[1:-1]) == len(word[1:-1]) and word[0] == s[0] and word[-1] == s[-1]
    
    def sol(word="antidisestablishmentarianism", max_len=10):
        """
        Abbreviate strings longer than a given length by replacing everything but the first and last characters by
        an integer indicating how many characters there were in between them.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if len(word) <= max_len:
            return word
        return f"{word[0]}{len(word) - 2}{word[-1]}"
    ```
    
    </details>
    
* <a name="squaretiles"></a>**SquareTiles** Inspired by [Codeforces Problem 1 A](https://codeforces.com/problemset/problem/1/A) (5 instances)
    
    ```python
    def sat(corners: List[List[int]], m=10, n=9, a=5, target=4):
        covered = {(i + x, j + y) for i, j in corners for x in range(a) for y in range(a)}
        assert len(covered) == len(corners) * a * a, "Double coverage"
        return len(corners) <= target and covered.issuperset({(x, y) for x in range(m) for y in range(n)})
    
    def sol(m=10, n=9, a=5, target=4):
        """Find a minimal list of corner locations for a×a tiles that covers [0, m] × [0, n] and does not double-cover
        squares.
    
        Sample Input:
        m = 10
        n = 9
        a = 5
        target = 4
    
        Sample Output:
        [[0, 0], [0, 5], [5, 0], [5, 5]]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [[x, y] for x in range(0, m, a) for y in range(0, n, a)]
    ```
    
    </details>
    
* <a name="easytwos"></a>**EasyTwos** Inspired by [Codeforces Problem 231 A](https://codeforces.com/problemset/problem/231/A) (5 instances)
    
    ```python
    def sat(lb: List[bool], trips=[[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]):
        return len(lb) == len(trips) and all(
            (b is True) if sum(s) >= 2 else (b is False) for b, s in zip(lb, trips))
    
    def sol(trips=[[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]):
        """
        Given a list of lists of triples of integers, return True for each list with a total of at least 2 and
        False for each other list.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [sum(s) >= 2 for s in trips]
    ```
    
    </details>
    
* <a name="decreasingcountcomparison"></a>**DecreasingCountComparison** Inspired by [Codeforces Problem 158 A](https://codeforces.com/problemset/problem/158/A) (5 instances)
    
    ```python
    def sat(n: int, scores=[100, 95, 80, 70, 65, 9, 9, 9, 4, 2, 1], k=6):
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), "Hint: scores are non-decreasing"
        return all(s >= scores[k] and s > 0 for s in scores[:n]) and all(s < scores[k] or s <= 0 for s in scores[n:])
    
    def sol(scores=[100, 95, 80, 70, 65, 9, 9, 9, 4, 2, 1], k=6):
        """
        Given a list of non-increasing integers and given an integer k, determine how many positive integers in the list
        are at least as large as the kth.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        threshold = max(scores[k], 1)
        return sum(s >= threshold for s in scores)
    ```
    
    </details>
    
* <a name="voweldrop"></a>**VowelDrop** Inspired by [Codeforces Problem 118 A](https://codeforces.com/problemset/problem/118/A) (5 instances)
    
    ```python
    def sat(t: str, s="Problems"):
        i = 0
        for c in s.lower():
            if c in "aeiouy":
                continue
            assert t[i] == ".", f"expecting `.` at position {i}"
            i += 1
            assert t[i] == c, f"expecting `{c}`"
            i += 1
        return i == len(t)
    
    def sol(s="Problems"):
        """
        Given an alphabetic string s, remove all vowels (aeiouy/AEIOUY), insert a "." before each remaining letter
        (consonant), and make everything lowercase.
    
        Sample Input:
        s = "Problems"
    
        Sample Output:
        .p.r.b.l.m.s
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "".join("." + c for c in s.lower() if c not in "aeiouy")
    ```
    
    </details>
    
* <a name="dominotile"></a>**DominoTile** Inspired by [Codeforces Problem 50 A](https://codeforces.com/problemset/problem/50/A) (5 instances)
    
    ```python
    def sat(squares: List[List[int]], m=10, n=5, target=50):
        covered = []
        for i1, j1, i2, j2 in squares:
            assert (0 <= i1 <= i2 < m) and (0 <= j1 <= j2 < n) and (j2 - j1 + i2 - i1 == 1)
            covered += [(i1, j1), (i2, j2)]
        return len(set(covered)) == len(covered) == target
    
    def sol(m=10, n=5, target=50):
        """Tile an m x n checkerboard with 2 x 1 tiles. The solution is a list of fourtuples [i1, j1, i2, j2] with
        i2 == i1 and j2 == j1 + 1 or i2 == i1 + 1 and j2 == j1 with no overlap."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if m % 2 == 0:
            ans = [[i, j, i + 1, j] for i in range(0, m, 2) for j in range(n)]
        elif n % 2 == 0:
            ans = [[i, j, i, j + 1] for i in range(m) for j in range(0, n, 2)]
        else:
            ans = [[i, j, i + 1, j] for i in range(1, m, 2) for j in range(n)]
            ans += [[0, j, 0, j + 1] for j in range(0, n - 1, 2)]
        return ans
    ```
    
    </details>
    
* <a name="incdec"></a>**IncDec** Inspired by [Codeforces Problem 282 A](https://codeforces.com/problemset/problem/282/A)
    
    This straightforward problem is a little harder than the Codeforces one. (5 instances)
    
    ```python
    def sat(n: int, ops=['x++', '--x', '--x'], target=19143212):
        for op in ops:
            if op in ["++x", "x++"]:
                n += 1
            else:
                assert op in ["--x", "x--"]
                n -= 1
        return n == target
    
    def sol(ops=['x++', '--x', '--x'], target=19143212):
        """
        Given a sequence of operations "++x", "x++", "--x", "x--", and a target value, find initial value so that the
        final value is the target value.
    
        Sample Input:
        ops = ["x++", "--x", "--x"]
        target = 12
    
        Sample Output:
        13
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return target - ops.count("++x") - ops.count("x++") + ops.count("--x") + ops.count("x--")
    ```
    
    </details>
    
* <a name="compareinanycase"></a>**CompareInAnyCase** Inspired by [Codeforces Problem 112 A](https://codeforces.com/problemset/problem/112/A) (5 instances)
    
    ```python
    def sat(n: int, s="aaAab", t="aAaaB"):
        if n == 0:
            return s.lower() == t.lower()
        if n == 1:
            return s.lower() > t.lower()
        if n == -1:
            return s.lower() < t.lower()
        return False
    
    def sol(s="aaAab", t="aAaaB"):
        """Ignoring case, compare s, t lexicographically. Output 0 if they are =, -1 if s < t, 1 if s > t."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if s.lower() == t.lower():
            return 0
        if s.lower() > t.lower():
            return 1
        return -1
    ```
    
    </details>
    
* <a name="slidingone"></a>**SlidingOne** Inspired by [Codeforces Problem 263 A](https://codeforces.com/problemset/problem/263/A) (4 instances)
    
    ```python
    def sat(s: str, matrix=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], max_moves=3):
        matrix = [m[:] for m in matrix]  # copy
        for c in s:
            if c in "01234":
                i = "01234".index(c)
                matrix[i], matrix[i + 1] = matrix[i + 1], matrix[i]
            if c in "abcde":
                j = "abcde".index(c)
                for row in matrix:
                    row[j], row[j + 1] = row[j + 1], row[j]
    
        return len(s) <= max_moves and matrix[2][2] == 1
    
    def sol(matrix=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], max_moves=3):
        """
        We are given a 5x5 matrix with a single 1 like:
    
        0 0 0 0 0
        0 0 0 0 1
        0 0 0 0 0
        0 0 0 0 0
        0 0 0 0 0
    
        Find a (minimal) sequence of row and column swaps to move the 1 to the center. A move is a string
        in "0"-"4" indicating a row swap and "a"-"e" indicating a column swap
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        i = [sum(row) for row in matrix].index(1)
        j = matrix[i].index(1)
        ans = ""
        while i > 2:
            ans += str(i - 1)
            i -= 1
        while i < 2:
            ans += str(i)
            i += 1
        while j > 2:
            ans += "abcde"[j - 1]
            j -= 1
        while j < 2:
            ans += "abcde"[j]
            j += 1
        return ans
    ```
    
    </details>
    
* <a name="sortplusplus"></a>**SortPlusPlus** Inspired by [Codeforces Problem 339 A](https://codeforces.com/problemset/problem/339/A) (5 instances)
    
    ```python
    def sat(s: str, inp="1+1+3+1+3+2+2+1+3+1+2"):
        return all(s.count(c) == inp.count(c) for c in inp + s) and all(s[i - 2] <= s[i] for i in range(2, len(s), 2))
    
    def sol(inp="1+1+3+1+3+2+2+1+3+1+2"):
        """Sort numbers in a sum of digits, e.g., 1+3+2+1 -> 1+1+2+3"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "+".join(sorted(inp.split("+")))
    ```
    
    </details>
    
* <a name="capitalizefirstletter"></a>**CapitalizeFirstLetter** Inspired by [Codeforces Problem 281 A](https://codeforces.com/problemset/problem/281/A) (5 instances)
    
    ```python
    def sat(s: str, word="konjac"):
        for i in range(len(word)):
            if i == 0:
                if s[i] != word[i].upper():
                    return False
            else:
                if s[i] != word[i]:
                    return False
        return True
    
    def sol(word="konjac"):
        """Capitalize the first letter of word"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return word[0].upper() + word[1:]
    ```
    
    </details>
    
* <a name="longestsubsetstring"></a>**LongestSubsetString** Inspired by [Codeforces Problem 266 A](https://codeforces.com/problemset/problem/266/A) (5 instances)
    
    ```python
    def sat(t: str, s="abbbcabbac", target=7):
        i = 0
        for c in t:
            while c != s[i]:
                i += 1
            i += 1
        return len(t) >= target and all(t[i] != t[i + 1] for i in range(len(t) - 1))
    
    def sol(s="abbbcabbac", target=7):
        """
        You are given a string consisting of a's, b's and c's, find any longest substring containing no repeated
        consecutive characters.
    
        Sample Input:
        `"abbbc"`
    
        Sample Output:
        `"abc"`
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        # target is ignored
        return s[:1] + "".join([b for a, b in zip(s, s[1:]) if b != a])
    ```
    
    </details>
    
* <a name="findhomogeneoussubstring"></a>**FindHomogeneousSubstring** Inspired by [Codeforces Problem 96 A](https://codeforces.com/problemset/problem/96/A) (5 instances)
    
    ```python
    def sat(n: int, s="0000101111111000010", k=5):
        return s[n:n + k] == s[n] * k
    
    def sol(s="0000101111111000010", k=5):
        """
        You are given a string consisting of 0's and 1's. Find an index after which the subsequent k characters are
        all 0's or all 1's.
    
        Sample Input:
        s = 0000111111100000, k = 5
    
        Sample Output:
        4
        (or 5 or 6 or 11)
        """
    ```
    <details><summary>4 hand-written solutions </summary>
    
    ```python
        return s.index("0" * k if "0" * k in s else "1" * k)
    ```
    
    ```python
        import re
        return re.search(r"([01])\1{" + str(k - 1) + "}", s).span()[0]
    ```
    
    ```python
        if "0" * k in s:
            return s.index("0" * k)
        else:
            return s.index("1" * k)
    ```
    
    ```python
        try:
            return s.index("0" * k)
        except:
            return s.index("1" * k)
    ```
    
    </details>
    
* <a name="triple0"></a>**Triple0** Inspired by [Codeforces Problem 630 A](https://codeforces.com/problemset/problem/69/A) (5 instances)
    
    ```python
    def sat(delta: List[int], nums=[[1, 2, 3], [9, -2, 8], [17, 2, 50]]):
        return all(sum(vec[i] for vec in nums) + delta[i] == 0 for i in range(3))
    
    def sol(nums=[[1, 2, 3], [9, -2, 8], [17, 2, 50]]):
        """Find the missing triple of integers to make them all add up to 0 coordinatewise"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [-sum(vec[i] for vec in nums) for i in range(3)]
    ```
    
    </details>
    
* <a name="totaldifference"></a>**TotalDifference** Inspired by [Codeforces Problem 546 A](https://codeforces.com/problemset/problem/546/A) (5 instances)
    
    ```python
    def sat(n: int, a=17, b=100, c=20):
        return n + a == sum([b * i for i in range(c)])
    
    def sol(a=17, b=100, c=20):
        """Find n such that n + a == b * (the sum of the first c integers)"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return -a + sum([b * i for i in range(c)])
    ```
    
    </details>
    
* <a name="tripledouble"></a>**TripleDouble** Inspired by [Codeforces Problem 791 A](https://codeforces.com/problemset/problem/791/A) (5 instances)
    
    ```python
    def sat(n: int, v=17, w=100):
        for i in range(n):
            assert v <= w
            v *= 3
            w *= 2
        return v > w
    
    def sol(v=17, w=100):
        """Find the smallest n such that if v is tripled n times and w is doubled n times, v exceeds w."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        i = 0
        while v <= w:
            v *= 3
            w *= 2
            i += 1
        return i
    ```
    
    </details>
    
* <a name="repeatdec"></a>**RepeatDec** Inspired by [Codeforces Problem 977 A](https://codeforces.com/problemset/problem/977/A) (5 instances)
    
    ```python
    def sat(res: int, m=1234578987654321, n=4):
        for i in range(n):
            m = (m - 1 if m % 10 else m // 10)
        return res == m
    
    def sol(m=1234578987654321, n=4):
        """
        Find the result of applying the following operation to integer m, n times: if the last digit is zero, remove
        the zero, otherwise subtract 1.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        for i in range(n):
            m = (m - 1 if m % 10 else m // 10)
        return m
    ```
    
    </details>
    
* <a name="shortestdecdelta"></a>**ShortestDecDelta** Inspired by [Codeforces Problem 617 A](https://codeforces.com/problemset/problem/617/A) (5 instances)
    
    ```python
    def sat(li: List[int], n=149432, upper=14943):
        return len(li) <= upper and all(abs(a - b) <= 10 for a, b in zip([1] + li, li + [n]))
    
    def sol(n=149432, upper=14943):
        """
        Find a the shortest sequence of integers going from 1 to n where each difference is at most 10.
        Do not include 1 or n in the sequence.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        m = 1
        ans = []
        while True:
            m = min(n, m + 10)
            if m >= n:
                return ans
            ans.append(m)
    ```
    
    </details>
    
* <a name="maxdelta"></a>**MaxDelta** Inspired by [Codeforces Problem 116 A](https://codeforces.com/problemset/problem/116/A) (5 instances)
    
    ```python
    def sat(n: int, pairs=[[3, 0], [17, 1], [9254359, 19], [123, 9254359], [0, 123]]):
        assert sum(p - m for p, m in pairs) == 0, "oo"
        tot = 0
        success = False
        for p, m in pairs:
            tot -= m
            tot += p
            assert tot <= n
            if tot == n:
                success = True
        return success
    
    def sol(pairs=[[3, 0], [17, 1], [9254359, 19], [123, 9254359], [0, 123]]):
        """
        Given a sequence of integer pairs, p_i, m_i, where \sum p_i-m_i = 0, find the maximum value, over t, of
        p_{t+1} + \sum_{i=1}^t p_i - m_i
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        tot = 0
        n = 0
        for p, m in pairs:
            tot += p - m
            if tot > n:
                n = tot
        return n
    ```
    
    </details>
    
* <a name="commoncase"></a>**CommonCase** Inspired by [Codeforces Problem 59 A](https://codeforces.com/problemset/problem/59/A)
    
    This is a trivial puzzle, especially if the AI realizes that it can can just copy the solution from
    the problem (5 instances)
    
    ```python
    def sat(s_case: str, s="CanYouTellIfItHASmoreCAPITALS"):
        caps = 0
        for c in s:
            if c != c.lower():
                caps += 1
        return s_case == (s.upper() if caps > len(s) // 2 else s.lower())
    
    def sol(s="CanYouTellIfItHASmoreCAPITALS"):
        """
        Given a word, replace it either with an upper-case or lower-case depending on whether or not it has more
        capitals or lower-case letters. If it has strictly more capitals, use upper-case, otherwise, use lower-case.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        caps = 0
        for c in s:
            if c != c.lower():
                caps += 1
        return (s.upper() if caps > len(s) // 2 else s.lower())  # duh, just take sat and return the answer checked for
    ```
    
    </details>
    
* <a name="sssuubbstriiingg"></a>**Sssuubbstriiingg** Inspired by [Codeforces Problem 58 A](https://codeforces.com/problemset/problem/58/A) (5 instances)
    
    ```python
    def sat(inds: List[int], string="Sssuubbstrissiingg"):
        return inds == sorted(inds) and "".join(string[i] for i in inds) == "substring"
    
    def sol(string="Sssuubbstrissiingg"):
        """Find increasing indices to make the substring "substring"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        target = "substring"
        j = 0
        ans = []
        for i in range(len(string)):
            while string[i] == target[j]:
                ans.append(i)
                j += 1
                if j == len(target):
                    return ans
    ```
    
    </details>
    
* <a name="sstriiinggssuubb"></a>**Sstriiinggssuubb** Inspired by [Codeforces Problem 58 A](https://codeforces.com/problemset/problem/58/A) (5 instances)
    
    ```python
    def sat(inds: List[int], string="enlightenment"):
        return inds == sorted(inds) and "".join(string[i] for i in inds) == "intelligent"
    
    def sol(string="enlightenment"):
        """Find increasing indices to make the substring "intelligent" (with a surprise twist)"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        target = "intelligent"
        j = 0
        ans = []
        for i in range(-len(string), len(string)):
            while string[i] == target[j]:
                ans.append(i)
                j += 1
                if j == len(target):
                    return ans
    ```
    
    </details>
    
* <a name="moving0s"></a>**Moving0s** Inspired by [Codeforces Problem 266 B](https://codeforces.com/problemset/problem/266/B) (5 instances)
    
    ```python
    def sat(seq: List[int], target=[1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], n_steps=4):
        s = seq[:]  # copy
        for step in range(n_steps):
            for i in range(len(seq) - 1):
                if (s[i], s[i + 1]) == (0, 1):
                    (s[i], s[i + 1]) = (1, 0)
        return s == target
    
    def sol(target=[1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], n_steps=4):
        """
        Find a sequence of 0's and 1's so that, after n_steps of swapping each adjacent (0, 1), the target sequence
        is achieved.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        s = target[:]  # copy
        for step in range(n_steps):
            for i in range(len(target) - 2, -1, -1):
                if (s[i], s[i + 1]) == (1, 0):
                    (s[i], s[i + 1]) = (0, 1)
        return s
    ```
    
    </details>
    
* <a name="factor47"></a>**Factor47** Inspired by [Codeforces Problem 122 A](https://codeforces.com/problemset/problem/122/A) (5 instances)
    
    ```python
    def sat(d: int, n=6002685529):
        return n % d == 0 and all(i in "47" for i in str(d))
    
    def sol(n=6002685529):
        """Find a integer factor of n whose decimal representation consists only of 7's and 4's."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        def helper(so_far, k):
            if k > 0:
                return helper(so_far * 10 + 4, k - 1) or helper(so_far * 10 + 7, k - 1)
            return (n % so_far == 0) and so_far
    
        for length in range(1, len(str(n)) // 2 + 2):
            ans = helper(0, length)
            if ans:
                return ans
    ```
    
    </details>
    
* <a name="count47"></a>**Count47** Inspired by [Codeforces Problem 110 A](https://codeforces.com/problemset/problem/110/A) (5 instances)
    
    ```python
    def sat(d: int, n=123456789):
        return d > n and all(i in "47" for i in str(str(d).count("4") + str(d).count("7")))
    
    def sol(n=123456789):
        """
        Find a number bigger than n whose decimal representation has k 4's and 7's where k's decimal representation
        consists only of 4's and 7's
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return int("4444" + "0" * (len(str(n)) - 3))
    ```
    
    </details>
    
* <a name="maybereversed"></a>**MaybeReversed** Inspired by [Codeforces Problem 41 A](https://codeforces.com/problemset/problem/41/A) (5 instances)
    
    ```python
    def sat(s: str, target="reverse me", reverse=True):
        return (s[::-1] == target) == reverse
    
    def sol(target="reverse me", reverse=True):
        """Either reverse a string or don't based on the reverse flag"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return target[::-1] if reverse else target + "x"
    ```
    
    </details>
    
* <a name="minbigger"></a>**MinBigger** Inspired by [Codeforces Problem 160 A](https://codeforces.com/problemset/problem/160/A) (5 instances)
    
    ```python
    def sat(taken: List[int], val_counts=[[4, 3], [5, 2], [9, 3], [13, 13], [8, 11], [56, 1]], upper=11):
        advantage = 0
        assert len(taken) == len(val_counts) and sum(taken) <= upper
        for i, (val, count) in zip(taken, val_counts):
            assert 0 <= i <= count
            advantage += val * i - val * count / 2
        return advantage > 0
    
    def sol(val_counts=[[4, 3], [5, 2], [9, 3], [13, 13], [8, 11], [56, 1]], upper=11):
        """
        The list of numbers val_counts represents multiple copies of integers, e.g.,
        val_counts=[[3, 2], [4, 6]] corresponds to 3, 3, 4, 4, 4, 4, 4, 4
        For each number, decide how many to take so that the total number taken is <= upper and the sum of those
        taken exceeds half the total sum.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        n = len(val_counts)
        pi = sorted(range(n), key=lambda i: val_counts[i][0])
        needed = sum(a * b for a, b in val_counts) / 2 + 0.1
        ans = [0] * n
        while needed > 0:
            while val_counts[pi[-1]][1] == ans[pi[-1]]:
                pi.pop()
            i = pi[-1]
            ans[i] += 1
            needed -= val_counts[i][0]
        return ans
    ```
    
    </details>
    
* <a name="dada"></a>**Dada** Inspired by [Codeforces Problem 734 A](https://codeforces.com/problemset/problem/734/A) (5 instances)
    
    ```python
    def sat(s: str, a=5129, d=17):
        return s.count("a") == a and s.count("d") == d and len(s) == a + d
    
    def sol(a=5129, d=17):
        """Find a string with a given number of a's and d's"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "a" * a + "d" * d
    ```
    
    </details>
    
* <a name="distinctdigits"></a>**DistinctDigits** Inspired by [Codeforces Problem 271 A](https://codeforces.com/problemset/problem/271/A) (5 instances)
    
    ```python
    def sat(nums: List[int], a=100, b=1000, count=648):
        assert all(len(str(n)) == len(set(str(n))) and a <= n <= b for n in nums)
        return len(set(nums)) >= count
    
    def sol(a=100, b=1000, count=648):
        """Find a list of count or more different numbers each between a and b that each have no repeated digits"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [n for n in range(a, b + 1) if len(str(n)) == len(set(str(n)))]
    ```
    
    </details>
    
* <a name="easysum"></a>**EasySum** Inspired by [Codeforces Problem 677 A](https://codeforces.com/problemset/problem/677/A) (5 instances)
    
    ```python
    def sat(tot: int, nums=[2, 8, 25, 18, 99, 11, 17, 16], thresh=17):
        return tot == sum(1 if i < thresh else 2 for i in nums)
    
    def sol(nums=[2, 8, 25, 18, 99, 11, 17, 16], thresh=17):
        """Add up 1 or 2 for numbers in a list depending on whether they exceed a threshold"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sum(1 if i < thresh else 2 for i in nums)
    ```
    
    </details>
    
* <a name="gimmechars"></a>**GimmeChars** Inspired by [Codeforces Problem 133 A](https://codeforces.com/problemset/problem/133/A), easy (5 instances)
    
    ```python
    def sat(s: str, chars=['o', 'h', 'e', 'l', ' ', 'w', '!', 'r', 'd']):
        for c in chars:
            if c not in s:
                return False
        return True
    
    def sol(chars=['o', 'h', 'e', 'l', ' ', 'w', '!', 'r', 'd']):
        """Find a string with certain characters"""
    ```
    0 hand-written solutions 
    
* <a name="halfpairs"></a>**HalfPairs** Inspired by [Codeforces Problem 467 A](https://codeforces.com/problemset/problem/467/A) (4 instances)
    
    ```python
    def sat(ans: List[List[int]], target=17):
        for i in range(len(ans)):
            a, b = ans[i]
            if b - a >= 2:
                target -= 1
        return target == 0
    
    def sol(target=17):
        """
        Find a list of pairs of integers where the number of pairs in which the second number is more than
        two greater than the first number is a given constant
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [[0, 2]] * target
    ```
    
    </details>
    
* <a name="invertindices"></a>**InvertIndices** Inspired by [Codeforces Problem 136 A](https://codeforces.com/problemset/problem/136/A) (5 instances)
    
    ```python
    def sat(indexes: List[int], target=[1, 3, 4, 2, 5, 6, 7, 13, 12, 11, 9, 10, 8]):
        for i in range(1, len(target) + 1):
            if target[indexes[i - 1] - 1] != i:
                return False
        return True
    
    def sol(target=[1, 3, 4, 2, 5, 6, 7, 13, 12, 11, 9, 10, 8]):
        """Given a list of integers representing a permutation, invert the permutation."""
    ```
    0 hand-written solutions 
    
* <a name="fivepowers"></a>**FivePowers** Inspired by [Codeforces Problem 630 A](https://codeforces.com/problemset/problem/630/A) (5 instances)
    
    ```python
    def sat(s: str, n=7012):
        return int(str(5 ** n)[:-2] + s) == 5 ** n
    
    def sol(n=7012):
        """What are the last two digits of 5^n?"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return ("1" if n == 0 else "5" if n == 1 else "25")
    ```
    
    </details>
    
* <a name="combinationlock"></a>**CombinationLock** Inspired by [Codeforces Problem 540 A](https://codeforces.com/problemset/problem/540/A) (5 instances)
    
    ```python
    def sat(states: List[str], start="424", combo="778", target_len=12):
        assert all(len(s) == len(start) for s in states) and all(c in "0123456789" for s in states for c in s)
        for a, b in zip([start] + states, states + [combo]):
            assert sum(i != j for i, j in zip(a, b)) == 1
            assert all(abs(int(i) - int(j)) in {0, 1, 9} for i, j in zip(a, b))
    
        return len(states) <= target_len
    
    def sol(start="424", combo="778", target_len=12):
        """
        Shortest Combination Lock Path
    
        Given a starting a final lock position, find the (minimal) intermediate states, where each transition
        involves increasing or decreasing a single digit (mod 10).
    
        Example:
        start = "012"
        combo = "329"
        output: ['112', '212', '312', '322', '321', '320']
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        n = len(start)
        ans = []
        a, b = [[int(c) for c in x] for x in [start, combo]]
        for i in range(n):
            while a[i] != b[i]:
                a[i] = (a[i] - 1 if (a[i] - b[i]) % 10 < 5 else a[i] + 1) % 10
                if a != b:
                    ans.append("".join(str(i) for i in a))
        return ans
    ```
    
    </details>
    
* <a name="combinationlockobfuscated"></a>**CombinationLockObfuscated** Inspired by [Codeforces Problem 540 A](https://codeforces.com/problemset/problem/540/A)
    This an obfuscated version of CombinationLock above, can the AI figure out what is being asked or that
    it is the same puzzle? (5 instances)
    
    ```python
    def sat(states: List[str], start="424", combo="778", target_len=12):
        return all(sum((int(a[i]) - int(b[i])) ** 2 % 10 for i in range(len(start))) == 1
                   for a, b in zip([start] + states, states[:target_len] + [combo]))
    
    def sol(start="424", combo="778", target_len=12):
        """Figure out what this does only from the code"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        n = len(start)
        ans = []
        a, b = [[int(c) for c in x] for x in [start, combo]]
        for i in range(n):
            while a[i] != b[i]:
                a[i] = (a[i] - 1 if (a[i] - b[i]) % 10 < 5 else a[i] + 1) % 10
                if a != b:
                    ans.append("".join(str(i) for i in a))
        return ans
    ```
    
    </details>
    
* <a name="invertpermutation"></a>**InvertPermutation** Inspired by [Codeforces Problem 474 A](https://codeforces.com/problemset/problem/474/A) (5 instances)
    
    ```python
    def sat(s: str, perm="qwertyuiopasdfghjklzxcvbnm", target="hello are you there?"):
        return "".join((perm[(perm.index(c) + 1) % len(perm)] if c in perm else c) for c in s) == target
    
    def sol(perm="qwertyuiopasdfghjklzxcvbnm", target="hello are you there?"):
        """Find a string that, when a given permutation of characters is applied, has a given result."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "".join((perm[(perm.index(c) - 1) % len(perm)] if c in perm else c) for c in target)
    ```
    
    </details>
    
* <a name="samedifferent"></a>**SameDifferent** Inspired by [Codeforces Problem 1335 C](https://codeforces.com/problemset/problem/1335/C) (5 instances)
    
    ```python
    def sat(lists: List[List[int]], items=[5, 4, 9, 4, 5, 5, 5, 1, 5, 5], length=4):
        a, b = lists
        assert len(a) == len(b) == length
        assert len(set(a)) == len(a)
        assert len(set(b)) == 1
        for i in a + b:
            assert (a + b).count(i) <= items.count(i)
        return True
    
    def sol(items=[5, 4, 9, 4, 5, 5, 5, 1, 5, 5], length=4):
        """
        Given a list of integers and a target length, create of the given length such that:
            * The first list must be all different numbers.
            * The second must be all the same number.
            * The two lists together comprise a sublist of all the list items
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        from collections import Counter
        [[a, count]] = Counter(items).most_common(1)
        assert count >= length
        seen = {a}
        dedup = [i for i in items if i not in seen and not seen.add(i)]
        return [(dedup + [a])[:length], [a] * length]
    ```
    
    </details>
    
* <a name="onesandtwos"></a>**OnesAndTwos** Inspired by [Codeforces Problem 476 A](https://codeforces.com/problemset/problem/476/A) (5 instances)
    
    ```python
    def sat(seq: List[int], n=10000, length=5017):
        return all(i in [1, 2] for i in seq) and sum(seq) == n and len(seq) == length
    
    def sol(n=10000, length=5017):
        """Find a sequence of 1's and 2's of a given length that that adds up to n"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [2] * (n - length) + [1] * (2 * length - n)
    ```
    
    </details>
    
* <a name="minconsecutivesum"></a>**MinConsecutiveSum** Inspired by [Codeforces Problem 363 B](https://codeforces.com/problemset/problem/363/B) (5 instances)
    
    ```python
    def sat(start: int, k=3, upper=6, seq=[17, 1, 2, 65, 18, 91, -30, 100, 3, 1, 2]):
        return 0 <= start <= len(seq) - k and sum(seq[start:start + k]) <= upper
    
    def sol(k=3, upper=6, seq=[17, 1, 2, 65, 18, 91, -30, 100, 3, 1, 2]):
        """Find a sequence of k consecutive indices whose sum is minimal"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return min(range(len(seq) - k + 1), key=lambda start: sum(seq[start:start + k]))
    ```
    
    </details>
    
* <a name="maxconsecutivesum"></a>**MaxConsecutiveSum** Inspired by [Codeforces Problem 363 B](https://codeforces.com/problemset/problem/363/B) (5 instances)
    
    ```python
    def sat(start: int, k=3, lower=150, seq=[3, 1, 2, 65, 18, 91, -30, 100, 0, 19, 52]):
        return 0 <= start <= len(seq) - k and sum(seq[start:start + k]) >= lower
    
    def sol(k=3, lower=150, seq=[3, 1, 2, 65, 18, 91, -30, 100, 0, 19, 52]):
        """Find a sequence of k consecutive indices whose sum is maximal"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return max(range(len(seq) - k + 1), key=lambda start: sum(seq[start:start + k]))
    ```
    
    </details>
    
* <a name="maxconsecutiveproduct"></a>**MaxConsecutiveProduct** Inspired by [Codeforces Problem 363 B](https://codeforces.com/problemset/problem/363/B) (5 instances)
    
    ```python
    def sat(start: int, k=3, lower=100000, seq=[91, 1, 2, 64, 18, 91, -30, 100, 3, 65, 18]):
        prod = 1
        for i in range(start, start + k):
            prod *= seq[i]
        return prod >= lower
    
    def sol(k=3, lower=100000, seq=[91, 1, 2, 64, 18, 91, -30, 100, 3, 65, 18]):
        """Find a sequence of k consecutive indices whose product is maximal, possibly looping around"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        def prod(start):
            ans = 1
            for i in range(start, start + k):
                ans *= seq[i]
            return ans
    
        return max(range(-len(seq), len(seq) - k + 1), key=prod)
    ```
    
    </details>
    
* <a name="distinctoddsum"></a>**DistinctOddSum** Inspired by [Codeforces Problem 1327 A](https://codeforces.com/problemset/problem/1327/A) (5 instances)
    
    ```python
    def sat(nums: List[int], tot=12345, n=5):
        return len(nums) == len(set(nums)) == n and sum(nums) == tot and all(i >= i % 2 > 0 for i in nums)
    
    def sol(tot=12345, n=5):
        """Find n distinct positive odd integers that sum to tot"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return list(range(1, 2 * n - 1, 2)) + [tot - sum(range(1, 2 * n - 1, 2))]
    ```
    
    </details>
    
* <a name="minrotations"></a>**MinRotations** Inspired by [Codeforces Problem 731 A](https://codeforces.com/problemset/problem/731/A) (5 instances)
    
    ```python
    def sat(rotations: List[int], target="wonderful", upper=69):
        s = "abcdefghijklmnopqrstuvwxyz"
        assert len(rotations) == len(target)
        for r, c in zip(rotations, target):
            s = s[r:] + s[:r]
            assert s[0] == c
    
        return sum(abs(r) for r in rotations) <= upper
    
    def sol(target="wonderful", upper=69):
        """
        We begin with the string `"a...z"`
    
        An `r`-rotation of a string means shifting it to the right (positive) or left (negative) by `r` characters and
        cycling around. Given a target string of length n, find the n rotations that put the consecutive characters
        of that string at the beginning of the r-rotation, with minimal sum of absolute values of the `r`'s.
    
        For example if the string was `'dad'`, the minimal rotations would be `[3, -3, 3]` with a total of `9`.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        s = "abcdefghijklmnopqrstuvwxyz"
        ans = []
        for c in target:
            i = s.index(c)
            r = min([i, i - len(s)], key=abs)
            ans.append(r)
            s = s[r:] + s[:r]
            assert s[0] == c
        return ans
    ```
    
    </details>
    
* <a name="billsums"></a>**BillSums** Inspired by [Codeforces Problem 996 A](https://codeforces.com/problemset/problem/996/A)
    
    We make it much harder when the denominations are non-American so the greedy algorithm doesn't work. (5 instances)
    
    ```python
    def sat(bills: List[int], denominations=[1, 25, 35, 84], n=980, max_len=14):
        return sum(bills) == n and all(b in denominations for b in bills) and len(bills) <= max_len
    
    def sol(denominations=[1, 25, 35, 84], n=980, max_len=14):
        """
        Find the shortest sequence (length <= max_len) that sum to n, where each number is in denominations
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        """
        This solution uses dynamic programming, I believe it could be further sped up without having to count
        all the way up to denominations.
        """
        denominations = sorted(set(denominations)) # remove duplicates
        seqs = [[0 for _ in denominations] +[0]]  # vectors
        for i in range(1, n + 1):
            _, j, k = min((seqs[i - k][-1], j, k) for j, k in enumerate(denominations) if k <= i)
            s = seqs[i - k]
            seqs.append([*s[:j], s[j] + 1, *s[j + 1:-1], s[-1] + 1])
    
        return [k for k, count in zip(denominations, seqs[-1]) for _ in range(count)]
    ```
    
    </details>
    
* <a name="boxvolume"></a>**BoxVolume** (Also) inspired by [Codeforces Problem 996 A](https://codeforces.com/problemset/problem/996/A)
    
    We make it much much harder by making it a multiplication problem where the greedy algorithm doesn't work. (5 instances)
    
    ```python
    def sat(sides: List[int], options=[2, 512, 1024], n=340282366920938463463374607431768211456, max_dim=13):
        prod = 1
        for b in sides:
            prod *= b
        return prod == n and set(sides) <= set(options) and len(sides) <= max_dim
    
    def sol(options=[2, 512, 1024], n=340282366920938463463374607431768211456, max_dim=13):
        """
        Find the side lengths of a box in fewest dimensions (dimension <= max_dim) whose volume is n,
         where each side length is in options
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        options = sorted(set(options))
        base = options[0]
        logs = []
        for i in options + [n]:
            j = 1
            log = 0
            while j < i:
                log +=1
                j *= base
            assert j == i, "All numbers must be a power of the smallest number"
            logs.append(log)
        denominations, n = logs[:-1], logs[-1]
    
        seqs = [[0 for _ in denominations] +[0]]  # vectors
        for i in range(1, n + 1):
            _, j, k = min((seqs[i - k][-1], j, k) for j, k in enumerate(denominations) if k <= i)
            s = seqs[i - k]
            seqs.append([*s[:j], s[j] + 1, *s[j + 1:-1], s[-1] + 1])
    
        return [base ** k for k, count in zip(denominations, seqs[-1]) for _ in range(count)]
    ```
    
    </details>
    
## algebra

Roots of polynomials

* <a name="quadraticroot"></a>**QuadraticRoot** See [quadratic equations](https://en.wikipedia.org/wiki/Quadratic_formula) (5 instances)
    
    ```python
    def sat(x: float, coeffs=[2.5, 1.3, -0.5]):
        a, b, c = coeffs
        return abs(a * x ** 2 + b * x + c) < 1e-6
    
    def sol(coeffs=[2.5, 1.3, -0.5]):
        """
        Find any (real) solution to:  a x^2 + b x + c where coeffs = [a, b, c].
        For example, since x^2 - 3x + 2 has a root at 1, sat(x = 1., coeffs = [1., -3., 2.]) is True.
        """
    ```
    <details><summary>2 hand-written solutions </summary>
    
    ```python
        a, b, c = coeffs
        if a == 0:
            ans = -c / b if b != 0 else 0.0
        else:
            ans = ((-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a))
        return ans
    ```
    
    ```python
        a, b, c = coeffs
        if a == 0:
            ans = -c / b if b != 0 else 0.0
        else:
            ans = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
        return ans
    ```
    
    </details>
    
* <a name="allquadraticroots"></a>**AllQuadraticRoots** See [quadratic equations](https://en.wikipedia.org/wiki/Quadratic_formula). (5 instances)
    
    ```python
    def sat(roots: List[float], coeffs=[1.3, -0.5]):
        b, c = coeffs
        r1, r2 = roots
        return abs(r1 + r2 + b) + abs(r1 * r2 - c) < 1e-6
    
    def sol(coeffs=[1.3, -0.5]):
        """Find all (real) solutions to: x^2 + b x + c (i.e., factor into roots), here coeffs = [b, c]"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        b, c = coeffs
        delta = (b ** 2 - 4 * c) ** 0.5
        return [(-b + delta) / 2, (-b - delta) / 2]
    ```
    
    </details>
    
* <a name="cubicroot"></a>**CubicRoot** See [cubic equation](https://en.wikipedia.org/wiki/Cubic_formula). (5 instances)
    
    ```python
    def sat(x: float, coeffs=[2.0, 1.0, 0.0, 8.0]):
        return abs(sum(c * x ** (3 - i) for i, c in enumerate(coeffs))) < 1e-6
    
    def sol(coeffs=[2.0, 1.0, 0.0, 8.0]):
        """
        Find any (real) solution to: a x^3 + b x^2 + c x + d where coeffs = [a, b, c, d]
        For example, since (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6, sat(x = 1., coeffs = [-6., 11., -6.]) is True.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
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
    ```
    
    </details>
    
* <a name="allcubicroots"></a>**AllCubicRoots** See [cubic equation](https://en.wikipedia.org/wiki/Cubic_formula). (5 instances)
    
    ```python
    def sat(roots: List[float], coeffs=[1.0, -2.0, -1.0]):
        r1, r2, r3 = roots
        a, b, c = coeffs
        return abs(r1 + r2 + r3 + a) + abs(r1 * r2 + r1 * r3 + r2 * r3 - b) + abs(r1 * r2 * r3 + c) < 1e-6
    
    def sol(coeffs=[1.0, -2.0, -1.0]):
        """Find all 3 distinct real roots of x^3 + a x^2 + b x + c, i.e., factor into (x-r1)(x-r2)(x-r3).
        coeffs = [a, b, c]. For example, since (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6,
        sat(roots = [1., 2., 3.], coeffs = [-6., 11., -6.]) is True.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
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
    ```
    
    </details>
    
## basic

Problems testing basic knowledge -- easy to solve if you understand what is being asked

* <a name="sumofdigits"></a>**SumOfDigits**  (5 instances)
    
    ```python
    def sat(x: str, s=679):
        return s == sum([int(d) for d in x])
    
    def sol(s=679):
        """Find a number that its digits sum to a specific value."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return int(s / 9) * '9' + str(s % 9)
    ```
    
    </details>
    
* <a name="floatwithdecimalvalue"></a>**FloatWithDecimalValue**  (5 instances)
    
    ```python
    def sat(z: float, v=9, d=0.0001):
        return int(z * 1 / d % 10) == v
    
    def sol(v=9, d=0.0001):
        """Create a float with a specific decimal."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return v * d
    ```
    
    </details>
    
* <a name="arithmeticsequence"></a>**ArithmeticSequence**  (5 instances)
    
    ```python
    def sat(x: List[int], a=7, s=5, e=200):
        return x[0] == a and x[-1] <= e and (x[-1] + s > e) and all([x[i] + s == x[i + 1] for i in range(len(x) - 1)])
    
    def sol(a=7, s=5, e=200):
        """Create a list that is a subrange of an arithmetic sequence."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return list(range(a, e + 1, s))
    ```
    
    </details>
    
* <a name="geometricsequence"></a>**GeometricSequence**  (5 instances)
    
    ```python
    def sat(x: List[int], a=8, r=2, l=50):
        return x[0] == a and len(x) == l and all([x[i] * r == x[i + 1] for i in range(len(x) - 1)])
    
    def sol(a=8, r=2, l=50):
        """Create a list that is a subrange of an gemoetric sequence."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [a * r ** i for i in range(l)]
    ```
    
    </details>
    
* <a name="lineintersection"></a>**LineIntersection**  (5 instances)
    
    ```python
    def sat(e: List[int], a=2, b=-1, c=1, d=2021):
        x = e[0] / e[1]
        return abs(a * x + b - c * x - d) < 10 ** -5
    
    def sol(a=2, b=-1, c=1, d=2021):
        """
        Find the intersection of two lines.
        Solution should be a list of the (x,y) coordinates.
        Accuracy of fifth decimal digit is required.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [d - b, a - c]
    ```
    
    </details>
    
* <a name="ifproblem"></a>**IfProblem**  (5 instances)
    
    ```python
    def sat(x: int, a=324554, b=1345345):
        if a < 50:
            return x + a == b
        else:
            return x - 2 * a == b
    
    def sol(a=324554, b=1345345):
        """Satisfy a simple if statement"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if a < 50:
            return b - a
        else:
            return b + 2 * a
    ```
    
    </details>
    
* <a name="ifproblemwithand"></a>**IfProblemWithAnd**  (5 instances)
    
    ```python
    def sat(x: int, a=9384594, b=1343663):
        if x > 0 and a > 50:
            return x - a == b
        else:
            return x + a == b
    
    def sol(a=9384594, b=1343663):
        """Satisfy a simple if statement with an and clause"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if a > 50 and b > a:
            return b + a
        else:
            return b - a
    ```
    
    </details>
    
* <a name="ifproblemwithor"></a>**IfProblemWithOr**  (5 instances)
    
    ```python
    def sat(x: int, a=253532, b=1230200):
        if x > 0 or a > 50:
            return x - a == b
        else:
            return x + a == b
    
    def sol(a=253532, b=1230200):
        """Satisfy a simple if statement with an or clause"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if a > 50 or b > a:
            return b + a
        else:
            return b - a
    ```
    
    </details>
    
* <a name="ifcases"></a>**IfCases**  (5 instances)
    
    ```python
    def sat(x: int, a=4, b=54368639):
        if a == 1:
            return x % 2 == 0
        elif a == -1:
            return x % 2 == 1
        else:
            return x + a == b
    
    def sol(a=4, b=54368639):
        """Satisfy a simple if statement with multiple cases"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if a == 1:
            x = 0
        elif a == -1:
            x = 1
        else:
            x = b - a
        return x
    ```
    
    </details>
    
* <a name="listpossum"></a>**ListPosSum**  (5 instances)
    
    ```python
    def sat(x: List[int], n=5, s=19):
        return len(x) == n and sum(x) == s and all([a > 0 for a in x])
    
    def sol(n=5, s=19):
        """Find a list of n non-negative integers that sum up to s"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        x = [1] * n
        x[0] = s - n + 1
        return x
    ```
    
    </details>
    
* <a name="listdistinctsum"></a>**ListDistinctSum**  (5 instances)
    
    ```python
    def sat(x: List[int], n=4, s=2021):
        return len(x) == n and sum(x) == s and len(set(x)) == n
    
    def sol(n=4, s=2021):
        """Construct a list of n distinct integers that sum up to s"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        a = 1
        x = []
        while len(x) < n - 1:
            x.append(a)
            a = -a
            if a in x:
                a += 1
    
        if s - sum(x) in x:
            x = [i for i in range(n - 1)]
    
        x = x + [s - sum(x)]
        return x
    ```
    
    </details>
    
* <a name="concatstrings"></a>**ConcatStrings**  (5 instances)
    
    ```python
    def sat(x: str, s=['a', 'b', 'c', 'd', 'e', 'f'], n=4):
        return len(x) == n and all([x[i] == s[i] for i in range(n)])
    
    def sol(s=['a', 'b', 'c', 'd', 'e', 'f'], n=4):
        """Concatenate the list of characters in s"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return ''.join([s[i] for i in range(n)])
    ```
    
    </details>
    
* <a name="sublistsum"></a>**SublistSum**  (5 instances)
    
    ```python
    def sat(x: List[int], t=677, a=43, e=125, s=10):
        non_zero = [z for z in x if z != 0]
        return t == sum([x[i] for i in range(a, e, s)]) and len(set(non_zero)) == len(non_zero) and all(
            [x[i] != 0 for i in range(a, e, s)])
    
    def sol(t=677, a=43, e=125, s=10):
        """Sum values of sublist by range specifications"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        x = [0] * e
        for i in range(a, e, s):
            x[i] = i
        correction = t - sum(x) + x[i]
        if correction in x:
            x[correction] = -1 * correction
            x[i] = 3 * correction
        else:
            x[i] = correction
        return x
    ```
    
    </details>
    
* <a name="cumulativesum"></a>**CumulativeSum**  (5 instances)
    
    ```python
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
    
    def sol(t=50, n=10):
        """Find how many values have cumulative sum less than target"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [1] * n + [t]
    ```
    
    </details>
    
* <a name="basicstrcounts"></a>**BasicStrCounts**  (5 instances)
    
    ```python
    def sat(s: str, s1="a", s2="b", count1=50, count2=30):
        return s.count(s1) == count1 and s.count(s2) == count2 and s[:10] == s[-10:]
    
    def sol(s1="a", s2="b", count1=50, count2=30):
        """
        Find a string that has count1 occurrences of s1 and count2 occurrences of s2 and starts and ends with
        the same 10 characters
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if s1 == s2:
            ans = (s1 + "?") * count1
        elif s1.count(s2):
            ans = (s1 + "?") * count1
            ans += (s2 + "?") * (count2 - ans.count(s2))
        else:
            ans = (s2 + "?") * count2
            ans += (s1 + "?") * (count1 - ans.count(s1))
        return "?" * 10 + ans + "?" * 10
    ```
    
    </details>
    
* <a name="zipstr"></a>**ZipStr**  (5 instances)
    
    ```python
    def sat(s: str, substrings=['foo', 'bar', 'baz', 'oddball']):
        return all(sub in s[i::len(substrings)] for i, sub in enumerate(substrings))
    
    def sol(substrings=['foo', 'bar', 'baz', 'oddball']):
        """
        Find a string that contains each string in substrings alternating, e.g., 'cdaotg' for 'cat' and 'dog'
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        m = max(len(s) for s in substrings)
        return "".join([(s[i] if i < len(s) else " ") for i in range(m) for s in substrings])
    ```
    
    </details>
    
* <a name="reversecat"></a>**ReverseCat**  (5 instances)
    
    ```python
    def sat(s: str, substrings=['foo', 'bar', 'baz']):
        return all(sub in s and sub[::-1] in s for sub in substrings)
    
    def sol(substrings=['foo', 'bar', 'baz']):
        """
        Find a string that contains all the substrings reversed and forward
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "".join(substrings + [s[::-1] for s in substrings])
    ```
    
    </details>
    
* <a name="engineernumbers"></a>**EngineerNumbers**  (5 instances)
    
    ```python
    def sat(ls: List[str], n=100, a="bar", b="foo"):
        return len(ls) == len(set(ls)) == n and ls[0] == a and ls[-1] == b and ls == sorted(ls)
    
    def sol(n=100, a="bar", b="foo"):
        """
        Find a list of n strings, in alphabetical order, starting with a and ending with b.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sorted([a] + [a + chr(0) + str(i) for i in range(n - 2)] + [b])
    ```
    
    </details>
    
* <a name="penultimatestring"></a>**PenultimateString**  (5 instances)
    
    ```python
    def sat(s: str, strings=['cat', 'dog', 'bird', 'fly', 'moose']):
        return s in strings and sum(t > s for t in strings) == 1
    
    def sol(strings=['cat', 'dog', 'bird', 'fly', 'moose']):
        """Find the alphabetically second to last last string in a list."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sorted(strings)[-2]
    ```
    
    </details>
    
* <a name="penultimaterevstring"></a>**PenultimateRevString**  (5 instances)
    
    ```python
    def sat(s: str, strings=['cat', 'dog', 'bird', 'fly', 'moose']):
        return s[::-1] in strings and sum(t < s[::-1] for t in strings) == 1
    
    def sol(strings=['cat', 'dog', 'bird', 'fly', 'moose']):
        """Find the reversed version of the alphabetically second string in a list."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return sorted(strings)[1][::-1]
    ```
    
    </details>
    
* <a name="centeredstring"></a>**CenteredString**  (5 instances)
    
    ```python
    def sat(s: str, target="foobarbazwow", length=6):
        return target[(len(target) - length) // 2:(len(target) + length) // 2] == s
    
    def sol(target="foobarbazwow", length=6):
        """Find a substring of the given length centered within the target string."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return target[(len(target) - length) // 2:(len(target) + length) // 2]
    ```
    
    </details>
    
* <a name="substrcount"></a>**SubstrCount**  (5 instances)
    
    ```python
    def sat(substring: str, string="moooboooofasd", count=2):
        return string.count(substring) == count
    
    def sol(string="moooboooofasd", count=2):
        """Find a substring with a certain count in a given string"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        for i in range(len(string)):
            for j in range(i+1, len(string)):
                substring = string[i:j]
                c = string.count(substring)
                if c == count:
                    return substring
                if c < count:
                    break
        assert False
    ```
    
    </details>
    
* <a name="completeparens"></a>**CompleteParens**  (5 instances)
    
    ```python
    def sat(t: str, s="))(Add)some))parens()to()(balance(()(()(me!)(((("):
        for i in range(len(t) + 1):
            depth = t[:i].count("(") - t[:i].count(")")
            assert depth >= 0
        return depth == 0 and s in t
    
    def sol(s="))(Add)some))parens()to()(balance(()(()(me!)(((("):
        """Add parentheses to the beginning and end of s to make all parentheses balanced"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "(" * s.count(")") + s + ")" * s.count("(")
    ```
    
    </details>
    
## chess

Classic chess puzzles

* <a name="eightqueensorfewer"></a>**EightQueensOrFewer** Eight (or fewer) Queens Puzzle
    
    See Wikipedia entry on
    [Eight Queens puzzle](https://en.wikipedia.org/w/index.php?title=Eight_queens_puzzle).
    
    See the MoreQueens puzzle below for another (longer but clearer) equivalent definition of sat
    
    Hint: a brute force approach works on this puzzle. (5 instances)
    
    ```python
    def sat(squares: List[List[int]], m=8, n=8):
        k = min(m, n)
        assert all(i in range(m) and j in range(n) for i, j in squares) and len(squares) == k
        return 4 * k == len({t for i, j in squares for t in [('row', i), ('col', j), ('SE', i + j), ('NE', i - j)]})
    
    def sol(m=8, n=8):
        """Position min(m, n) <= 8 queens on an m x n chess board so that no pair is attacking each other."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        # brute force
        k = min(m, n)
    
        from itertools import permutations
        for p in permutations(range(k)):
            if 4 * k == len(
                    {t for i, j in enumerate(p) for t in [('row', i), ('col', j), ('SE', i + j), ('NE', i - j)]}):
                return [[i, j] for i, j in enumerate(p)]
    ```
    
    </details>
    
* <a name="morequeens"></a>**MoreQueens** See Wikipedia entry on [Eight Queens puzzle](https://en.wikipedia.org/w/index.php?title=Eight_queens_puzzle).
    
    A brute force approach will not work on many of these problems. (5 instances)
    
    ```python
    def sat(squares: List[List[int]], m=9, n=9):
        k = min(m, n)
        assert all(i in range(m) and j in range(n) for i, j in squares), "queen off board"
        assert len(squares) == k, "Wrong number of queens"
        assert len({i for i, j in squares}) == k, "Queens on same row"
        assert len({j for i, j in squares}) == k, "Queens on same file"
        assert len({i + j for i, j in squares}) == k, "Queens on same SE diagonal"
        assert len({i - j for i, j in squares}) == k, "Queens on same NE diagonal"
        return True
    
    def sol(m=9, n=9):
        """
        Position min(m, n) > 8 queens on an m x n chess board so that no pair is attacking each other.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        t = min(m, n)
        ans = []
        if t % 2 == 1:  # odd k, put a queen in the lower right corner (and decrement k)
            ans.append([t - 1, t - 1])
            t -= 1
        if t % 6 == 2:  # do something special for 8x8, 14x14 etc:
            ans += [[i, (2 * i + t // 2 - 1) % t] for i in range(t // 2)]
            ans += [[i + t // 2, (2 * i - t // 2 + 2) % t] for i in range(t // 2)]
        else:
            ans += [[i, 2 * i + 1] for i in range(t // 2)]
            ans += [[i + t // 2, 2 * i] for i in range(t // 2)]
        return ans
    ```
    
    </details>
    
* <a name="knightstour"></a>**KnightsTour** See Wikipedia entry on [Knight's tour](https://en.wikipedia.org/w/index.php?title=Knight%27s_tour) (5 instances)
    
    ```python
    def sat(tour: List[List[int]], m=8, n=8):
        assert all({abs(i1 - i2), abs(j1 - j2)} == {1, 2} for [i1, j1], [i2, j2] in zip(tour, tour[1:])), 'legal moves'
        return sorted(tour) == [[i, j] for i in range(m) for j in range(n)]  # cover every square once
    
    def sol(m=8, n=8):
        """Find an (open) tour of knight moves on an m x n chess-board that visits each square once."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        # using Warnsdorff's heuristic, breaking ties randomly
        import random
        for seed in range(100):
            r = random.Random(seed)
            ans = [(0, 0)]
            free = {(i, j) for i in range(m) for j in range(n)} - {(0, 0)}
    
            def possible(i, j):
                moves = [(i + s * a, j + t * b) for (a, b) in [(1, 2), (2, 1)] for s in [-1, 1] for t in [-1, 1]]
                return [z for z in moves if z in free]
    
            while True:
                if not free:
                    return [[a, b] for (a, b) in ans]
                candidates = possible(*ans[-1])
                if not candidates:
                    break
                ans.append(min(candidates, key=lambda z: len(possible(*z)) + r.random()))
                free.remove(ans[-1])
    ```
    
    </details>
    
* <a name="uncrossedknightspath"></a>**UncrossedKnightsPath** Uncrossed Knights Path (known solvable, but no solution given)
    
    The goal of these problems is to match the nxn_records from [http://ukt.alex-black.ru/](http://ukt.alex-black.ru/)
    (accessed 2020-11-29).
    
    A more precise description is in this
    [Wikipedia article](https://en.wikipedia.org/w/index.php?title=Longest_uncrossed_knight%27s_path). (5 instances)
    
    ```python
    def sat(path: List[List[int]], m=8, n=8, target=35):
        def legal_move(m):
            (a, b), (i, j) = m
            return {abs(i - a), abs(j - b)} == {1, 2}
    
        def legal_quad(m1, m2):  # non-overlapping test: parallel or bounding box has (width - 1) * (height - 1) >= 5
            (i1, j1), (i2, j2) = m1
            (a1, b1), (a2, b2) = m2
            return (len({(i1, j1), (i2, j2), (a1, b1), (a2, b2)}) < 4  # adjacent edges in path, ignore
                    or (i1 - i2) * (b1 - b2) == (j1 - j2) * (a1 - a2)  # parallel
                    or (max(a1, a2, i1, i2) - min(a1, a2, i1, i2)) * (max(b1, b2, j1, j2) - min(b1, b2, j1, j2)) >= 5
                    # far
                    )
    
        assert all(i in range(m) and j in range(n) for i, j in path), "move off board"
        assert len({(i, j) for i, j in path}) == len(path), "visited same square twice"
    
        moves = list(zip(path, path[1:]))
        assert all(legal_move(m) for m in moves), "illegal move"
        assert all(legal_quad(m1, m2) for m1 in moves for m2 in moves), "intersecting move pair"
    
        return len(path) >= target
    
    def sol(m=8, n=8, target=35):
        """Find a long (open) tour of knight moves on an m x n chess-board whose edges don't cross."""
    ```
    0 hand-written solutions 
    
* <a name="unsolved_uncrossedknightspath"></a>**UNSOLVED_UncrossedKnightsPath** Uncrossed Knights Path (open problem, unsolved)
    
    Similar to above, but the goal of these problems is to *beat* the nxn_records from
    [http://ukt.alex-black.ru/](http://ukt.alex-black.ru/)
    (accessed 2020-11-29).
    
    A more precise description is in this
    [Wikipedia article](https://en.wikipedia.org/w/index.php?title=Longest_uncrossed_knight%27s_path). (4 instances)
    
    ```python
    def sat(path: List[List[int]], m=10, n=10, target=62):
        def legal_move(m):
            (a, b), (i, j) = m
            return {abs(i - a), abs(j - b)} == {1, 2}
    
        def legal_quad(m1, m2):  # non-overlapping test: parallel or bounding box has (width - 1) * (height - 1) >= 5
            (i1, j1), (i2, j2) = m1
            (a1, b1), (a2, b2) = m2
            return (len({(i1, j1), (i2, j2), (a1, b1), (a2, b2)}) < 4  # adjacent edges in path, ignore
                    or (i1 - i2) * (b1 - b2) == (j1 - j2) * (a1 - a2)  # parallel
                    or (max(a1, a2, i1, i2) - min(a1, a2, i1, i2)) * (max(b1, b2, j1, j2) - min(b1, b2, j1, j2)) >= 5
                    # far
                    )
    
        assert all(i in range(m) and j in range(n) for i, j in path), "move off board"
        assert len({(i, j) for i, j in path}) == len(path), "visited same square twice"
    
        moves = list(zip(path, path[1:]))
        assert all(legal_move(m) for m in moves), "illegal move"
        assert all(legal_quad(m1, m2) for m1 in moves for m2 in moves), "intersecting move pair"
    
        return len(path) >= target
    
    def sol(m=10, n=10, target=62):
        """Find a long (open) tour of knight moves on an m x n chess-board whose edges don't cross."""
    ```
    0 hand-written solutions 
    
## compression

Puzzles relating to de/compression.

* <a name="lzw"></a>**LZW** We have provided a simple version of the *decompression* algorithm of
    [Lempel-Ziv-Welch](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch)
    so the solution is the *compression* algorithm. (5 instances)
    
    ```python
    def sat(seq: List[int], compressed_len=17, text="Hellooooooooooooooooooooo world!"):
        index = [chr(i) for i in range(256)]
        pieces = [""]
        for i in seq:
            pieces.append((pieces[-1] + pieces[-1][0]) if i == len(index) else index[i])
            index.append(pieces[-2] + pieces[-1][0])
        return "".join(pieces) == text and len(seq) <= compressed_len
    
    def sol(compressed_len=17, text="Hellooooooooooooooooooooo world!"):
        """
        Find a (short) compression that decompresses to the given string for the provided implementation of the
        Lempel-Ziv decompression algorithm from https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        # compressed_len is ignored
        index = {chr(i): i for i in range(256)}
        seq = []
        buffer = ""
        for c in text:
            if buffer + c in index:
                buffer += c
                continue
            seq.append(index[buffer])
            index[buffer + c] = len(index) + 1
            buffer = c
    
        if text != "":
            seq.append(index[buffer])
    
        return seq
    ```
    
    </details>
    
* <a name="packingham"></a>**PackingHam** This packing problem a [classic problem](https://en.wikipedia.org/wiki/Sphere_packing#Other_spaces)
    in coding theory. (5 instances)
    
    ```python
    def sat(words: List[str], num=100, bits=100, dist=34):
        assert len(words) == num and all(len(word) == bits and set(word) <= {"0", "1"} for word in words)
        return all(sum([a != b for a, b in zip(words[i], words[j])]) >= dist for i in range(num) for j in range(i))
    
    def sol(num=100, bits=100, dist=34):
        """Pack a certain number of binary strings so that they have a minimum hamming distance between each other."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        import random  # key insight, use randomness!
        r = random.Random(0)
        while True:
            seqs = [r.getrandbits(bits) for _ in range(num)]
            if all(bin(seqs[i] ^ seqs[j]).count("1") >= dist for i in range(num) for j in range(i)):
                return [bin(s)[2:].rjust(bits, '0') for s in seqs]
    ```
    
    </details>
    
## conways_game_of_life

Conway's Game of Life problems (see https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)

* <a name="oscillators"></a>**Oscillators** Oscillators (including some unsolved, open problems)
    
    This problem is *unsolved* for periods 19, 38, and 41.
    
    See
    [discussion](https://en.wikipedia.org/wiki/Oscillator_%28cellular_automaton%29#:~:text=Game%20of%20Life )
    in Wikipedia article on Cellular Automaton Oscillators. (4 instances)
    
    ```python
    def sat(init: List[List[int]], period=3):
        target = {x + y * 1j for x, y in init}  # complex numbers encode live cells
    
        deltas = (1j, -1j, 1, -1, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j)
        live = target
        for t in range(period):
            visible = {z + d for z in live for d in deltas}
            live = {z for z in visible if sum(z + d in live for d in deltas) in ([2, 3] if z in live else [3])}
            if live == target:
                return t + 1 == period
    
    def sol(period=3):
        """
        Find a pattern in Conway's Game of Life https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life that repeats
        with a certain period https://en.wikipedia.org/wiki/Oscillator_%28cellular_automaton%29#:~:text=Game%20of%20Life
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        # # generate random patterns, slow solution
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
    ```
    
    </details>
    
* <a name="reverselifestep"></a>**ReverseLifeStep** Unsolvable for "Garden of Eden" positions, but we only generate solvable examples (5 instances)
    
    ```python
    def sat(position: List[List[int]], target=[[1, 3], [1, 4], [2, 5]]):
        live = {x + y * 1j for x, y in position}  # complex numbers encode live cells
        deltas = (1j, -1j, 1, -1, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j)
        visible = {z + d for z in live for d in deltas}
        next_step = {z for z in visible if sum(z + d in live for d in deltas) in ([2, 3] if z in live else [3])}
        return next_step == {x + y * 1j for x, y in target}
    
    def sol(target=[[1, 3], [1, 4], [2, 5]]):
        """
        Given a target pattern in Conway's Game of Life (see https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life ),
        specified by [x,y] coordinates of live cells, find a position that leads to that pattern on the next step.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        # fixed-temperature MC optimization
        TEMP = 0.05
        import random
        rand = random.Random(0)  # set seed but don't interfere with other random uses
        target = {x + y * 1j for x, y in target}
        deltas = (1j, -1j, 1, -1, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j)
    
        def distance(live):
            visible = {z + d for z in live for d in deltas}
            next_step = {z for z in visible if sum(z + d in live for d in deltas) in ([2, 3] if z in live else [3])}
            return len(next_step.symmetric_difference(target))
    
        for step in range(10 ** 5):
            if step % 10000 == 0:
                pos = target.copy()  # start with the target position
                cur_dist = distance(pos)
    
            if cur_dist == 0:
                return [[int(z.real), int(z.imag)] for z in pos]
            z = rand.choice([z + d for z in pos.union(target) for d in deltas])
            dist = distance(pos.symmetric_difference({z}))
            if rand.random() <= TEMP ** (dist - cur_dist):
                pos.symmetric_difference_update({z})
                cur_dist = dist
        print('Failed', len(target), step)
    ```
    
    </details>
    
* <a name="spaceship"></a>**Spaceship** Spaceship (including *unsolved*, open problems)
    
    Find a [spaceship](https://en.wikipedia.org/wiki/Spaceship_%28cellular_automaton%29) in
    [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
    with a certain period.
    
    This is an *unsolved* problem for periods 33, 34. (4 instances)
    
    ```python
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
    
    def sol(period=4):
        """
        Find a "spaceship" (see https://en.wikipedia.org/wiki/Spaceship_%28cellular_automaton%29 ) in Conway's
        Game of Life see https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life with a certain period
        """
    ```
    0 hand-written solutions 
    
## games


Some two-player game problems and hard game theory problems


* <a name="nim"></a>**Nim** Compute optimal play for the classic two-player game [Nim](https://en.wikipedia.org/wiki/Nim)
    
    Nim has an elegant theory for optimal play based on the xor of the bits in the heaps.
    
    Instead of writing a program that plays the game interactively (since interaction is not allowed), we require
    them to determine winning states or beat a certain opponent. (5 instances)
    
    ```python
    def sat(moves: List[List[int]], initial_state=[5, 9, 3, 11, 18, 25, 1, 2, 4, 1]):
    
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
    
    def sol(initial_state=[5, 9, 3, 11, 18, 25, 1, 2, 4, 1]):
        """
        Beat a bot at Nim, a two-player game involving a number of heaps of objects. Players alternate, in each turn
        removing one or more objects from a single non-empty heap. The player who takes the last object wins.
        - initial_state is list of numbers of objects in each heap
        - moves is a list of your moves: [heap, number of objects to take]
        - you play first
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
    
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
    ```
    
    </details>
    
* <a name="mastermind"></a>**Mastermind** Compute a strategy for winning in [mastermind](https://en.wikipedia.org/wiki/Mastermind_%28board_game%29)
    in a given number of guesses.
    
    Instead of writing a program that plays the game interactively (since interaction is not allowed), we require
    them to provide a provable winning game tree. (3 instances)
    
    ```python
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
    
    def sol(max_moves=10):
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
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
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
    ```
    
    </details>
    
* <a name="tictactoex"></a>**TicTacToeX** Since we don't have interaction, this problem asks for a full tie-guranteeing strategy. (1 instance)
    
    ```python
    def sat(good_boards: List[str]):
        board_bit_reps = {tuple(sum(1 << i for i in range(9) if b[i] == c) for c in "XO") for b in good_boards}
        win = [any(i & w == w for w in [7, 56, 73, 84, 146, 273, 292, 448]) for i in range(512)]
    
        def tie(x, o):  # returns True if X has a forced tie/win assuming it's X's turn to move.
            x |= 1 << [i for i in range(9) if (x | (1 << i), o) in board_bit_reps][0]
            return not win[o] and (win[x] or all((x | o) & (1 << i) or tie(x, o | (1 << i)) for i in range(9)))
    
        return tie(0, 0)
    
    def sol():
        """
        Compute a strategy for X (first player) in tic-tac-toe that guarantees a tie. That is a strategy for X that,
        no matter what the opponent does, X does not lose.
    
        A board is represented as a 9-char string like an X in the middle would be "....X...." and a
        move is an integer 0-8. The answer is a list of "good boards" that X aims for, so no matter what O does there
        is always good board that X can get to with a single move.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
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
    ```
    
    </details>
    
* <a name="tictactoeo"></a>**TicTacToeO** Same as above but for 2nd player (1 instance)
    
    ```python
    def sat(good_boards: List[str]):
        board_bit_reps = {tuple(sum(1 << i for i in range(9) if b[i] == c) for c in "XO") for b in good_boards}
        win = [any(i & w == w for w in [7, 56, 73, 84, 146, 273, 292, 448]) for i in range(512)]
    
        def tie(x, o):  # returns True if O has a forced tie/win. It's O's turn to move.
            if o | x != 511:  # complete board
                o |= 1 << [i for i in range(9) if (x, o | (1 << i)) in board_bit_reps][0]
            return not win[x] and (win[o] or all((x | o) & (1 << i) or tie(x | (1 << i), o) for i in range(9)))
    
        return all(tie(1 << i, 0) for i in range(9))
    
    def sol():
        """
        Compute a strategy for O (second player) in tic-tac-toe that guarantees a tie. That is a strategy for O that,
        no matter what the opponent does, O does not lose.
    
        A board is represented as a 9-char string like an X in the middle would be "....X...." and a
        move is an integer 0-8. The answer is a list of "good boards" that O aims for, so no matter what X does there
        is always good board that O can get to with a single move.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
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
    ```
    
    </details>
    
* <a name="rockpaperscissors"></a>**RockPaperScissors**  (1 instance)
    
    ```python
    def sat(probs: List[float]):
        assert len(probs) == 3 and abs(sum(probs) - 1) < 1e-6
        return max(probs[(i + 2) % 3] - probs[(i + 1) % 3] for i in range(3)) < 1e-6
    
    def sol():
        """Find optimal probabilities for playing Rock-Paper-Scissors zero-sum game, with best worst-case guarantee"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [1 / 3] * 3
    ```
    
    </details>
    
* <a name="nash"></a>**Nash** Computing a [Nash equilibrium](https://en.wikipedia.org/wiki/Nash_equilibrium) for a given
    [bimatrix game](https://en.wikipedia.org/wiki/Bimatrix_game) is known to be
    PPAD-hard in general. However, the challenge is be much easier for an approximate
    [eps-equilibrium](https://en.wikipedia.org/wiki/Epsilon-equilibrium) and of course for small games. (5 instances)
    
    ```python
    def sat(strategies: List[List[float]], A=[[1.0, -1.0], [-1.3, 0.8]], B=[[-0.9, 1.1], [0.7, -0.8]], eps=0.01):
        m, n = len(A), len(A[0])
        p, q = strategies
        assert len(B) == m and all(len(row) == n for row in A + B), "inputs are a bimatrix game"
        assert len(p) == m and len(q) == n, "solution is a pair of strategies"
        assert sum(p) == sum(q) == 1.0 and min(p + q) >= 0.0, "strategies must be non-negative and sum to 1"
        v = sum(A[i][j] * p[i] * q[j] for i in range(m) for j in range(n))
        w = sum(B[i][j] * p[i] * q[j] for i in range(m) for j in range(n))
        return (all(sum(A[i][j] * q[j] for j in range(n)) <= v + eps for i in range(m)) and
                all(sum(B[i][j] * p[i] for i in range(m)) <= w + eps for j in range(n)))
    
    def sol(A=[[1.0, -1.0], [-1.3, 0.8]], B=[[-0.9, 1.1], [0.7, -0.8]], eps=0.01):
        """
        Find an eps-Nash-equilibrium for a given two-player game with payoffs described by matrices A, B.
        For example, for the classic Prisoner dilemma:
           A=[[-1., -3.], [0., -2.]], B=[[-1., 0.], [-3., -2.]], and strategies = [[0, 1], [0, 1]]
    
        eps is the error tolerance
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
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
    ```
    
    </details>
    
* <a name="zerosum"></a>**ZeroSum** Compute minimax optimal strategies for a given
    [zero-sum game](https://en.wikipedia.org/wiki/Zero-sum_game). This problem is known to be equivalent to
    Linear Programming. Note that the provided instances are all quite easy---harder solutions could readily
    be made by decreasing the accuracy tolerance `eps` at which point the solution we provided would fail and
    more efficient algorithms would be needed. (5 instances)
    
    ```python
    def sat(strategies: List[List[float]], A=[[0.0, -0.5, 1.0], [0.75, 0.0, -1.0], [-1.0, 0.4, 0.0]], eps=0.01):
        m, n = len(A), len(A[0])
        p, q = strategies
        assert all(len(row) == n for row in A), "inputs are a matrix"
        assert len(p) == m and len(q) == n, "solution is a pair of strategies"
        assert sum(p) == sum(q) == 1.0 and min(p + q) >= 0.0, "strategies must be non-negative and sum to 1"
        v = sum(A[i][j] * p[i] * q[j] for i in range(m) for j in range(n))
        return (all(sum(A[i][j] * q[j] for j in range(n)) <= v + eps for i in range(m)) and
                all(sum(A[i][j] * p[i] for i in range(m)) >= v - eps for j in range(n)))
    
    def sol(A=[[0.0, -0.5, 1.0], [0.75, 0.0, -1.0], [-1.0, 0.4, 0.0]], eps=0.01):
        """
        Compute minimax optimal strategies for a given zero-sum game up to error tolerance eps.
        For example, rock paper scissors has
        A = [[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]] and strategies = [[0.33, 0.33, 0.34]] * 2
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
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
    ```
    
    </details>
    
## graphs

Problems related to graphs such as Conway's 99 problem, finding
[cliques](https://en.wikipedia.org/wiki/Clique_(graph_theory)) of various sizes, shortest path (Dijkstra) 

* <a name="conway99"></a>**Conway99** Conway's 99-graph problem (*unsolved*, open problem)
    
    Conway's 99-graph problem is an unsolved problem in graph theory.
    In Conway's terminology, from [Five $1,000 Problems (Update 2017)](https://oeis.org/A248380/a248380.pdf)
    "Is there a graph with 99 vertices in which every edge (i.e. pair of joined vertices) belongs to a unique
    triangle and every nonedge (pair of unjoined vertices) to a unique quadrilateral?"
    
    See also this [Wikipedia article](https://en.wikipedia.org/w/index.php?title=Conway%27s_99-graph_problem). (1 instance)
    
    ```python
    def sat(edges: List[List[int]]):
        # first compute neighbors sets, N:
        N = {i: {j for j in range(99) if j != i and ([i, j] in edges or [j, i] in edges)} for i in range(99)}
        return all(len(N[i].intersection(N[j])) == (1 if j in N[i] else 2) for i in range(99) for j in range(i))
    
    def sol():
        """
        Find an undirected graph with 99 vertices, in which each two adjacent vertices have exactly one common
        neighbor, and in which each two non-adjacent vertices have exactly two common neighbors.
        """
    ```
    0 hand-written solutions 
    
* <a name="anyedge"></a>**AnyEdge** Trivial [graph](https://en.wikipedia.org/w/index.php?title=Graph_(discrete_mathematics)) problem. (5 instances)
    
    ```python
    def sat(e: List[int], edges=[[0, 217], [40, 11], [17, 29], [11, 12], [31, 51]]):
        return e in edges
    
    def sol(edges=[[0, 217], [40, 11], [17, 29], [11, 12], [31, 51]]):
        """Find any edge in edges."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return edges[0]
    ```
    
    </details>
    
* <a name="anytriangle"></a>**AnyTriangle** Easy [graph](https://en.wikipedia.org/w/index.php?title=Graph_(discrete_mathematics)) problem,
    see [triangle](https://en.wikipedia.org/w/index.php?title=Triangle_graph) (5 instances)
    
    ```python
    def sat(tri: List[int], edges=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):
        a, b, c = tri
        return [a, b] in edges and [b, c] in edges and [c, a] in edges and a != b != c != a
    
    def sol(edges=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):
        """Find any triangle in the given directed graph."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        from collections import defaultdict
        outs = defaultdict(set)
        ins = defaultdict(set)
        for i, j in edges:
            if j != i:
                outs[i].add(j)
                ins[j].add(i)
        for i in outs:
            for j in outs[i]:
                try:
                    if j in outs:
                        k = min(outs[j].intersection(ins[i]))
                        return [i, j, k]
                except ValueError:
                    pass
    ```
    
    </details>
    
* <a name="plantedclique"></a>**PlantedClique** Find a [planted clique](https://en.wikipedia.org/w/index.php?title=Planted_clique) of a given size
    in an undirected graph. Finding a polynomial-time algorithm for this problem has been *unsolved* for
    some time. (5 instances)
    
    ```python
    def sat(nodes: List[int], size=3, edges=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):
        assert len(nodes) == len(set(nodes)) >= size
        edge_set = {(a, b) for (a, b) in edges}
        for a in nodes:
            for b in nodes:
                assert a == b or (a, b) in edge_set or (b, a) in edge_set
    
        return True
    
    def sol(size=3, edges=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):
        """Find a clique of the given size in the given undirected graph. It is guaranteed that such a clique exists."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        # brute force (finds list in increasing order), but with a tiny bit of speedup
        if size == 0:
            return []
        from collections import defaultdict
        neighbors = defaultdict(set)
        n = max(max(e) for e in edges)
        for (a, b) in edges:
            if a != b:
                neighbors[a].add(b)
                neighbors[b].add(a)
        pools = [list(range(n + 1))]
        indices = [-1]
        while pools:
            indices[-1] += 1
            if indices[-1] >= len(pools[-1]) - size + len(pools):  # since list is increasing order
                indices.pop()
                pools.pop()
                continue
            if len(pools) == size:
                return [pool[i] for pool, i in zip(pools, indices)]
            a = (pools[-1])[indices[-1]]
            pools.append([i for i in pools[-1] if i > a and i in neighbors[a]])
            indices.append(-1)
        assert False, f"No clique of size {size}"
    ```
    
    </details>
    
* <a name="shortestpath"></a>**ShortestPath** Shortest Path, see (Dijkstra's algorithm)[https://en.wikipedia.org/w/index.php?title=Dijkstra%27s_algorithm] (5 instances)
    
    ```python
    def sat(path: List[int], weights=[{1: 20, 2: 1}, {2: 2, 3: 5}, {1: 10}], bound=11):
        return path[0] == 0 and path[-1] == 1 and sum(weights[a][b] for a, b in zip(path, path[1:])) <= bound
    
    def sol(weights=[{1: 20, 2: 1}, {2: 2, 3: 5}, {1: 10}], bound=11):
        """
        Find a path from node 0 to node 1, of length at most bound, in the given digraph.
        weights[a][b] is weight on edge [a,b] for (int) nodes a, b
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        # Dijkstra's algorithm (bound is ignored)
        u, v = 0, 1  # go from 0 to 1
        import heapq
        queue = [(0, u, u)]  # distance, node, trail
    
        trails = {}
    
        while queue:
            dist, i, j = heapq.heappop(queue)
            if i in trails:
                continue
            trails[i] = j
            if i == v:
                break
            for j in weights[i]:
                if j not in trails:
                    heapq.heappush(queue, (dist + weights[i][j], j, i))
        if v in trails:
            rev_path = [v]
            while rev_path[-1] != u:
                rev_path.append(trails[rev_path[-1]])
            return rev_path[::-1]
    ```
    
    </details>
    
* <a name="unweightedshortestpath"></a>**UnweightedShortestPath** Unweighted Shortest Path
    
    See (Dijkstra's algorithm)[https://en.wikipedia.org/w/index.php?title=Dijkstra%27s_algorithm] (5 instances)
    
    ```python
    def sat(path: List[int], edges=[[0, 11], [0, 7], [7, 5], [0, 22], [11, 22], [11, 33], [22, 33]], u=0, v=33, bound=3):
        assert path[0] == u and path[-1] == v and all([i, j] in edges for i, j in zip(path, path[1:]))
        return len(path) <= bound
    
    def sol(edges=[[0, 11], [0, 7], [7, 5], [0, 22], [11, 22], [11, 33], [22, 33]], u=0, v=33, bound=3):
        """Find a path from node u to node v, of a bounded length, in the given digraph on vertices 0, 1,..., n."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        # Dijkstra's algorithm
        import heapq
        from collections import defaultdict
        queue = [(0, u, u)]  # distance, node, trail
    
        trails = {}
        neighbors = defaultdict(set)
        for (i, j) in edges:
            neighbors[i].add(j)
    
        while queue:
            dist, i, j = heapq.heappop(queue)
            if i in trails:
                continue
            trails[i] = j
            if i == v:
                break
            for j in neighbors[i]:
                if j not in trails:
                    heapq.heappush(queue, (dist + 1, j, i))
        if v in trails:
            rev_path = [v]
            while rev_path[-1] != u:
                rev_path.append(trails[rev_path[-1]])
            return rev_path[::-1]
    ```
    
    </details>
    
* <a name="anypath"></a>**AnyPath** Any Path (5 instances)
    
    ```python
    def sat(path: List[int], edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [1, 2]]):
        for i in range(len(path) - 1):
            assert [path[i], path[i + 1]] in edges
        assert path[0] == 0
        assert path[-1] == max(max(edge) for edge in edges)
        return True
    
    def sol(edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [1, 2]]):
        """ Find any path from node 0 to node n in a given digraph on vertices 0, 1,..., n."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        n = max(max(edge) for edge in edges)
        paths = {0: [0]}
        for _ in range(n + 1):
            for i, j in edges:
                if i in paths and j not in paths:
                    paths[j] = paths[i] + [j]
        return paths.get(n)
    ```
    
    </details>
    
* <a name="evenpath"></a>**EvenPath**  (5 instances)
    
    ```python
    def sat(path: List[int], edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [1, 2]]):
        assert path[0] == 0 and path[-1] == max(max(e) for e in edges)
        assert all([[a, b] in edges for a, b in zip(path, path[1:])])
        return len(path) % 2 == 0
    
    def sol(edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [1, 2]]):
        """Find a path with an even number of nodes from nodes 0 to n in the given digraph on vertices 0, 1,..., n."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        even_paths = {}
        odd_paths = {0: [0]}
        n = max(max(e) for e in edges)
        for _ in range(n + 1):
            for i, j in edges:
                if i in even_paths and j not in odd_paths:
                    odd_paths[j] = even_paths[i] + [j]
                if i in odd_paths and j not in even_paths:
                    even_paths[j] = odd_paths[i] + [j]
        return even_paths.get(n)
    ```
    
    </details>
    
* <a name="oddpath"></a>**OddPath** To make it even more different than EvenPath, we changed to go from node 0 to node *1*. (5 instances)
    
    ```python
    def sat(p: List[int], edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [6, 1]]):
        return p[0] == 0 and p[-1] == 1 == len(p) % 2 and all([[a, b] in edges for a, b in zip(p, p[1:])])
    
    def sol(edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [6, 1]]):
        """Find a path with an even number of nodes from nodes 0 to 1 in the given digraph on vertices 0, 1,..., n."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        even_paths = {}
        odd_paths = {0: [0]}
        n = 1
        for _ in range(max(max(e) for e in edges) + 1):
            for i, j in edges:
                if i in even_paths and j not in odd_paths:
                    odd_paths[j] = even_paths[i] + [j]
                if i in odd_paths and j not in even_paths:
                    even_paths[j] = odd_paths[i] + [j]
        return odd_paths.get(n)
    ```
    
    </details>
    
* <a name="zarankiewicz"></a>**Zarankiewicz** [Zarankiewicz problem](https://en.wikipedia.org/wiki/Zarankiewicz_problem) (3 instances)
    
    ```python
    def sat(edges: List[List[int]], z=20, n=5, t=3):
        from itertools import combinations
        edges = {(a, b) for a, b in edges if a in range(n) and b in range(n)}  # convert to a set for efficiency
        assert len(edges) >= z
    
        return all(
            any((a, b) not in edges for a in left for b in right)
            for left in combinations(range(n), t)
            for right in combinations(range(n), t)
        )
    
    def sol(z=20, n=5, t=3):
        """Find a bipartite graph with n vertices on each side, z edges, and no K_3,3 subgraph."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        from itertools import combinations
        all_edges = [(a, b) for a in range(n) for b in range(n)]
        for edges in combinations(all_edges, z):
            edge_set = set(edges)
            if all(any((a, b) not in edge_set for a in left for b in right)
                   for left in combinations(range(n), t)
                   for right in combinations(range(n), t)):
                return [[a, b] for a, b in edges]
    ```
    
    </details>
    
* <a name="graphisomorphism"></a>**GraphIsomorphism** The classic [Graph Isomorphism](https://en.wikipedia.org/wiki/Graph_isomorphism) problem.
    It is unknown whether or not there exists a polynomial-time algorithm
    for this problem, though an unpublished quasi-polynomial-time algorithm has been announced by Babai.
    
    The classic version is a decision problem: given two graphs, determine whether or not they are isomorphic.
    However, it is polynomial-time equivalent to the one below through a standard reduction. In particular, if you
    could solve the search problem below (finding the actual bijection), then you can decide isomorphism because the
    search solver would simply fail on non-isomorphic graphs. Conversely, if you could solve the decision problem,
    then you can find a bijection as follows: if the decider determines that the graphs are isomorphic, for each node
    in the first graph, find a corresponding node in the second graph as follows. Add N self-edges from the node to
    itself where N is the maximum degree in the graph + 1, and do that for each candidate node in the second graph.
    For each of these additions, test isomorphism. If the graphs are isomorphic then there must be a bijection that maps
    the first node to the second. Repeat this for each node until you have found a bijection. (If self-loops are not
    allowed, one can do this by adding N additional nodes for each test. (5 instances)
    
    ```python
    def sat(bi: List[int], g1=[[0, 1], [1, 2], [2, 3], [3, 4], [2, 5]], g2=[[0, 4], [1, 5], [4, 1], [1, 2], [2, 3]]):
        return len(bi) == len(set(bi)) and {(i, j) for i, j in g1} == {(bi[i], bi[j]) for i, j in g2}
    
    def sol(g1=[[0, 1], [1, 2], [2, 3], [3, 4], [2, 5]], g2=[[0, 4], [1, 5], [4, 1], [1, 2], [2, 3]]):
        """
        You are given two graphs which are permutations of one another and the goal is to find the permutation.
        Each graph is specified by a list of edges where each edge is a pair of integer vertex numbers.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        # exponentially slow
        from itertools import permutations
        n = max(i for g in [g1, g2] for e in g for i in e) + 1
        g1_set = {(i, j) for i, j in g1}
        for pi in permutations(range(n)):
            if all((pi[i], pi[j]) in g1_set for i, j in g2):
                return list(pi)
        assert False, f"Graphs are not isomorphic {g1}, {g2}"
    ```
    
    </details>
    
* <a name="shortintegerpath"></a>**ShortIntegerPath** This is a more interesting version of Study_20 with an additional length constraint. One can think of the graph
    defined by the integer pairs. (1 instance)
    
    ```python
    def sat(li: List[int]):
        return all(j in {i - 1, i + 1, 3 * i} for i, j in zip([0] + li, li + [128])) and len(li) == 9
    
    def sol():
        """
        Find a list of nine integers, starting with 0 and ending with 128, such that each integer either differs from
        the previous one by one or is thrice the previous one.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [1, 3, 4, 12, 13, 14, 42, 126, 127]
    ```
    
    </details>
    
## ICPC


Problems inspired by the [International Collegiate Programming Contest](https://icpc.global) (ICPC).


* <a name="bipermutations"></a>**BiPermutations** Inspired by
    [ICPC 2019 Problem A: Azulejos](https://icpc.global/worldfinals/problems/2019%20ACM-ICPC%20World%20Finals/icpc2019.pdf)
    which is 2,287 characters. (5 instances)
    
    ```python
    def sat(perms: List[List[int]], prices0=[7, 7, 9, 5, 3, 7, 1, 2], prices1=[5, 5, 5, 4, 2, 5, 1, 1], heights0=[2, 4, 9, 3, 8, 5, 5, 4], heights1=[1, 3, 8, 1, 5, 4, 4, 2]):
        n = len(prices0)
        perm0, perm1 = perms
        assert sorted(perm0) == sorted(perm1) == list(range(n)), "Solution must be two permutations"
        for i in range(n - 1):
            assert prices0[perm0[i]] <= prices0[perm0[i + 1]], "Permuted prices must be nondecreasing (row 0)"
            assert prices1[perm1[i]] <= prices1[perm1[i + 1]], "Permuted prices must be nondecreasing (row 1)"
        return all(heights0[i] > heights1[j] for i, j in zip(perm0, perm1))
    
    def sol(prices0=[7, 7, 9, 5, 3, 7, 1, 2], prices1=[5, 5, 5, 4, 2, 5, 1, 1], heights0=[2, 4, 9, 3, 8, 5, 5, 4], heights1=[1, 3, 8, 1, 5, 4, 4, 2]):
        """
        There are two rows of objects. Given the length-n integer arrays of prices and heights of objects in each
        row, find a permutation of both rows so that the permuted prices are non-decreasing in each row and
        so that the first row is taller than the second row.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        n = len(prices0)
        prices = [prices0, prices1]
        orders = [sorted(range(n), key=lambda i: (prices0[i], heights0[i])),
                  sorted(range(n), key=lambda i: (prices1[i], -heights1[i]))]
        jumps = [1, 1]  # next price increase locations
        for i in range(n):
            for r, (p, o) in enumerate(zip(prices, orders)):
                while jumps[r] < n and p[o[jumps[r]]] == p[o[i]]:
                    jumps[r] += 1
    
            to_fix = orders[jumps[0] < jumps[1]]
            j = i
            while heights0[orders[0][i]] <= heights1[orders[1][i]]:
                j += 1
                to_fix[i], to_fix[j] = to_fix[j], to_fix[i]
    
        return orders
    ```
    
    </details>
    
* <a name="optimalbridges"></a>**OptimalBridges** Inspired by
    [ICPC 2019 Problem B: Bridges](https://icpc.global/worldfinals/problems/2019%20ACM-ICPC%20World%20Finals/icpc2019.pdf)
    which is 3,003 characters. (5 instances)
    
    ```python
    def sat(indices: List[int], H=60, alpha=18, beta=2, xs=[0, 10, 20, 30, 50, 80, 100, 120, 160, 190, 200], ys=[0, 30, 10, 30, 50, 40, 10, 20, 20, 55, 10], thresh=26020):
        assert sorted({0, len(xs) - 1, *indices}) == indices, f"Ans. should be sorted list [0, ..., {len(xs) - 1}]"
        cost = alpha * (H - ys[0])
        for i, j in zip(indices, indices[1:]):
            a, b, r = xs[i], xs[j], (xs[j] - xs[i]) / 2
            assert max(ys[i], ys[j]) + r <= H, "Bridge too tall"
            assert all(ys[k] <= H - r + ((b - xs[k]) * (xs[k] - a)) ** 0.5 for k in range(i + 1, j)), \
                "Bridge too short"
            cost += alpha * (H - ys[j]) + beta * (b - a) ** 2
        return cost <= thresh
    
    def sol(H=60, alpha=18, beta=2, xs=[0, 10, 20, 30, 50, 80, 100, 120, 160, 190, 200], ys=[0, 30, 10, 30, 50, 40, 10, 20, 20, 55, 10], thresh=26020):
        """
        You are to choose locations for bridge bases from among a given set of mountain peaks located at
        `xs, ys`, where `xs` and `ys` are lists of n integers of the same length. Your answer should be a sorted
        list of indices starting at 0 and ending at n-1. The goal is to minimize building costs such that the bridges
        are feasible. The bridges are all semicircles placed on top of the pillars. The feasibility constraints are that:
        * The bridges may not extend above a given height `H`. Mathematically, if the distance between the two xs
        of adjacent pillars is d, then the semicircle will have radius `d/2` and therefore the heights of the
        selected mountain peaks must both be at most `H - d/2`.
        *  The bridges must clear all the mountain peaks, which means that the semicircle must lie above the tops of the
        peak. See the code for how this is determined mathematically.
        * The total cost of all the bridges must be at most `thresh`, where the cost is parameter alpha * (the sum of
        all pillar heights) + beta * (the sum of the squared diameters)
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        # thresh is ignored
        n = len(xs)
        cost = [-1] * n
        prior = [n] * n
        cost[0] = beta * (H - ys[0])
        for i in range(n):
            if cost[i] == -1:
                continue
            min_d = 0
            max_d = 2 * (H - ys[i])
            for j in range(i + 1, n):
                d = xs[j] - xs[i]
                h = H - ys[j]
                if d > max_d:
                    break
                if 2 * h <= d:
                    min_d = max(min_d, 2 * d + 2 * h - int((8 * d * h) ** 0.5))
                max_d = min(max_d, 2 * d + 2 * h + int((8 * d * h) ** 0.5))
                if min_d > max_d:
                    break
                if min_d <= d <= max_d:
                    new_cost = cost[i] + alpha * h + beta * d * d
                    if cost[j] == -1 or cost[j] > new_cost:
                        cost[j] = new_cost
                        prior[j] = i
        rev_ans = [n - 1]
        while rev_ans[-1] != 0:
            rev_ans.append(prior[rev_ans[-1]])
        return rev_ans[::-1]
    ```
    
    </details>
    
* <a name="checkersposition"></a>**CheckersPosition** Inspired by
    [ICPC 2019 Problem C: Checks Post Facto](https://icpc.global/worldfinals/problems/2019%20ACM-ICPC%20World%20Finals/icpc2019.pdf)
    
    Nobody solved this problem during the competition -- it is pretty difficult! (5 instances)
    
    ```python
    def sat(position: List[List[int]], transcript=[[[3, 3], [5, 5], [3, 7]], [[5, 3], [6, 4]]]):
        board = {(x, y): 0 for x in range(8) for y in range(8) if (x + y) % 2 == 0}  # empty board, 0 = empty
        for x, y, p in position:
            assert -2 <= p <= 2 and board[x, y] == 0  # -1, 1 is regular piece, -2, 2 is king
            board[x, y] = p
    
        def has_a_jump(x, y):
            p = board[x, y]  # piece to move
            deltas = [(dx, dy) for dx in [-1, 1] for dy in [-1, 1] if dy != -p]  # don't check backwards for non-kings
            return any(board.get((x + 2 * dx, y + 2 * dy)) == 0 and board[x + dx, y + dy] * p < 0 for dx, dy in deltas)
    
        sign = 1  # player 1 moves first
        for move in transcript:
            start, end = tuple(move[0]), tuple(move[-1])
            p = board[start]  # piece to move
            assert p * sign > 0, "Moving square must be non-empty and players must be alternate signs"
            assert all(board[x, y] == 0 for x, y in move if [x, y] != move[0]), "Moved to an occupied square"
    
            for (x1, y1), (x2, y2) in zip(move, move[1:]):
                assert abs(p) != 1 or (y2 - y1) * p > 0, "Non-kings can only move forward (in direction of sign)"
                if abs(x2 - x1) == 1:  # non-jump
                    assert not any(has_a_jump(*a) for a in board if board[a] * p > 0), "Must make a jump if possible"
                    break
                mid = ((x1 + x2) // 2, (y1 + y2) // 2)
                assert board[mid] * p < 0, "Can only jump over piece of opposite sign"
                board[mid] = 0
            board[start], board[end] = 0, p
            assert abs(x2 - x1) == 1 or not has_a_jump(*end)
            if abs(p) == 1 and any(y in {0, 7} for x, y in move[1:]):
                board[end] *= 2  # king me at the end of turn after any jumps are done!
            sign *= -1
    
        return True
    
    def sol(transcript=[[[3, 3], [5, 5], [3, 7]], [[5, 3], [6, 4]]]):
        """
        You are given a partial transcript a checkers game. Find an initial position such that the transcript
        would be a legal set of moves. The board positions are [x, y] pairs with 0 <= x, y < 8 and x + y even.
        There are two players which we call -1 and 1 for convenience, and player 1 must move first in transcript.
        The initial position is represented as a list [x, y, piece] where piece means:
        * 0 is empty square
        * 1 or -1 is piece that moves only in the y = 1 or y = -1 dir, respectively
        * 2 or -2 is king for player 1 or player 2 respectively
    
        Additional rules:
        * You must jump if you can, and you must continue jumping until one can't any longer.
        * You cannot start the position with any non-kings on your last rank.
        * Promotion happens after the turn ends
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        START_PLAYER = 1  # assumed
    
        class InitOpts:
            def __init__(self, x, y):
                self.x, self.y = x, y
                self.opts = {-2, -1, 0, 1, 2}
                if y == 0:
                    self.opts.remove(-1)
                if y == 7:
                    self.opts.remove(1)
                self.promoted = 2 ** 63  # on which step was it promoted t >= 0
                self.jumped = 2 ** 63  # on which step was it jumped t >= 0
    
        # def board2str(board):  # for debugging
        #     mapping = ".bBWw"
        #     ans = ""
        #     for y in range(7, -1, -1):
        #         ans += "".join(" " if (x+y)%2 else mapping[board[x,y]] for x in range(8)) + "\n"
        #     return ans
    
        init_opts = {(x, y): InitOpts(x, y) for x in range(8) for y in range(8) if (x + y) % 2 == 0}
        # board = {(x, y): (1 if y < 3 else -1 if y > 4 else 0) for x in range(8) for y in range(8) if
        #          (x + y) % 2 == 0}  # new board
    
        transcript = [[tuple(a) for a in move] for move in transcript]
    
        permuted_opts = init_opts.copy()
        sign = START_PLAYER
        for t, move in enumerate(transcript):
            start, end = tuple(move[0]), tuple(move[-1])
            p = permuted_opts[start]  # opts to move
            assert p.jumped >= t
            p.opts -= {-sign, -2 * sign, 0}
            if any((y2 - y1) * sign < 0 for (x1, y1), (x2, y2) in zip(move, move[1:])):  # backward move!
                if p.promoted >= t:
                    p.opts -= {sign}  # must be a king!
    
            for a, b in zip(move, move[1:]):
                if permuted_opts[b].jumped >= t:
                    permuted_opts[b].opts -= {-2, -1, 1, 2}  # must be empty
                assert permuted_opts[a].jumped >= t
                permuted_opts[a], permuted_opts[b] = permuted_opts[b], permuted_opts[a]
                # board[a], board[b] = board[b], board[a]
                (x1, y1), (x2, y2) = a, b
                if abs(x2 - x1) == 2:  # jump
                    mid = ((x1 + x2) // 2, (y1 + y2) // 2)
                    assert permuted_opts[mid].jumped >= t
                    permuted_opts[mid].opts -= {0, sign, 2 * sign}  # Can only jump over piece of opposite sign
                    permuted_opts[mid].jumped = t
                    # board[mid] = 0
    
            if any(y in {0, 7} for x, y in move[1:]):
                if p.promoted > t:
                    p.promoted = t
                # if abs(board[x2, y2]) == 1:
                #     board[x2, y2] *= 2
    
            sign *= -1
    
        for y in range(7, -1, -1):
            for x in range(8):
                if (x, y) in init_opts:
                    s = init_opts[x, y].opts
                    if {1, 2} <= s:
                        s.remove(2)
                    if {-1, -2} <= s:
                        s.remove(-2)
    
        def helper():  # returns True if success and store everything, otherwise None
            my_opts = init_opts.copy()
            sign = START_PLAYER  # player 1 always starts
    
            for t, move in enumerate(transcript):
                if abs(move[0][0] - move[1][0]) == 1:  # not a jump
                    check_no_jumps = [a for a, p in my_opts.items() if p.jumped >= t and p.opts <= {sign, 2 * sign}]
                else:
                    for a, b in zip(move, move[1:]):
                        my_opts[a], my_opts[b] = my_opts[b], my_opts[a]
                    check_no_jumps = [b]
    
                for x, y in check_no_jumps:
                    p = my_opts[x, y]
                    [o] = p.opts
                    assert o * sign > 0
                    dys = [o] if (abs(o) == 1 and p.promoted >= t) else [-1, 1]  # only check forward jumps
                    for dx in [-1, 1]:
                        for dy in dys:
                            target_o = my_opts.get((x + 2 * dx, y + 2 * dy))
                            if target_o is not None and (0 in target_o.opts or target_o.jumped < t):
                                mid_o = my_opts[x + dx, y + dy]
                                if mid_o.jumped > t and mid_o.opts <= {-sign, -2 * sign}:  # ok if jumped at t
                                    if target_o.jumped < t or target_o.opts == {0}:
                                        return False
                                    old_opts = target_o.opts
                                    for v in target_o.opts:
                                        if v != 0:
                                            target_o.opts = {v}
                                            h = helper()
                                            if h:
                                                return True
                                    target_o.opts = old_opts
                                    return False
    
                if abs(move[0][0] - move[1][0]) == 1:  # not a jump
                    a, b = move[0], move[1]
                    my_opts[a], my_opts[b] = my_opts[b], my_opts[a]
    
                sign *= -1
            return True
    
        res = helper()
        assert res
    
        def get_opt(opts):
            if 0 in opts.opts:
                return 0
            assert len(opts.opts) == 1
            return list(opts.opts)[0]
    
        return [[x, y, get_opt(opts)] for (x, y), opts in init_opts.items()]
    ```
    
    </details>
    
* <a name="matchingmarkers"></a>**MatchingMarkers** Inspired by
    [ICPC 2019 Problem D: Circular DNA](https://icpc.global/worldfinals/problems/2019%20ACM-ICPC%20World%20Finals/icpc2019.pdf)
    
    This is trivial in quadratic time, but the challenge is to solve it quickly (i.e., linear time). (5 instances)
    
    ```python
    def sat(cut_position: int, ring="yRrsmOkLCHSDJywpVDEDsjgCwSUmtvHMefxxPFdmBIpM", lower=5):
        line = ring[cut_position:] + ring[:cut_position]
        matches = {c: 0 for c in line.lower()}
        for c in line:
            if c.islower():
                matches[c] -= (1 if matches[c] > 0 else len(line))
            else:
                matches[c.lower()] += 1
        return sum(i == 0 for i in matches.values()) >= lower
    
    def sol(ring="yRrsmOkLCHSDJywpVDEDsjgCwSUmtvHMefxxPFdmBIpM", lower=5):
        """
        The input is a string of start and end markers "aaBAcGeg" where upper-case characters indicate start markers
        and lower-case characters indicate ending markers. The string indicates a ring (joined at the ends) and the goal is
        to find a location to split the ring so that there are a maximal number of matched start/end chars where a character
        (like "a"/"A") is matched if starting at the split and going around the ring, the start-end pairs form a valid
        nesting like nested parentheses. Can you solve it in linear time?
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        cumulatives = {c: [(0, 0)] for c in ring.lower()}
        n = len(ring)
        for i, c in enumerate(ring):
            v = cumulatives[c.lower()]
            v.append((i, v[-1][1] + (-1 if c.islower() else 1)))
    
        scores = [0]*n
        cumulatives = {c: v for c, v in cumulatives.items() if v[-1][1]==0}
        for c, v in cumulatives.items():
            if v[-1][1] != 0: # ignore things with unequal numbers of opens and closes
                continue
            m = min(t for i, t in v)
            for (i, t), (i2, t2) in zip(v, v[1:] + [(n, 0)]):
                if t == m:
                    for j in range(i+1, i2+1):
                        scores[j % n] += 1
        b = max(scores)
        for i in range(n):
            if scores[i] == b:
                return i
    ```
    
    </details>
    
## IMO

Problems inspired by the
[International Mathematical Olympiad](https://en.wikipedia.org/wiki/International_Mathematical_Olympiad)
[problems](https://www.imo-official.org/problems.aspx)

* <a name="exponentialcoinmoves"></a>**ExponentialCoinMoves** This problem has *long* answers, not that the code to solve it is long but that what the solution outputs is long.
    
    The version below uses only 5 boxes (unlike the IMO problem with 6 boxes since 2010^2010^2010 is too big
    for computers) but the solution is quite similar to the solution to the IMO problem. Because the solution
    requires exponential many moves, our representation allows combining multiple Type-1 (advance) operations
    into a single step.
    
    Inspired by [IMO 2010 Problem 5](https://www.imo-official.org/problems.aspx) (5 instances)
    
    ```python
    def sat(states: List[List[int]], n=16385):
        assert states[0] == [1] * 5 and all(len(li) == 5 for li in states) and all(i >= 0 for li in states for i in li)
        for prev, cur in zip(states, states[1:]):
            for i in range(5):
                if cur[i] != prev[i]:
                    break
            assert cur[i] < prev[i]
            assert (
                    cur[i + 1] - prev[i + 1] == 2 * (prev[i] - cur[i]) and cur[i + 2:] == prev[i + 2:]  # k decrements
                    or
                    cur[i:i + 3] == [prev[i] - 1, prev[i + 2], prev[i + 1]] and cur[i + 3:] == prev[i + 3:]  # swap
            )
    
        return states[-1][-1] == 2 ** n
    
    def sol(n=16385):
        """
        There are five boxes each having one coin initially. Two types of moves are allowed:
        * (advance) remove `k > 0` coins from box `i` and add `2k` coins to box `i + 1`
        * (swap) remove a coin from box `i` and swap the contents of boxes `i+1` and `i+2`
        Given `0 <= n <= 16385`, find a sequence of states that result in 2^n coins in the last box.
        Note that `n` can be as large as 16385 yielding 2^16385 coins (a number with 4,933 digits) in the last
        box. Encode each state as a list of the numbers of coins in the five boxes.
    
        Sample Input:
        `n = 2`
    
        Sample Output:
        `[[1, 1, 1, 1, 1], [0, 3, 1, 1, 1], [0, 1, 5, 1, 1], [0, 1, 4, 1, 1], [0, 0, 1, 4, 1], [0, 0, 0, 1, 4]]`
    
        The last box now has 2^2 coins. This is a sequence of two advances followed by three swaps.
    
        states is encoded by lists of 5 coin counts
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        assert n >= 1
        ans = [[1] * 5, [0, 3, 1, 1, 1], [0, 2, 3, 1, 1], [0, 2, 2, 3, 1], [0, 2, 2, 0, 7], [0, 2, 1, 7, 0],
               [0, 2, 1, 0, 14], [0, 2, 0, 14, 0], [0, 1, 14, 0, 0]]
    
        def exp_move():  # shifts last 3 [..., a, 0, 0] to [..., 0, 2^a, 0] for a>0
            state = ans[-1][:]
            state[2] -= 1
            state[3] += 2
            ans.append(state[:])
            while state[2]:
                state[3], state[4] = 0, 2 * state[3]
                ans.append(state[:])
                state[2:] = [state[2] - 1, state[4], 0]
                ans.append(state[:])
    
        exp_move()
        assert ans[-1] == [0, 1, 0, 2 ** 14, 0]
        ans.append([0, 0, 2 ** 14, 0, 0])
        if n <= 16:
            ans.append([0, 0, 0, 2 ** 15, 0])
        else:
            exp_move()
            assert ans[-1] == [0, 0, 0, 2 ** (2 ** 14), 0]
        state = ans[-1][:]
        state[-2] -= 2 ** (n - 1)
        state[-1] = 2 ** n
        ans.append(state)
        return ans
    ```
    
    </details>
    
* <a name="norelativeprimes"></a>**NoRelativePrimes** Inspired by [IMO 2016 Problem 4](https://www.imo-official.org/problems.aspx)
    
    Question: Is there a more efficient solution than the brute-force one we give, perhaps using the Chinese remainder
    theorem? (5 instances)
    
    ```python
    def sat(nums: List[int], b=7, m=6):
        assert len(nums) == len(set(nums)) == m and min(nums) >= 0
    
        def gcd(i, j):
            r, s = max(i, j), min(i, j)
            while s >= 1:
                r, s = s, (r % s)
            return r
    
        for a in nums:
            nums = [(a + i + 1) ** 2 + (a + i + 1) + 1 for i in range(b)]
            assert all(any(i != j and gcd(i, j) > 1 for j in nums) for i in nums)
    
        return True
    
    def sol(b=7, m=6):
        """
        Let P(n) = n^2 + n + 1.
    
        Given b>=6 and m>=1, find m non-negative integers for which the set {P(a+1), P(a+2), ..., P(a+b)} has
        the property that there is no element that is relatively prime to every other element.
    
        Sample input:
        b = 6
        m = 2
    
        Sample output:
        [195, 196]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        ans = []
    
        seen = set()
        deltas = set()
    
        def go(a):
            if a < 0 or a in seen or len(ans) == m:
                return
            seen.add(a)
            nums = [(a + i + 1) ** 2 + (a + i + 1) + 1 for i in range(b)]
            if all(any(i != j and gcd(i, j) > 1 for j in nums) for i in nums):
                new_deltas = [abs(a - a2) for a2 in ans if a != a2 and abs(a - a2) not in deltas]
                ans.append(a)
                for delta in new_deltas:
                    for a2 in ans:
                        go(a2 + delta)
                        go(a2 - delta)
                deltas.update(new_deltas)
                for delta in sorted(deltas):
                    go(a + delta)
    
        def gcd(i, j):
            r, s = max(i, j), min(i, j)
            while s >= 1:
                r, s = s, (r % s)
            return r
    
        a = 0
    
        while len(ans) < m:
            go(a)
            a += 1
    
        return ans
    ```
    
    </details>
    
* <a name="findrepeats"></a>**FindRepeats** Note: This problem is much easier than the IMO problem which also required a proof that it is impossible
    for a_0 not divisible by 3.
    
    Inspired by [IMO 2017 Problem 1](https://www.imo-official.org/problems.aspx) (5 instances)
    
    ```python
    def sat(indices: List[int], a0=123):
        assert a0 >= 0 and a0 % 3 == 0, "Hint: a_0 is a multiple of 3."
        s = [a0]
        for i in range(max(indices)):
            s.append(int(s[-1] ** 0.5) if int(s[-1] ** 0.5) ** 2 == s[-1] else s[-1] + 3)
        return len(indices) == len(set(indices)) == 1000 and min(indices) >= 0 and len({s[i] for i in indices}) == 1
    
    def sol(a0=123):
        """
        Find a repeating integer in an infinite sequence of integers, specifically the indices for which the same value
        occurs 1000 times. The sequence is defined by a starting value a_0 and each subsequent term is:
        a_{n+1} = the square root of a_n if the a_n is a perfect square, and a_n + 3 otherwise.
    
        For a given a_0 (that is a multiple of 3), the goal is to find 1000 indices where the a_i's are all equal.
    
        Sample input:
        9
    
        Sample output:
        [0, 3, 6, ..., 2997]
    
        The sequence starting with a0=9 is [9, 3, 6, 9, 3, 6, 9, ...] thus a_n at where n is a multiple of 3 are
        all equal in this case.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        n = a0
        ans = []
        i = 0
        while len(ans) < 1000:
            if n == 3:  # use the fact that 3 will repeat infinitely often
                ans.append(i)
            n = int(n ** 0.5) if int(n ** 0.5) ** 2 == n else n + 3
            i += 1
        return ans
    ```
    
    </details>
    
* <a name="picknearneighbors"></a>**PickNearNeighbors** Inspired by [IMO 2017 Problem 5](https://www.imo-official.org/problems.aspx)
    
    The puzzle solution follows the judge's proof closely. (5 instances)
    
    ```python
    def sat(keep: List[bool], heights=[10, 2, 14, 1, 8, 19, 16, 6, 12, 3, 17, 0, 9, 18, 5, 7, 11, 13, 15, 4]):
        n = int(len(heights) ** 0.5)
        assert sorted(heights) == list(range(n * n + n)), "hint: heights is a permutation of range(n * n + n)"
        kept = [i for i, k in zip(heights, keep) if k]
        assert len(kept) == 2 * n, "must keep 2n items"
        pi = sorted(range(2 * n), key=lambda i: kept[i])  # the sort indices
        return all(abs(pi[2 * i] - pi[2 * i + 1]) == 1 for i in range(n))
    
    def sol(heights=[10, 2, 14, 1, 8, 19, 16, 6, 12, 3, 17, 0, 9, 18, 5, 7, 11, 13, 15, 4]):
        """
        Given a permutation of the integers up to n(n+1) as a list, choose 2n numbers to keep (in the same order)
        so that the remaining list of numbers satisfies:
        * its largest number is next to its second largest number
        * its third largest number is next to its fourth largest number
        ...
        * its second smallest number is next to its smallest number
    
        Sample input:
        [4, 0, 5, 3, 1, 2]
        n = 2
    
        Sample output:
        [True, False, True, False, True, True]
    
        Keeping these indices results in the sublist [4, 5, 1, 2] where 4 and 5 are adjacent as are 1 and 2.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        # Based on the judge's solution.
        n = int(len(heights) ** 0.5)
        assert sorted(heights) == list(range(n * (n + 1)))
        groups = [h // (n + 1) for h in heights]
        ans = [False] * len(heights)
        a = 0
        used_groups = set()
        while sum(ans) < 2 * n:
            group_tracker = {}
            b = a
            while groups[b] not in group_tracker or groups[b] in used_groups:
                group_tracker[groups[b]] = b
                b += 1
            ans[group_tracker[groups[b]]] = True
            ans[b] = True
            used_groups.add(groups[b])
            a = b + 1
        return ans
    ```
    
    </details>
    
* <a name="findproductivelist"></a>**FindProductiveList** Note: This problem is easier than the IMO problem because the hard part is proving that sequences do not
    exists for non-multiples of 3.
    
    Inspired by [IMO 2010 Problem 5](https://www.imo-official.org/problems.aspx) (5 instances)
    
    ```python
    def sat(li: List[int], n=18):
        assert n % 3 == 0, "Hint: n is a multiple of 3"
        return len(li) == n and all(li[(i + 2) % n] == 1 + li[(i + 1) % n] * li[i] for i in range(n))
    
    def sol(n=18):
        """
        Given n, find n integers such that li[i] * li[i+1] + 1 == li[i+2], for i = 0, 1, ..., n-1
        where indices >= n "wrap around". Note: only n multiples of 3 are given since this is only possible for n
        that are multiples of 3 (as proven in the IMO problem).
    
        Sample input:
        6
    
        Sample output:
        [_, _, _, _, _, _]
    
        (Sample output hidden because showing sample output would give away too much information.)
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [-1, -1, 2] * (n // 3)
    ```
    
    </details>
    
* <a name="halftag"></a>**HalfTag** Inspired by [IMO 2020 Problem 3](https://www.imo-official.org/problems.aspx) (5 instances)
    
    ```python
    def sat(li: List[int], tags=[3, 0, 3, 2, 0, 1, 0, 3, 1, 1, 2, 2, 0, 2, 1, 3]):
        n = max(tags) + 1
        assert sorted(tags) == sorted(list(range(n)) * 4), "hint: each tag occurs exactly four times"
        assert len(li) == len(set(li)) and min(li) >= 0
        return sum(li) * 2 == sum(range(4 * n)) and sorted([tags[i] for i in li]) == [i // 2 for i in range(2 * n)]
    
    def sol(tags=[3, 0, 3, 2, 0, 1, 0, 3, 1, 1, 2, 2, 0, 2, 1, 3]):
        """
        The input tags is a list of 4n integer tags each in range(n) with each tag occurring 4 times.
        The goal is to find a subset (list) li of half the indices such that:
        * The sum of the indices equals the sum of the sum of the missing indices.
        * The tags of the chosen indices contains exactly each number in range(n) twice.
    
        Sample input:
        n = 3
        tags = [0, 1, 2, 0, 0, 1, 1, 1, 2, 2, 0, 2]
    
        Sample output:
        [0, 3, 5, 6, 8, 11]
    
        Note the sum of the output is 33 = (0+1+2+...+11)/2 and the selected tags are [0, 0, 1, 1, 2, 2]
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        n = max(tags) + 1
        pairs = {(i, 4 * n - i - 1) for i in range(2 * n)}
        by_tag = {tag: [] for tag in range(n)}
        for p in pairs:
            a, b = [tags[i] for i in p]
            by_tag[a].append(p)
            by_tag[b].append(p)
        cycles = []
        cycle = []
        while pairs:
            if not cycle:  # start new cycle
                p = pairs.pop()
                pairs.add(p)  # just to pick a tag
                tag = tags[p[0]]
                # print("Starting cycle with tag", tag)
            p = by_tag[tag].pop()
            a, b = [tags[i] for i in p]
            # print(p, a, b)
            tag = a if a != tag else b
            by_tag[tag].remove(p)
            cycle.append(p if tag == b else p[::-1])
            pairs.remove(p)
            if not by_tag[tag]:
                cycles.append(cycle)
                cycle = []
    
        while any(len(c) % 2 for c in cycles):
            cycle_tags = [{tags[k] for p in c for k in p} for c in cycles]
            merged = False
            for i in range(len(cycles)):
                for j in range(i):
                    intersection = cycle_tags[i].intersection(cycle_tags[j])
                    if intersection:
                        c = intersection.pop()
                        # print(f"Merging cycle {i} and cycle {j} at tag {c}", cycles)
                        cycle_i = cycles.pop(i)
                        for i1, p in enumerate(cycle_i):
                            if tags[p[0]] == c:
                                break
                        for j1, p in enumerate(cycles[j]):
                            if tags[p[0]] == c:
                                break
                        cycles[j][j1:j1] = cycle_i[i1:] + cycle_i[:i1]
                        merged = True
                        break
                if merged:
                    break
    
        ans = []
        for c in cycles:
            for i, p in enumerate(c):
                if i % 2:
                    ans += p
    
        return ans
    ```
    
    </details>
    
## lattices

Lattice problems with and without noise

* <a name="learnparity"></a>**LearnParity** Parity learning (Gaussian elimination)
    
    The canonical solution to this 
    [Parity learning problem](https://en.wikipedia.org/w/index.php?title=Parity_learning)
    is to use 
    [Gaussian Elimination](https://en.wikipedia.org/w/index.php?title=Gaussian_elimination).
    
    The vectors are encoded as binary integers for succinctness. (5 instances)
    
    ```python
    def sat(inds: List[int], vecs=[8543342634111025532, 8335192666369313368, 2359039407982105779, 4172548441791366513, 1256349095522986569, 3754463859322679595, 1562879970152915618, 1933016518061876369, 5920060919607788629, 8545759471656960221, 2934241949774725291, 559495833580308526, 5239436672544732707, 5865707252111994906, 8310678944230832071, 4595527784831581592, 4348871153851862010, 5198370132175169882, 3748480974791545460, 1215135748294622536, 4321487173746421746, 9012812639700145153, 588387599697000986, 5003829835901037543, 7754881381173342129, 2635789994388296837, 3222773777603033590, 5790284924977099989, 7540575369379211274, 7898971930608516039, 27260728996582582, 1792453914477410383, 8726418386455953809, 9193001185022172125, 3515388340741601364, 6217726337930929836, 1038687698871580494, 1892601486162604802, 3633356355444530940, 108334555669330693, 1955821183884414243, 5681081121990060330, 5791800194327455183, 8459367068223249929, 4271428016720060690, 913733008909519396, 2233236350093301187, 6538503022239131288, 5292485269677307644, 4615671355181378169, 2605305508625596241, 4954529961471509975, 2312963580097644831, 888555840551788245, 4152336321587083789, 8978251650218883651, 2567641184250287470, 2168893575221172018, 4358821646257958779, 3102433300308778243, 4185793889128296420, 6687096428156463254, 4143873353280484310, 8454616559174688585, 6589014033410725016, 5903549622062684554, 2388718494916838667, 8850145667696469408, 5068285804151890745, 2981241929741282230, 79408177335937724, 1711542430102927280]):
        return all(sum((v >> i) & 1 for i in inds) % 2 == 1 for v in vecs)
    
    def sol(vecs=[8543342634111025532, 8335192666369313368, 2359039407982105779, 4172548441791366513, 1256349095522986569, 3754463859322679595, 1562879970152915618, 1933016518061876369, 5920060919607788629, 8545759471656960221, 2934241949774725291, 559495833580308526, 5239436672544732707, 5865707252111994906, 8310678944230832071, 4595527784831581592, 4348871153851862010, 5198370132175169882, 3748480974791545460, 1215135748294622536, 4321487173746421746, 9012812639700145153, 588387599697000986, 5003829835901037543, 7754881381173342129, 2635789994388296837, 3222773777603033590, 5790284924977099989, 7540575369379211274, 7898971930608516039, 27260728996582582, 1792453914477410383, 8726418386455953809, 9193001185022172125, 3515388340741601364, 6217726337930929836, 1038687698871580494, 1892601486162604802, 3633356355444530940, 108334555669330693, 1955821183884414243, 5681081121990060330, 5791800194327455183, 8459367068223249929, 4271428016720060690, 913733008909519396, 2233236350093301187, 6538503022239131288, 5292485269677307644, 4615671355181378169, 2605305508625596241, 4954529961471509975, 2312963580097644831, 888555840551788245, 4152336321587083789, 8978251650218883651, 2567641184250287470, 2168893575221172018, 4358821646257958779, 3102433300308778243, 4185793889128296420, 6687096428156463254, 4143873353280484310, 8454616559174688585, 6589014033410725016, 5903549622062684554, 2388718494916838667, 8850145667696469408, 5068285804151890745, 2981241929741282230, 79408177335937724, 1711542430102927280]):
        """
        Parity learning: Given binary vectors in a subspace, find the secret set S of indices such that:
        $\\sum_{i \in S} x_i = 1 (mod 2)$
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        # Gaussian elimination
        d = 0  # decode vectors into arrays
        m = max(vecs)
        while m:
            m >>= 1
            d += 1
        vecs = [[(n >> i) & 1 for i in range(d)] for n in vecs]
        ans = []
        pool = [[0] * (d + 1) for _ in range(d)] + [v + [1] for v in vecs]
        for i in range(d):
            pool[i][i] = 1
    
        for i in range(d):  # zero out bit i
            for v in pool[d:]:
                if v[i] == 1:
                    break
            if v[i] == 0:
                v = pool[i]
            assert v[i] == 1  # found a vector with v[i] = 1, subtract it off from those with a 1 in the ith coordinate
            w = v[:]
            for v in pool:
                if v[i] == 1:
                    for j in range(d + 1):
                        v[j] ^= w[j]
    
        return [i for i in range(d) if pool[i][-1]]
    ```
    
    </details>
    
* <a name="learnparitywithnoise"></a>**LearnParityWithNoise** Learn parity with noise (*unsolved*)
    
    The fastest known algorithm to this
    [Parity learning problem](https://en.wikipedia.org/w/index.php?title=Parity_learning)
    runs in time $2^(d/(log d))$ (5 instances)
    
    ```python
    def sat(inds: List[int], vecs=[2874444459419665109, 3571416480966091062, 3627516422625241827, 2417762213996395207, 4371357242721531635, 1396026910505373292, 6671557086560014752, 9066082518122683098, 5240053591369828114, 8556210480838058892, 7302977584273736381, 8938278934736014411, 4398671200512032996, 6147375266514044469, 6609538006889421793, 2297823643430705118, 7583979108118079257, 2498392101379258437, 7893501751515236283, 2027235323873165116, 925357965000140266, 9009345166609418406, 5689450111800001849, 2079746314404416253, 4228649029862868917, 5819371323838727219, 102386757609774316, 5480808186035115654, 3001738569073502536, 9059061077086189682, 681271298018419415, 5616731111115463763, 2722737236861682531, 4918075690687573998, 7125583379861998376, 7968096465923567867, 898679944061110348, 1140358409311167922, 6077294650144352445, 587776882127248609, 2018954969823094844, 1618480274277140739, 8884189689498565225, 4084721521520724931, 4718438135662169666, 8411612063174086200, 8726374275365985960, 3135872851883336005, 1091802941995014823, 4944178741300545327, 6970959994566965947, 2911632933598497473, 8638215954009823387, 7438975146059987571, 3486356869336916018, 4935404783245269300, 3492912640500734004, 7903591215281799872, 4616161610863395412, 875020887047334808, 2721628497281503934, 6882639287577667047, 6274957618887284536, 3575443754501116278, 2031604067526359716, 4433373641914130623, 6204772769819600658, 8509292558066435714, 1857073904365563798, 7875287918949902618, 5205034693823928900, 4943396962875355147, 2805601192218759148, 8976171820624983460, 5930936964665834653, 949687393644726240, 6466615045398392331, 423404729770342491, 2720698610804800422, 7479269416044676778, 7869290888646534505, 6327163107872545492, 476579447640475544, 1218066186129904051, 7630726053076756205, 7741086216563432736, 5225376670650457287, 7040078265943665053, 2162853338175426448, 5633819254572300801, 92334600849454176, 9098183941628882647, 3481731752092062852, 5473741255745389738, 7266470290696653678, 3090338455353169956, 4358343354422765853, 3623553173494979282, 8328390749513844747, 2287762878756609646, 4126189061710502597, 5829472669961813184, 7342395882491704275, 5030578088617810038, 2210525427289006508, 6161187897225224000, 5601573223749212224, 6539026784581543793, 3571032801838391198, 4813662449014287760, 6577243754700968179, 4401899289452367605, 305529480505303551, 1548494450097231731, 6926707725781258948, 6357305518384676781, 6357665620505806556, 1554358231697328409, 7871587375269472810, 2094942344314098945, 1452972368095860063, 3210274450167364491, 6901356410911155351, 7609098874470545378, 6955802737127492446, 6919896432783547538, 5423154486785623318, 3105394980859157674, 8438962979748731599, 4110730383299136510, 6718356757580670867]):
        return sum(sum((v >> i) & 1 for i in inds) % 2 for v in vecs) >= len(vecs) * 3 / 4
    
    def sol(vecs=[2874444459419665109, 3571416480966091062, 3627516422625241827, 2417762213996395207, 4371357242721531635, 1396026910505373292, 6671557086560014752, 9066082518122683098, 5240053591369828114, 8556210480838058892, 7302977584273736381, 8938278934736014411, 4398671200512032996, 6147375266514044469, 6609538006889421793, 2297823643430705118, 7583979108118079257, 2498392101379258437, 7893501751515236283, 2027235323873165116, 925357965000140266, 9009345166609418406, 5689450111800001849, 2079746314404416253, 4228649029862868917, 5819371323838727219, 102386757609774316, 5480808186035115654, 3001738569073502536, 9059061077086189682, 681271298018419415, 5616731111115463763, 2722737236861682531, 4918075690687573998, 7125583379861998376, 7968096465923567867, 898679944061110348, 1140358409311167922, 6077294650144352445, 587776882127248609, 2018954969823094844, 1618480274277140739, 8884189689498565225, 4084721521520724931, 4718438135662169666, 8411612063174086200, 8726374275365985960, 3135872851883336005, 1091802941995014823, 4944178741300545327, 6970959994566965947, 2911632933598497473, 8638215954009823387, 7438975146059987571, 3486356869336916018, 4935404783245269300, 3492912640500734004, 7903591215281799872, 4616161610863395412, 875020887047334808, 2721628497281503934, 6882639287577667047, 6274957618887284536, 3575443754501116278, 2031604067526359716, 4433373641914130623, 6204772769819600658, 8509292558066435714, 1857073904365563798, 7875287918949902618, 5205034693823928900, 4943396962875355147, 2805601192218759148, 8976171820624983460, 5930936964665834653, 949687393644726240, 6466615045398392331, 423404729770342491, 2720698610804800422, 7479269416044676778, 7869290888646534505, 6327163107872545492, 476579447640475544, 1218066186129904051, 7630726053076756205, 7741086216563432736, 5225376670650457287, 7040078265943665053, 2162853338175426448, 5633819254572300801, 92334600849454176, 9098183941628882647, 3481731752092062852, 5473741255745389738, 7266470290696653678, 3090338455353169956, 4358343354422765853, 3623553173494979282, 8328390749513844747, 2287762878756609646, 4126189061710502597, 5829472669961813184, 7342395882491704275, 5030578088617810038, 2210525427289006508, 6161187897225224000, 5601573223749212224, 6539026784581543793, 3571032801838391198, 4813662449014287760, 6577243754700968179, 4401899289452367605, 305529480505303551, 1548494450097231731, 6926707725781258948, 6357305518384676781, 6357665620505806556, 1554358231697328409, 7871587375269472810, 2094942344314098945, 1452972368095860063, 3210274450167364491, 6901356410911155351, 7609098874470545378, 6955802737127492446, 6919896432783547538, 5423154486785623318, 3105394980859157674, 8438962979748731599, 4110730383299136510, 6718356757580670867]):
        """
        Learning parity with noise: Given binary vectors, find the secret set $S$ of indices such that, for at least
        3/4 of the vectors, $$sum_{i \in S} x_i = 1 (mod 2)$$
        """
    ```
    0 hand-written solutions 
    
## number_theory

Number theory problems

* <a name="fermatslasttheorem"></a>**FermatsLastTheorem** [Fermat's last theorem](https://en.wikipedia.org/w/index.php?title=Fermat%27s_Last_Theorem)
    
    Supposedly unsolvable, but how confident are really in the super-complicated proof?
    
    See [Wiles, Andrew. "Modular elliptic curves and Fermat's last theorem." Annals of mathematics 141.3 (1995): 443-551.](https://www.jstor.org/stable/2118559) (1 instance)
    
    ```python
    def sat(nums: List[int]):
        a, b, c, n = nums
        return (a ** n + b ** n == c ** n) and min(a, b, c) > 0 and n > 2
    
    def sol():
        """Find integers a,b,c > 0, n > 2, such such that a^n + b^n == c^n"""
    ```
    0 hand-written solutions 
    
* <a name="gcd"></a>**GCD** [Greatest Common Divisor](https://en.wikipedia.org/w/index.php?title=Greatest_common_divisor&oldid=990943381)
    (GCD)
    
    See also the [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm) (5 instances)
    
    ```python
    def sat(n: int, a=15482, b=23223, lower_bound=5):
        return a % n == 0 and b % n == 0 and n >= lower_bound
    
    def sol(a=15482, b=23223, lower_bound=5):
        """Find a large common divisor of two integers."""
    ```
    <details><summary>2 hand-written solutions </summary>
    
    ```python
        m, n = min(a, b), max(a, b)
        while m > 0:
            m, n = n % m, m
        return n
    ```
    
    ```python
        def gcd(m, n):
            if m > n:
                return gcd(n, m)
            if m == 0:
                return n
            return gcd(n % m, m)
    
        return gcd(a, b)
    ```
    
    </details>
    
* <a name="gcd_multi"></a>**GCD_multi** [Greatest Common Divisor](https://en.wikipedia.org/w/index.php?title=Greatest_common_divisor&oldid=990943381)
    (GCD)
    
    See also the [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm) (5 instances)
    
    ```python
    def sat(n: int, nums=[77410, 23223, 54187], lower_bound=2):
        return all(i % n == 0 for i in nums) and n >= lower_bound
    
    def sol(nums=[77410, 23223, 54187], lower_bound=2):
        """Find a large common divisor of the list of integers."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        n = 0
        for i in nums:
            m, n = min(i, n), max(i, n)
            while m > 0:
                m, n = n % m, m
        return n
    ```
    
    </details>
    
* <a name="lcm"></a>**LCM** [Least Common Multiple](https://en.wikipedia.org/wiki/Least_common_multiple)
    (LCM)
    
    See also the [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm) (5 instances)
    
    ```python
    def sat(n: int, a=15, b=27, upper_bound=150):
        return n % a == 0 and n % b == 0 and 0 < n <= upper_bound
    
    def sol(a=15, b=27, upper_bound=150):
        """Find a small common multiple of two integers."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        m, n = min(a, b), max(a, b)
        while m > 0:
            m, n = n % m, m
        return a * (b // n)
    ```
    
    </details>
    
* <a name="lcm_multi"></a>**LCM_multi** [Least Common Multiple](https://en.wikipedia.org/wiki/Least_common_multiple)
    (LCM)
    
    See also the [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm) (5 instances)
    
    ```python
    def sat(n: int, nums=[15, 27, 102], upper_bound=5000):
        return all(n % i == 0 for i in nums) and 0 < n <= upper_bound
    
    def sol(nums=[15, 27, 102], upper_bound=5000):
        """Find a small common multiple of a list of integers."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        ans = 1
        for i in nums:
            m, n = min(i, ans), max(i, ans)
            while m > 0:
                m, n = n % m, m
            ans *= (i // n)
        return ans
    ```
    
    </details>
    
* <a name="smallexponentbigsolution"></a>**SmallExponentBigSolution** Small exponent, big solution
    
    Problems have small b and target but solution is typically a large n.
    Some of them are really hard, for example, for `b=2, target=3`, the smallest solution is `n=4700063497`
    
    See [Richard K. Guy "The strong law of small numbers", (problem 13)](https://doi.org/10.2307/2322249) (4 instances)
    
    ```python
    def sat(n: int, b=2, target=5):
        return (b ** n) % n == target
    
    def sol(b=2, target=5):
        """Solve for n: b^n = target (mod n)"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        for n in range(1, 10 ** 5):
            if pow(b, n, n) == target:
                return n
    ```
    
    </details>
    
* <a name="threecubes"></a>**ThreeCubes** Sum of three cubes
    
    Given `n`, find integers `a`, `b`, `c` such that `a**3 + b**3 + c**3 = n`. This is unsolvable for `n % 9 in {4, 5}`.
    Conjectured to be true for all other n, i.e., `n % 9 not in {4, 5}`.
    `a`, `b`, `c` may be positive or negative
    
    See [wikipedia entry](https://en.wikipedia.org/wiki/Sums_of_three_cubes) or
    [Andrew R. Booker, Andrew V. Sutherland (2020). "On a question of Mordell."](https://arxiv.org/abs/2007.01209) (5 instances)
    
    ```python
    def sat(nums: List[int], target=983):
        assert target % 9 not in [4, 5], "Hint"
        return len(nums) == 3 and sum([i ** 3 for i in nums]) == target
    
    def sol(target=983):
        """Given n, find integers a, b, c such that a^3 + b^3 + c^3 = n."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        assert target % 9 not in {4, 5}
        for i in range(20):
            for j in range(i + 1):
                for k in range(-20, j + 1):
                    n = i ** 3 + j ** 3 + k ** 3
                    if n == target:
                        return [i, j, k]
                    if n == -target:
                        return [-i, -j, -k]
    ```
    
    </details>
    
* <a name="foursquares"></a>**FourSquares** Sum of four squares
    
    [Lagrange's Four Square Theorem](https://en.wikipedia.org/w/index.php?title=Lagrange%27s_four-square_theorem)
    
    Given a non-negative integer `n`, a classic theorem of Lagrange says that `n` can be written as the sum of four
    integers. The problem here is to find them. This is a nice problem and we give an elementary solution
    that runs in time 	ilde{O}(n),
    which is not "polynomial time" because it is not polynomial in log(n), the length of n. A poly-log(n)
    algorithm using quaternions is described in the book:
    ["Randomized algorithms in number theory" by Michael O. Rabin and Jeffery O. Shallit (1986)](https://doi.org/10.1002/cpa.3160390713)
    
    The first half of the problems involve small numbers and the second half involve some numbers up to 50 digits. (5 instances)
    
    ```python
    def sat(nums: List[int], n=12345):
        return len(nums) <= 4 and sum(i ** 2 for i in nums) == n
    
    def sol(n=12345):
        """Find four integers whose squares sum to n"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        m = n
        squares = {i ** 2: i for i in range(int(m ** 0.5) + 2) if i ** 2 <= m}
        sums_of_squares = {i + j: [a, b] for i, a in squares.items() for j, b in squares.items()}
        for s in sums_of_squares:
            if m - s in sums_of_squares:
                return sums_of_squares[m - s] + sums_of_squares[s]
        assert False, "Should never reach here"
    ```
    
    </details>
    
* <a name="factoring"></a>**Factoring** [Factoring](https://en.wikipedia.org/w/index.php?title=Integer_factorization) and
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
    unless considerable advances are made in integer factorization or computational power in the near future. (5 instances)
    
    ```python
    def sat(i: int, n=241864633):
        return 1 < i < n and n % i == 0
    
    def sol(n=241864633):
        """Find a non-trivial factor of integer n"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if n % 2 == 0:
            return 2
    
        for i in range(3, int(n ** 0.5) + 1, 2):
            if n % i == 0:
                return i
    
        assert False, "problem defined for composite n only"
    ```
    
    </details>
    
* <a name="discretelog"></a>**DiscreteLog** Discrete Log
    
    The discrete logarithm problem is (given `g`, `t`, and `p`) to find n such that:
    
    `g ** n % p == t`
    
    From [Wikipedia article](https://en.wikipedia.org/w/index.php?title=Discrete_logarithm_records):
    
    "Several important algorithms in public-key cryptography base their security on the assumption
    that the discrete logarithm problem over carefully chosen problems has no efficient solution."
    
    The problem is *unsolved* in the sense that no known polynomial-time algorithm has been found.
    
    We include McCurley's discrete log challenge from
    [Weber D., Denny T. (1998) "The solution of McCurley's discrete log challenge."](https://link.springer.com/content/pdf/10.1007/BFb0055747.pdf) (5 instances)
    
    ```python
    def sat(n: int, g=44337, p=69337, t=38187):
        return pow(g, n, p) == t
    
    def sol(g=44337, p=69337, t=38187):
        """Find n such that g^n is congruent to t mod n"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        for n in range(p):
            if pow(g, n, p) == t:
                return n
        assert False, f"unsolvable discrete log problem g={g}, t={t}, p={p}"
    ```
    
    </details>
    
* <a name="gcd17"></a>**GCD17** According to [this article](https://primes.utm.edu/glossary/page.php?sort=LawOfSmall), the smallest
    solution is 8424432925592889329288197322308900672459420460792433 (1 instance)
    
    ```python
    def sat(n: int):
        i = n ** 17 + 9
        j = (n + 1) ** 17 + 9
    
        while i != 0:  # compute gcd using Euclid's algorithm
            (i, j) = (j % i, i)
    
        return n >= 0 and j != 1
    
    def sol():
        """Find n for which gcd(n^17+9, (n+1)^17+9) != 1"""
    ```
    0 hand-written solutions 
    
* <a name="znam"></a>**Znam** [Znam's Problem](https://en.wikipedia.org/wiki/Zn%C3%A1m%27s_problem)
    
    For example [2, 3, 7, 47, 395] is a solution for k=5 (4 instances)
    
    ```python
    def sat(li: List[int], k=5):
        def prod(nums):
            ans = 1
            for i in nums:
                ans *= i
            return ans
    
        return min(li) > 1 and len(li) == k and all((1 + prod(li[:i] + li[i + 1:])) % li[i] == 0 for i in range(k))
    
    def sol(k=5):
        """Find k positive integers such that each integer divides (the product of the rest plus 1)."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        n = 2
        prod = 1
        ans = []
        while len(ans) < k:
            ans.append(n)
            prod *= n
            n = prod + 1
        return ans
    ```
    
    </details>
    
* <a name="collatzcycleunsolved"></a>**CollatzCycleUnsolved** Collatz Conjecture
    
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
    there is either a cycle or a sequence that increases without bound. This problem seeks a cycle. (1 instance)
    
    ```python
    def sat(n: int):
        m = n
        while n > 4:
            n = 3 * n + 1 if n % 2 else n // 2
            if n == m:
                return True
    
    def sol():
        """
        Consider the following process. Start with an integer `n` and repeatedly applying the operation:
        * if n is even, divide n by 2,
        * if n is odd, multiply n by 3 and add 1
        Find n > 4 which is part of a cycle of this process
        """
    ```
    0 hand-written solutions 
    
* <a name="collatzgeneralizedunsolved"></a>**CollatzGeneralizedUnsolved** Generalized Collatz Conjecture
    
    This version, permits negative n and seek a cycle with a number of magnitude greater than 1000,
    which would disprove the Generalized conjecture that states that the only cycles are the known 5 cycles
    (which don't have positive integers).
    
    See the [Wikipedia article](https://en.wikipedia.org/wiki/Collatz_conjecture) (1 instance)
    
    ```python
    def sat(start: int):
        n = start  # could be positive or negative ...
        while abs(n) > 1000:
            n = 3 * n + 1 if n % 2 else n // 2
            if n == start:
                return True
    
    def sol():
        """
        Consider the following process. Start with an integer `n` and repeatedly applying the operation:
        * if n is even, divide n by 2,
        * if n is odd, multiply n by 3 and add 1
        Find n which is part of a cycle of this process that has |n| > 1000
        """
    ```
    0 hand-written solutions 
    
* <a name="collatzdelay"></a>**CollatzDelay** Collatz Delay
    
    Consider the following process. Start with an integer `n` and repeatedly applying the operation:
    * if n is even, divide n by 2,
    * if n is odd, multiply n by 3 and add 1
    Find `0 < n < upper` so that it takes exactly `t` steps to reach 1.
    
    
    For instance,
    the number `n=9780657630` takes 1,132 steps and the number `n=93,571,393,692,802,302` takes
    2,091 steps, according to the [Wikipedia article](https://en.wikipedia.org/wiki/Collatz_conjecture)
    
    Now, this problem can be solved trivially by taking exponentially large `n = 2 ** t` so we also bound the
    number of bits of the solution to be upper.
    
    See [this webpage](http://www.ericr.nl/wondrous/delrecs.html) for up-to-date records. (4 instances)
    
    ```python
    def sat(n: int, t=197, upper=20):
        m = n
        for i in range(t):
            if n <= 1:
                return False
            n = 3 * n + 1 if n % 2 else n // 2
        return n == 1 and m <= 2 ** upper
    
    def sol(t=197, upper=20):
        """
        Consider the following process. Start with an integer `n` and repeatedly applying the operation:
        * if n is even, divide n by 2,
        * if n is odd, multiply n by 3 and add 1
        Find `0 < n < upper` so that it takes exactly `t` steps to reach 1.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
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
    ```
    
    </details>
    
* <a name="lehmer"></a>**Lehmer** Lehmer puzzle
    
    According to [The Strong Law of Large Numbers](https://doi.org/10.2307/2322249) Richard K. Guy states that
        D. H. & Emma Lehmer discovered that 2^n = 3 (mod n) for n = 4700063497,
        but for no smaller n > 1 (1 instance)
    
    ```python
    def sat(n: int):
        return pow(2, n, n) == 3
    
    def sol():
        """Find n  such that 2^n mod n = 3"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return 4700063497
    ```
    
    </details>
    
## probability

Probability problems

* <a name="birthdayparadox"></a>**BirthdayParadox** Adaptation of the classic
    [Birthday Problem](https://en.wikipedia.org/wiki/Birthday_problem (Mathematical Problems category)).
    
    The year length is year_len (365 is earth, while Neptune year is 60,182). (4 instances)
    
    ```python
    def sat(n: int, year_len=365):
        prob = 1.0
        for i in range(n):
            prob *= (year_len - i) / year_len
        return (prob - 0.5) ** 2 <= 1/year_len
    
    def sol(year_len=365):
        """Find n such that the probability of two people having the same birthday in a group of n is near 1/2."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        n = 1
        distinct_prob = 1.0
        best = (0.5, 1)  # (difference between probability and 1/2, n)
        while distinct_prob > 0.5:
            distinct_prob *= (year_len - n) / year_len
            n += 1
            best = min(best, (abs(0.5 - distinct_prob), n))
    
        return best[1]
    ```
    
    </details>
    
* <a name="birthdayparadoxmontecarlo"></a>**BirthdayParadoxMonteCarlo** A slower, Monte Carlo version of the above Birthday Paradox problem. (4 instances)
    
    ```python
    def sat(n: int, year_len=365):
        import random
        random.seed(0)
        K = 1000  # number of samples
        prob = sum(len({random.randrange(year_len) for i in range(n)}) < n for j in range(K)) / K
        return (prob - 0.5) ** 2 <= year_len
    
    def sol(year_len=365):
        """Find n such that the probability of two people having the same birthday in a group of n is near 1/2."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        n = 1
        distinct_prob = 1.0
        best = (0.5, 1)  # (difference between probability and 1/2, n)
        while distinct_prob > 0.5:
            distinct_prob *= (year_len - n) / year_len
            n += 1
            best = min(best, (abs(0.5 - distinct_prob), n))
    
        return best[1]
    ```
    
    </details>
    
* <a name="ballotproblem"></a>**BallotProblem** See the [Wikipedia article](https://en.wikipedia.org/wiki/Bertrand%27s_ballot_theorem) or
    or  [Addario-Berry L., Reed B.A. (2008) Ballot Theorems, Old and New. In: Gyori E., Katona G.O.H., Lovász L.,
    Sági G. (eds) Horizons of Combinatorics. Bolyai Society Mathematical Studies, vol 17.
    Springer, Berlin, Heidelberg.](https://doi.org/10.1007/978-3-540-77200-2_1) (5 instances)
    
    ```python
    def sat(counts: List[int], target_prob=0.5):
        m, n = counts  # m = num 1's, n = num -1's
        probs = [1.0] + [0.0] * n  # probs[n] is probability for current m, starting with m = 1
        for i in range(2, m + 1):  # compute probs using dynamic programming for m = i
            old_probs = probs
            probs = [1.0] + [0.0] * n
            for j in range(1, min(n + 1, i)):
                probs[j] = (
                        j / (i + j) * probs[j - 1]  # last element is a -1 so use probs
                        +
                        i / (i + j) * old_probs[j]  # last element is a 1 so use old_probs, m = i - 1
                )
        return abs(probs[n] - target_prob) < 1e-6
    
    def sol(target_prob=0.5):
        """
        Suppose a list of m 1's and n -1's are permuted at random.
        What is the probability that all of the cumulative sums are positive?
        The goal is to find counts = [m, n] that make the probability of the ballot problem close to target_prob.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        for m in range(1, 10000):
            n = round(m * (1 - target_prob) / (1 + target_prob))
            if abs(target_prob - (m - n) / (m + n)) < 1e-6:
                return [m, n]
    ```
    
    </details>
    
* <a name="binomialprobabilities"></a>**BinomialProbabilities** See [Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution) (5 instances)
    
    ```python
    def sat(counts: List[int], p=0.5, target_prob=0.0625):
        from itertools import product
        a, b = counts
        n = a + b
        prob = (p ** a) * ((1-p) ** b)
        tot = sum([prob for sample in product([0, 1], repeat=n) if sum(sample) == a])
        return abs(tot - target_prob) < 1e-6
    
    def sol(p=0.5, target_prob=0.0625):
        """Find counts = [a, b] so that the probability of  a H's and b T's among a + b coin flips is ~ target_prob."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        probs = [1.0]
        q = 1 - p
        while len(probs) < 20:
            probs = [(p * a + q * b) for a, b in zip([0] + probs, probs + [0])]
            answers = [i for i, p in enumerate(probs) if abs(p - target_prob) < 1e-6]
            if answers:
                return [answers[0], len(probs) - 1 - answers[0]]
    ```
    
    </details>
    
* <a name="exponentialprobability"></a>**ExponentialProbability** See [Exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution) (5 instances)
    
    ```python
    def sat(p_stop: float, steps=10, target_prob=0.5):
        prob = sum(p_stop*(1-p_stop)**t for t in range(steps))
        return abs(prob - target_prob) < 1e-6
    
    def sol(steps=10, target_prob=0.5):
        """
        Find p_stop so that the probability of stopping in steps or fewer time steps is the given target_prob if you
        stop each step with probability p_stop
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return 1 - (1 - target_prob) ** (1.0/steps)
    ```
    
    </details>
    
## trivial_inverse

Trivial problems. Typically for any function, you can construct a trivial example.
For instance, for the len function you can ask for a string of len(s)==100 etc.


* <a name="helloworld"></a>**HelloWorld** Trivial example, no solutions provided (1 instance)
    
    ```python
    def sat(s: str):
        return s + 'world' == 'Hello world'
    
    def sol():
        """Find a string that when concatenated onto 'world' gives 'Hello world'."""
    ```
    0 hand-written solutions 
    
* <a name="backworlds"></a>**BackWorlds** We provide two solutions (1 instance)
    
    ```python
    def sat(s: str):
        return s[::-1] + 'world' == 'Hello world'
    
    def sol():
        """Find a string that when reversed and concatenated onto 'world' gives 'Hello world'."""
    ```
    <details><summary>2 hand-written solutions </summary>
    
    ```python
        return ' olleH'
    ```
    
    ```python
        # solution methods must begin with 'sol'
        return 'Hello '[::-1]
    ```
    
    </details>
    
* <a name="stradd"></a>**StrAdd**  (5 instances)
    
    ```python
    def sat(st: str, a="world", b="Hello world"):
        return st + a == b
    
    def sol(a="world", b="Hello world"):
        """Solve simple string addition problem."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return b[:len(b) - len(a)]
    ```
    
    </details>
    
* <a name="strsetlen"></a>**StrSetLen**  (4 instances)
    
    ```python
    def sat(s: str, dups=2021):
        return len(set(s)) == len(s) - dups
    
    def sol(dups=2021):
        """Find a string with dups duplicate chars"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "a" * (dups + 1)
    ```
    
    </details>
    
* <a name="strmul"></a>**StrMul**  (5 instances)
    
    ```python
    def sat(s: str, target="foofoofoofoo", n=2):
        return s * n == target
    
    def sol(target="foofoofoofoo", n=2):
        """Find a string which when repeated n times gives target"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if n == 0:
            return ''
        return target[:len(target) // n]
    ```
    
    </details>
    
* <a name="strmul2"></a>**StrMul2**  (5 instances)
    
    ```python
    def sat(n: int, target="foofoofoofoo", s="foofoo"):
        return s * n == target
    
    def sol(target="foofoofoofoo", s="foofoo"):
        """Find n such that s repeated n times gives target"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if len(s) == 0:
            return 1
        return len(target) // len(s)
    ```
    
    </details>
    
* <a name="strlen"></a>**StrLen**  (5 instances)
    
    ```python
    def sat(s: str, n=1000):
        return len(s) == n
    
    def sol(n=1000):
        """Find a string of length n"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return 'a' * n
    ```
    
    </details>
    
* <a name="strat"></a>**StrAt**  (5 instances)
    
    ```python
    def sat(i: int, s="cat", target="a"):
        return s[i] == target
    
    def sol(s="cat", target="a"):
        """Find the index of target in string s"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return s.index(target)
    ```
    
    </details>
    
* <a name="strnegat"></a>**StrNegAt**  (5 instances)
    
    ```python
    def sat(i: int, s="cat", target="a"):
        return s[i] == target and i < 0
    
    def sol(s="cat", target="a"):
        """Find the index of target in s using a negative index."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return - (len(s) - s.index(target))
    ```
    
    </details>
    
* <a name="strslice"></a>**StrSlice**  (5 instances)
    
    ```python
    def sat(inds: List[int], s="hello world", target="do"):
        i, j, k = inds
        return s[i:j:k] == target
    
    def sol(s="hello world", target="do"):
        """Find the three slice indices that give the specific target in string s"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        from itertools import product
        for i, j, k in product(range(-len(s) - 1, len(s) + 1), repeat=3):
            try:
                if s[i:j:k] == target:
                    return [i, j, k]
            except (IndexError, ValueError):
                pass
    ```
    
    </details>
    
* <a name="strindex"></a>**StrIndex**  (5 instances)
    
    ```python
    def sat(s: str, big_str="foobar", index=2):
        return big_str.index(s) == index
    
    def sol(big_str="foobar", index=2):
        """Find a string whose *first* index in big_str is index"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return big_str[index:]
    ```
    
    </details>
    
* <a name="strindex2"></a>**StrIndex2**  (5 instances)
    
    ```python
    def sat(big_str: str, sub_str="foobar", index=2):
        return big_str.index(sub_str) == index
    
    def sol(sub_str="foobar", index=2):
        """Find a string whose *first* index of sub_str is index"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        i = ord('A')
        while chr(i) in sub_str:
            i += 1
        return chr(i) * index + sub_str
    ```
    
    </details>
    
* <a name="strin"></a>**StrIn**  (5 instances)
    
    ```python
    def sat(s: str, a="hello", b="yellow", length=4):
        return len(s) == length and s in a and s in b
    
    def sol(a="hello", b="yellow", length=4):
        """Find a string of length length that is in both strings a and b"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        for i in range(len(a) - length + 1):
            if a[i:i + length] in b:
                return a[i:i + length]
    ```
    
    </details>
    
* <a name="strin2"></a>**StrIn2**  (5 instances)
    
    ```python
    def sat(substrings: List[str], s="hello", count=15):
        return len(substrings) == len(set(substrings)) >= count and all(sub in s for sub in substrings)
    
    def sol(s="hello", count=15):
        """Find a list of >= count distinct strings that are all contained in s"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [""] + sorted({s[j:i] for i in range(len(s) + 1) for j in range(i)})
    ```
    
    </details>
    
* <a name="strcount"></a>**StrCount**  (5 instances)
    
    ```python
    def sat(string: str, substring="a", count=10, length=100):
        return string.count(substring) == count and len(string) == length
    
    def sol(substring="a", count=10, length=100):
        """Find a string with a certain number of copies of a given substring and of a given length"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        c = chr(1 + max(ord(c) for c in (substring or "a")))  # a character not in substring
        return substring * count + (length - len(substring) * count) * '^'
    ```
    
    </details>
    
* <a name="strsplit"></a>**StrSplit**  (5 instances)
    
    ```python
    def sat(x: str, parts=['I', 'love', 'dumplings', '!'], length=100):
        return len(x) == length and x.split() == parts
    
    def sol(parts=['I', 'love', 'dumplings', '!'], length=100):
        """Find a string of a given length with a certain split"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        joined = " ".join(parts)
        return joined + " " * (length - len(joined))
    ```
    
    </details>
    
* <a name="strsplitter"></a>**StrSplitter**  (5 instances)
    
    ```python
    def sat(x: str, parts=['I', 'love', 'dumplings', '!', ''], string="I_love_dumplings_!_"):
        return string.split(x) == parts
    
    def sol(parts=['I', 'love', 'dumplings', '!', ''], string="I_love_dumplings_!_"):
        """Find a separator that when used to split a given string gives a certain result"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if len(parts) <= 1:
            return string * 2
        length = (len(string) - len("".join(parts))) // (len(parts) - 1)
        start = len(parts[0])
        return string[start:start + length]
    ```
    
    </details>
    
* <a name="strjoiner"></a>**StrJoiner**  (5 instances)
    
    ```python
    def sat(x: str, parts=['I!!', '!love', 'dumplings', '!', ''], string="I!!!!!love!!dumplings!!!!!"):
        return x.join(parts) == string
    
    def sol(parts=['I!!', '!love', 'dumplings', '!', ''], string="I!!!!!love!!dumplings!!!!!"):
        """
        Find a separator that when used to join a given string gives a certain result.
        This is related to the previous problem but there are some edge cases that differ.
        """
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if len(parts) <= 1:
            return ""
        length = (len(string) - len("".join(parts))) // (len(parts) - 1)
        start = len(parts[0])
        return string[start:start + length]
    ```
    
    </details>
    
* <a name="strparts"></a>**StrParts**  (5 instances)
    
    ```python
    def sat(parts: List[str], sep="!!", string="I!!!!!love!!dumplings!!!!!"):
        return sep.join(parts) == string and all(sep not in p for p in parts)
    
    def sol(sep="!!", string="I!!!!!love!!dumplings!!!!!"):
        """Find parts that when joined give a specific string."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return string.split(sep)
    ```
    
    </details>
    
* <a name="listsetlen"></a>**ListSetLen**  (5 instances)
    
    ```python
    def sat(li: List[int], dups=42155):
        return len(set(li)) == len(li) - dups
    
    def sol(dups=42155):
        """Find a list with a certain number of duplicate items"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [1] * (dups + 1)
    ```
    
    </details>
    
* <a name="listmul"></a>**ListMul**  (5 instances)
    
    ```python
    def sat(li: List[int], target=[17, 9, -1, 17, 9, -1], n=2):
        return li * n == target
    
    def sol(target=[17, 9, -1, 17, 9, -1], n=2):
        """Find a list that when multiplied n times gives the target list"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if n == 0:
            return []
        return target[:len(target) // n]
    ```
    
    </details>
    
* <a name="listlen"></a>**ListLen**  (5 instances)
    
    ```python
    def sat(li: List[int], n=85012):
        return len(li) == n
    
    def sol(n=85012):
        """Find a list of a given length n"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [1] * n
    ```
    
    </details>
    
* <a name="listat"></a>**ListAt**  (5 instances)
    
    ```python
    def sat(i: int, li=[17, 31, 91, 18, 42, 1, 9], target=18):
        return li[i] == target
    
    def sol(li=[17, 31, 91, 18, 42, 1, 9], target=18):
        """Find the index of an item in a list. Any such index is fine."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return li.index(target)
    ```
    
    </details>
    
* <a name="listnegat"></a>**ListNegAt**  (5 instances)
    
    ```python
    def sat(i: int, li=[17, 31, 91, 18, 42, 1, 9], target=91):
        return li[i] == target and i < 0
    
    def sol(li=[17, 31, 91, 18, 42, 1, 9], target=91):
        """Find the index of an item in a list using negative indexing."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return li.index(target) - len(li)
    ```
    
    </details>
    
* <a name="listslice"></a>**ListSlice**  (5 instances)
    
    ```python
    def sat(inds: List[int], li=[42, 18, 21, 103, -2, 11], target=[-2, 21, 42]):
        i, j, k = inds
        return li[i:j:k] == target
    
    def sol(li=[42, 18, 21, 103, -2, 11], target=[-2, 21, 42]):
        """Find three slice indices to achieve a given list slice"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        from itertools import product
        for i, j, k in product(range(-len(li) - 1, len(li) + 1), repeat=3):
            try:
                if li[i:j:k] == target:
                    return [i, j, k]
            except (IndexError, ValueError):
                pass
    ```
    
    </details>
    
* <a name="listindex"></a>**ListIndex**  (5 instances)
    
    ```python
    def sat(item: int, li=[17, 2, 3, 9, 11, 11], index=4):
        return li.index(item) == index
    
    def sol(li=[17, 2, 3, 9, 11, 11], index=4):
        """Find the item whose first index in li is index"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return li[index]
    ```
    
    </details>
    
* <a name="listindex2"></a>**ListIndex2**  (5 instances)
    
    ```python
    def sat(li: List[int], i=29, index=10412):
        return li.index(i) == index
    
    def sol(i=29, index=10412):
        """Find a list that contains i first at index index"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [i - 1] * index + [i]
    ```
    
    </details>
    
* <a name="listin"></a>**ListIn**  (5 instances)
    
    ```python
    def sat(s: str, a=['cat', 'dot', 'bird'], b=['tree', 'fly', 'dot']):
        return s in a and s in b
    
    def sol(a=['cat', 'dot', 'bird'], b=['tree', 'fly', 'dot']):
        """Find an item that is in both lists a and b"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return next(s for s in b if s in a)
    ```
    
    </details>
    
* <a name="intneg"></a>**IntNeg**  (5 instances)
    
    ```python
    def sat(x: int, a=93252338):
        return -x == a
    
    def sol(a=93252338):
        """Solve a unary negation problem"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return - a
    ```
    
    </details>
    
* <a name="intsum"></a>**IntSum**  (5 instances)
    
    ```python
    def sat(x: int, a=1073258, b=72352549):
        return a + x == b
    
    def sol(a=1073258, b=72352549):
        """Solve a sum problem"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return b - a
    ```
    
    </details>
    
* <a name="intsub"></a>**IntSub**  (5 instances)
    
    ```python
    def sat(x: int, a=-382, b=14546310):
        return x - a == b
    
    def sol(a=-382, b=14546310):
        """Solve a subtraction problem"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return a + b
    ```
    
    </details>
    
* <a name="intsub2"></a>**IntSub2**  (5 instances)
    
    ```python
    def sat(x: int, a=8665464, b=-93206):
        return a - x == b
    
    def sol(a=8665464, b=-93206):
        """Solve a subtraction problem"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return a - b
    ```
    
    </details>
    
* <a name="intmul"></a>**IntMul**  (5 instances)
    
    ```python
    def sat(n: int, a=14302, b=5):
        return b * n + (a % b) == a
    
    def sol(a=14302, b=5):
        """Solve a multiplication problem"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return a // b
    ```
    
    </details>
    
* <a name="intdiv"></a>**IntDiv**  (5 instances)
    
    ```python
    def sat(n: int, a=3, b=23463462):
        return b // n == a
    
    def sol(a=3, b=23463462):
        """Solve a division problem"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        if a == 0:
            return 2 * b
        for n in [b // a, b // a - 1, b // a + 1]:
            if b // n == a:
                return n
    ```
    
    </details>
    
* <a name="intdiv2"></a>**IntDiv2**  (5 instances)
    
    ```python
    def sat(n: int, a=345346363, b=10):
        return n // b == a
    
    def sol(a=345346363, b=10):
        """Find n that when divided by b is a"""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return a * b
    ```
    
    </details>
    
* <a name="intsquareroot"></a>**IntSquareRoot**  (5 instances)
    
    ```python
    def sat(x: int, a=10201202001):
        return x ** 2 == a
    
    def sol(a=10201202001):
        """Compute an integer that when squared equals perfect-square a."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return int(a ** 0.5)
    ```
    
    </details>
    
* <a name="intnegsquareroot"></a>**IntNegSquareRoot**  (5 instances)
    
    ```python
    def sat(n: int, a=10000200001):
        return a == n * n and n < 0
    
    def sol(a=10000200001):
        """Find a negative integer that when squared equals perfect-square a."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return -int(a ** 0.5)
    ```
    
    </details>
    
* <a name="floatsquareroot"></a>**FloatSquareRoot**  (5 instances)
    
    ```python
    def sat(x: float, a=1020):
        return abs(x ** 2 - a) < 10 ** -3
    
    def sol(a=1020):
        """Find a number that when squared is close to a."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return a ** 0.5
    ```
    
    </details>
    
* <a name="floatnegsquareroot"></a>**FloatNegSquareRoot**  (5 instances)
    
    ```python
    def sat(x: float, a=1020):
        return abs(x ** 2 - a) < 10 ** -3 and x < 0
    
    def sol(a=1020):
        """Find a negative number that when squared is close to a."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return -a ** 0.5
    ```
    
    </details>
    
## tutorial


A few example puzzles that were presented with solutions to participants of the study.


* <a name="tutorial1"></a>**Tutorial1**  (1 instance)
    
    ```python
    def sat(s: str):
        return "Hello " + s == "Hello world"
    
    def sol():
        """Find a string that when concatenated onto 'Hello ' gives 'Hello world'."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "world"
    ```
    
    </details>
    
* <a name="tutorial2"></a>**Tutorial2**  (1 instance)
    
    ```python
    def sat(s: str):
        return "Hello " + s[::-1] == "Hello world"
    
    def sol():
        """Find a string that when reversed and concatenated onto 'Hello ' gives 'Hello world'."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return "world"[::-1]
    ```
    
    </details>
    
* <a name="tutorial3"></a>**Tutorial3**  (1 instance)
    
    ```python
    def sat(x: List[int]):
        return len(x) == 2 and sum(x) == 3
    
    def sol():
        """Find a list of two integers whose sum is 3."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return [1, 2]
    ```
    
    </details>
    
* <a name="tutorial4"></a>**Tutorial4**  (1 instance)
    
    ```python
    def sat(s: List[str]):
        return len(set(s)) == 1000 and all((x.count("a") > x.count("b")) and ('b' in x) for x in s)
    
    def sol():
        """Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b'."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return ["a" * (i + 2) + "b" for i in range(1000)]
    ```
    
    </details>
    
* <a name="tutorial5"></a>**Tutorial5**  (1 instance)
    
    ```python
    def sat(n: int):
        return str(n * n).startswith("123456789")
    
    def sol():
        """Find an integer whose perfect square begins with 123456789 in its decimal representation."""
    ```
    <details><summary>1 hand-written solution </summary>
    
    ```python
        return int(int("123456789" + "0" * 9) ** 0.5) + 1
    ```
    
    </details>
    
