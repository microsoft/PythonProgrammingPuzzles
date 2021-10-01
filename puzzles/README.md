# Summary of Puzzles and codex solutions
We ran OpenAI's recently released [codex](https://arxiv.org/abs/2107.03374) API on the problems for 10,000 attempts per problem. We used a 5-shot prompt based on the Tutorial (see our [paper](https://arxiv.org/abs/2106.05784)). We report the number of successes and the shortest and longest valid solutions below.
This document summarizes the dataset stored in the `puzzles.json` file in this directory. 
These files are generated from the `generators/*.py` files.
The only import for puzzles is `from typing import List` but you should also pass a candidate solution 
through `type_check` from `puzzle_generator.py` before certifying correctness. 

## Puzzles by module: <!-- descriptions come from the module docstring --> 

- [study.py](../generators/study.py), [30 problems](#study): Puzzles used for the study
- [classic_puzzles.py](../generators/classic_puzzles.py), [22 problems](#classic_puzzles): Classic puzzles
- [human_eval.py](../generators/human_eval.py), [107 problems](#human_eval): Problems inspired by [HumanEval dataset](https://github.com/openai/human-eval) described
in the [codex paper](https://arxiv.org/abs/2107.03374), specifically,
[this](https://github.com/openai/human-eval/blob/fa06031e684fbe1ee429c7433809460c159b66ad/data/HumanEval.jsonl.gz)
version released 7/7/21
- [codeforces.py](../generators/codeforces.py), [45 problems](#codeforces): Problems inspired by the popular programming competition site [codeforces.com](https://codeforces.com)
- [algebra.py](../generators/algebra.py), [4 problems](#algebra): Roots of polynomials
- [basic.py](../generators/basic.py), [22 problems](#basic): Problems testing basic knowledge -- easy to solve if you understand what is being asked
- [chess.py](../generators/chess.py), [5 problems](#chess): Classic chess puzzles
- [compression.py](../generators/compression.py), [3 problems](#compression): Puzzles relating to de/compression
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

Total (337 problems, 2,781 instances)


----

The puzzles in each module are sorted by number of codex solutions



## study


Puzzles used for the study.


* <a name="study_24"></a>**Study_24**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        """Find a list of integers with exactly three occurrences of seventeen and at least two occurrences of three."""
        return li.count(17) == 3 and li.count(3) >= 2
    ```
    <details><summary>4,867 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return [3,17]*3
    
    ```
    
    ```python
    def sol():
        """Generate list of five consecutive integers which each have either exactly 17 or exactly 3 as the number of occurrences of their first digit."""
        li = []
        for i in range(5):
            x = i
            while x%10 != 3 and x%10 != 7: x += 1
            li.append(x)
        return [17]*3 + [3]*2
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return [17] * 3 + [3] * 2
    ```
    
    </details>
    
* <a name="study_26"></a>**Study_26**  (1 instance)
    
    ```python
    def sat(ls: List[str]):
        """Divide the decimal representation of 8^88 up into strings of length eight."""
        return "".join(ls) == str(8 ** 88) and all(len(s) == 8 for s in ls)
    ```
    <details><summary>2,290 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return [str(8**88)[i*8:i*8+8] for i in range(10)]
    
    ```
    
    ```python
    def sol():
        x = 8 ** 88
        d = str(x)
        m = len(d) % 8
        ls = []
        if m != 0:
            ls.append(d[:m])
            ls.extend([d[m+8*i:m+8*(i+1)] for i in range((len(d)-m) // 8)])
        else:
            ls.extend([d[8*i:8*(i+1)] for i in range(len(d) // 8)])
        return ls
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return [str(8 ** 88)[i:i + 8] for i in range(0, len(str(8 ** 88)), 8)]
    ```
    
    </details>
    
* <a name="study_8"></a>**Study_8**  (1 instance)
    
    ```python
    def sat(ls: List[str]):
        """Find a list of more than 1235 strings such that the 1234th string is a proper substring of the 1235th."""
        return ls[1234] in ls[1235] and ls[1234] != ls[1235]
    ```
    <details><summary>1,007 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return ['a', 'ab'] * 1000
    
    ```
    
    ```python
    def sol():
        # Generate a list of strings such that each string is a proper substring of a string in the last 1000 items
        # in this list.
        result = ["_" + i*"a" for i in range(1000)]
        for i in range(1234):
            result.append(result[i])
        # Now append the "1234th" string.
        result.append("_" + (1234*"a"))
        # Now append the "1234th" string again.
        result.append("_" + (1234*"a"))
        return result
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return [''] * 1235 + ['a']
    ```
    
    </details>
    
* <a name="study_4"></a>**Study_4**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        """Find a list of length 10 where the fourth element occurs exactly twice."""
        return len(li) == 10 and li.count(li[3]) == 2
    ```
    <details><summary>822 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return list(range(5))*2
    
    ```
    
    ```python
    def sol():
        # Note that for this test case, a randomly generated list won't work. I've chosen the ordering 
        # to be slightly less than optimal, in order to increase the chances of failing this test case.
        return [1, 1, 2, 0, 0, 2, 1, 2, 2, 2]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return list(range(10 // 2)) * 2
    ```
    
    </details>
    
* <a name="study_6"></a>**Study_6**  (1 instance)
    
    ```python
    def sat(i: int):
        """Find an integer greater than 10^10 which is 4 mod 123."""
        return i % 123 == 4 and i > 10 ** 10
    ```
    <details><summary>785 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return 13 ** 12
    
    ```
    
    ```python
    def sol():
        m = 900719925474055 # 5 * 10^10
        ll = 50 # initial lower_limit
        ul = 50*m # initial upper_limit
    
        while True:
            mid = (ll + ul) // 2
            if mid * mid * ul > m*m*ll:
                ll = mid
            else:
                ul = mid
            if ll == ul or ll == ul - 1:
                return ll % 123 if ll < m else ll
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return 4 + 10 ** 10 + 123 - 10 ** 10 % 123
    ```
    
    </details>
    
* <a name="study_14"></a>**Study_14**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        """Find a list of integers such that the sum of the first i integers is i, for i=0, 1, 2, ..., 19."""
        return all([sum(li[:i]) == i for i in range(20)])
    ```
    <details><summary>516 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return [1]*20
    
    ```
    
    ```python
    def sol():
        ret = []
        i = 1; j = 0
        while len(ret) != 20:
            if i > 9:
                j += 1; i = 0
            if i == j:
                ret.append(i)
                continue
            if j % 10 == 9:
                ret.append(i)
            i += 1
        return ret
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return [1] * 20
    ```
    
    </details>
    
* <a name="study_28"></a>**Study_28** 9/15/2021: updated to a list since sets were removed from puzzle formats (1 instance)
    
    ```python
    def sat(li: List[int]):
        """Find a list of one hundred integers between 0 and 999 which all differ by at least ten from one another."""
        return all(i in range(1000) and abs(i - j) >= 10 for i in li for j in li if i != j) and len(set(li)) == 100
    ```
    <details><summary>462 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return [*range(0, 1000, 10)]
    
    ```
    
    ```python
    def sol():
        li = list(range(0, 999, 10))
        n = 10000
        i = n
        j = n
        while len(li) < 100:
            if sat([li[i % n] + (i // n)]):
                li.append(li[i % n] + (i // n))
                i = n
                n *= 10
            if sat([li[j % n] - (j // n)]):
                li.append(li[j % n] - (j // n))
                j = n
                n *= 10
        return li
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return list(range(0, 1000, 10))
    ```
    
    </details>
    
* <a name="study_15"></a>**Study_15**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        """Find a list of integers such that the sum of the first i integers is 2^i -1, for i = 0, 1, 2, ..., 19."""
        return all(sum(li[:i]) == 2 ** i - 1 for i in range(20))
    ```
    <details><summary>425 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return [1<<i for i in range(20)]
    
    ```
    
    ```python
    def sol():
        result = []
        prev_mod = 1
        for i in range(20):
            new_mod = (prev_mod * 2) % 1000000000
            result.append((new_mod - prev_mod) % 1000000000)
            prev_mod = new_mod
        return result
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return [(2 ** i) for i in range(20)]
    ```
    
    </details>
    
* <a name="study_9"></a>**Study_9**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        """
        Find a way to rearrange the letters in the pangram "The quick brown fox jumps over the lazy dog" to get
        the pangram "The five boxing wizards jump quickly". The answer should be represented as a list of index
        mappings.
        """
        return ["The quick brown fox jumps over the lazy dog"[i] for i in li] == list(
            "The five boxing wizards jump quickly")
    ```
    <details><summary>307 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return list("The quick brown fox jumps over the lazy dog".find(x) for x in "The five boxing wizards jump quickly")
    
    ```
    
    ```python
    def sol():
        def pangramcmp(n):
            "Compares characters in the pangram to the characters in the solution"
            _pangramcmp = {
                "The quick brown fox jumps over the lazy dog":     "The five boxing wizards jump quickly",
                "The five boxing wizards jump quickly":            "The quick brown fox jumps over the lazy dog"
            }
            return _pangramcmp[n]
        return [
            "The quick brown fox jumps over the lazy dog".find(x)
            for x in "The five boxing wizards jump quickly"
        ]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return ['The quick brown fox jumps over the lazy dog'.index(t)
                for t in 'The five boxing wizards jump quickly']
    ```
    
    </details>
    
* <a name="study_21"></a>**Study_21**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        """
        Find a list integers containing exactly three distinct values, such that no integer repeats
        twice consecutively among the first eleven entries. (So the list needs to have length greater than ten.)
        """
        return all([li[i] != li[i + 1] for i in range(10)]) and len(set(li)) == 3
    ```
    <details><summary>301 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return [1, 2, 3]*4
    
    ```
    
    ```python
    def sol():
        return [i for i in range(3)] + [i for i in range(3)] + [i for i in range(3)] + [i for i in range(3)] + [i for i in range(3)] + [i for i in range(3)] + [i for i in range(3)] + [i for i in range(3)] + [i for i in range(3)] + [i for i in range(3)] + [i for i in range(3)]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return list(range(3)) * 10
    ```
    
    </details>
    
* <a name="study_5"></a>**Study_5**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        """Find a list integers such that the integer i occurs i times, for i = 0, 1, 2, ..., 9."""
        return all([li.count(i) == i for i in range(10)])
    ```
    <details><summary>257 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return [i for j in range(10) for i in [j]*j]
    
    ```
    
    ```python
    def sol():
        """One possible implementation of g6 is to generate concatenated strings of 0's and 1's then convert those to integers."""
        return list(map(int, ("".join([str(i) * i for i in range(10)]))))
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return [i for i in range(10) for j in range(i)]
    ```
    
    </details>
    
* <a name="study_20"></a>**Study_20** A more interesting version of this puzzle with a length constraint is ShortIntegerPath in graphs.py (1 instance)
    
    ```python
    def sat(li: List[int]):
        """
        Find a list of integers, starting with 0 and ending with 128, such that each integer either differs from
        the previous one by one or is thrice the previous one.
        """
        return all(j in {i - 1, i + 1, 3 * i} for i, j in zip([0] + li, li + [128]))
    ```
    <details><summary>191 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return list(range(128))
    
    ```
    
    ```python
    def sol():
        """
        Python v3 solution
        (found this at http://code.activestate.com/recipes/579107-one-liner-to-generate-integer-list-starts-with-0-and-1/)
    
        I originally tried to solve this without using any lookup table.  I changed it to use a lookup table after
        reading the solution.
        """
        return [i for d, i in enumerate(range(128)) if i - d not in [1, 2, 3 * d]];
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return [1, 3, 4, 12, 13, 14, 42, 126, 127]
    ```
    
    </details>
    
* <a name="study_23"></a>**Study_23**  (1 instance)
    
    ```python
    def sat(ls: List[str]):
        """
        Find a list of characters which are aligned at the same indices of the three strings 'dee', 'doo', and 'dah!'.
        """
        return tuple(ls) in zip('dee', 'doo', 'dah!')
    ```
    <details><summary>170 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return ['d']*3
    
    ```
    
    ```python
    def sol():
        s1, s2, s3 = 'dee', 'doo', 'dah!'
        zipped = zip(s1, s2, s3)
        initial = list(next(zipped))
        for x, y, z in zipped:
            if x in initial:
                index = initial.index(x)
                initial[index] = '0'
                if y in initial:
                    index = initial.index(y)
                    initial[index] = '1'
                else:
                    initial[index] = '2'
        return initial
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return list(next(zip('dee', 'doo', 'dah!')))
    ```
    
    </details>
    
* <a name="study_13"></a>**Study_13**  (1 instance)
    
    ```python
    def sat(x: float):
        """Find a real number which, when you subtract 3.1415, has a decimal representation starting with 123.456."""
        return str(x - 3.1415).startswith("123.456")
    ```
    <details><summary>148 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return 3.1415 + 123.456
    
    ```
    
    ```python
    def sol():
        epsilon = 0.001
        x = 3.1415
        while True:
            xplus = x + epsilon
            if str(xplus - 3.1415).startswith("123.456"):
                return xplus
            else:
                x = xplus
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return 123.456 + 3.1415
    ```
    
    </details>
    
* <a name="study_18"></a>**Study_18**  (1 instance)
    
    ```python
    def sat(ls: List[str]):
        """
        Find a list of strings that when you combine them in all pairwise combinations gives the six strings:
        'berlin', 'berger', 'linber', 'linger', 'gerber', 'gerlin'
        """
        return [s + t for s in ls for t in ls if s != t] == 'berlin berger linber linger gerber gerlin'.split()
    ```
    <details><summary>124 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return ["ber","lin","ger"]
    
    ```
    
    ```python
    def sol():
        return ["ber", "lin", "ger"]
        
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
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
    
* <a name="study_3"></a>**Study_3**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        """Find a permutation of [0, 1, ..., 998] such that the ith element is *not* i, for all i=0, 1, ..., 998."""
        return sorted(li) == list(range(999)) and all(li[i] != i for i in range(len(li)))
    ```
    <details><summary>88 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return [*range(1, 999), 0]
    
    ```
    
    ```python
    def sol():
        a = list(range(999))
        for i in range(999):
            k = -1
            for j in range(i, 999):
                if a[j] > i:
                    k = j
                    break
            a[i], a[k] = a[k], a[i]
        return a
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return [((i + 1) % 999) for i in range(999)]
    ```
    
    </details>
    
* <a name="study_27"></a>**Study_27**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        """
        Consider a digraph where each node has exactly one outgoing edge. For each edge (u, v), call u the parent and
        v the child. Then find such a digraph where the grandchildren of the first and second nodes differ but they
        share the same great-grandchildren. Represented this digraph by the list of children indices.
        """
        return li[li[0]] != li[li[1]] and li[li[li[0]]] == li[li[li[1]]]
    ```
    <details><summary>82 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return [1, -1, 2]
    
    ```
    
    ```python
    def sol():
        # first create a small digraph with two nodes
        li = [1, -1]
        i = 2
        # then extend the digraph to a larger digraph with a fixed number of nodes
        # such that the grandchildren of the two nodes in the original digraph are different but share the same
        # great-grandchildren
        while len(li) < 1000:
            li.append(i)
            i += 1
        return li[::]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return [1, 2, 3, 3]
    ```
    
    </details>
    
* <a name="study_30"></a>**Study_30**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        """
        Define f(n) to be the residue of 123 times n mod 1000. Find a list of integers such that the first twenty one
        are between 0 and 999, inclusive, and are strictly increasing in terms of f(n).
        """
        return all([123 * li[i] % 1000 < 123 * li[i + 1] % 1000 and li[i] in range(1000) for i in range(20)])
    ```
    <details><summary>65 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return sorted(range(1000), key = lambda x: 123*x % 1000)
    
    ```
    
    ```python
    def sol():
        s = set()
        for i in range(21):
            v = 123 * i % 1000
            for n in s:
                if v == n:
                    v = n + 1
                    while v in s:
                        v += 1
            s.add(v)
        return sorted(s, key=lambda n: 123 * n % 1000)[:21]
    
    ```
    
    </details>
    
    <details><summary>2 hand-written solutions </summary>
    
    ```python
    def sol():
        return sorted(range(1000), key=lambda n: 123 * n % 1000)[:21]
    ```
    
    ```python
    def sol():
        return list(range(1000))[::8][::-1]
    ```
    
    </details>
    
* <a name="study_16"></a>**Study_16**  (1 instance)
    
    ```python
    def sat(s: str):
        """Find a real number such that when you add the length of its decimal representation to it, you get 4.5.
        Your answer should be the string form of the number in its decimal representation."""
        return float(s) + len(s) == 4.5
    ```
    <details><summary>47 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return "1.5"
    
    ```
    
    ```python
    def sol():
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    q = i + 0.1*j + 0.01*k
                    if q + len("%s" % q) == 4.5:
                        return "%s" % q
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return str(4.5 - len(str(4.5)))
    ```
    
    </details>
    
* <a name="study_7"></a>**Study_7**  (1 instance)
    
    ```python
    def sat(s: str):
        """Find a three-digit pattern  that occurs more than 8 times in the decimal representation of 8^2888."""
        return str(8 ** 2888).count(s) > 8 and len(s) == 3
    ```
    <details><summary>44 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        c = 0
        while not sat(str(c)):
            c += 1
        return str(c)
    
    ```
    
    ```python
    def sol():
        i = 0
        while True:
            s = str(100 * int("1" + str(i).zfill(3)) ** 2)
            if s[2] == s[5]:
                if sat(s[2:5]):
                    return s[2:5]
                if s[1] == s[4]:
                    if sat(s[1:4]):
                        return s[1:4]
                    if s[0] == s[3]:
                        if sat(s[0:3]):
                            return s[0:3]
            i += 1
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        s = str(8 ** 2888)
        return max({s[i: i + 3] for i in range(len(s) - 2)}, key=lambda t: s.count(t))
    ```
    
    </details>
    
* <a name="study_11"></a>**Study_11**  (1 instance)
    
    ```python
    def sat(ls: List[str]):
        """
        Find a list of strings whose length (viewed as a string) is equal to the lexicographically largest element
        and is equal to the lexicographically smallest element.
        """
        return min(ls) == max(ls) == str(len(ls))
    ```
    <details><summary>24 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return ["1"]
    
    ```
    
    ```python
    def sol():
        l = ""
        for i in range(10000):
            if str(i) == str(i * i)[::-1]:
                l += str(i) * i
        return [l[i:i+12] for i in range(0, len(l), 12)]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return ['1']
    ```
    
    </details>
    
* <a name="study_10"></a>**Study_10**  (1 instance)
    
    ```python
    def sat(s: str):
        """Find a palindrome of length greater than 11 in the decimal representation of 8^1818."""
        return s in str(8 ** 1818) and s == s[::-1] and len(s) > 11
    ```
    <details><summary>16 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        x = str(8 ** 1818)
        return [x[i:(i+13)] for i in range(len(x)-11) if x[i:(i+13)] == x[i:(i+13)][::-1]][0]
    
    ```
    
    ```python
    def sol():
        s = str(8 ** 1818)
        n = len(s)
    
        pal = set()
        for i in range(11, 55):
            for j in range(n-i+1):
                pal.add(s[j:j+i])
    
        for p in pal:
            if p == p[::-1] and len(p) > 11:
                return p
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        s = str(8 ** 1818)
        return next(s[i: i + le]
                    for le in range(12, len(s) + 1)
                    for i in range(len(s) - le + 1)
                    if s[i: i + le] == s[i: i + le][::-1]
                    )
    ```
    
    </details>
    
* <a name="study_1"></a>**Study_1**  (1 instance)
    
    ```python
    def sat(s: str):
        """Find a string with 1000 'o's but no two adjacent 'o's."""
        return s.count('o') == 1000 and s.count('oo') == 0
    ```
    <details><summary>15 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return "'o' "*1000
    
    ```
    
    ```python
    def sol():
        s = ["o" for _ in range(1000)]
        for i in range(999):
            s[i] = s[i] + " "
        return "".join(s)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return ('h' + 'o') * 1000
    ```
    
    </details>
    
* <a name="study_2"></a>**Study_2**  (1 instance)
    
    ```python
    def sat(s: str):
        """Find a string with 1000 'o's, 100 pairs of adjacent 'o's and 801 copies of 'ho'."""
        return s.count('o') == 1000 and s.count('oo') == 100 and s.count('ho') == 801
    ```
    <details><summary>9 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return 'ho' * 701 + 'o' * 199 + 'ho' * 100
    
    ```
    
    ```python
    def sol():
        a = ['o' for i in range(100)]
        b = ['ho' for i in range(801)]
        c = ['o' for i in range(1000 - 100 - 801)]
        return ''.join(a + b + c)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return 'ho' * (800 + 1) + 'o' * (100 * 2 - 1)
    ```
    
    </details>
    
* <a name="study_29"></a>**Study_29** 9/15/2021: updated to a list since sets were removed from puzzle formats (1 instance)
    
    ```python
    def sat(l: List[int]):
        """
        Find a list of more than 995 distinct integers between 0 and 999, inclusive, such that each pair of integers
        have squares that differ by at least 10.
        """
        return all(i in range(1000) and abs(i * i - j * j) >= 10 for i in l for j in l if i != j) and len(set(l)) > 995
    ```
    <details><summary>9 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        l = []
        for i in range(1000):
            if all(abs(i*i - j*j) >= 10 for j in l):
                l.append(i)
        return l
    
    ```
    
    ```python
    def sol():
        l = [0]
        while True:
            for i in range(1000):
                if i in l:
                    continue
                if all(abs(j*j - i*i) >= 10 for j in l):
                    l.append(i)
                if len(l) > 995:
                    return l
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return [0, 4] + list(range(6, 1000))
    ```
    
    </details>
    
* <a name="study_12"></a>**Study_12**  (1 instance)
    
    ```python
    def sat(li: List[int]):
        """Find a list of 1,000 integers where every two adjacent integers sum to 9, and where the first
        integer plus 4 is 9."""
        return all(i + j == 9 for i, j in zip([4] + li, li)) and len(li) == 1000
    ```
    <details><summary>1 AI solution from codex</summary>
    
    ```python
    def sol():
        return [5 if i % 2 == 0 else 4 for i in range(1000)]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return [9 - 4, 4] * (1000 // 2)
    ```
    
    </details>
    
* <a name="study_19"></a>**Study_19** 9/15/2021 Updated to take a list rather than a set because it was the only puzzle in the repo with Set argument. (1 instance)
    
    ```python
    def sat(li: List[int]):
        """
        Find a list of integers whose pairwise sums make the set {0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 34}.
        That is find L such that, { i + j | i, j in L } = {0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 34}.
        """
        return {i + j for i in li for j in li} == {0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 34}
    ```
    <details><summary>1 AI solution from codex</summary>
    
    ```python
    def sol():
        return [1, 0, 1, 2, 2, 3, 17, 3]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return [0, 1, 2, 3, 17]
    ```
    
    </details>
    
* <a name="study_17"></a>**Study_17**  (1 instance)
    
    ```python
    def sat(i: int):
        """Find a number whose decimal representation is *a longer string* when you add 1,000 to it than when you add 1,001."""
        return len(str(i + 1000)) > len(str(i + 1001))
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return -1001
    ```
    
    </details>
    
* <a name="study_22"></a>**Study_22**  (1 instance)
    
    ```python
    def sat(s: str):
        """
        Find a string s containing exactly five distinct characters which also contains as a substring every other
        character of s (e.g., if the string s were 'parrotfish' every other character would be 'profs').
        """
        return s[::2] in s and len(set(s)) == 5
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return """abacadaeaaaaaaaaaa"""
    ```
    
    </details>
    
* <a name="study_25"></a>**Study_25**  (1 instance)
    
    ```python
    def sat(s: str):
        """Find a permutation of the string 'Permute me true' which is a palindrome."""
        return sorted(s) == sorted('Permute me true') and s == s[::-1]
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        s = sorted('Permute me true'[1:])[::2]
        return "".join(s + ['P'] + s[::-1])
    ```
    
    </details>
    
## classic_puzzles

Classic puzzles


* <a name="necklacesplit"></a>**NecklaceSplit** [Necklace Splitting Problem](https://en.wikipedia.org/wiki/Necklace_splitting_problem) (10 instances)
    
    ```python
    def sat(n: int, lace="bbrbrbbbbbbrrrrrrrbrrrrbbbrbrrbbbrbrrrbrrbrrbrbbrrrrrbrbbbrrrbbbrbbrbbbrbrbb"):
        """
        Find a split dividing the given red/blue necklace in half at n so that each piece has an equal number of
        reds and blues.
        """
        sub = lace[n: n + len(lace) // 2]
        return n >= 0 and lace.count("r") == 2 * sub.count("r") and lace.count("b") == 2 * sub.count("b")
    ```
    <details><summary>1,502 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(lace="bbrbrbbbbbbrrrrrrrbrrrrbbbrbrrbbbrbrrrbrrbrrbrbbrrrrrbrbbbrrrbbbrbbrbbbrbrbb"):
        return len(lace)//2 - 1
    
    ```
    
    ```python
    def sol(lace="bbrbrbbbbbbrrrrrrrbrrrrbbbrbrrbbbrbrrrbrrbrrbrbbrrrrrbrbbbrrrbbbrbbrbbbrbrbb"):
        """Red/blue split of given necklace, where each piece has equal number of red and blue"""
        # Proof: each piece can be done recursively:
        #   * first return the other color and the other color and the other color and so on
        #   * then return red and red and red and ...
        #   * the pieces are the same color bc they are opposites of each other, with halved counts of the
        #     color each piece started with.
        l = len(lace)
        for i in range(l):
            if sat(i, lace):
                return i
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(lace="bbrbrbbbbbbrrrrrrrbrrrrbbbrbrrbbbrbrrrbrrbrrbrbbrrrrrbrbbbrrrbbbrbbrbbbrbrbb"):
        if lace == "":
            return 0
        return next(n for n in range(len(lace) // 2) if lace[n: n + len(lace) // 2].count("r") == len(lace) // 4)
    ```
    
    </details>
    
* <a name="quine"></a>**Quine** [Quine](https://en.wikipedia.org/wiki/Quine_%28computing%29) (1 instance)
    
    ```python
    def sat(quine: str):
        """Find a string that when evaluated as a Python expression is that string itself."""
        return eval(quine) == quine
    ```
    <details><summary>615 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return "sol()"
    
    ```
    
    ```python
    def sol():
        # Don't use "return quine", as this would cause the surrounding `eval` to be triggered.
        return "quine"
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return "(lambda x: f'({x})({chr(34)}{x}{chr(34)})')(\"lambda x: f'({x})({chr(34)}{x}{chr(34)})'\")"
    ```
    
    </details>
    
* <a name="clockangle"></a>**ClockAngle** [Clock Angle Problem](https://en.wikipedia.org/wiki/Clock_angle_problem), easy variant (10 instances)
    
    ```python
    def sat(hands: List[int], target_angle=45):
        """Find clock hands = [hour, min] such that the angle is target_angle degrees."""
        h, m = hands
        assert 0 < h <= 12 and 0 <= m < 60
        hour_angle = 30 * h + m / 2
        minute_angle = 6 * m
        return abs(hour_angle - minute_angle) in [target_angle, 360 - target_angle]
    ```
    <details><summary>200 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(target_angle=45):
        return [7,30]
    
    ```
    
    ```python
    def sol(target_angle=45):
        hands = [3, 0]  # at 9:00
        if sat(hands, target_angle):
            return hands
        # Add an hour then try again
        hands[0] += 1
        for m in range(59):  # next minute
            hands[1] = m
            if sat(hands, target_angle):
                return hands
        hands[0] += 1  # next hour
        for h in range(11):
            for m in range(59):
                hands = [h, m]
                if sat(hands, target_angle):
                    return hands
        # Increment by an hour until it
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(target_angle=45):
        for h in range(1, 13):
            for m in range(60):
                hour_angle = 30 * h + m / 2
                minute_angle = 6 * m
                if abs(hour_angle - minute_angle) % 360 in [target_angle, 360 - target_angle]:
                    return [h, m]
    ```
    
    </details>
    
* <a name="easy63"></a>**Easy63** An easy puzzle to make 63 using two 8's and one 1's. (1 instance)
    
    ```python
    def sat(s: str):
        """Find a formula using two 8s and two 1's and -+*/ that evaluates to 1."""
        return set(s) <= set("18-+*/") and s.count("8") == 2 and s.count("1") == 1 and eval(s) == 63
    ```
    <details><summary>79 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return "8*8-1"
    
    ```
    
    ```python
    def sol():
        for op1 in "-", "+", "/", "*":
            for op2 in "-", "+", "/", "*":
                for x1 in "1", "8":
                    for x2 in "1", "8":
                        n = x1 + op1 + x2 + op2 + "1"
                        if sat(n): return n
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return "8*8-1"
    ```
    
    </details>
    
* <a name="pandigitalsquare"></a>**PandigitalSquare** [Pandigital](https://en.wikipedia.org/wiki/Pandigital_number) Square (1 instance)
    
    ```python
    def sat(n: int):
        """Find an integer whose square has all digits 0-9 once."""
        s = str(n * n)
        for i in "0123456789":
            assert s.count(i) == 1
        return True
    ```
    <details><summary>32 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        i = int("0"*10) + 1
        while not (set(str(i*i)) == set("0123456789")):
            i += 1
        return i
    
    ```
    
    ```python
    def sol():
        n = 1
        while True:
            n += 1
            x = str(n * n)
            if x.count("0") == 1 and x.count("1") == 1 and x.count("2") == 1 and x.count("3") == 1 and x.count("4") == 1 and x.count("5") == 1 and x.count("6") == 1 and x.count("7") == 1 and x.count("8") == 1 and x.count("9") == 1:
                return n
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        for n in range(10 ** 5):
            if sorted([int(s) for s in str(n * n)]) == list(range(10)):
                return n
    ```
    
    </details>
    
* <a name="verbalarithmetic"></a>**VerbalArithmetic** Find a substitution of digits for characters to make the numbers add up in a sum like this:
    SEND + MORE = MONEY
    
    The first digit in any number cannot be 0. In this example the solution is `9567 + 1085 = 10652`.
    See [Wikipedia article](https://en.wikipedia.org/wiki/Verbal_arithmetic) (10 instances)
    
    ```python
    def sat(li: List[int], words=['SEND', 'MORE', 'MONEY']):
        """
        Find a list of integers corresponding to the given list of strings substituting a different digit for each
        character, so that the last string corresponds to the sum of the previous numbers.
        """
        assert len(li) == len(words) and all(i > 0 and len(str(i)) == len(w) for i, w in zip(li, words))
        assert len({c for w in words for c in w}) == len({(d, c) for i, w in zip(li, words) for d, c in zip(str(i), w)})
        return sum(li[:-1]) == li[-1]
    ```
    <details><summary>28 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(words=['SEND', 'MORE', 'MONEY']):
        return [9567, 1085, 10652]
    
    ```
    
    ```python
    def sol(words=['SEND', 'MORE', 'MONEY']):
        """
        The given list of words consists of three distinct words 'SEND', 'MORE', and 'MONEY' and we have to
        find the corresponding list of integers such that the digit substitutions make the last word correspond to the
        sum of the previous two words.
        """
        assert all(w in words for w in ['SEND', 'MORE', 'MONEY'])
        return [9567, 1085, 10652]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(words=['SEND', 'MORE', 'MONEY']):
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
    
* <a name="monkeyandcoconuts"></a>**MonkeyAndCoconuts** [The Monkey and the Coconuts](https://en.wikipedia.org/wiki/The_monkey_and_the_coconuts) (1 instance)
    
    ```python
    def sat(n: int):
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
        for i in range(5):
            assert n % 5 == 1
            n -= 1 + (n - 1) // 5
        return n > 0 and n % 5 == 1
    ```
    <details><summary>22 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return int("9"*99) * 5 + 1
    
    ```
    
    ```python
    def sol():
        n = 1000000000
        while True:
            n += 1
            try:
                if sat(n):
                    return n
            except AssertionError:
                pass
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
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
    
* <a name="postagestamp"></a>**PostageStamp** [Postage stamp problem](https://en.wikipedia.org/wiki/Postage_stamp_problem) (10 instances)
    
    ```python
    def sat(stamps: List[int], target=80, max_stamps=4, options=[10, 32, 8]):
        """Find a selection of at most max_stamps stamps whose total worth is the target value."""
        for s in stamps:
            assert s in options
        return len(stamps) <= max_stamps and sum(stamps) == target
    ```
    <details><summary>8 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(target=80, max_stamps=4, options=[10, 32, 8]):
        return [32, 32, 8, 8]
    
    ```
    
    ```python
    def sol(target=80, max_stamps=4, options=[10, 32, 8]):
        stamps = []
        current = target
        remaining = max_stamps
        while current > 0:
            # Include largest possible stamp
            if current >= max(options):
                stamps.append(max(options))
                current -= max(options)
            # Include other stamps
            else:
                for option in reversed(options):
                    if current - option >= 0:
                        stamps.append(option)
                        current -= option
                        break
        return stamps
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(target=80, max_stamps=4, options=[10, 32, 8]):
        from itertools import combinations_with_replacement
        for n in range(max_stamps + 1):
            for c in combinations_with_replacement(options, n):
                if sum(c) == target:
                    return list(c)
    ```
    
    </details>
    
* <a name="towersofhanoi"></a>**TowersOfHanoi** [Towers of Hanoi](https://en.wikipedia.org/w/index.php?title=Tower_of_Hanoi)
    
    In this classic version one must move all 8 disks from the first to third peg. (1 instance)
    
    ```python
    def sat(moves: List[List[int]]):
        """
        Eight disks of sizes 1-8 are stacked on three towers, with each tower having disks in order of largest to
        smallest. Move [i, j] corresponds to taking the smallest disk off tower i and putting it on tower j, and it
        is legal as long as the towers remain in sorted order. Find a sequence of moves that moves all the disks
        from the first to last towers.
        """
        rods = ([8, 7, 6, 5, 4, 3, 2, 1], [], [])
        for [i, j] in moves:
            rods[j].append(rods[i].pop())
            assert rods[j][-1] == min(rods[j]), "larger disk on top of smaller disk"
        return rods[0] == rods[1] == []
    ```
    <details><summary>6 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        def move(n, i, j, k):
            if n == 1:
                return [[i, j]]
            return move(n-1, i, k, j) + [[i, j]] + move(n-1, k, j, i)
        return move(8, 0, 2, 1)
    
    ```
    
    ```python
    def sol():
        def hanoi(moves, num_disks, source, aux, target):
            if num_disks > 0:
                hanoi(moves, num_disks-1, source, target, aux)
                moves.append([source, target])
                hanoi(moves, num_disks-1, aux, source, target)
        num_disks = 8
        moves = []
        hanoi(moves, num_disks, 0, 1, 2)
        return moves
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        def helper(m, i, j):
            if m == 0:
                return []
            k = 3 - i - j
            return helper(m - 1, i, k) + [[i, j]] + helper(m - 1, k, j)
    
        return helper(8, 0, 2)
    ```
    
    </details>
    
* <a name="cardgame24"></a>**CardGame24** [24 Game](https://en.wikipedia.org/wiki/24_Game)
    
    In this game one is given four numbers from the range 1-13 (Ace-King) and one needs to combine them with
        + - * / (and parentheses)
    to make the number 24.
    The solution to this tricky example is `7 * (3 + 3 / 7)` (10 instances)
    
    ```python
    def sat(expr: str, nums=[3, 7, 3, 7]):
        """Find a formula with two 3's and two 7's and + - * / (and parentheses) that evaluates to 24."""
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
    ```
    <details><summary>2 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[3, 7, 3, 7]):
        return "((({}/{})+{})*{})".format(*nums)
    
    ```
    
    ```python
    def sol(nums=[3, 7, 3, 7]):
        return "(((   {} /  {}) +  {}) *  {})".format(*nums)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[3, 7, 3, 7]):
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
    
* <a name="towersofhanoiarbitrary"></a>**TowersOfHanoiArbitrary** [Towers of Hanoi](https://en.wikipedia.org/w/index.php?title=Tower_of_Hanoi)
    
    In this version one must transform a given source state to a target state. (10 instances)
    
    ```python
    def sat(moves: List[List[int]], source=[[0, 7], [4, 5, 6], [1, 2, 3, 8]], target=[[0, 1, 2, 3, 8], [4, 5], [6, 7]]):
        """
        A state is a partition of the integers 0-8 into three increasing lists. A move is pair of integers i, j in
        {0, 1, 2} corresponding to moving the largest number from the end of list i to list j, while preserving the
        order of list j. Find a sequence of moves that transform the given source to target states.
        """
        state = [s[:] for s in source]
    
        for [i, j] in moves:
            state[j].append(state[i].pop())
            assert state[j] == sorted(state[j])
    
        return state == target
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(source=[[0, 7], [4, 5, 6], [1, 2, 3, 8]], target=[[0, 1, 2, 3, 8], [4, 5], [6, 7]]):
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
    where the goal is to find a substring with characters in sorted order. (10 instances)
    
    ```python
    def sat(x: List[int], length=13, s="Dynamic programming solves this puzzle!!!"):
        """
        Remove as few characters as possible from s so that the characters of the remaining string are alphebetical.
        Here x is the list of string indices that have not been deleted.
        """
        return all(s[x[i]] <= s[x[i + 1]] and x[i + 1] > x[i] >= 0 for i in range(length - 1))
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(length=13, s="Dynamic programming solves this puzzle!!!"):  # O(N^2) method. Todo: add binary search solution which is O(n log n)
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
    
* <a name="longestmonotonicsubstringtricky"></a>**LongestMonotonicSubstringTricky** The same as the above problem, but with a twist! (10 instances)
    
    ```python
    def sat(x: List[int], length=20, s="Dynamic programming solves this classic job-interview puzzle!!!"):
        """Find the indices of the longest substring with characters in sorted order"""
        return all(s[x[i]] <= s[x[i + 1]] and x[i + 1] > x[i] for i in range(length - 1))
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(length=20, s="Dynamic programming solves this classic job-interview puzzle!!!"):  # O(N^2) method. Todo: add binary search solution which is O(n log n)
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
    
* <a name="revquine"></a>**RevQuine** Reverse [Quine](https://en.wikipedia.org/wiki/Quine_%28computing%29). The solution we give is from GPT3. (1 instance)
    
    ```python
    def sat(rev_quine: str):
        """Find a string that, when reversed and evaluated gives you back that same string."""
        return eval(rev_quine[::-1]) == rev_quine
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return "rev_quine"[::-1]  # thanks GPT-3!
    ```
    
    </details>
    
* <a name="booleanpythagoreantriples"></a>**BooleanPythagoreanTriples** [Boolean Pythagorean Triples Problem](https://en.wikipedia.org/wiki/Boolean_Pythagorean_triples_problem) (9 instances)
    
    ```python
    def sat(colors: List[int], n=100):
        """
        Color the first n integers with one of two colors so that there is no monochromatic Pythagorean triple.
        A monochromatic Pythagorean triple is a triple of numbers i, j, k such that i^2 + j^2 = k^2 that
        are all assigned the same color. The input, colors, is a list of 0/1 colors of length >= n.
        """
        assert set(colors) <= {0, 1} and len(colors) >= n
        squares = {i ** 2: colors[i] for i in range(1, len(colors))}
        return not any(c == d == squares.get(i + j) for i, c in squares.items() for j, d in squares.items())
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=100):
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
    
* <a name="kirkman"></a>**Kirkman** [Kirkman's problem](https://en.wikipedia.org/wiki/Kirkman%27s_schoolgirl_problem) (1 instance)
    
    ```python
    def sat(daygroups: List[List[List[int]]]):
        """
        Arrange 15 people into groups of 3 each day for seven days so that no two people are in the same group twice.
        """
        assert len(daygroups) == 7
        assert all(len(groups) == 5 and {i for g in groups for i in g} == set(range(15)) for groups in daygroups)
        assert all(len(g) == 3 for groups in daygroups for g in groups)
        return len({(i, j) for groups in daygroups for g in groups for i in g for j in g}) == 15 * 15
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
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
    
* <a name="no3colinear"></a>**No3Colinear** [No three-in-a-line](https://en.wikipedia.org/wiki/No-three-in-line_problem) (9 instances)
    
    ```python
    def sat(coords: List[List[int]], side=10, num_points=20):
        """Find num_points points in an side x side grid such that no three points are collinear."""
        for i1 in range(len(coords)):
            x1, y1 = coords[i1]
            assert 0 <= x1 < side and 0 <= y1 < side
            for i2 in range(i1):
                x2, y2 = coords[i2]
                for i3 in range(i2):
                    x3, y3 = coords[i3]
                    assert x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) != 0
        return len({(a, b) for a, b in coords}) == len(coords) >= num_points
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(side=10, num_points=20):
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
    
* <a name="squaringthesquare"></a>**SquaringTheSquare** [Squaring the square](https://en.wikipedia.org/wiki/Squaring_the_square)
    Wikipedia gives a minimal [solution with 21 squares](https://en.wikipedia.org/wiki/Squaring_the_square)
    due to Duijvestijn (1978). (1 instance)
    
    ```python
    def sat(xy_sides: List[List[int]]):
        """
        Partition a square into smaller squares with unique side lengths. A perfect squared path has distinct sides.
        xy_sides is a List of (x, y, side)
        """
        n = max(x + side for x, y, side in xy_sides)
        assert len({side for x, y, side in xy_sides}) == len(xy_sides) > 1
        for x, y, s in xy_sides:
            assert 0 <= y < y + s <= n and 0 <= x
            for x2, y2, s2 in xy_sides:
                assert s2 <= s or x2 >= x + s or x2 + s2 <= x or y2 >= y + s or y2 + s2 <= y
    
        return sum(side ** 2 for x, y, side in xy_sides) == n ** 2
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return [[0, 0, 50], [0, 50, 29], [0, 79, 33], [29, 50, 25], [29, 75, 4], [33, 75, 37], [50, 0, 35],
                [50, 35, 15], [54, 50, 9], [54, 59, 16], [63, 50, 2], [63, 52, 7], [65, 35, 17], [70, 52, 18],
                [70, 70, 42], [82, 35, 11], [82, 46, 6], [85, 0, 27], [85, 27, 8], [88, 46, 24], [93, 27, 19]]
    ```
    
    </details>
    
* <a name="allpandigitalsquares"></a>**AllPandigitalSquares** All [Pandigital](https://en.wikipedia.org/wiki/Pandigital_number) Squares (1 instance)
    
    ```python
    def sat(nums: List[int]):
        """Find all 174 integers whose 10-digit square has all digits 0-9 just once."""
        return [sorted([int(s) for s in str(n * n)]) for n in set(nums)] == [list(range(10))] * 174
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return [i for i in range(-10 ** 5, 10 ** 5) if sorted([int(s) for s in str(i * i)]) == list(range(10))]
    ```
    
    </details>
    
* <a name="harder63"></a>**Harder63** An harder puzzle to make 63 using three 8's and one 1's. (1 instance)
    
    ```python
    def sat(s: str):
        """Find an expression using two 8s and two 1's and -+*/ that evaluates to 1."""
        return set(s) <= set("18-+*/") and s.count("8") == 3 and s.count("1") == 1 and eval(s) == 63
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return "8*8-1**8"
    ```
    
    </details>
    
* <a name="waterpouring"></a>**WaterPouring** [Water pouring puzzle](https://en.wikipedia.org/w/index.php?title=Water_pouring_puzzle&oldid=985741928) (10 instances)
    
    ```python
    def sat(moves: List[List[int]], capacities=[8, 5, 3], init=[8, 0, 0], goal=[4, 4, 0]):
        """
        Given an initial state of water quantities in jugs and jug capacities, find a sequence of moves (pouring
        one jug into another until it is full or the first is empty) to reaches the given goal state.
        moves is list of [from, to] pairs
        """
        state = init.copy()
    
        for [i, j] in moves:
            assert min(i, j) >= 0, "Indices must be non-negative"
            assert i != j, "Cannot pour from same state to itself"
            n = min(capacities[j], state[i] + state[j])
            state[i], state[j] = state[i] + state[j] - n, n
    
        return state == goal
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(capacities=[8, 5, 3], init=[8, 0, 0], goal=[4, 4, 0]):
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
    
* <a name="slidingpuzzle"></a>**SlidingPuzzle** [Sliding puzzle](https://en.wikipedia.org/wiki/15_puzzle)
    The 3-, 8-, and 15-sliding puzzles are classic examples of A* search.
    The problem is NP-hard but the puzzles can all be solved with A* and an efficient representation. (10 instances)
    
    ```python
    def sat(moves: List[int], start=[[5, 0, 2, 3], [1, 9, 6, 7], [4, 14, 8, 11], [12, 13, 10, 15]]):
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
    
        locs = {i: [x, y] for y, row in enumerate(start) for x, i in enumerate(row)}  # locations, 0 stands for blank
        for i in moves:
            assert abs(locs[0][0] - locs[i][0]) + abs(locs[0][1] - locs[i][1]) == 1
            locs[0], locs[i] = locs[i], locs[0]
        return all(locs[i] == [i % len(start[0]), i // len(start)] for i in locs)
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(start=[[5, 0, 2, 3], [1, 9, 6, 7], [4, 14, 8, 11], [12, 13, 10, 15]]):
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

* <a name="maxint"></a>**MaxInt** Inspired by [HumanEval](https://github.com/openai/human-eval) \#35 (10 instances)
    
    ```python
    def sat(m: int, hello=[1, 31, 3, 2, 0, 18, 32, -4, 2, -1000, 35, 35, 21, 18, 2, 60]):
        """
        Find the largest integer in a sequence
    
        Sample Input:
        [8, 0, 1, 4, 9, 3, 4, -2, 8, 3]
    
        Sample Output:
        9
        """
        return m in hello and not any(m < i for i in hello)
    ```
    <details><summary>9,469 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(hello=[1, 31, 3, 2, 0, 18, 32, -4, 2, -1000, 35, 35, 21, 18, 2, 60]):
        return 60
    
    ```
    
    ```python
    def sol(hello=[1, 31, 3, 2, 0, 18, 32, -4, 2, -1000, 35, 35, 21, 18, 2, 60]):
        """
        Option 1:
        Recursive heuristic: pick any integer in the list as your max
        Subproblem: maximum of A[1:], maximum of A[:len(A)-1]
    
        Option 2:
        Try to recursively solve and then save a memo of the max for different slices
    
        Option 3:
        Sort the list and then pick the max as the last element
    
        Option 4:
        Iterate through the list, looking at each element and keeping track of the max, then return the max
        """
        return max(hello)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(hello=[1, 31, 3, 2, 0, 18, 32, -4, 2, -1000, 35, 35, 21, 18, 2, 60]):
        return max(hello)
    ```
    
    </details>
    
* <a name="findextensions"></a>**FindExtensions** Inspired by [HumanEval](https://github.com/openai/human-eval) \#29 (10 instances)
    
    ```python
    def sat(extensions: List[str], strings=['cat', 'dog', 'shatter', 'donut', 'at', 'todo'], prefix="do"):
        """
        Find the strings in a list starting with a given prefix
    
        Sample Input:
        ['cat', 'car', 'fear', 'center'], 'ca'
    
        Sample Output:
        ['cat', 'car']
        """
        i = 0
        for s in strings:
            if s.startswith(prefix):
                assert extensions[i] == s
                i += 1
        return i == len(extensions)
    ```
    <details><summary>9,057 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(strings=['cat', 'dog', 'shatter', 'donut', 'at', 'todo'], prefix="do"):
        return ["dog", "donut"]
    
    ```
    
    ```python
    def sol(strings=['cat', 'dog', 'shatter', 'donut', 'at', 'todo'], prefix="do"):
        """
        Sample Input:
        ['cat', 'car', 'fear', 'center'], 'ca'
        """
        # The solution can be as simple as this:
        return [str for str in strings if str.startswith(prefix)]
    
        # The heuristics used in CatLab are the following:
        #   goal is a list of strings starting with "prefix"
        #   The probability of a string having "prefix" at the start is proportional to the total 
        #       number of characters that can come after "prefix"
        #   The probability of a string having "prefix" at the start is proportional to the number
        #
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(strings=['cat', 'dog', 'shatter', 'donut', 'at', 'todo'], prefix="do"):
        return [s for s in strings if s.startswith(prefix)]
    ```
    
    </details>
    
* <a name="catstrings"></a>**CatStrings** Inspired by [HumanEval](https://github.com/openai/human-eval) \#28 (10 instances)
    
    ```python
    def sat(cat: str, strings=['Will', 'i', 'am', 'Now', 'here']):
        """
        Concatenate a list of strings
    
        Sample input
        ---
        ['cat', 'dog', 'bird']
    
        Sample output
        ---
        'catdogbird'
        """
        i = 0
        for s in strings:
            for c in s:
                assert cat[i] == c
                i += 1
        return i == len(cat)
    ```
    <details><summary>8,813 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(strings=['Will', 'i', 'am', 'Now', 'here']):
        return ''.join(strings)
    ```
    
    ```python
    def sol(strings=['Will', 'i', 'am', 'Now', 'here']):
        """
        Given a list of strings, returns a string containing all of the strings concatenated together
        Note: it may not take exactly as input as long as it concatenates all of the strings together
    
        Sample input
        ---
        ['Will', 'i', 'am', 'Now', 'here']
    
        Sample output
        ---
        'WilliamNowhere'
        """
        return "".join(strings)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(strings=['Will', 'i', 'am', 'Now', 'here']):
        return "".join(strings)
    ```
    
    </details>
    
* <a name="findcontainers"></a>**FindContainers** Inspired by [HumanEval](https://github.com/openai/human-eval) \#7 (10 instances)
    
    ```python
    def sat(containers: List[str], strings=['cat', 'dog', 'shatter', 'bear', 'at', 'ta'], substring="at"):
        """
        Find the strings in a list containing a given substring
    
        Sample Input:
        ['cat', 'dog', 'bear'], 'a'
    
        Sample Output:
        ['cat', 'bear']
        """
        i = 0
        for s in strings:
            if substring in s:
                assert containers[i] == s
                i += 1
        return i == len(containers)
    ```
    <details><summary>8,755 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(strings=['cat', 'dog', 'shatter', 'bear', 'at', 'ta'], substring="at"):
        return [x for x in strings if "at" in x]
    
    ```
    
    ```python
    def sol(strings=['cat', 'dog', 'shatter', 'bear', 'at', 'ta'], substring="at"):
    
        # map the name of the string to all versions of the substring
        substring_map: Dict[str, Set[str]] = {}
        for s in strings:
            substring_map[s] = set()
    
            for ss in range(len(substring)):
                for i in range(len(s)-(len(substring)-ss)):
                    substring_map[s].add(s[i:i+(ss+1)])
    
        containers = []
        for k, v in substring_map.items():
            if substring in v:
                containers.append(k)
    
        return containers
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(strings=['cat', 'dog', 'shatter', 'bear', 'at', 'ta'], substring="at"):
        return [s for s in strings if substring in s]
    ```
    
    </details>
    
* <a name="belowthreshold"></a>**BelowThreshold** Inspired by [HumanEval](https://github.com/openai/human-eval) \#52 (10 instances)
    
    ```python
    def sat(indexes: List[int], nums=[0, 2, 17, 4, 4213, 322, 102, 29, 15, 39, 55], thresh=100):
        """
        Find the indexes of numbers below a given threshold
        """
        j = 0
        for i, n in enumerate(nums):
            if n < thresh:
                assert indexes[j] == i
                j += 1
        assert j == len(indexes)
        return True
    ```
    <details><summary>8,591 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[0, 2, 17, 4, 4213, 322, 102, 29, 15, 39, 55], thresh=100):
        return [i for i,n in enumerate(nums) if n<thresh]
    
    ```
    
    ```python
    def sol(nums=[0, 2, 17, 4, 4213, 322, 102, 29, 15, 39, 55], thresh=100):
        """
        ------------------------------------------------------
        Find the indexes of numbers below a given threshold
        ------------------------------------------------------
        PSEUDO:
    
        For each number i find nums[i] < thresh.
        """
        indices = []
        j = 0
        for i, n in enumerate(nums):
            if n < thresh:
                indices.append(i)
                j += 1
        return indices
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[0, 2, 17, 4, 4213, 322, 102, 29, 15, 39, 55], thresh=100):
        return [i for i, n in enumerate(nums) if n < thresh]
    ```
    
    </details>
    
* <a name="longeststr"></a>**LongestStr** Inspired by [HumanEval](https://github.com/openai/human-eval) \#12 (10 instances)
    
    ```python
    def sat(ans: str, words=['these', 'are', 'some', 'pretty', 'long', 'words']):
        """
        Find the longest of a list of strings
    
        Sample Input:
        ["cat", "dog", "sheep", "chimp"]
    
        Sample Output:
        "sheep"
        """
        return ans in words and all(len(ans) >= len(w) for w in words)
    ```
    <details><summary>8,260 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(words=['these', 'are', 'some', 'pretty', 'long', 'words']):
        return words[3]
    
    ```
    
    ```python
    def sol(words=['these', 'are', 'some', 'pretty', 'long', 'words']):
        """
        A greedy string is defined as a string v such that if there are at least two strings of the same length
        and containing equal numbers of 'a' and 'b' characters, v will always start with either a single
        'a' or 'b' character.
    
        Find the longest such string from the given list of strings.
    
        Sample Input:
        ["aabbccddee", "aabcde", "baaaad", "bbaaad", "aaaaaa"]
    
        Sample Output:
        "baaaad"
        """
        return sorted(words, key=lambda x: (-len(x), x))[0]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(words=['these', 'are', 'some', 'pretty', 'long', 'words']):
        return max(words, key=len)
    ```
    
    </details>
    
* <a name="spaceyrange"></a>**SpaceyRange** Inspired by [HumanEval](https://github.com/openai/human-eval) \#15 (10 instances)
    
    ```python
    def sat(ans: str, n=15):
        """
        Find a string consisting of the non-negative integers up to n inclusive
    
        Sample Input:
        4
    
        Sample Output:
        '0 1 2 3 4'
        """
        return [int(i) for i in ans.split(' ')] == list(range(n + 1))
    ```
    <details><summary>8,252 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=15):
        return " ".join(map(str,range(n+1)))
    
    ```
    
    ```python
    def sol(n=15):
        """
        We produce a string by writing a function that takes an argument x,
        which is a non-negative integer. Its output is the same as f6.
    
        Sample Input:
        4
    
        Sample Output:
        '0 1 2 3 4'
        """
        return " ".join([str(i) for i in list(range(n + 1))])
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=15):
        return ' '.join(str(i) for i in range(n + 1))
    ```
    
    </details>
    
* <a name="flipcase"></a>**FlipCase** Inspired by [HumanEval](https://github.com/openai/human-eval) \#27 (10 instances)
    
    ```python
    def sat(ans: str, s="FlIp ME!"):
        """
        Flip case
    
        Sample input
        ---
        'cAt'
    
        Sample output
        ---
        'CaT'
        """
        return len(ans) == len(s) and all({c, d} == {d.upper(), d.lower()} for c, d in zip(ans, s))
    ```
    <details><summary>7,991 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="FlIp ME!"):
        return s.swapcase()
    ```
    
    ```python
    def sol(s="FlIp ME!"):
        """
        Flip case
    
        Complete this function
    
        Remove the pass statement. You want to return the same value as you get in the input.
    
        Look up the docstring for the function. It will tell you a bit about the type annotations.
        """
        return "".join(c.upper() if c.islower() else c.lower() for c in s)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="FlIp ME!"):
        return "".join(c.lower() if c.upper() == c else c.upper() for c in s)
    ```
    
    </details>
    
* <a name="charsum"></a>**CharSum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#66 (10 instances)
    
    ```python
    def sat(tot: int, s="Add ME uP AND YOU WILL GET A BIG NUMBER!"):
        """
        Compute the sum of the ASCII values of the upper-case characters in the string.
        """
        for c in s:
            if c.isupper():
                tot -= ord(c)
        return tot == 0
    ```
    <details><summary>7,244 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="Add ME uP AND YOU WILL GET A BIG NUMBER!"):
        return sum(map(ord,filter(str.isupper,s)))
    
    ```
    
    ```python
    def sol(s="Add ME uP AND YOU WILL GET A BIG NUMBER!"):
        """
        'Add ME uP AND YOU WILL GET A BIG NUMBER!',
        'Repeatedly add the ASCII values of the upper-case characters in the string.',
        """
        tot = 0
        for c in s:
            if c.isupper():
                tot += ord(c)
        return tot
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="Add ME uP AND YOU WILL GET A BIG NUMBER!"):
        return sum(ord(c) for c in s if c.isupper())
    ```
    
    </details>
    
* <a name="strlength"></a>**StrLength** Inspired by [HumanEval](https://github.com/openai/human-eval) \#23 (10 instances)
    
    ```python
    def sat(length: int, s="pneumonoultramicroscopicsilicovolcanoconiosis"):
        """
        Find the length of a non-empty string
    
        Sample input
        ---
        "foo"
    
        Sample output
        ---
        3
        """
        try:
            s[length]
        except IndexError:
            s[length - 1]
            return True
    ```
    <details><summary>7,188 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="pneumonoultramicroscopicsilicovolcanoconiosis"):
        return 45
    
    ```
    
    ```python
    def sol(s="pneumonoultramicroscopicsilicovolcanoconiosis"):
        min_length = 1
        max_length = len(s)
        while True:
            if sat(max_length):
                return max_length
            if sat(min_length):
                return min_length
            if not sat(math.floor((min_length + max_length) / 2)):
                max_length = math.floor((min_length + max_length) / 2)
            else:
                min_length = math.floor((min_length + max_length) / 2)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="pneumonoultramicroscopicsilicovolcanoconiosis"):
        return len(s)
    ```
    
    </details>
    
* <a name="findpositives"></a>**FindPositives** Inspired by [HumanEval](https://github.com/openai/human-eval) \#30 (10 instances)
    
    ```python
    def sat(positives: List[int], nums=[2, 2342, -2, 32, -8, -5, 2342, 0, -9, 44, 11]):
        """
        Find the positive integers in a list
    
        Sample Input:
        [-1, 3, 19, -2, 0, 44, 0, 44, 11]
    
        Sample Output:
        [3, 19, 44, 44, 11]
        """
        stack = positives[::-1]
        for n in nums:
            assert n <= 0 or n == stack.pop()
        return stack == []
    ```
    <details><summary>7,055 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[2, 2342, -2, 32, -8, -5, 2342, 0, -9, 44, 11]):
        return [n for n in nums if n>0]
    
    ```
    
    ```python
    def sol(nums=[2, 2342, -2, 32, -8, -5, 2342, 0, -9, 44, 11]):
        """
        Find the positive integers in a list
    
        Sample Input:
        [-1, 3, 19, -2, 0, 44, 0, 44, 11]
    
        Sample Output:
        [3, 19, 44, 44, 11]
    
        Explanation:
        Stack is initialized with the items in `positives` in reverse order.
        For each element in `nums`, we check if the element is <= 0 or == stack.pop().
        This checks if this element is a positive integer in `nums`.
        """
        return [n for n in nums if n > 0]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[2, 2342, -2, 32, -8, -5, 2342, 0, -9, 44, 11]):
        return [i for i in nums if i > 0]
    ```
    
    </details>
    
* <a name="rollingmax"></a>**RollingMax** Inspired by [HumanEval](https://github.com/openai/human-eval) \#9 (10 instances)
    
    ```python
    def sat(maxes: List[int], nums=[1, 4, 3, -6, 19]):
        """
        Find a list whose ith element is the maximum of the first i elements of the input list.
    
        Sample Input:
        [2, 8, 2]
    
        Sample Output:
        [2, 8, 8]
        """
        assert len(maxes) == len(nums)
        for i in range(len(nums)):
            if i > 0:
                assert maxes[i] == max(maxes[i - 1], nums[i])
            else:
                assert maxes[0] == nums[0]
        return True
    ```
    <details><summary>6,985 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[1, 4, 3, -6, 19]):
        return [max(nums[:i+1]) for i in range(5)]
    
    ```
    
    ```python
    def sol(nums=[1, 4, 3, -6, 19]):
        """
        Find a list whose ith element is the maximum of the first i elements of the input list.
    
        Sample Input:
        [2, 8, 2]
    
        Sample Output:
        [2, 8, 8]
        """
        res = []
        prev = None
        for num in nums:
            if prev is None:
                res.append(num)
                prev = num
            else:
                if num > prev:
                    res.append(num)
                    prev = num
                else:
                    res.append(prev)
        return res
    
    ```
    
    </details>
    
    <details><summary>2 hand-written solutions </summary>
    
    ```python
    def sol(nums=[1, 4, 3, -6, 19]):
        return [max(nums[:i]) for i in range(1, len(nums) + 1)]
    ```
    
    ```python
    def sol(nums=[1, 4, 3, -6, 19]):
        ans = []
        if nums:
            m = nums[0]
            for n in nums:
                m = max(n, m)
                ans.append(m)
        return ans
    ```
    
    </details>
    
* <a name="removevowels"></a>**RemoveVowels** Inspired by [HumanEval](https://github.com/openai/human-eval) \#51 (10 instances)
    
    ```python
    def sat(txt: str, text="Hello, world!"):
        """
        Remove the vowels from the original string.
        """
        n = 0
        for c in text:
            if c.lower() not in "aeiou":
                assert txt[n] == c
                n += 1
        assert n == len(txt)
        return True
    ```
    <details><summary>6,866 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(text="Hello, world!"): return "Hll, wrld!"
    
    ```
    
    ```python
    def sol(text="Hello, world!"):
        n = len(text)
    
        # Put the vowels at the end.
        for c in "aeiouAEIOU":
            while c in text:
                text = text[:text.index(c)] + text[text.index(c)+1:]
                n -= 1
    
        # Put the non-vowels in the beginning.
        vowels = "aeiou"
        txt = ""
        for c in text:
            if c.lower() not in vowels:
                txt += c
        txt += text[n:]
        return txt
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(text="Hello, world!"):
        return "".join(c for c in text if c.lower() not in "aeiou")
    ```
    
    </details>
    
* <a name="frac"></a>**Frac** Inspired by [HumanEval](https://github.com/openai/human-eval) \#2 (10 instances)
    
    ```python
    def sat(x: float, v=523.12892):
        """
        Given a floating point number, find its fractional part.
    
        Sample Input:
        4.175
    
        Sample Output:
        0.175
        """
        return 0 <= x < 1 and (v - x).is_integer()
    ```
    <details><summary>6,545 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(v=523.12892):
        return v%1
    
    ```
    
    ```python
    def sol(v=523.12892):
        """
        Given a fixed arbitrary floating point number, find a random floating point number whose fractional part
        is the same as the provided number.
    
        For example:
        v = 523.12892
        g6(v) == v because 523.12892 - (523.12892 % 1) == g6(v) % 1
    
        """
        return v % 1
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(v=523.12892):
        return v % 1.0
    ```
    
    </details>
    
* <a name="negcumulative_trivial"></a>**NegCumulative_Trivial** Inspired by [HumanEval](https://github.com/openai/human-eval) \#3
    (see also FirstNegCumulative above which is not as trivial)
    This version is a more direct translation of the problem but it can of course
    be solved trivially just by trying both neg=True and neg=False (10 instances)
    
    ```python
    def sat(neg: bool, balances=[2, 7, -2, 4, 3, -15, 10, -45, 3]):
        """
        Given a list of numbers which represent bank deposits and withdrawals,
        determine if the cumulative sum is negative.
    
        Sample Input:
        [12, -5, 3, -99, 14, 88, -99]
    
        Sample Output:
        True
        """
        total = 0
        for b in balances:
            total += b
            if total < 0:
                return neg == True
        return neg == False
    ```
    <details><summary>6,163 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(balances=[2, 7, -2, 4, 3, -15, 10, -45, 3]):
        return True
    
    ```
    
    ```python
    def sol(balances=[2, 7, -2, 4, 3, -15, 10, -45, 3]):
        """
        Given a list of numbers which represent bank deposits and withdrawals,
        determine if the cumulative sum is negative.
    
        Sample Input:
        [12, -5, 3, -99, 14, 88, -99]
    
        Sample Output:
        True
    
        The additive inverse of any positive integer is negative.
        Adding the additive inverse to another negative number doesn't change that
        number.  Adding it to zero would make it negative, and adding it to a
        positive number would take it towards zero.
        """
        return sum(x for x in balances if x < 0) <= sum(x for x in balances if x > 0)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(balances=[2, 7, -2, 4, 3, -15, 10, -45, 3]):
        total = 0
        for b in balances:
            total += b
            if total < 0:
                return True
        return False
    ```
    
    </details>
    
* <a name="unevenfind"></a>**UnevenFind** Inspired by [HumanEval](https://github.com/openai/human-eval) \#87 (10 instances)
    
    ```python
    def sat(indices: List[List[int]], uneven=[[1, 3, 2, 32, 17], [17, 2, 48, 17], [], [9, 35, 4], [3, 17]], target=17):
        """Find the indices of all occurrences of target in the uneven matrix"""
        for i, j in indices:
            assert uneven[i][j] == target
        for i, row in enumerate(uneven):
            for j, n in enumerate(row):
                assert n != target or [i, j] in indices
        return True
    ```
    <details><summary>6,059 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(uneven=[[1, 3, 2, 32, 17], [17, 2, 48, 17], [], [9, 35, 4], [3, 17]], target=17):
        return [[i,j] for i, row in enumerate(uneven) for j, n in enumerate(row) if n == 17]
    
    ```
    
    ```python
    def sol(uneven=[[1, 3, 2, 32, 17], [17, 2, 48, 17], [], [9, 35, 4], [3, 17]], target=17):
        return [[i, j] for  i in range(len(uneven))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    for j in range(len(uneven[i])) if uneven[i][j] == target]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(uneven=[[1, 3, 2, 32, 17], [17, 2, 48, 17], [], [9, 35, 4], [3, 17]], target=17):
        return [[i, j] for i, row in enumerate(uneven) for j, n in enumerate(row) if n == target]
    ```
    
    </details>
    
* <a name="overlappingcount"></a>**OverlappingCount** Inspired by [HumanEval](https://github.com/openai/human-eval) \#18 (10 instances)
    
    ```python
    def sat(ans: List[int], s="Bananannanaannanaanananananana", sub="anan", count=7):
        """
        Find occurrences of a substring in a parent string *including overlaps*
    
        Sample Input:
        'helllo', 'll'
    
        Sample Output:
        [2, 3]
        """
        return all(sub == s[i:i + len(sub)] and i >= 0 for i in ans) and len(set(ans)) >= count
    ```
    <details><summary>5,796 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="Bananannanaannanaanananananana", sub="anan", count=7):
        return [s.find(sub,i) for i in range(len(s)-len(sub))]
    
    ```
    
    ```python
    def sol(s="Bananannanaannanaanananananana", sub="anan", count=7):
        """s: the string from which to find occurrences of sub
           sub: the substring to find in s
           count: the number of occurrences of sub in s, including overlaps
           Returns: a list of start indices where sub occurs in s, including overlaps.
           This should not include duplicates!
        """
        result = []
        for i in range(len(s)-len(sub)+1):
            for j in range(len(sub)):
                if s[i+j] != sub[j]:
                    break
                if j == len(sub)-1:
                    result.append(i)
        return result
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="Bananannanaannanaanananananana", sub="anan", count=7):
        ans = []
        for i in range(len(s) + 1):
            if s[i:i + len(sub)] == sub:
                ans.append(i)
        return ans
    ```
    
    </details>
    
* <a name="binarystrxor"></a>**BinaryStrXOR** Inspired by [HumanEval](https://github.com/openai/human-eval) \#11 (10 instances)
    
    ```python
    def sat(str_num: str, nums=['100011101100001', '100101100101110']):
        """
        Find a the XOR of two given strings interpreted as binary numbers.
    
        Sample Input:
        "0001", "1011"
    
        Sample Output:
        "1010"
        """
        a, b = nums
        return int(str_num, 2) == int(a, 2) ^ int(b, 2)
    ```
    <details><summary>5,465 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=['100011101100001', '100101100101110']):
        return f"{int(nums[0],2)^int(nums[1],2):b}"
    
    ```
    
    ```python
    def sol(nums=['100011101100001', '100101100101110']):
        """
        Find a the XOR of two given strings interpreted as binary numbers.
    
        Sample Input:
        "0001", "1011"
    
        Sample Output:
        "1010"
        """
        a, b = nums
        """
        s = ""
        for i in range(len(a)):
            if a[i] == b[i]:
                s += "0"
            else: s += "1"
        return s
        """
        return ''.join(map(lambda x: "0" if x[0] == x[1] else "1", zip(a, b)))
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=['100011101100001', '100101100101110']):
        a, b = nums
        ans = int(a, 2) ^ int(b, 2)
        return format(ans, "b")
    ```
    
    </details>
    
* <a name="palindrome_trivial"></a>**Palindrome_Trivial** Inspired by [HumanEval](https://github.com/openai/human-eval) \#48 (10 instances)
    
    ```python
    def sat(p: bool, s="This problem is trivial but common"):
        """
        Test whether the given string is a palindrome
        """
        return p == (s == s[::-1])
    ```
    <details><summary>5,338 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="This problem is trivial but common"):
        return False
    
    ```
    
    ```python
    def sol(s="This problem is trivial but common"):
        ans = "abcdcba"
        if s:
            # If the first half of the string is equal to the reverse of the second half, the string is a palindrome
            return s[:(len(s)+1)//2] == s[(len(s)+1)//2:][::-1]
        else:
            ans = ""
            i = 0
            while i < 100:
                ans += "abcdefghijklmnopqrstuvwxyz"[i % 26]
                i += 1
            # If the first half of the string is equal to the reverse of the second half, the string is a pal
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="This problem is trivial but common"):
        return s == s[::-1]
    ```
    
    </details>
    
* <a name="evenodd"></a>**EvenOdd** Inspired by [HumanEval](https://github.com/openai/human-eval) \#85 (10 instances)
    
    ```python
    def sat(even_odd_sum: int, nums=[2341, 125146894, 12521, -12451293476325, 535284623934, 132974693614350]):
        """Find the sum of the even elements that are at odd indices"""
        for i in nums[1::2]:
            if i % 2 == 0:
                even_odd_sum -= i
        return even_odd_sum == 0
    ```
    <details><summary>5,098 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[2341, 125146894, 12521, -12451293476325, 535284623934, 132974693614350]):
        return sum(nums[1::2][::2])
    
    ```
    
    ```python
    def sol(nums=[2341, 125146894, 12521, -12451293476325, 535284623934, 132974693614350]):
        """Find the sum of the even elements that are at odd indices.
        ```
        >>> g6()
        ```
        """
        i = 0
        even_odd_sum = 0
        while i < len(nums):
            if i % 2 == 1: #odd
                if nums[i] % 2 == 0: #even
                    even_odd_sum += nums[i]
            i += 1
        return even_odd_sum
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[2341, 125146894, 12521, -12451293476325, 535284623934, 132974693614350]):
        return sum(i for i in nums[1::2] if i % 2 == 0)
    ```
    
    </details>
    
* <a name="monotonic"></a>**Monotonic** Inspired by [HumanEval](https://github.com/openai/human-eval) \#57 (10 instances)
    
    ```python
    def sat(direction: str, nums=[2, 4, 17, 29, 31, 1000, 416629]):
        """
        Determine the direction ('increasing' or 'decreasing') of monotonic sequence nums
        """
        if direction == "increasing":
            return all(nums[i] < nums[i + 1] for i in range(len(nums) - 1))
        if direction == "decreasing":
            return all(nums[i + 1] < nums[i] for i in range(len(nums) - 1))
    ```
    <details><summary>5,026 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[2, 4, 17, 29, 31, 1000, 416629]):
        return "increasing"
    
    ```
    
    ```python
    def sol(nums=[2, 4, 17, 29, 31, 1000, 416629]):
        """
        Determine the direction ('increasing' or 'decreasing') of monotonic sequence nums
    
        An alternative implementation of f6.
        """
        direction = None
        for i in range(len(nums) - 1):
            if nums[i] == nums[i + 1]:
                return "static"
            elif nums[i] < nums[i + 1]:
                if direction == "decreasing":
                    return "static"
                direction = "increasing"
            else:
                if direction == "increasing":
                    return "static"
                direction = "decreasing"
        return direction
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[2, 4, 17, 29, 31, 1000, 416629]):
        return "increasing" if len(nums) > 1 and nums[1] > nums[0] else "decreasing"
    ```
    
    </details>
    
* <a name="narrowerlist"></a>**NarrowerList** Inspired by [HumanEval](https://github.com/openai/human-eval) \#74 (10 instances)
    
    ```python
    def sat(li: List[str], lists=[['this', 'list', 'is', 'narrow'], ['I', 'am', 'shorter but wider']]):
        """
        Find the list that has fewer total characters (including repetitions)
        """
        width = sum(len(s) for s in li)
        for li2 in lists:
            assert width <= sum(len(s) for s in li2)
        return li in lists
    ```
    <details><summary>4,786 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(lists=[['this', 'list', 'is', 'narrow'], ['I', 'am', 'shorter but wider']]):
       return lists[0]
    
    ```
    
    ```python
    def sol(lists=[['this', 'list', 'is', 'narrow'], ['I', 'am', 'shorter but wider']]):
        """
        Find lists that have more characters than the second-shortest one but less than the longest one.
        """
        lists = sorted(lists, key=lambda li: sum(len(s) for s in li))
        if len(lists) == 2:
            return lists[0] if sum(len(s) for s in lists[0]) > len(lists[1]) else lists[1]
        else:
            return lists[1] if sum(len(s) for s in lists[0]) > len(lists[1]) else lists[2]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(lists=[['this', 'list', 'is', 'narrow'], ['I', 'am', 'shorter but wider']]):
        return min(lists, key=lambda x: sum(len(i) for i in x))
    ```
    
    </details>
    
* <a name="filterints"></a>**FilterInts** Inspired by [HumanEval](https://github.com/openai/human-eval) \#22 (1 instance)
    
    ```python
    def sat(indexes: List[int], li=['Hello', '5', '10', 'bye'], num=2):
        """
        Find the indices of valid python integers in a list of strings
    
        Sample input
        ---
        ["18.5", "-1", "2+2", "7", "foo"]
    
        Sample output
        ---
        [1, 3]
        """
        [int(li[i]) for i in indexes]
        return len(set(indexes)) >= num and min(indexes) >= 0
    ```
    <details><summary>4,763 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(li=['Hello', '5', '10', 'bye'], num=2):
        return [1,2]
    
    ```
    
    ```python
    def sol(li=['Hello', '5', '10', 'bye'], num=2):
        """
        Create a sample solution whose return value is a list of integers
        """
        index_list = []
        for i in range(len(li)):
            if li[i] in ["-1", "foo", "10", "5", "bye", "Hello"]:
                try:
                    int(li[i])
                    if i not in index_list:
                        index_list.append(i)
                except ValueError:
                    continue
    
        while len(index_list) < num:
            index_list.append(0)
    
        return index_list
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(li=['Hello', '5', '10', 'bye'], num=2):
        ans = []
        for i in range(len(li)):
            try:
                int(li[i])
                ans.append(i)
            except:
                pass
        return ans
    ```
    
    </details>
    
* <a name="commonnumbers"></a>**CommonNumbers** Inspired by [HumanEval](https://github.com/openai/human-eval) \#58 (10 instances)
    
    ```python
    def sat(common: List[int], a=[2, 416629, 2, 4, 17, 29, 31, 1000], b=[31, 2, 4, 17, 29, 41205]):
        """
        Find numbers common to a and b
        """
        return all((i in common) == (i in a and i in b) for i in a + b + common)
    ```
    <details><summary>4,728 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=[2, 416629, 2, 4, 17, 29, 31, 1000], b=[31, 2, 4, 17, 29, 41205]):
        return list({*a} & {*b})
    
    ```
    
    ```python
    def sol(a=[2, 416629, 2, 4, 17, 29, 31, 1000], b=[31, 2, 4, 17, 29, 41205]):
        a = sorted(a)
        b = sorted(b)
        i = 0
        j = 0
        common = []
        while True:
            if i == len(a):
                break
            if j == len(b):
                break
            if a[i] < b[j]:
                i += 1
                continue
            elif a[i] > b[j]:
                j += 1
                continue
            else:
                if a[i] not in common:
                    common.append(a[i])
                i += 1
                j += 1
        return common
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=[2, 416629, 2, 4, 17, 29, 31, 1000], b=[31, 2, 4, 17, 29, 41205]):
        return sorted(set(a).intersection(set(b)))
    ```
    
    </details>
    
* <a name="fibonacci"></a>**Fibonacci** Inspired by [HumanEval](https://github.com/openai/human-eval) \#55 (10 instances)
    
    ```python
    def sat(nums: List[int], n=1402):
        """
        Find the first n Fibonacci numbers
        """
        return nums[0] == nums[1] == 1 and all(nums[i + 2] == nums[i + 1] + nums[i] for i in range(n - 2))
    ```
    <details><summary>4,583 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=1402):
        x = [1, 1]
        while len(x) < n: x.append(sum(x[-2:]))
        return x
    
    ```
    
    ```python
    def sol(n=1402):
        """
        We know that 1 is the first Fibonacci number, and we know that every Fibonacci number after that is
        the sum of the previous two. It's likely that if we started with a large integer and worked our way down
        we would arrive at the first n Fibonacci numbers. So let's try to find that value
        """
        nums = [1] * n
        for i in range(2, n):
            nums[i] = nums[i - 1] + nums[i - 2]
        return nums
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=1402):
        ans = [1, 1]
        while len(ans) < n:
            ans.append(ans[-1] + ans[-2])
        return ans
    ```
    
    </details>
    
* <a name="investigatecrash"></a>**InvestigateCrash** Inspired by [HumanEval](https://github.com/openai/human-eval) \#72 (10 instances)
    
    ```python
    def sat(problem: int, weights=[1, 2, 5, 2, 1, 17], max_weight=100):
        """
        An object will "fly" if its weights are a palindrome and sum to <= max_weight. The given object won't fly.
        You have to determine why. Find index where the weights aren't a palindrome or -1 if weights are too big.
        """
        if problem == -1:
            return sum(weights) > max_weight
        return weights[problem] != weights[- 1 - problem]
    ```
    <details><summary>4,274 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(weights=[1, 2, 5, 2, 1, 17], max_weight=100):
        return 1
    ```
    
    ```python
    def sol(weights=[1, 2, 5, 2, 1, 17], max_weight=100):
        """
        Submitting a solution with a greater value for `weights` will result in a better score, which is good for
        the series, but this program is a contestant in a two-player competition meant to find the smallest problem
        in `f6` such that the "fly" object doesn't fly. The series is meant to train contestants to find the
        smallest problem using a brute force algorithm given a problem specification.
        """
        
        problem_number = -1
        while True:
            problem_number += 1
            if sat(problem_number, weights, max_weight):
                break
        return problem_number
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(weights=[1, 2, 5, 2, 1, 17], max_weight=100):
        if sum(weights) > max_weight:
            return -1
        return next(i for i, w in enumerate(weights) if weights[-i - 1] != weights[i])
    ```
    
    </details>
    
* <a name="fermatcomposite"></a>**FermatComposite** Inspired by [HumanEval](https://github.com/openai/human-eval) \#31 (10 instances)
    
    ```python
    def sat(certificate: int, n=1449):
        """
        Find a Fermat composite certificate for a number n > 1
    
        Sample Input:
        1469
    
        Sample Output:
        3  # because (3 ** 1468) % 1469 != 1
        """
        return pow(certificate, n - 1, n) > 1
    ```
    <details><summary>4,235 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=1449):
        return 6
    
    ```
    
    ```python
    def sol(n=1449):
        sat = int(n ** 0.5) + 1
        while True:
            # Find a value for x which suffices to test candidate certificates
            x = sat + 1
            while pow(x, n - 1, n) == 1:
                x += 1
            # Check all possible candidates
            for x in range(x, sat, -1):
                candidate = pow(x, n - 1, n)
                if candidate != 1:
                    # Save found optimum
                    sat = x
                    # Break out of inner loop
                    break
            # Check termination criterion
            if sat == x:
                return sat
            # Check termination criterion
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=1449):
        return next(i for i in range(2, n) if pow(i, n - 1, n) > 1)
    ```
    
    </details>
    
* <a name="rescale"></a>**Rescale** Inspired by [HumanEval](https://github.com/openai/human-eval) \#21 (10 instances)
    
    ```python
    def sat(ans: List[float], nums=[13.0, 17.0, 17.0, 15.5, 2.94]):
        """
        Rescale and shift numbers so that they cover the range [0, 1]
    
        Sample input
        ---
        [18.5, 17.0, 18.0, 19.0, 18.0]
    
        Sample output
        ---
        [0.75, 0.0, 0.5, 1.0, 0.5]
        """
        assert min(ans) == 0.0 and max(ans) == 1.0
        a = min(nums)
        b = max(nums)
        for i in range(len(nums)):
            x = a + (b - a) * ans[i]
            assert abs(nums[i] - x) < 1e-6
        return True
    ```
    <details><summary>4,124 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[13.0, 17.0, 17.0, 15.5, 2.94]):
        return [1.0 * (x - 2.94)/(17.0 - 2.94) for x in nums]
    
    ```
    
    ```python
    def sol(nums=[13.0, 17.0, 17.0, 15.5, 2.94]):
        """
        Rescale and shift numbers so that they cover the range [0, 1]
    
        Modified version of
        https://gist.github.com/Sheljohn/68ca3be74139f66dbc6127784f638920
        """
        #
        a = min(nums)
        b = max(nums)
        #
        def get_part(x, start, length):
            return (x - start) / length
        #
        new_nums = [get_part(x, a, b - a) for x in nums]
        return new_nums
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[13.0, 17.0, 17.0, 15.5, 2.94]):
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
    
* <a name="minsquareddeviation"></a>**MinSquaredDeviation** Loosely inspired by [HumanEval](https://github.com/openai/human-eval) \#4
    
    The HumanEval problem was simply to compute the mean absolute deviation. This problem is more interesting.
    It requires minimizing the sum of squared deviations, which turns out to be the mean `mu`. Moreover, if
    `mu` is the mean of the numbers then a simple calculation shows that:
    
    `sum((mu - n) ** 2 for n in nums) == sum((m - n) ** 2 for m in nums for n in nums) / (2 * len(nums))`
    
    We use 0.501 rather than 1/2 to deal with rounding errors. (10 instances)
    
    ```python
    def sat(x: float, nums=[12, -2, 14, 3, -15, 10, -45, 3, 30]):
        """
        Given a list of numbers, find x that minimizes mean squared deviation.
    
        Sample Input:
        [4, -5, 17, -9, 14, 108, -9]
    
        Sample Output:
        17.14285
        """
        return sum((n - x) ** 2 for n in nums) <= sum((m - n) ** 2 for m in nums for n in nums) * 0.501 / len(nums)
    ```
    <details><summary>4,106 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[12, -2, 14, 3, -15, 10, -45, 3, 30]):
        return 1.7
    
    ```
    
    ```python
    def sol(nums=[12, -2, 14, 3, -15, 10, -45, 3, 30]):
        def mean(nums):
            return (sum(nums) / len(nums))
    
        def average_msd(nums, avg):
            return (sum([(n - avg) ** 2 for n in nums]) / len(nums))
    
        trial = mean(nums)
        minimum = average_msd(nums, trial)
        while True:
            trial = (trial + mean(nums)) / 2
            tv = average_msd(nums, trial)
            if tv < minimum:
                minimum = tv
                continue
            else:
                break
        return trial
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[12, -2, 14, 3, -15, 10, -45, 3, 30]):
        return sum(nums) / len(nums)  # mean minimizes mean squared deviation
    ```
    
    </details>
    
* <a name="median"></a>**Median** One definition of the median is a number that minimizes the sum of absolute deviations.
    
    Inspired by [HumanEval](https://github.com/openai/human-eval) \#47 (10 instances)
    
    ```python
    def sat(x: int, nums=[132666041, 237412, 28141, -12, 11939, 912414, 17], upper=133658965):
        """
        Find an integer that minimizes the sum of absolute deviations with respect to the given numbers.
        """
        dev = sum(n - x for n in nums)
        return dev <= upper
    ```
    <details><summary>4,070 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[132666041, 237412, 28141, -12, 11939, 912414, 17], upper=133658965):
        return upper
    
    ```
    
    ```python
    def sol(nums=[132666041, 237412, 28141, -12, 11939, 912414, 17], upper=133658965):
        """Use a binary search to find the minimum value that satisfies the above condition."""
        assert len(nums) == 7
    
        def b(lower, upper):
            if lower == upper:
                return lower
            else:
                pos = int((lower + upper) / 2)
                dev = sum(n - pos for n in nums)
                if dev == upper:
                    return pos
                else:
                    if dev > upper:
                        return b(lower, pos - 1)
                    else:
                        return b(pos + 1, upper)
    
        return b(int(nums[0]), int(upper))
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[132666041, 237412, 28141, -12, 11939, 912414, 17], upper=133658965):
        return sorted(nums)[len(nums) // 2] if nums else 0
    ```
    
    </details>
    
* <a name="largestdivisor"></a>**LargestDivisor** Inspired by [HumanEval](https://github.com/openai/human-eval) \#24 (10 instances)
    
    ```python
    def sat(d: int, n=123456):
        """
        Find the largest integer divisor of a number n that is less than n
    
        Sample input
        ---
        1000
    
        Sample output
        ---
        500
        """
        return n % d == 0 and d < n and all(n % e for e in range(d + 1, n))
    ```
    <details><summary>3,752 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=123456): return n // 2
    
    ```
    
    ```python
    def sol(n=123456):
        """
        Find the largest integer divisor of a number n that is less than n
        This version uses a basic trial and bound approach of testing every divisor
        d from 2 to n - 1. This implementation is intended for educational purposes.
        """
        lo, hi = 2, n - 1
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if sat(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=123456):
        return next(d for d in range(n - 1, 0, -1) if n % d == 0)
    ```
    
    </details>
    
* <a name="substitutioncypher"></a>**SubstitutionCypher** Inspired by [HumanEval](https://github.com/openai/human-eval) \#89 (10 instances)
    
    ```python
    def sat(encrypted: str, orig="Hello, world!"):
        """Apply a substitution cypher in which each character is advanced by two multiplied by two places."""
        assert len(encrypted) == len(orig)
        return all(chr(ord(a) - 2 * 2) == b for a, b in zip(encrypted, orig))
    ```
    <details><summary>3,568 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(orig="Hello, world!"):
        return "".join(chr(ord(c)+4) for c in orig)
    
    ```
    
    ```python
    def sol(orig="Hello, world!"):
        """Form a substitution cypher in which each character is advanced by two multiplied by two places."""
        a = [x for x in orig]
        for i, x in enumerate(a):
            a[i] = chr(ord(x) + 2 * 2)
        return "".join(a)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(orig="Hello, world!"):
        return "".join(chr(ord(b) + 2 * 2) for b in orig)
    ```
    
    </details>
    
* <a name="allprefixes"></a>**AllPrefixes** Inspired by [HumanEval](https://github.com/openai/human-eval) \#14 (10 instances)
    
    ```python
    def sat(prefixes: List[str], s="donesezichethofalij"):
        """
        Find all prefixes of a given string
    
        Sample Input:
        "aabcd"
    
        Sample Output:
        ["", "a", "aa", "aab", "aabc", "aabcd"]
        """
        return all(s.startswith(p) for p in prefixes) and len(set(prefixes)) > len(s)
    ```
    <details><summary>3,271 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="donesezichethofalij"): return [s[:i] for i in range(len(s) + 1)]
    
    ```
    
    ```python
    def sol(s="donesezichethofalij"):
        """
        Generate all of the possible prefixes of a given string s.
    
        Return a list of length s + 1, where each entry is either a prefix of s (with its own prefix appended),
        or the empty string.
    
        Sample Input:
        "aabcd"
    
        Sample Output:
        ["", "a", "aa", "aab", "aabc", "aabcd"]
        """
        return [""] + [s[:i] for i in range(1, len(s)+1)]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="donesezichethofalij"):
        return [s[:i] for i in range(len(s) + 1)]
    ```
    
    </details>
    
* <a name="sortnumbers"></a>**SortNumbers** Inspired by [HumanEval](https://github.com/openai/human-eval) \#19 (10 instances)
    
    ```python
    def sat(ans: str, s="six one four three two nine eight"):
        """
        Sort numbers based on strings
    
        Sample input
        ---
        "six one four"
    
        Sample output
        ---
        "one four six"
        """
        nums = 'zero one two three four five six seven eight nine'.split()
        return [nums.index(x) for x in ans.split(" ")] == sorted([nums.index(x) for x in s.split(" ")])
    ```
    <details><summary>3,253 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="six one four three two nine eight"):
        return ' '.join([x for x in 'zero one two three four five six seven eight nine'.split() if x in s.split()])
    
    ```
    
    ```python
    def sol(s="six one four three two nine eight"):
        nums = 'zero one two three four five six seven eight nine'.split()
        numbers = [nums.index(x) for x in s.split(" ")]
        result = []
        while len(numbers) > 0:
            smallest = min(numbers)
            for i in [x for x in nums if x != nums[smallest]]:
                if i in numbers:
                    smallest = i
                    break
            result.append(nums[smallest])
            numbers.remove(smallest)
        return ' '.join(result)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="six one four three two nine eight"):
        nums = 'zero one two three four five six seven eight nine'.split()
        arr = [nums.index(x) for x in s.split()]
        arr.sort()
        ans = " ".join([nums[i] for i in arr])
        return ans
    ```
    
    </details>
    
* <a name="binarize"></a>**Binarize** Inspired by [HumanEval](https://github.com/openai/human-eval) \#79 (10 instances)
    
    ```python
    def sat(b: str, n=5324680297138495285):
        """Write n base 2 followed and preceded by 'bits'"""
        assert b[:4] == b[-4:] == 'bits'
        inside = b[4:-4]
        assert all(c in "01" for c in inside)
        assert inside[0] == "1" or len(inside) == 1
        m = 0
        for c in inside:
            m = 2 * m + int(c)
        return m == n
    ```
    <details><summary>3,008 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=5324680297138495285):
        return f"bits{n:b}bits"
    
    ```
    
    ```python
    def sol(n=5324680297138495285):
        def find_length():
            def check(length):
                inside = bin(n)[2:].rjust(length, '0')
                return sat("bits" + inside + "bits")
            try:
                return next(length for length in range(1, 1000) if check(length))
            except StopIteration:
                raise ValueError("Cannot find proper length")
        return "bits" + bin(n)[2:].rjust(find_length(), '0') + "bits"
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=5324680297138495285):
        s = bin(n)[2:]
        return f'bits{s}bits'
    ```
    
    </details>
    
* <a name="derivative"></a>**Derivative** Inspired by [HumanEval](https://github.com/openai/human-eval) \#62 (10 instances)
    
    ```python
    def sat(derivative: List[int], poly=[2, 1, 0, 4, 19, 231, 0, 5]):
        """
        Find the derivative of the given polynomial, with coefficients in order of increasing degree
        """
    
        def val(poly, x):
            return sum(coeff * (x ** i) for i, coeff in enumerate(poly))
    
        return all(abs(val(poly, x + 1e-8) - val(poly, x) - 1e-8 * val(derivative, x)) < 1e-4 for x in range(len(poly)))
    ```
    <details><summary>2,837 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(poly=[2, 1, 0, 4, 19, 231, 0, 5]):
        return [i*poly[i] for i in range(1,8)]
    
    ```
    
    ```python
    def sol(poly=[2, 1, 0, 4, 19, 231, 0, 5]):
        """
        Find the derivative of the given polynomial, with coefficients in order of increasing degree
        """
    
        def val(poly, x):
            return sum(coeff * (x ** i) for i, coeff in enumerate(poly))
    
        #can't use enumerate because multivariate polynomials map more than one variable per term
        def derive_single_var(poly):
            return [coeff * (i+1) for i, coeff in enumerate(poly[1:])]
    
        return derive_single_var(poly)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(poly=[2, 1, 0, 4, 19, 231, 0, 5]):
        return [i * poly[i] for i in range(1, len(poly))]
    ```
    
    </details>
    
* <a name="findvowels"></a>**FindVowels** Inspired by [HumanEval](https://github.com/openai/human-eval) \#64 (10 instances)
    
    ```python
    def sat(vowels: str, text="Hello, world!"):
        """
        Find the vowels from the original string.
        """
        i = 0
        for j, c in enumerate(text):
            if c.lower() in "aeiou" or c.lower() == 'y' and j == len(text) - 1:
                assert vowels[i] == c
                i += 1
        return i == len(vowels)
    ```
    <details><summary>2,667 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(text="Hello, world!"):
        return "".join(c for c in text if c in "aeiou")
    
    ```
    
    ```python
    def sol(text="Hello, world!"):
        """
        Find the vowels from the original string.
        """
        vowels = ""
        i = 0
        while i < len(text):
            c = text[i]
            if c.lower() in "aeiou":
                vowels += c
            else:
                if c.lower() == 'y' and i != len(text) - 1:
                    next_c = text[i + 1]
                    if next_c.lower() in "aeiou":
                        vowels += c
                        i += 1
                    else:
                        break
            i += 1
        return vowels
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(text="Hello, world!"):
        return "".join(c for c in text if c.lower() in "aeiou") + (text[-1] if text[-1].lower() == "y" else "")
    ```
    
    </details>
    
* <a name="numpasses"></a>**NumPasses** Inspired by [HumanEval](https://github.com/openai/human-eval) \#41 (10 instances)
    
    ```python
    def sat(count: int, n=981):
        """
        Given n cars traveling East and n cars traveling West on a road, how many passings will there be?
        A passing is when one car passes another. The East-bound cars all begin further West than the West-bound cars.
        """
        for i in range(n):
            for j in range(n):
                count -= 1
        return count == 0
    ```
    <details><summary>2,524 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=981):
       return n*n
    
    ```
    
    ```python
    def sol(n=981):
        """
        Given n cars traveling East and n cars traveling West on a road, how many passings will there be?
        A passing is when one car passes another. The East-bound cars all begin further West than the West-bound cars.
        "But wait!" you say, "the cars are staggered, so even if two cars are East-bound and never pass one another,
        they may pass each other later on in the road."
        """
        count: int = 0
        for i in range(n):
            for j in range(n):
                count += 1
        return count
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=981):
        return n ** 2
    ```
    
    </details>
    
* <a name="hexprimes"></a>**HexPrimes** Inspired by [HumanEval](https://github.com/openai/human-eval) \#78 (10 instances)
    
    ```python
    def sat(primes: List[bool], n="A4D4455214122CE192CCBE3"):
        """Determine which characters of a hexidecimal correspond to prime numbers"""
        return all(primes[i] == (c in "2357BD") for i, c in enumerate(n))
    ```
    <details><summary>2,507 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n="A4D4455214122CE192CCBE3"):
      return [c in "2357BD" for c in n]
    
    ```
    
    ```python
    def sol(n="A4D4455214122CE192CCBE3"):
        """Given a hexadecimal, return a list of bool which indicates whether each character is prime"""
        def is_prime(x):
            if x <= 1:
                return False
            elif x <= 3:
                return True
            elif x % 2 == 0:
                return False
    
            i = 3
            while i**2 <= x:
                if x % i == 0:
                    return False
                i += 2
            return True
    
        return [is_prime(int(x, 16)) for x in n]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n="A4D4455214122CE192CCBE3"):
        return [c in "2357BD" for c in n]
    ```
    
    </details>
    
* <a name="distinctchars"></a>**DistinctChars** Inspired by [HumanEval](https://github.com/openai/human-eval) \#16 (10 instances)
    
    ```python
    def sat(ans: List[str], s="The quick brown fox jumps over the lazy dog!", n=28):
        """
        Find the set of distinct characters in a string, ignoring case
    
        Sample Input:
        'HELlo', 4
    
        Sample Output:
        ['h', 'e', 'l', 'o']
        """
        assert all(ans.count(c.lower()) == 1 for c in s)
        assert all(c == c.lower() for c in ans)
        assert all(c in s.lower() for c in ans)
        return True
    ```
    <details><summary>2,321 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="The quick brown fox jumps over the lazy dog!", n=28):
        return list(set(s))[:n]
    
    ```
    
    ```python
    def sol(s="The quick brown fox jumps over the lazy dog!", n=28):
        """
        Given a string, find a set of distinct characters in that string. Ignore case.
    
        Sample Input:
        'HELlo'
    
        Sample Output:
        ['h', 'e', 'l', 'o']
        """
        result = []
        for c in s.lower():
            if len(result) >= n:
                break
            if c in result:
                continue
            result.append(c)
        return result
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="The quick brown fox jumps over the lazy dog!", n=28):
        return list(set(s.lower()))
    ```
    
    </details>
    
* <a name="largestprimefactor"></a>**LargestPrimeFactor** Inspired by [HumanEval](https://github.com/openai/human-eval) \#59 (10 instances)
    
    ```python
    def sat(p: int, n=101076):
        """
        Find the largest prime factor of n.
        """
    
        def is_prime(m):
            return all(m % i for i in range(2, m - 1))
    
        return is_prime(p) and n % p == 0 and p > 0 and all(n % i or not is_prime(i) for i in range(p + 1, n))
    ```
    <details><summary>2,245 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=101076):
        return next(i for i in range(2, n) if sat(i))
    
    ```
    
    ```python
    def sol(n=101076):
        def is_prime(n: int) -> bool:
            if n % 2 == 0:
                return False
            for x in range(3, int(n**0.5) + 1, 2):
                if n % x == 0:
                    return False
            return True
    
        if is_prime(n):
            return n
        else:
            ps = [2, 3]
            m = 3
            while m < n:
                m += 1
                if is_prime(m):
                    ps.append(m)
                    if n % m == 0:
                        return m
            return max(ps)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=101076):
        def is_prime(m):
            return all(m % i for i in range(2, m - 1))
    
        return next(n // i for i in range(1, n) if n % i == 0 and is_prime(n // i))
    ```
    
    </details>
    
* <a name="greatesthindex"></a>**GreatestHIndex** Inspired by [HumanEval](https://github.com/openai/human-eval) \#69 (10 instances)
    
    ```python
    def sat(h: int, seq=[3, 1, 4, 17, 5, 17, 2, 1, 41, 32, 2, 5, 5, 5, 5]):
        """
        Find the h-index, the largest positive number h such that that h occurs in the sequence at least h times.
        h = -1 if there is no such positive number.
        """
        for i in seq:
            assert not (i > 0 and i > h and seq.count(i) >= i)
        return h == -1 or seq.count(h) >= h > 0
    ```
    <details><summary>2,175 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(seq=[3, 1, 4, 17, 5, 17, 2, 1, 41, 32, 2, 5, 5, 5, 5]):
        return 5
    
    ```
    
    ```python
    def sol(seq=[3, 1, 4, 17, 5, 17, 2, 1, 41, 32, 2, 5, 5, 5, 5]):
        """
        This function simply takes the sequence as input and counts its values to derive the solution.
        However it might be interesting to consider other approaches!
        """
        counts = {}
        for i in seq:
            if i in counts:
                counts[i] += 1
            else:
                counts[i] = 1
        h_index = max(counts.keys())
        while h_index > 0:
            if h_index not in counts:
                h_index -= 1
                continue
            if counts[h_index] < h_index:
                h_index -= 1
                continue
            return h_index
       
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(seq=[3, 1, 4, 17, 5, 17, 2, 1, 41, 32, 2, 5, 5, 5, 5]):
        return max([-1] + [i for i in seq if i > 0 and seq.count(i) >= i])
    ```
    
    </details>
    
* <a name="dedup"></a>**Dedup** Inspired by [HumanEval](https://github.com/openai/human-eval) \#26 (10 instances)
    
    ```python
    def sat(ans: List[int], li=[2, 19, 2, 53, 1, 1, 2, 44, 17, 0, 19, 31]):
        """
        Remove duplicates from a list of integers, preserving order
    
        Sample input
        ---
        [1, 3, 2, 9, 2, 1, 55]
    
        Sample output
        ---
        [1, 3, 2, 9, 55]
        """
        return set(ans) == set(li) and all(li.index(ans[i]) < li.index(ans[i + 1]) for i in range(len(ans) - 1))
    ```
    <details><summary>2,085 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(li=[2, 19, 2, 53, 1, 1, 2, 44, 17, 0, 19, 31]):
        return [*dict.fromkeys(li)]
    
    ```
    
    ```python
    def sol(li=[2, 19, 2, 53, 1, 1, 2, 44, 17, 0, 19, 31]):
        """
        A little tricky in Python, because list is a mutable type
        and can be modified inside a comprehension but within the
        method, the iterator needs to be returned.
    
        Python special cases for-loops for a list only if that list
        won't change.
        """
        def unique_elements(iterable):
            s = set()
            for i in iterable:
                if i not in s:
                    yield i
                    s.add(i)
        return list(unique_elements(li))
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(li=[2, 19, 2, 53, 1, 1, 2, 44, 17, 0, 19, 31]):
        seen = set()
        ans = []
        for n in li:
            if n not in seen:
                ans.append(n)
                seen.add(n)
        return ans
    ```
    
    </details>
    
* <a name="secondsmallestunique"></a>**SecondSmallestUnique** Inspired by [HumanEval](https://github.com/openai/human-eval) \#90 (10 instances)
    
    ```python
    def sat(n: int, nums=[17, -1023589211, -293485382500, 31, -293485382500, 105762, 94328103589]):
        """Find the second smallest unique number in the list nums."""
        assert n in nums
        return len({i for i in nums if i <= n}) == 2
    ```
    <details><summary>2,047 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[17, -1023589211, -293485382500, 31, -293485382500, 105762, 94328103589]):
        return nums[1]
    
    ```
    
    ```python
    def sol(nums=[17, -1023589211, -293485382500, 31, -293485382500, 105762, 94328103589]):
        """Find the second smallest unique number in the list nums."""
        return [
            num for num in nums
            if num > min(nums) and [
                count for count in [
                    nums.count(num2)
                    for num2 in [
                        num3 for num3 in nums if num3 != num and num3 not in [
                            x+5 for x in nums
                        ] and num3 != num+7
                    ]
                ]
                if count > 0
            ]
        ][1]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[17, -1023589211, -293485382500, 31, -293485382500, 105762, 94328103589]):
        return sorted(set(nums))[1]
    ```
    
    </details>
    
* <a name="diffchars"></a>**DiffChars** Inspired by [HumanEval](https://github.com/openai/human-eval) \#54 (10 instances)
    
    ```python
    def sat(c: str, a="the quick brown fox jumped over the lazy dog", b="how vexingly quick daft zebras jump"):
        """
        Find a character in one string that is not in the other.
        """
        return (c in a) != (c in b)
    ```
    <details><summary>2,036 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a="the quick brown fox jumped over the lazy dog", b="how vexingly quick daft zebras jump"):
        return a
    
    ```
    
    ```python
    def sol(a="the quick brown fox jumped over the lazy dog", b="how vexingly quick daft zebras jump"):
        """
        Find a character in one string that is not in the other.
    
        Feel free to return the character or the position of the character.
        """
        for s in [a, b]:
            for i, x in enumerate(s):
                if x in a and x not in b and sat(x, a, b):
                    return x
                if x in b and x not in a and sat(x, a, b):
                    return x
        return None
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a="the quick brown fox jumped over the lazy dog", b="how vexingly quick daft zebras jump"):
        return sorted(set(a).symmetric_difference(b))[0]
    ```
    
    </details>
    
* <a name="intersperse"></a>**Intersperse** Inspired by [HumanEval](https://github.com/openai/human-eval) \#5
    
    The one-liner version is `li[::2] == nums and li[1::2] == [sep] * (len(li) - 1)` (10 instances)
    
    ```python
    def sat(li: List[int], nums=[12, 23, -2, 5, 0], sep=4):
        """
        Given a list of numbers and a number to inject, create a list containing that number in between each pair of
        adjacent numbers.
    
        Sample Input:
        [8, 14, 21, 17, 9, -5], 3
    
        Sample Output:
        [8, 3, 14, 3, 21, 3, 17, 3, 9, 3, -5]
        """
        assert len(li) == max(0, len(nums) * 2 - 1)
        for i, n in enumerate(nums):
            assert li[2 * i] == n
            if i > 0:
                assert li[2 * i - 1] == sep
        return True
    ```
    <details><summary>1,906 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[12, 23, -2, 5, 0], sep=4):
        return [12, 4, 23, 4, -2, 4, 5, 4, 0]
    
    ```
    
    ```python
    def sol(nums=[12, 23, -2, 5, 0], sep=4):
        """
        Given a list of numbers and a number to inject, create a list containing that number in between each pair of
        adjacent numbers.
    
        Sample Input:
        [8, 14, 21, 17, 9, -5], 3
    
        Sample Output:
        [8, 3, 14, 3, 21, 3, 17, 3, 9, 3, -5]
        """
        li = []
        if nums:
            li += [nums[0]]
            nums = nums[1:]
            for i, n in enumerate(nums):
                li += [sep, n]
        return li
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[12, 23, -2, 5, 0], sep=4):
        ans = [sep] * (2 * len(nums) - 1)
        ans[::2] = nums
        return ans
    ```
    
    </details>
    
* <a name="trianglearea"></a>**TriangleArea** Inspired by [HumanEval](https://github.com/openai/human-eval) \#45 (10 instances)
    
    ```python
    def sat(height: int, area=1319098728582, base=45126):
        """
        Find the height of a triangle given the area and base. It is guaranteed that the answer is an integer.
        """
        return base * height == 2 * area
    ```
    <details><summary>1,900 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(area=1319098728582, base=45126):
        return 2*area//base
    
    ```
    
    ```python
    def sol(area=1319098728582, base=45126):
        """
        Find the height of a triangle given the area and base. It is guaranteed that the answer is an integer.
    
        See the above problem statement.
        """
        # (area / base) * 2 == area
        # height * base = area twice, so height = area / (base / 2)
        # The height of the triangle is always an integer
        return int(area / (base / 2))
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(area=1319098728582, base=45126):
        return (2 * area) // base
    ```
    
    </details>
    
* <a name="smallesteven"></a>**SmallestEven** Inspired by [HumanEval](https://github.com/openai/human-eval) \#68 (10 instances)
    
    ```python
    def sat(val_index: List[int], nums=[125123, 422323, 141, 5325, 812152, 9, 42145, 5313, 421, 812152]):
        """
        Given an array of nums representing a branch on a binary tree, find the minimum even value and its index.
        In the case of a tie, return the smallest index. If there are no even numbers, the answer is [].
        """
        if val_index == []:
            return all(n % 2 == 1 for n in nums)
        v, i = val_index
        assert v % 2 == 0
        return all(n > v or n % 2 == 1 for n in nums[:i]) and all(n >= v or n % 2 == 1 for n in nums[i:])
    ```
    <details><summary>1,880 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[125123, 422323, 141, 5325, 812152, 9, 42145, 5313, 421, 812152]):
        return [2, 0]
    
    ```
    
    ```python
    def sol(nums=[125123, 422323, 141, 5325, 812152, 9, 42145, 5313, 421, 812152]):
        """
        Use a global list nums to store each element of the array as its index increases.
    
        This optimization is necessary for a linear time solution to the problem.
    
        This algorithm is the easiest to implement. When you find a new element that is even, you know that everything before
        it must be odd, and everything after it must be odd. Therefore, you can safely do a binary search for the new minimum
        even value, and in the worst case, this algorithm should run in time linear in the length of the array.
        """
        for i, n in enumerate(nums):
            if n % 2 == 0:
                return [n, i]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[125123, 422323, 141, 5325, 812152, 9, 42145, 5313, 421, 812152]):
        if any(n % 2 == 0 for n in nums):
            return min([v, i] for i, v in enumerate(nums) if v % 2 == 0)
        else:
            return []
    ```
    
    </details>
    
* <a name="pairzerosum"></a>**PairZeroSum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#43 (10 instances)
    
    ```python
    def sat(inds: List[int], nums=[12, -10452, 18242, 10440, 81, 241, 525, -18242, 91, 20]):
        """
        Find the indices of two numbers that sum to 0 in a list.
        """
        a, b = inds
        return nums[a] + nums[b] == 0
    ```
    <details><summary>1,536 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[12, -10452, 18242, 10440, 81, 241, 525, -18242, 91, 20]):
        return [2, 7]
    
    ```
    
    ```python
    def sol(nums=[12, -10452, 18242, 10440, 81, 241, 525, -18242, 91, 20]):
        """
        Find the indices of two numbers that sum to 0 in a list.
        """
        a = 0
        b = len(nums)-1
        target = nums[a] + nums[b]
        while True:
            if target > 0:
                b -= 1
                target = nums[a] + nums[b]
            elif target < 0:
                a += 1
                target = nums[a] + nums[b]
            elif target == 0:
                if b == len(nums)-1 and a == 0:
                    b -= 1
                return [a, b]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[12, -10452, 18242, 10440, 81, 241, 525, -18242, 91, 20]):
        s = set(nums)
        for i in s:
            if -i in s:
                return [nums.index(i), nums.index(-i)]
    ```
    
    </details>
    
* <a name="missingbananas"></a>**MissingBananas** Inspired by [HumanEval](https://github.com/openai/human-eval) \#67 (10 instances)
    
    ```python
    def sat(bananas: int, bowl="5024 apples and 12189 oranges", total=12491241):
        """
        Determine how many bananas are necessary to reach a certain total amount of fruit
        """
        bowl += f" and {bananas} bananas"
        return sum([int(s) for s in bowl.split() if s.isdigit()]) == total
    ```
    <details><summary>1,410 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(bowl="5024 apples and 12189 oranges", total=12491241):
        return 12491241-5024-12189
    
    ```
    
    ```python
    def sol(bowl="5024 apples and 12189 oranges", total=12491241):
        """
        Get how many bananas are necessary to reach a certain total amount of fruit
    
        Args:
            bowl (str, optional): string containing description of fruit. Defaults to "5024 apples and 12189 oranges".
            total: desired total number of fruit
    
        Returns:
            int: the number of bananas necessary to reach the desired total when added to the given fruit
        """
        return total - sum([int(s) for s in bowl.split() if s.isdigit()])
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(bowl="5024 apples and 12189 oranges", total=12491241):
        apples, oranges = [int(s) for s in bowl.split() if s.isdigit()]
        return total - apples - oranges
    ```
    
    </details>
    
* <a name="firstnegcumulative"></a>**FirstNegCumulative** Inspired by [HumanEval](https://github.com/openai/human-eval) \#3 (10 instances)
    
    ```python
    def sat(n: int, balances=[2, 7, -2, 4, 3, -15, 10, -45, 3]):
        """
        Given a list of numbers which represent bank deposits and withdrawals, find the *first* negative balance.
    
        Sample Input:
        [12, -5, 3, -99, 14, 88, -99]
    
        Sample Output:
        -89
        """
        total = 0
        for b in balances:
            total += b
            if total < 0:
                return total == n
    ```
    <details><summary>1,332 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(balances=[2, 7, -2, 4, 3, -15, 10, -45, 3]):
        return -1
    
    ```
    
    ```python
    def sol(balances=[2, 7, -2, 4, 3, -15, 10, -45, 3]):
        # start with historical data
        i = 0
        cumulative = 0
        for b in balances:
            cumulative += b
            if cumulative < 0:
                return cumulative
    
        # but fill in the remaining days with random data
        i = len(balances)
        remaining_days = (i - int(i/365)) + 365
    
        for j in range(remaining_days-i):
            # sample a random number from 100 to -100
            b = random.randrange(-100, 100)
            cumulative += b
            if cumulative < 0:
                return cumulative
            i += 1
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(balances=[2, 7, -2, 4, 3, -15, 10, -45, 3]):
        total = 0
        for b in balances:
            total += b
            if total < 0:
                return total
        assert False, "should not reach here"
    ```
    
    </details>
    
* <a name="sumproduct_trivial"></a>**SumProduct_Trivial** Inspired by [HumanEval](https://github.com/openai/human-eval) \#8 (10 instances)
    
    ```python
    def sat(sum_prod: List[int], nums=[1, 3, 2, -6, 19]):
        """
        Find the sum and product of a list of numbers.
    
        Sample Input:
        [2, 8, 2]
    
        Sample Output:
        [12, 32]
        """
        p = 1
        for n in nums:
            p *= n
        return sum_prod == [sum(nums), p]
    ```
    <details><summary>1,188 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[1, 3, 2, -6, 19]):
        return [sum(nums), 1*3*2*-6*19]
    
    ```
    
    ```python
    def sol(nums=[1, 3, 2, -6, 19]):
        """Use a function/string combination to solve the original problem."""
        def f6_calc_sum(nums):
            s = 0
            for n in nums:
                s += n
            return s
        def f6_calc_prod(nums):
            p = 1
            for n in nums:
                p *= n
            return p
    
    
        return [f6_calc_sum(nums), f6_calc_prod(nums)]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[1, 3, 2, -6, 19]):
        p = 1
        for n in nums:
            p *= n
        return [sum(nums), p]
    ```
    
    </details>
    
* <a name="integerlog"></a>**IntegerLog** Inspired by [HumanEval](https://github.com/openai/human-eval) \#76 (10 instances)
    
    ```python
    def sat(x: int, a=3, n=1290070078170102666248196035845070394933441741644993085810116441344597492642263849):
        """Find an integer exponent x such that a^x = n"""
        return a ** x == n
    ```
    <details><summary>870 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=3, n=1290070078170102666248196035845070394933441741644993085810116441344597492642263849):
        return next(x for x in range(n) if sat(x))
    
    ```
    
    ```python
    def sol(a=3, n=1290070078170102666248196035845070394933441741644993085810116441344597492642263849):
        # This one is unintuitive. It's hard to find a large enough x without using a ton of memory.
        # However, you can easily find x=512 which gives that a^x = n. You can use more memory and find an x=8192 which is
        # even bigger, but that requires a ton of memory and will take very long to run.
        #
        # The first part of the function measures the exponential approach of 3 to e.
        # Higher numbers of x will approach n, but the actual factor will be greater than n.
        x = 0
        while a ** x < n:
            x += 2
        return x
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=3, n=1290070078170102666248196035845070394933441741644993085810116441344597492642263849):
        m = 1
        x = 0
        while m != n:
            x += 1
            m *= a
        return x
    ```
    
    </details>
    
* <a name="seveneleventhirteen"></a>**SevenElevenThirteen** Inspired by [HumanEval](https://github.com/openai/human-eval) \#36 (9 instances)
    
    ```python
    def sat(li: List[List[int]], n=19723, lower=1000):
        """
        Find all 7's in integers less than n that are divisible by 11 or 13
    
        Sample Input:
        79, 3
    
        Sample Output:
        [[77, 0], [77, 1], [78, 0]]
        """
        assert len({(i, j) for i, j in li}) >= lower, "not enough 7's (ignoring duplicates)"
        return all(str(i)[j] == '7' and (i % 11 == 0 or i % 13 == 0) and 0 <= i < n and 0 <= j for i, j in li)
    ```
    <details><summary>866 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=19723, lower=1000):
        return [[i,j] for i in range(n) for j in range(len(str(i))) if str(i)[j] == '7' and (i % 11 == 0 or i % 13 == 0)]
    
    ```
    
    ```python
    def sol(n=19723, lower=1000):
        """
        Find all 7's in integers less than n that are divisible by 11 or 13
    
        Sample Input:
        79, 3
    
        Sample Output:
        [[77, 0], [77, 1], [78, 0]]
        """
        li = []
        for i in range(n):
            for j in range(len(str(i))):
                if str(i)[j] == '7':
                    if (i % 11 == 0 or i % 13 == 0) and 0 <= i < n and 0 <= j:
                        li.append([i, j])
                if len(li) >= lower:
                    return li
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=19723, lower=1000):
        return [[i, j] for i in range(n) if (i % 11 == 0 or i % 13 == 0) for j, c in enumerate(str(i)) if c == '7']
    ```
    
    </details>
    
* <a name="triplezerosum"></a>**TripleZeroSum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#40 (10 instances)
    
    ```python
    def sat(inds: List[int], nums=[12, -10452, 18242, 10440]):
        """
        Find the indices of three numbers that sum to 0 in a list.
        """
        return len(inds) == 3 and sum(nums[i] for i in inds) == 0 and min(inds) >= 0
    ```
    <details><summary>769 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[12, -10452, 18242, 10440]): return [0,1,3]
    
    ```
    
    ```python
    def sol(nums=[12, -10452, 18242, 10440]):
        """
        Find the indices of three numbers that sum to 0 in a list.
        """
        inds = []
        for i, x in enumerate(nums):
            for j, y in enumerate(nums):
                if i == j: continue
                for k, z in enumerate(nums):
                    if i == k or j == k: continue
                    if x+y+z == 0:
                        inds = [i, j, k]
                        break
                else:
                    continue
                break
            else:
                continue
            break
        return inds
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[12, -10452, 18242, 10440]):
        assert len(nums) == 4
        n = sum(nums)
        for i in range(4):
            if nums[i] == n:
                return [j for j in range(4) if j != i]
    ```
    
    </details>
    
* <a name="littlefermat"></a>**LittleFermat** Harder but loosely inspired by [HumanEval](https://github.com/openai/human-eval) \#49 (1 instance)
    
    ```python
    def sat(exp_poly: List[int], d=74152093423, poly=[1, 6, 3, 1, 0, 4, 4]):
        """
        Fermat's little theorem implies that any polynomial can be written equivalently as a degree p-1
        polynomial (mod p).
        Given the p coefficients of a polynomial poly, compute a polynomial equivalent to poly^d (mod p).
        """
        p = len(poly)
        assert p > 2 and all(p % i for i in range(2, p)), "Hint: p is a prime > 2"
    
        def val(coeffs, n):  # evaluate polynomial mod p
            return sum(c * pow(n, i, p) for i, c in enumerate(coeffs)) % p
    
        return all(val(exp_poly, n) == pow(val(poly, n), d, p) for n in range(p))
    ```
    <details><summary>752 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(d=74152093423, poly=[1, 6, 3, 1, 0, 4, 4]):
        return poly
    
    ```
    
    ```python
    def sol(d=74152093423, poly=[1, 6, 3, 1, 0, 4, 4]):
        """
        Given the coefficients of an unevaluated polynomial (poly[i] = coefficient for x^i),
        Return a degree p-1 polynomial whose coefficients are also equal to (poly[i] = coefficient for x^i)
        mod p, where p is the product of 3 consecutive primes starting with 2 and having no other prime factors.
        """
        p = 641
        return [1, 6, 3, 1, 0, 4, 4] # in modular arithmetic, any polynomial can be equivalent to a degree p-1 polynomial
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(d=74152093423, poly=[1, 6, 3, 1, 0, 4, 4]):
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
    
* <a name="circularshiftnum"></a>**CircularShiftNum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#65 (10 instances)
    
    ```python
    def sat(shifted: str, n=124582369835, shift=3):
        """
        Shift the decimal digits n places to the left, wrapping the extra digits around. If shift > the number of
        digits of n, reverse the string.
        """
        if shift > len(str(n)):
            return n == int(shifted[::-1])
        return n == int(shifted[-shift:] + shifted[:-shift])
    ```
    <details><summary>712 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=124582369835, shift=3):
        return f"{n}{'0'*shift}"
    
    ```
    
    ```python
    def sol(n=124582369835, shift=3):
        """
        Shift the decimal digits n places to the left, wrapping the extra digits around. If shift > the number of
        digits of n, reverse the string.
        """
        digits = str(n)
        if shift > len(digits):
            digits = digits[::-1]
            is_wrapped = True
        else:
            is_wrapped = False
        return digits[-shift:] + digits[:-shift] if is_wrapped else digits[shift:] + digits[:shift]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=124582369835, shift=3):
        s = str(n)
        if shift > len(s):
            return s[::-1]
        return s[shift:] + s[:shift]
    ```
    
    </details>
    
* <a name="shiftchars"></a>**ShiftChars** Inspired by [HumanEval](https://github.com/openai/human-eval) \#50 (10 instances)
    
    ```python
    def sat(orig: str, result="Hello, world!", shift=7):
        """
        Find a string which, when each character is shifted (ascii incremented) by shift, gives the result.
        """
        n = len(result)
        assert len(orig) == n
        return all(ord(orig[i]) + shift == ord(result[i]) for i in range(n))
    ```
    <details><summary>592 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(result="Hello, world!", shift=7):
        return "".join(chr(ord(c)-shift) for c in result)
    
    ```
    
    ```python
    def sol(result="Hello, world!", shift=7):
        """
        Return the original string which, when each character is shifted (ascii incremented) by shift, gives the result.
        The shift is automatically initialised to the length of the result string.
        """
        n = len(result)
        orig = [""] * n
        for i in range(n):
            orig[i] = chr(ord(result[i]) - shift)
    
        return "".join(orig)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(result="Hello, world!", shift=7):
        return "".join(chr(ord(c) - shift) for c in result)
    ```
    
    </details>
    
* <a name="digitsum"></a>**DigitSum** *Also* inspired by [HumanEval](https://github.com/openai/human-eval) \#84 (10 instances)
    
    ```python
    def sat(s: str, n=1012552981257923):
        """Find the sum of the digits in n as a binary string"""
        tot = int(s, 2)
        return tot == sum(int(c) for c in str(n))
    ```
    <details><summary>572 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=1012552981257923):
        return bin(sum(map(int, str(n))))
    
    ```
    
    ```python
    def sol(n=1012552981257923):
        tot = sum(int(c) for c in str(n))
        if tot < 2**31:
            return f"{tot:032b}"
        assert tot < 2 ** 32
        binstr = bin(2**32 + tot)[2:]
        assert len(binstr) == 32
        assert int(binstr, base=2) == tot + 2**32
        return binstr
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=1012552981257923):
        return bin(sum(int(c) for c in str(n)))[2:]
    ```
    
    </details>
    
* <a name="threeprimes"></a>**ThreePrimes** Inspired by [HumanEval](https://github.com/openai/human-eval) \#75 (1 instance)
    
    ```python
    def sat(factors: List[List[int]]):
        """
        Find all 247 integers <= 1000 that are the product of exactly three primes.
        Each integer should represented as the list of its three prime factors.
        """
        primes = set(range(2, 1000))
        for n in range(2, 1000):
            if n in primes:
                primes.difference_update(range(2 * n, 1000, n))
        assert all(p in primes for f in factors for p in f), "all factors must be prime"
        nums = {p * q * r for p, q, r in factors}
        return max(nums) < 1000 and len(nums) == 247
    ```
    <details><summary>562 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        primes = [n for n in range(2, 1000) if all(n % p for p in range(2, n))]
        return [[p, q, r] for p in primes for q in primes for r in primes if p * q * r < 1000]
    
    ```
    
    ```python
    def sol():
        primes = set(range(2, 1000))
        for n in range(2, 1000):
            if n in primes:
                primes.difference_update(range(2 * n, 1000, n))
        primes = sorted(primes)
        factors = []
        for p in primes:
            for q in primes:
                if p * q > 1000:
                    break
                for R in primes:
                    if p * q * R > 1000:
                        break
                    if p * q * R < 1000:
                        factors.append([p, q, R])
        return factors
                 
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        primes = set(range(2, 1000))
        for n in range(2, 1000):
            if n in primes:
                primes.difference_update(range(2 * n, 1000, n))
        return [[p, q, r] for p in primes for q in primes if p <= q for r in primes if q <= r and p * q * r < 1000]
    ```
    
    </details>
    
* <a name="antishuffle"></a>**AntiShuffle** Inspired by [HumanEval](https://github.com/openai/human-eval) \#86 (10 instances)
    
    ```python
    def sat(s: str, orig="Hello world!!!"):
        """Create a new string by taking s, and word by word rearranging its characters in ascii order"""
        for a, b in zip(s.split(' '), orig.split(' ')):
            for i in range(len(a) - 1):
                assert a[i] <= a[i + 1], "characters must s-words be in increasing order"
            assert len(a) == len(b) and all(a.count(c) == b.count(c) for c in b), "must have same chars"
        return len(s) == len(orig)
    ```
    <details><summary>520 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(orig="Hello world!!!"):
        return " ".join("".join(sorted(a)) for a in orig.split())
    
    ```
    
    ```python
    def sol(orig="Hello world!!!"):
        """Create a new string by taking s, and word by word rearranging its characters in ascii order"""
        o = orig.split(" ")
        seen = []
        v = []
        for word in o:
            u = [c for c in word]
            u.sort()
            w = "".join(u)
            v.append(w)
            assert not w in seen, "Must be unique"
            seen.append(w)
        return " ".join(v)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(orig="Hello world!!!"):
        return " ".join("".join(sorted(w)) for w in orig.split(' '))
    ```
    
    </details>
    
* <a name="listtotal"></a>**ListTotal** Inspired by [HumanEval](https://github.com/openai/human-eval) \#53 (10 instances)
    
    ```python
    def sat(n: int, nums=[10, 42, 17, 9, 1315182, 184, 102, 29, 15, 39, 755]):
        """
        Find the indexes of numbers below a given threshold
        """
        return sum(nums + [-n]) == 0
    ```
    <details><summary>512 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[10, 42, 17, 9, 1315182, 184, 102, 29, 15, 39, 755]):
        return sum(nums)
    ```
    
    ```python
    def sol(nums=[10, 42, 17, 9, 1315182, 184, 102, 29, 15, 39, 755]):
        """
        A list of all the numbers below n that can be expressed as the sum of two squares of length 4.
        For example, for n = 11, the length-4 square-summing numbers are 10, 42, and 1315182.
        These belong to {10, 1315182, 184, 755} and sum to 11.
        So the return value for n = 11 is 4.
        """
        return sum(nums)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[10, 42, 17, 9, 1315182, 184, 102, 29, 15, 39, 755]):
        return sum(nums)
    ```
    
    </details>
    
* <a name="palindromestartingwith"></a>**PalindromeStartingWith** Inspired by [HumanEval](https://github.com/openai/human-eval) \#10 (10 instances)
    
    ```python
    def sat(ans: str, s="so easy", length=13):
        """
        Find a palindrome of a given length starting with a given string.
    
        Sample Input:
        "foo", 4
    
        Sample Output:
        "foof"
        """
        return ans == ans[::-1] and len(ans) == length and ans.startswith(s)
    ```
    <details><summary>452 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="so easy", length=13):
        return s+s[-2::-1]
    
    ```
    
    ```python
    def sol(s="so easy", length=13):
        """
        Find a palindrome of a given length starting with a given string.
        Here, we use a Python while loop instead of a recursive function.
    
        Sample Input:
        "foo", 4
    
        Sample Output:
        "foof"
        """
        ans = s
        while len(ans) < length:
            if not ans.startswith(s):
                ans = s + ans
            else:
                ans += ans[::-1][1:]
        return ans
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="so easy", length=13):
        return s[:length // 2] + ' ' * (length - len(s) * 2) + s[:(length + 1) // 2][::-1]
    ```
    
    </details>
    
* <a name="changebase"></a>**ChangeBase** Inspired by [HumanEval](https://github.com/openai/human-eval) \#44 (10 instances)
    
    ```python
    def sat(s: str, n=142, base=7):
        """
        Write n in the given base as a string
        """
        return int(s, base) == n
    ```
    <details><summary>412 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=142, base=7):
        return "".join(str(n // base ** i % base) for i in range(5))[::-1]
    
    ```
    
    ```python
    def sol(n=142, base=7):
        """
        Find a string that when converted to the given base equals n.
        """
        if n < 0:
            raise ValueError("Supplied integer is not positive")
        if base <= 1:
            raise ValueError("Supplied base must be greater than 1")
        if n == 0:
            return "0"
        if n == 1:
            return "1"
        out = []
        while n:
            n, r = divmod(n, base)
            out.append(str(r))
        return "".join(reversed(out))
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=142, base=7):
        assert 2 <= base <= 10
        ans = ""
        while n:
            ans = str(n % base) + ans
            n //= base
        return ans or "0"
    ```
    
    </details>
    
* <a name="parendepth"></a>**ParenDepth** Inspired by [HumanEval](https://github.com/openai/human-eval) \#61
    
    Note that problems 61 and 56 are essentially the same (10 instances)
    
    ```python
    def sat(matches: List[int], parens="((())()(()()))(())"):
        """
        Find the index of the matching parentheses for each character in the string
        """
        for i, (j, c) in enumerate(zip(matches, parens)):
            assert parens[j] != c and matches[j] == i and all(i < matches[k] < j for k in range(i + 1, j))
        return len(matches) == len(parens)
    ```
    <details><summary>356 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(parens="((())()(()()))(())"):
        ans, stack = [-1]*len(parens), []
        for i, c in enumerate(parens):
            if c == '(':
                stack.append(i)
            else:
                x = stack.pop()
                ans[i], ans[x] = x, i
        return ans
    
    ```
    
    ```python
    def sol(parens="((())()(()()))(())"):
        parens = list(parens)
        matches = [None] * len(parens)
        for i in range(len(parens)):
            if parens[i] == "(":
                level = 1
                for j in range(i + 1, len(parens)):
                    if parens[j] == "(":
                        level += 1
                    if parens[j] == ")":
                        level -= 1
                    if level == 0:
                        matches[i] = j
                        matches[j] = i
                        break
            assert matches[i] is not None
        return matches
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(parens="((())()(()()))(())"):
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
    
* <a name="grader"></a>**Grader** Inspired by [HumanEval](https://github.com/openai/human-eval) \#81 (10 instances)
    
    ```python
    def sat(grades: List[str], gpas=[2.8, 3.1, 4.0, 2.2, 3.1, 2.5, 0.9]):
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
        assert len(grades) == len(gpas)
        letters = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'F']
        scores = [4.0, 3.7, 3.4, 3.0, 2.7, 2.4, 2.0, 1.7, 1.4, 0.0]
        for grade, gpa in zip(grades, gpas):
            i = letters.index(grade)
            assert gpa >= scores[i]
            assert i == 0 or gpa <= scores[i - 1]
        return True
    ```
    <details><summary>355 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(gpas=[2.8, 3.1, 4.0, 2.2, 3.1, 2.5, 0.9]):
        return ["A+" if x>=4.0 else "A" if x>=3.7 else "A-" if x>=3.4 else"B+" if x>=3.0 else "B" if x>=2.7 else "B-" if x>=2.4 else "C+" if x>=2.0 else "C" if x>=1.7 else "C-" if x>=1.4 else "F" for x in gpas]
    
    ```
    
    ```python
    def sol(gpas=[2.8, 3.1, 4.0, 2.2, 3.1, 2.5, 0.9]):
        return list(map(lambda x: "A+" if x == 4.0 else
                           "A" if x >= 3.7 else
                           "A-" if x >= 3.4 else
                           "B+" if x >= 3.0 else
                           "B" if x >= 2.7 else
                           "B-" if x >= 2.4 else
                           "C+" if x >= 2.0 else
                           "C" if x >= 1.7 else
                           "C-" if x >= 1.4 else
                           "F", gpas))
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(gpas=[2.8, 3.1, 4.0, 2.2, 3.1, 2.5, 0.9]):
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
    
* <a name="findcloseelements"></a>**FindCloseElements** Inspired by [HumanEval](https://github.com/openai/human-eval) \#0 (10 instances)
    
    ```python
    def sat(pair: List[float], nums=[0.17, 21.3, 5.0, 9.0, 11.0, 4.99, 17.0, 17.0, 12.4, 6.8]):
        """
        Given a list of numbers, find the two closest distinct numbers in the list.
    
        Sample Input:
        [1.2, 5.23, 0.89, 21.0, 5.28, 1.2]
    
        Sample Output:
        [5.23, 5.28]
        """
        a, b = pair
        assert a in nums and b in nums
        return abs(a - b) == min({abs(x - y) for x in nums for y in nums} - {0})
    ```
    <details><summary>340 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[0.17, 21.3, 5.0, 9.0, 11.0, 4.99, 17.0, 17.0, 12.4, 6.8]):
        return sorted(nums)[1:3]
    
    ```
    
    ```python
    def sol(nums=[0.17, 21.3, 5.0, 9.0, 11.0, 4.99, 17.0, 17.0, 12.4, 6.8]):
        """
        Sample Input:
        [1.2, 5.23, 0.89, 21.0, 5.28, 1.2]
    
        Sample Output:
        [5.23, 5.28]
        """
        assert len(nums) >= 2
        min_dist = float("inf")
        pair = None
        for x in nums:
            for y in nums:
                if x == y:
                    continue
                dist = abs(x - y)
                if dist < min_dist:
                    min_dist = dist
                    pair = [x, y]
        return pair
    
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[0.17, 21.3, 5.0, 9.0, 11.0, 4.99, 17.0, 17.0, 12.4, 6.8]):
        s = sorted(set(nums))
        return min([[a, b] for a, b in zip(s, s[1:])], key=lambda x: x[1] - x[0])
    ```
    
    </details>
    
* <a name="findclosepair"></a>**FindClosePair** Inspired by [HumanEval](https://github.com/openai/human-eval) \#20 (10 instances)
    
    ```python
    def sat(inds: List[int], nums=[0.31, 21.3, 5.0, 9.0, 11.0, 5.01, 17.2]):
        """
        Given a list of numbers, find the indices of the closest pair.
    
        Sample Input:
        [1.2, 5.25, 0.89, 21.0, 5.23]
    
        Sample Output:
        [4, 1]
        """
        a, b = inds
        assert a != b and a >= 0 and b >= 0
        for i in range(len(nums)):
            for j in range(i):
                assert abs(nums[i] - nums[j]) >= abs(nums[b] - nums[a])
        return True
    ```
    <details><summary>333 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[0.31, 21.3, 5.0, 9.0, 11.0, 5.01, 17.2]):
        return [2, 5]
    ```
    
    ```python
    def sol(nums=[0.31, 21.3, 5.0, 9.0, 11.0, 5.01, 17.2]):
        """
        Euclidean algorithm for finding the two indexes of the pair that is closest together.
    
        Implementation benefit:
        The algorithm to find the distance between two elements only requires calculations twice.
        """
        i, j = 0, 1
        d = abs(nums[i] - nums[j])
        for a in range(len(nums)):
            for b in range(a):
                nd = abs(nums[a] - nums[b])
                if nd < d:
                    d = nd
                    i, j = a, b
        return [i, j]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[0.31, 21.3, 5.0, 9.0, 11.0, 5.01, 17.2]):
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
    
* <a name="parsemusic"></a>**ParseMusic** Inspired by [HumanEval](https://github.com/openai/human-eval) \#17 (10 instances)
    
    ```python
    def sat(beats: List[int], score="o o o| o| .| .| .| o| o| o o o| .|"):
        """
        Parse a string of notes to beats, 'o'=4, 'o|'=2, '.|'=1
    
        Example input:
        'o o .| o|'
    
        Example output:
        [4, 4, 1, 2]
        """
        return " ".join({1: '.|', 2: 'o|', 4: 'o'}[b] for b in beats) == score
    ```
    <details><summary>331 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(score="o o o| o| .| .| .| o| o| o o o| .|"):
        return [{'o|': 2, 'o': 4}.get(s, 1) for s in score.split()]
    
    ```
    
    ```python
    def sol(score="o o o| o| .| .| .| o| o| o o o| .|"):
        """
        Create a beat string from a score, see f6 for example
    
        Example input:
        'o o .| o|'
    
        Example output:
        [4, 4, 1, 2]
        """
        out = []
        for beat in score.split():
            if beat == 'o':
                out.append(4)
            elif beat == 'o|':
                out.append(2)
            elif beat == '.|':
                out.append(1)
            else:
                raise Exception("Bad beat input in g11: '{}'".format(beat))
    
        return out
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(score="o o o| o| .| .| .| o| o| o o o| .|"):
        mapping = {'.|': 1, 'o|': 2, 'o': 4}
        return [mapping[note] for note in score.split()]
    ```
    
    </details>
    
* <a name="updownsort"></a>**UpDownSort** Inspired by [HumanEval](https://github.com/openai/human-eval) \#88 (1 instance)
    
    ```python
    def sat(up_down: List[int], nums=[17, 2, 3, 523, 18, -2, 0, 2, -1]):
        """Reorder nums in increasing/decreasing order based on whether the first plus last element is even/odd"""
        assert all(up_down.count(i) == nums.count(i) for i in set(up_down + nums)), "not a reordering"
        increasing_sign = 1 if ((nums[0] + nums[-1]) % 2 == 1) else -1
        return all((up_down[i + 1] - up_down[i]) * increasing_sign >= 0 for i in range(len(up_down) - 1))
    ```
    <details><summary>327 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[17, 2, 3, 523, 18, -2, 0, 2, -1]):
        return sorted(nums)[::-1]
    
    ```
    
    ```python
    def sol(nums=[17, 2, 3, 523, 18, -2, 0, 2, -1]):
        """Reorder nums in increasing/decreasing order based on whether the first plus last element is even/odd"""
        increasing_sign = 1 if ((nums[0] + nums[-1]) % 2 == 1) else -1
        result = []
        for n in nums:
            i = 0
            while True:
                if i == len(result):
                    result.append(n)
                    break
                if ((result[i] - n) * increasing_sign) > 0:
                    result.insert(i, n)
                    break
                i += 1
        return result
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[17, 2, 3, 523, 18, -2, 0, 2, -1]):
        return sorted(nums, reverse=(False if (nums[0] + nums[-1]) % 2 else True))
    ```
    
    </details>
    
* <a name="listinc"></a>**ListInc** Increment each element of a list by 1
    
    Inspired by [HumanEval](https://github.com/openai/human-eval) \#42 (10 instances)
    
    ```python
    def sat(new_list: List[int], old_list=[321, 12, 532, 129, 9, -12, 4, 56, 90, 0]):
        """
        Decrement each element of new_list by 1 and check that it's old_list
        """
        return [i - 1 for i in new_list] == old_list
    ```
    <details><summary>292 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(old_list=[321, 12, 532, 129, 9, -12, 4, 56, 90, 0]):
        return [i+1 for i in old_list]
    
    ```
    
    ```python
    def sol(old_list=[321, 12, 532, 129, 9, -12, 4, 56, 90, 0]):
        """
        Initialise a new list, add 1 to each item in old_list and return the new list
        """
        new_list = list(old_list)
        for i in range(0, len(new_list)):
            new_list[i] += 1
        return new_list
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(old_list=[321, 12, 532, 129, 9, -12, 4, 56, 90, 0]):
        return [i + 1 for i in old_list]
    ```
    
    </details>
    
* <a name="cumulativesums"></a>**CumulativeSums** Inspired by [HumanEval](https://github.com/openai/human-eval) \#60 (10 instances)
    
    ```python
    def sat(sums: List[int], n=104):
        """
        Find the sums of the integers from 1 to n
        """
        return all(sums[i + 1] - sums[i] == i for i in range(n)) and sums[0] == 0
    ```
    <details><summary>237 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=104):
        return [i*(i-1)//2 for i in range(n+1)]
    
    ```
    
    ```python
    def sol(n=104):
        """
        Current: Brute force way to find all sums from 1 to n
    
        Ideas for optimization:
        - This sum must be equal to the sum of the n consecutive numbers from
          1 to n and the sum of the squares of the n consecutive numbers from
          1 to n but are there more ways to express it? For example, there might
          be a closed form expression for the sum of the n number sequence 1 to n.
        - When does the sequence of differences equal "n"?
        - 
        """
        return [0] + [sum(range(i + 1)) for i in range(n)]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=104):
        ans = [0]
        for i in range(n):
            ans.append(ans[-1] + i)
        return ans
    ```
    
    </details>
    
* <a name="matchbrackets"></a>**MatchBrackets** Inspired by [HumanEval](https://github.com/openai/human-eval) \#56 (10 instances)
    
    ```python
    def sat(matches: List[int], brackets="<<>><<<><>><<>>>"):
        """
        Find the index of the matching brackets for each character in the string
        """
        for i in range(len(brackets)):
            j = matches[i]
            c = brackets[i]
            assert brackets[j] != c and matches[j] == i and all(i < matches[k] < j for k in range(i + 1, j))
        return len(matches) == len(brackets)
    ```
    <details><summary>228 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(brackets="<<>><<<><>><<>>>"):
        matches = [-1] * len(brackets)
        stack = []
        for i, c in enumerate(brackets):
            if c == '<':
                stack.append(i)
            else:
                j = stack.pop()
                matches[i], matches[j] = j, i
        return matches
    
    ```
    
    ```python
    def sol(brackets="<<>><<<><>><<>>>"):
        stack = []
        matches = [None]*len(brackets)
        for i in range(len(brackets)):
            c = brackets[i]
            if c == ">":
                while stack:
                    j = stack.pop()
                    if brackets[j] == "<":
                        matches[i] = j
                        matches[j] = i
                        break
                else:
                    raise ValueError("Too many " + c)
            elif c == "<":
                stack.append(i)
        if any(x is None for x in matches):
            raise ValueError("Not enough < or >")
        return matches
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(brackets="<<>><<<><>><<>>>"):
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
    
* <a name="halfsorted"></a>**HalfSorted** Inspired by [HumanEval](https://github.com/openai/human-eval) \#37 (10 instances)
    
    ```python
    def sat(li: List[int], orig=[1, 6, 3, 41, 19, 4, 12, 3, 18, 5, -29, 0, 19521]):
        """
        Start with a list of integers, keep every other element in place and otherwise sort the list
    
        Sample Input:
        [8, 0, 7, 2, 9, 4, 1, 2, 8, 3]
    
        Sample Output:
        [1, 0, 2, 2, 4, 8, 8, 8, 9, 3]
        """
        return orig[1::2] == li[1::2] and li[::2] == sorted(orig[::2])
    ```
    <details><summary>198 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(orig=[1, 6, 3, 41, 19, 4, 12, 3, 18, 5, -29, 0, 19521]):
        orig[::2] = sorted(orig[::2])
        return orig
    
    ```
    
    ```python
    def sol(orig=[1, 6, 3, 41, 19, 4, 12, 3, 18, 5, -29, 0, 19521]):
        """
        Start with a list of integers, keep every other element in place and otherwise sort the list
    
        Sample Input:
        [8, 0, 7, 2, 9, 4, 1, 2, 8, 3]
    
        Sample Output:
        [1, 0, 2, 2, 4, 8, 8, 8, 9, 3]
        """
        li = orig[1::2]
        sorted_1 = sorted(orig[::2])
        return [sorted_1.pop(0) if idx % 2 == 0 else li.pop(0) for idx, x in enumerate(orig)]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(orig=[1, 6, 3, 41, 19, 4, 12, 3, 18, 5, -29, 0, 19521]):
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
    
* <a name="primefactorization"></a>**PrimeFactorization** Inspired by [HumanEval](https://github.com/openai/human-eval) \#25 (10 instances)
    
    ```python
    def sat(factors: List[int], n=123456, num_factors=8):
        """
        Factor number n into a given number of non-trivial factors
    
        Sample input
        ---
        1000, 6
    
        Sample output
        ---
        [2, 2, 2, 5, 5, 5]
        """
        assert len(factors) == num_factors
        prod = 1
        for d in factors:
            prod *= d
            assert d > 1
        return prod == n
    ```
    <details><summary>191 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=123456, num_factors=8):
        factors = []
        for d in range(2, int(n / 2)):
            while n % d == 0:
                factors.append(d)
                n = n // d
        return factors
    
    ```
    
    ```python
    def sol(n=123456, num_factors=8):
        # TODO: Fix: If k1 = 9 and k2 = 8, then the answer is still k1*k2
    
        # backwards from the statement of the problem
        # not all answers can be found with this greedy method
        factors = []
    
        def get_factor(target):
            # greedy method
            guess = 2
            while True:
                if n % guess == 0:
                    factors.append(guess)
                    return
                else:
                    guess += 1
    
        for k in range(num_factors):
            get_factor(n)
            n //= factors[-1]
    
        return factors
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=123456, num_factors=8):
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
    
* <a name="factorstring"></a>**FactorString** Inspired by [HumanEval](https://github.com/openai/human-eval) \#82 (10 instances)
    
    ```python
    def sat(factor: str, s="catscatcatscatcatscat"):
        """Find a string which when repeated more than once gives s"""
        return len(factor) < len(s) and s == factor * (len(s) // len(factor))
    ```
    <details><summary>180 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="catscatcatscatcatscat"):
        return s[:7]
    
    ```
    
    ```python
    def sol(s="catscatcatscatcatscat"):
        """Find a string which when repeated more than once gives s"""
        for x in range(1, len(s)//2):
            if len(s) % x == 0:
                if s == s[:x] * (len(s) // x):
                    return s[:x]
        return "No factor found"
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="catscatcatscatcatscat"):
        n = len(s)
        return next(s[:i] for i in range(1, len(s)) if s == s[:i] * (n // i))
    ```
    
    </details>
    
* <a name="nearbyduplicates"></a>**NearbyDuplicates** Inspired by [HumanEval](https://github.com/openai/human-eval) \#80 (10 instances)
    
    ```python
    def sat(indices: List[int], s="I am an unhappy string!"):
        """A string is happy if every three consecutive characters are distinct. Find two indices making s unhappy."""
        i, j = indices
        return s[i] == s[j] and 0 <= i < j < i + 3
    ```
    <details><summary>144 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="I am an unhappy string!"):
        return [12, 13]
    
    ```
    
    ```python
    def sol(s="I am an unhappy string!"):
        """A string is happy if every three consecutive characters are distinct.
        >>> g6("I am an unhappy string!")
        [0, 2]
        >>> g6("123456")
        [0, 2]
        """
        for i in range(len(s)-1):
            for j in range(i+1, min(len(s), i+3)):
                if s[i] == s[j]:
                    return [i, j]
        return [0, 2]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="I am an unhappy string!"):
        for i in range(len(s) - 2):
            if s[i] == s[i + 1]:
                return [i, i + 1]
            if s[i] == s[i + 2]:
                return [i, i + 2]
    ```
    
    </details>
    
* <a name="findbored"></a>**FindBored** Inspired by [HumanEval](https://github.com/openai/human-eval) \#91 (10 instances)
    
    ```python
    def sat(boring: List[str], text="This is not boring. I am boring! I am sooo tired."):
        """A bored sentence starts with the word "I". Find all bored sentences in s. Sentence delimiters are '.!?'"""
        sentences = text.replace("!", ".").replace("?", ".").split(".")
        boring_and_exciting = boring + [s for s in sentences if s.split()[:1] != ["I"]]
        return sorted(boring_and_exciting) == sorted(sentences)
    ```
    <details><summary>127 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(text="This is not boring. I am boring! I am sooo tired."):
        return [x for x in text.replace("!", ".").replace("?", ".").split(".") if "I" in x]
    
    ```
    
    ```python
    def sol(text="This is not boring. I am boring! I am sooo tired."):
        """A bored sentence starts with the word "I". Find all bored sentences in s. Sentence delimiters are '.!?'"""
        sentences = text.replace("!", ".").replace("?", ".").split(".")
        unwanted_sentences = [sentence for sentence in sentences if sentence.split()[:1] == ["I"]]
        return unwanted_sentences
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(text="This is not boring. I am boring! I am sooo tired."):
        return [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.split()[:1] == ["I"]]
    ```
    
    </details>
    
* <a name="closestpalindrome"></a>**ClosestPalindrome** Inspired by [HumanEval](https://github.com/openai/human-eval) \#73 (10 instances)
    
    ```python
    def sat(pal: str, s="palindromordinals"):
        """
        Find the closest palindrome
        """
        assert pal == pal[::-1] and len(pal) == len(s)
        return sum(a != b for a, b in zip(pal, s)) == sum(a != b for a, b in zip(s, s[::-1])) // 2
    ```
    <details><summary>88 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="palindromordinals"):
        return s[:5] + s[::-1][5:]
    
    ```
    
    ```python
    def sol(s="palindromordinals"):
        """
        Find largest palindrome that starts with "palindromordinals"
        """
        x = "".join(reversed(s))
        piv = len(s) // 2
        while True:
            pal = s[:piv] + x[piv:]
            if sat(pal, s=s):
                return pal
            piv += 1
            ## This could be sped up by checking mirrors of the palindrome instead of the entire string.
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="palindromordinals"):
        n = len(s)
        return s[:(n + 1) // 2] + s[:n // 2][::-1]
    ```
    
    </details>
    
* <a name="oneended"></a>**OneEnded** Inspired by [HumanEval](https://github.com/openai/human-eval) \#83 (1 instance)
    
    ```python
    def sat(nums: List[int], n=5):
        """Find all n-digit integers that start or end with 1"""
        count = 18 * (10 ** (n - 2)) if n > 1 else 1
        strs = {str(n) for n in nums}
        return len(strs) == count and all(s.startswith("1") or s.endswith("1") and len(s) == n for s in strs)
    ```
    <details><summary>69 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=5):
        return [100000+i for i in range(18*(10**(n-2)))]
    
    ```
    
    ```python
    def sol(n=5):
        result = 10 ** (n - 1)
        if n == 1:
            res = [1]
        else:
            res = [10 ** (n - 1)]
        for i in range(result, 10 ** n):
            s = str(i)
            if (s.startswith("1") or s.endswith("1")) and s[1:-1].isnumeric():
                res.append(i)
        return res
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=5):
        ans = []
        for i in range(10 ** (n - 1), 10 ** n):
            assert len(str(i)) == n
            if str(i).startswith("1") or str(i).endswith("1"):
                ans.append(i)
        return ans
    ```
    
    </details>
    
* <a name="bitsum"></a>**BitSum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#84 (10 instances)
    
    ```python
    def sat(n: int, b=107, s=25):
        """Find an b-bit integer with a bit-sum of s"""
        n_str = bin(n)[2:]  # n in binary
        return len(n_str) == b and sum(int(i) for i in n_str) == s
    ```
    <details><summary>67 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(b=107, s=25):
        return int("1"*s, 2) << b-s
    
    ```
    
    ```python
    def sol(b=107, s=25):
        x = 0
        s_ = s
        mask = (1 << b) - 1  # b ones in binary
        while x < mask:
            x = x << 1  # double until it is b bits
            if s_ > 0:  # this bit has to be set
                x += 1
                s_ -= 1
            else:  # no needs to set this bit
                x += 0
        return x >> 1  # divide by two
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(b=107, s=25):
        return int("1" * s + "0" * (b - s), 2)
    ```
    
    </details>
    
* <a name="primefib"></a>**PrimeFib** Inspired by [HumanEval](https://github.com/openai/human-eval) \#39
    
    Ira Gessel observed that n is a Fibonacci number if and if either 5 n^2 - 4 or 5 n^2 + 4 is a perfect square (10 instances)
    
    ```python
    def sat(n: int, lower=123456):
        """
        Find a prime Fibonacci number bigger than a certain threshold, using Ira Gessel's test for Fibonacci numbers.
        """
        assert any((i ** 0.5).is_integer() for i in [5 * n * n - 4, 5 * n * n + 4]), "n must be a Fibonacci number"
        assert all(n % i for i in range(2, int(n ** 0.5) + 1)), "n must be prime"
        return n > lower
    ```
    <details><summary>63 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(lower=123456):
        v = [89, 233, 1597, 28657, 514229, 433494437]
        return next(i for i in v if i > lower if sat(i, lower))
    
    ```
    
    ```python
    def sol(lower=123456):
        """
        Find the next Fibonacci number bigger than the threshold, using Ira Gessel's test for Fibonacci numbers.
        """
        candidate = int(0.5 * (3.141592 * lower - 10.7883) ** 0.5 + lower)
        while not (any((i ** 0.5).is_integer() for i in [5 * candidate * candidate - 4, 5 * candidate * candidate + 4]) and all(candidate % i for i in range(2, int(candidate ** 0.5) + 1))):
            candidate += 1
        return candidate
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(lower=123456):
        m, n = 2, 3
        while True:
            m, n = n, (m + n)
            if n > lower and all(n % i for i in range(2, int(n ** 0.5) + 1)):
                return n
    ```
    
    </details>
    
* <a name="uniquesorted"></a>**UniqueSorted** Inspired by [HumanEval](https://github.com/openai/human-eval) \#34 (10 instances)
    
    ```python
    def sat(li: List[int], orig=[1, 1, 3, 2, 0, 8, 32, -4, 0]):
        """
        Find an increasing sequence consisting of the elements of the original list.
    
        Sample Input:
        [8, 0, 7, 2, 9, 4, 4, -2, 8, 3]
    
        Sample Output:
        [-2, 0, 2, 3, 4, 7, 8, 9]
        """
        for i in range(len(li) - 1):
            assert li[i] < li[i + 1]
            assert li[i] in orig
        for n in orig:
            assert n in li
        return True
    ```
    <details><summary>57 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(orig=[1, 1, 3, 2, 0, 8, 32, -4, 0]):
        return sorted(set(orig))
    
    ```
    
    ```python
    def sol(orig=[1, 1, 3, 2, 0, 8, 32, -4, 0]):
        """
        Create an increasing sequence consisting of the elements of the original list.
    
        No duplicates in the sequence.
    
        Sample Input:
        [8, 0, 7, 2, 9, 4, 4, -2, 8, 3]
    
        Sample Output:
        [-2, 0, 2, 3, 4, 7, 8, 9]
        """
        li = [n for n in orig]
        while True:
            li.sort()
            for x in li:
                if li.count(x) > 1:
                    li.remove(x)
                    break
            else:
                break
        return li
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(orig=[1, 1, 3, 2, 0, 8, 32, -4, 0]):
        my_list = sorted(set(orig))
        return my_list
    ```
    
    </details>
    
* <a name="strangesort"></a>**StrangeSort** Inspired by [HumanEval](https://github.com/openai/human-eval) \#70 (10 instances)
    
    ```python
    def sat(strange: List[int], li=[30, 12, 42, 717, 45, 317, 200, -1, 491, 32, 15]):
        """
        Find the following strange sort of li: the first element is the smallest, the second is the largest of the
        remaining, the third is the smallest of the remaining, the fourth is the smallest of the remaining, etc.
        """
        if len(li) < 2:
            return strange == li
        bounds = strange[:2]  # lower, upper
        for i, n in enumerate(strange):
            assert bounds[0] <= n <= bounds[1]
            bounds[i % 2] = n
        return sorted(strange) == sorted(li)  # permutation check
    ```
    <details><summary>40 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(li=[30, 12, 42, 717, 45, 317, 200, -1, 491, 32, 15]):
        s = sorted(li)
        return [s.pop() if i % 2 else s.pop(0) for i in range(len(li))]
    
    ```
    
    ```python
    def sol(li=[30, 12, 42, 717, 45, 317, 200, -1, 491, 32, 15]):
        """
        Given an arbitrary list of integers, find a strange sort of it.
        Keeps the initial list in its original state.
        """
        if len(li) <= 1:
            return li
        strange = []
        li_cp = [x for x in li]  # copy
        while li_cp:
            x = min(li_cp)
            strange.append(x)
            li_cp.remove(x)
            if li_cp:
                x = max(li_cp)
                strange.append(x)
                li_cp.remove(x)
        return strange
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(li=[30, 12, 42, 717, 45, 317, 200, -1, 491, 32, 15]):
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
    
* <a name="palindromecontaining"></a>**PalindromeContaining** Inspired by [HumanEval](https://github.com/openai/human-eval) \#10 (10 instances)
    
    ```python
    def sat(ans: str, s="so easy", length=20):
        """
        Find a palindrome of a given length containing a given string.
    
        Sample Input:
        "abba", 6
    
        Sample Output:
        "cabbac"
        """
        return ans == ans[::-1] and len(ans) == length and s in ans
    ```
    <details><summary>35 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="so easy", length=20):
        return s + "0"*(length-2*len(s)) + s[::-1]
    
    ```
    
    ```python
    def sol(s="so easy", length=20):
        """Given a string and a length, constructs a sufficiently long palindrome
        which contains that string."""
        palindrome = []
        for j in range(length // 2):
            palindrome.append(s[j % len(s)])
        if length % 2:
            palindrome.append("")
    
        for j in reversed(range(length // 2)):
            palindrome.append(s[j % len(s)])
    
        return "".join(palindrome)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="so easy", length=20):
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
    
* <a name="separateparengroups"></a>**SeparateParenGroups** Inspired by [HumanEval](https://github.com/openai/human-eval) \#1 (10 instances)
    
    ```python
    def sat(ls: List[str], combined="() (()) ((() () ())) (() )"):
        """
        Given a string consisting of whitespace and groups of matched parentheses, split it
        into groups of perfectly matched parentheses without any whitespace.
    
        Sample Input:
        '( ()) ((()()())) (()) ()'
    
        Sample Output:
        ['(())', '((()()()))', '(())', '()']
        """
        assert ''.join(ls) == combined.replace(' ', '')
        for s in ls:  # check that s is not further divisible
            depth = 0
            for c in s[:-1]:
                if c == '(':
                    depth += 1
                else:
                    assert c == ')'
                    depth -= 1
                    assert depth >= 1
            assert depth == 1 and s[-1] == ')'
        return True
    ```
    <details><summary>34 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(combined="() (()) ((() () ())) (() )"):
        return ['(())', '((()()()))', '(())', '()'][::-1]
    
    ```
    
    ```python
    def sol(combined="() (()) ((() () ())) (() )"):
        ls = []
        tmp = ''
        depth = 0
        for c in combined:
            if c == '(':
                if depth == 0:
                    if tmp:
                        ls.append(tmp)
                        tmp = ''
                depth += 1
                tmp += c
            elif c == ')':
                assert depth > 0
                depth -= 1
                tmp += c
            else:
                assert c == ' '
                if depth == 0:
                    if tmp:
                        ls.append(tmp)
                        tmp = ''
        if tmp:
            ls.append(tmp)
        return ls
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(combined="() (()) ((() () ())) (() )"):
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
    
* <a name="deepestparens"></a>**DeepestParens** Inspired by [HumanEval](https://github.com/openai/human-eval) \#6 (10 instances)
    
    ```python
    def sat(depths: List[int], parens="() (()) ((()()())) (())"):
        """
        Given a string consisting of groups of matched nested parentheses separated by parentheses,
        compute the depth of each group.
    
        Sample Input:
        '(()) ((()()())) (()) ()'
    
        Sample Output:
        [2, 3, 2, 1]
        """
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
    ```
    <details><summary>21 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(parens="() (()) ((()()())) (())"):
        return [len(sol.split(')')[0]) for sol in parens.split()]
    
    ```
    
    ```python
    def sol(parens="() (()) ((()()())) (())"):
        groups = parens.split()
        depths = []
        depth = 0
        for group in groups:
            balance = 0
            for c in group:
                if c == '(':
                    balance += 1
                else:
                    balance -= 1
                assert balance >= 0
                if balance == 0 and depth > 0:
                    depths.append(depth)
                    depth = 0
                else:
                    depth = max(depth, balance)
        return depths
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(parens="() (()) ((()()())) (())"):
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
    
* <a name="cuberoot"></a>**CubeRoot** Inspired by [HumanEval](https://github.com/openai/human-eval) \#77
    
    We made it harder by giving very large n for which `round(n ** (1/3))` (10 instances)
    
    ```python
    def sat(x: int, n=42714774173606970182754018064350848294149432972747296768):
        """Find an integer that when cubed is n"""
        return x ** 3 == n
    ```
    <details><summary>18 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=42714774173606970182754018064350848294149432972747296768):
        x = int(n**(1./3.))
        while not sat(x, n):
            x += 1
        return x
    
    ```
    
    ```python
    def sol(n=42714774173606970182754018064350848294149432972747296768):
        a = int(n**(1./3)) + 1
        while True:
            x0 = a ** 3
            if x0 == n:
                return a
            x1 = x0 + 2 * a + 1
            if x1 == n:
                return a + 1
            x2 = x1 + 2 * a + 1
            if x2 == n:
                return a + 2
            a += 1
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=42714774173606970182754018064350848294149432972747296768):  # Using Newton's method
        m = abs(n)
        x = round(abs(n) ** (1 / 3))
        while x ** 3 != m:
            x += (m - x ** 3) // (3 * x ** 2)
        return -x if n < 0 else x
    ```
    
    </details>
    
* <a name="certifiedgcd"></a>**CertifiedGCD** Inspired by [HumanEval](https://github.com/openai/human-eval) \#13 (10 instances)
    
    ```python
    def sat(ans: List[int], m=200004931, n=66679984):
        """
        Find the greatest common divisor of two integers m, n and a certificate a, b such that m*a + n*b = gcd
    
        Sample Input:
        20, 30
    
        Sample Output:
        10, -1, 1
        """
        gcd, a, b = ans
        return m % gcd == n % gcd == 0 and a * m + b * n == gcd and gcd > 0
    ```
    <details><summary>7 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(m=200004931, n=66679984):
        def egcd(a, b):
            if a == 0:
                return (b, 0, 1)
            else:
                sol, x, y = egcd(b % a, a)
                return (sol, y - (b // a) * x, x)
        sol, a, b = egcd(m, n)
    
        return [sol, a, b]
    
    ```
    
    ```python
    def sol(m=200004931, n=66679984):
        # Based on Extended Euclidean algorithm.
        # https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
    
        def extendedEuclid(m, n):
            if n > 0:
                s, t, gcd = extendedEuclid(n, m % n)
                return (t, s - (m // n) * t, gcd)
            else:
                return (1, 0, m)
    
        (a, b, gcd) = extendedEuclid(m, n)
        return [gcd, a, b]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(m=200004931, n=66679984):
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
    
* <a name="odddegreepolynomialroot"></a>**OddDegreePolynomialRoot** Polynomials of odd degree always have a real solution.
    
    Inspired by [HumanEval](https://github.com/openai/human-eval) \#32 (10 instances)
    
    ```python
    def sat(root: float, coeffs=[1, 2, 3, 17]):
        """
        Find a real root of an odd degree polynomial from its coefficients
    
        Sample Input:
        [1, 0, 8]
    
        Sample Output:
        -2.0  # 1*(-2.0)^3 + 8 == 0
        """
        return abs(sum(coeff * (root ** i) for i, coeff in enumerate(coeffs))) < 1e-4
    ```
    <details><summary>5 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(coeffs=[1, 2, 3, 17]):
        root = -10.0
        while root < 10.0:
            root += 0.00001
            if sat(root, coeffs=coeffs):
                #print(root)
                break
        else:
            root = None
        return root
    
    ```
    
    ```python
    def sol(coeffs=[1, 2, 3, 17]):
        """Find the real root using the bisection method.
    
        This is hardcoded for the input [1, 0, 8]
        """
        low = -1.0
        high = 0
        while (high - low) > 1e-4:
            mid = 0.5 * (high + low)
            if (sum(c * (mid ** i) for i, c in enumerate(coeffs)) > 0):
                high = mid
            elif (sum(c * (mid ** i) for i, c in enumerate(coeffs)) < 0):
                low = mid
    
        return mid
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(coeffs=[1, 2, 3, 17]):
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
    
* <a name="twothirdssorted"></a>**TwoThirdsSorted** Inspired by [HumanEval](https://github.com/openai/human-eval) \#33 (10 instances)
    
    ```python
    def sat(li: List[int], orig=[1, -2, 3, 17, 8, 4, 12, 3, 18, 5, -29, 0, 0]):
        """
        Start with a list of integers, keep every third element in place and otherwise sort the list
    
        Sample Input:
        [8, 0, 7, 2, 9, 4, 1, 2, 8, 3]
    
        Sample Output:
        [8, 0, 2, 2, 4, 8, 1, 8, 9, 3]
        """
        assert orig[::3] == li[::3], "Keep every third entry fixed"
        assert sorted(li) == sorted(orig), "Not even a permutation"
        assert all(li[i] <= li[i + 1] for i in range(1, len(li) - 1, 3))
        assert all(li[i] <= li[i + 2] for i in range(2, len(li) - 2, 3))
        return True
    ```
    <details><summary>3 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(orig=[1, -2, 3, 17, 8, 4, 12, 3, 18, 5, -29, 0, 0]):
        li = orig[:]
        tmp = sorted(orig[1::3] + orig[2::3])
        i = 0
        for j in range(1, len(orig), 3):
            li[j] = tmp[i]
            li[j+1] = tmp[i+1]
            i += 2
        return li
    
    ```
    
    ```python
    def sol(orig=[1, -2, 3, 17, 8, 4, 12, 3, 18, 5, -29, 0, 0]):
        """
        I think this code can be written nicer using slices
        from https://stackoverflow.com/a/522578/2804231
        """
        inds = [i for i in range(len(orig)) if i % 3 != 0]
        seq = list(sorted(orig[i] for i in inds))
        return [orig[i] if i % 3 == 0 else seq.pop(0) for i in range(len(orig))]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(orig=[1, -2, 3, 17, 8, 4, 12, 3, 18, 5, -29, 0, 0]):
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
    
* <a name="threecycle"></a>**ThreeCycle** Inspired by [HumanEval](https://github.com/openai/human-eval) \#38 (10 instances)
    
    ```python
    def sat(s: str, target="Hello world"):
        """
        Given a target string, find a string s such that when each group of three consecutive characters is cycled
        forward one character, you achieve the target string.
        """
    
        def cycle3(trip):
            return trip if len(trip) != 3 else trip[2] + trip[:2]
    
        return target == "".join(cycle3(s[i: i + 3]) for i in range(0, len(s), 3))
    ```
    <details><summary>2 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(target="Hello world"):
        """
        Given a target string, find a string s such that when each group of three consecutive characters is cycled
        forward one character, you achieve the target string.
        """
    
        def cycle3(trip):
            return trip if len(trip) != 3 else trip[2] + trip[:2]
    
        s = target
        while not sat(s, target):
            s = "".join(cycle3(s[i: i + 3]) for i in range(0, len(s), 3))
        return s
    
    ```
    
    ```python
    def sol(target="Hello world"):
        """
        Given the parameters, find a string s such that when each group of three consecutive characters is cycled
        forward one character, you achieve the target string.
        """
    
        # also follows the above constraints, but demonstrates reverse-lookup
        def cycle3_r(trip):
            return trip if len(trip) != 3 else trip[1:] + trip[0]
    
        return "".join(cycle3_r(target[i: i + 3]) for i in range(0, len(target), 3))
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(target="Hello world"):
        def un_cycle3(trip):
            return trip if len(trip) != 3 else trip[1:3] + trip[0]
    
        return "".join(un_cycle3(target[i: i + 3]) for i in range(0, len(target), 3))
    ```
    
    </details>
    
* <a name="sumproduct"></a>**SumProduct** Inspired by [HumanEval](https://github.com/openai/human-eval) \#8 (10 instances)
    
    ```python
    def sat(nums: List[int], tot=14, prod=99):
        """
        Find a list of numbers with a given sum and a given product.
    
        Sample Input:
        12, 32
    
        Sample Output:
        [2, 8, 2]
        """
        assert sum(nums) == tot
        p = 1
        for n in nums:
            p *= n
        return p == prod
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(tot=14, prod=99):
        ans = [prod]
        while sum(ans) > tot:
            ans += [-1, -1]
        ans += [1] * (tot - sum(ans))
        return ans
    ```
    
    </details>
    
* <a name="fib4"></a>**Fib4** Inspired by [HumanEval](https://github.com/openai/human-eval) \#46
    
    Almost identical to problem 63 (10 instances)
    
    ```python
    def sat(init: List[int], target=2021):
        """
        Define a four-wise Fibonacci sequence to be a sequence such that each number is the sum of the previous
        four. Given a target number, find an initial four numbers such that the 100th number in the sequence is the
        given target number.
        """
        a, b, c, d = init
        for i in range(99):
            a, b, c, d = b, c, d, (a + b + c + d)
        return a == target
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(target=2021):
        nums = [target, 0, 0, 0]
        for i in range(99):
            x = nums[3] - sum(nums[:3])  # x is such that x + nums[:3] == nums[3]
            nums = [x] + nums[:3]
        return nums
    ```
    
    </details>
    
* <a name="fib3"></a>**Fib3** Inspired by [HumanEval](https://github.com/openai/human-eval) \#63
    
    Almost identical to problem 46 (10 instances)
    
    ```python
    def sat(init: List[int], target=124156):
        """
        Define a triple-Fibonacci sequence to be a sequence such that each number is the sum of the previous
        three. Given a target number, find an initial triple such that the 17th number in the sequence is the
        given target number.
        """
        a, b, c = init
        for i in range(16):
            a, b, c = b, c, (a + b + c)
        return a == target
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(target=124156):
        nums = [target, 0, 0]
        for i in range(16):
            x = nums[-1] - sum(nums[:-1])  # x is such that x + nums[:3] == nums[3]
            nums = [x] + nums[:-1]
        return nums
    ```
    
    </details>
    
* <a name="herontriangle"></a>**HeronTriangle** Inspired by [HumanEval](https://github.com/openai/human-eval) \#71
    
    That problem essentially asks for Heron's formula for the area of a triangle in terms of its three sides.
    In our version, we consider the related problem (also solved by Heron's formula) of finding 2d coordinates
    of a triangle with the given sides. If one knows the area, this is a straightforward calculation. (10 instances)
    
    ```python
    def sat(coords: List[List[float]], sides=[8.9, 10.8, 17.0]):
        """
        Find the coordinates of a triangle with the given side lengths
        """
        assert len(coords) == 3
        sides2 = [((x - x2) ** 2 + (y - y2) ** 2) ** 0.5 for i, (x, y) in enumerate(coords) for x2, y2 in coords[:i]]
        return all(abs(a - b) < 1e-6 for a, b in zip(sorted(sides), sorted(sides2)))
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(sides=[8.9, 10.8, 17.0]):
        a, b, c = sorted(sides)
    
        s = sum(sides) / 2  # semi-perimeter
        area = (s * (s - a) * (s - b) * (s - c)) ** 0.5  # Heron's formula
    
        y = 2 * area / a  # height
        x = (c ** 2 - y ** 2) ** 0.5
        return [[0.0, 0.0], [a, 0.0], [x, y]]
    ```
    
    </details>
    
* <a name="threesum"></a>**ThreeSum** Inspired by but harder than the [HumanEval](https://github.com/openai/human-eval) \#92
    
    This is a version of the classic [3SUM](https://en.wikipedia.org/wiki/3SUM) problem. (10 instances)
    
    ```python
    def sat(inds: List[int], nums=[1253532, 3920635, 332, -24, 235, 17, 2539, 39359, -3923425, 2790, 25, 0, -2, 17, 8]):
        """Find three (distinct) indices of nums that sum to 0."""
        return len(set(inds)) == len(inds) == 3 and min(inds) >= 0 and sum(nums[i] for i in inds) == 0
    ```
    codex was not run on this puzzle
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[1253532, 3920635, 332, -24, 235, 17, 2539, 39359, -3923425, 2790, 25, 0, -2, 17, 8]):  # \tilde{O}(n^2) algorithm
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
    
* <a name="weirddecodevowels"></a>**WeirdDecodeVowels** Inspired by [HumanEval](https://github.com/openai/human-eval) \#93 (10 instances)
    
    ```python
    def sat(s: str, target="Hello, world!"):
        """Find string s that, when case is flipped gives target where vowels are replaced by chars two later."""
        subs = {ord(c): ord(c) + 2 for c in "aeiouAEIOU"}
        return s.swapcase() == target.translate(subs)
    ```
    codex was not run on this puzzle
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(target="Hello, world!"):
        subs = {ord(c): ord(c) + 2 for c in "aeiouAEIOU"}
        return target.translate(subs).swapcase()
    ```
    
    </details>
    
* <a name="largestprimedigitsum"></a>**LargestPrimeDigitSum** Inspired by [HumanEval](https://github.com/openai/human-eval) \#94 (10 instances)
    
    ```python
    def sat(ans: List[int], nums=[23, 17, 201, 14, 10473, 43225, 421, 423, 11, 10, 2022, 342157]):
        """Find the index of the largest prime in the list and the sum of its digits"""
        i, digit_sum = ans
        n = nums[i]
    
        def is_prime(n):
            return n > 1 and all(n % j for j in range(2, int(n ** 0.5) + 1))
    
        return is_prime(n) and all(m <= n for m in nums if is_prime(m)) and digit_sum == sum(int(c) for c in str(n))
    ```
    codex was not run on this puzzle
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[23, 17, 201, 14, 10473, 43225, 421, 423, 11, 10, 2022, 342157]):
        def is_prime(n):
            return n > 1 and all(n % j for j in range(2, int(n ** 0.5) + 1))
    
        n, i = max((n, i) for i, n in enumerate(nums) if is_prime(n))
        return [i, sum(int(c) for c in str(n))]
    ```
    
    </details>
    
* <a name="oddcase"></a>**OddCase** Inspired by [HumanEval](https://github.com/openai/human-eval) \#95 (10 instances)
    
    ```python
    def sat(different: str, d={'cat': 'CAT', 'tree': 'T', 'pick me': 'not', 'OK': 'red', 'blah': 'blah', 'z': 'Z'}):
        """Find the dictionary key whose case is different than all other keys"""
        return different in d and all(k.islower() != different.islower() for k in d if k != different)
    ```
    codex was not run on this puzzle
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(d={'cat': 'CAT', 'tree': 'T', 'pick me': 'not', 'OK': 'red', 'blah': 'blah', 'z': 'Z'}):
        for different in d:
            if all(k.islower() != different.islower() for k in d if k != different):
                return different
    ```
    
    </details>
    
* <a name="primesupto"></a>**PrimesUpTo** Inspired by [HumanEval](https://github.com/openai/human-eval) \#96 (10 instances)
    
    ```python
    def sat(primes: List[int], n=1234):
        """Find all primes up to n"""
        assert all(1 < p for p in primes) and all(p % q for p in primes for q in primes if q < p)
        return len({i for p in primes for i in range(p, n, p)}) == max(n - 2, 0)
    ```
    codex was not run on this puzzle
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=1234):
        primes = []
        candidates = set(range(2, n))
        for i in range(2, n):
            if i in candidates:
                primes.append(i)
                candidates.difference_update(range(i, n, i))
    
        return primes
    ```
    
    </details>
    
* <a name="unitsproduct"></a>**UnitsProduct** Inspired by [HumanEval](https://github.com/openai/human-eval) \#97 (10 instances)
    
    ```python
    def sat(prod: int, nums=[17, 24, 39, 15, 11, 201, 97, 65, 18]):
        """Find the product of the units digits in the numbers"""
        if not all(nums):
            return prod == 0
        for n in nums:
            k = abs(n % 10)
            if k == 0:
                return prod == 0
            assert prod % k == 0
            prod //= k
        return prod == 1
    ```
    codex was not run on this puzzle
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[17, 24, 39, 15, 11, 201, 97, 65, 18]):
        prod = 1
        for n in nums:
            prod *= abs(n % 10)
        return prod
    ```
    
    </details>
    
* <a name="uppercaseeven"></a>**UppercaseEven** Inspired by [HumanEval](https://github.com/openai/human-eval) \#98 (10 instances)
    
    ```python
    def sat(positions: List[int], s="ThIs is A tEsT, Or *IS* iT?"):
        """Find the positions of all uppercase vowels (not counting Y) in even indices"""
        assert all(s[i] in "AEIOU" for i in positions)
        return all(i in positions or c not in "AEIOU" or i % 2 == 1 for i, c in enumerate(s))
    ```
    codex was not run on this puzzle
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="ThIs is A tEsT, Or *IS* iT?"):
        return [i for i, c in enumerate(s) if i % 2 == 0 and c in "AEIOU"]
    ```
    
    </details>
    
* <a name="closestinteger"></a>**ClosestInteger** Inspired by [HumanEval](https://github.com/openai/human-eval) \#99
    
    Since we can tolerate more than one answer per puzzle, we do not need to specify a tie-breaking rule. (10 instances)
    
    ```python
    def sat(n: int, x=329437923.5):
        """Round to nearest integer"""
        return abs(n - x) <= 0.5
    ```
    codex was not run on this puzzle
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(x=329437923.5):
        return round(x)
    ```
    
    </details>
    
* <a name="stonepiles"></a>**StonePiles** Inspired by [HumanEval](https://github.com/openai/human-eval) \#100 (10 instances)
    
    ```python
    def sat(li: List[int], n=909):
        """We are making n stone piles! The first pile has n stones. If n is even, then all piles have an even
        number of stones. If n is odd, all piles have an odd number of stones. Each pile must more stones
        than the previous pile but as few as possible. Return the number of stones in each pile."""
        return li[0] == n and len(li) == n and all(b - a == 2 for a, b in zip(li, li[1:]))
    ```
    codex was not run on this puzzle
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=909):
        return [n + 2 * i for i in range(n)]
    ```
    
    </details>
    
* <a name="completesplit"></a>**CompleteSplit** Inspired by [HumanEval](https://github.com/openai/human-eval) \#101 (10 instances)
    
    ```python
    def sat(splits: List[List[str]], string="Hello, world!  You look like you're on turtles."):
        """
        Split a string of words separated by commas and spaces into 2 lists: words and separators
    
        Sample input: "Hi there, Adam"
        Sample output: [["Hi", "there", "Adam"], [" ", ", "]]
        """
        words, separators = splits
        assert len(words) == len(separators) + 1
        merged = []
        for w, s in zip(words, separators + [" "]):
            assert s.count(" ") + s.count(",") == len(s) > 0
            assert w.count(" ") + w.count(",") == 0
            merged += [w, s]
        return "".join(merged[:-1]) == string
    ```
    codex was not run on this puzzle
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(string="Hello, world!  You look like you're on turtles."):
        import re
        merged = re.split(r"([ ,]+)", string)
        return [merged[::2], merged[1::2]]
    ```
    
    </details>
    
* <a name="biggesteven"></a>**BiggestEven** Inspired by [HumanEval](https://github.com/openai/human-eval) \#102 (10 instances)
    
    ```python
    def sat(x: int, a=145, b=24126846790974):
        """Return the biggest even number between a and b inclusive, or -1 if there is no such number"""
        if x == -1:
            return all(i % 2 == 1 for i in range(a, b + 1))
        return a <= x <= b and all(i % 2 == 1 for i in range(x + 1, b + 1))
    ```
    codex was not run on this puzzle
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=145, b=24126846790974):
        if a > b or (a == b and a % 2 == 1):
            return -1
        return b if b % 2 == 0 else b - 1
    ```
    
    </details>
    
## codeforces

Problems inspired by the popular programming competition site [codeforces.com](https://codeforces.com)

* <a name="easysum"></a>**EasySum** Inspired by [Codeforces Problem 677 A](https://codeforces.com/problemset/problem/677/A) (10 instances)
    
    ```python
    def sat(tot: int, nums=[2, 8, 25, 18, 99, 11, 17, 16], thresh=17):
        """Add up 1 or 2 for numbers in a list depending on whether they exceed a threshold"""
        return tot == sum(1 if i < thresh else 2 for i in nums)
    ```
    <details><summary>8,987 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[2, 8, 25, 18, 99, 11, 17, 16], thresh=17):
        return sum([1, 2][i<17] for i in nums)
    
    ```
    
    ```python
    def sol(nums=[2, 8, 25, 18, 99, 11, 17, 16], thresh=17):
        # Return the minimum value you can get by adding 1 or 2 to get a sum that exceeds the threshold
        # The way this is written, the output also serves as the "correct" set of numbers
        v = 0
        for i in nums:
            if i < thresh:
                v += 1
            else:
                v += 2
        return v
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[2, 8, 25, 18, 99, 11, 17, 16], thresh=17):
        return sum(1 if i < thresh else 2 for i in nums)
    ```
    
    </details>
    
* <a name="dada"></a>**Dada** Inspired by [Codeforces Problem 734 A](https://codeforces.com/problemset/problem/734/A) (10 instances)
    
    ```python
    def sat(s: str, a=5129, d=17):
        """Find a string with a given number of a's and d's"""
        return s.count("a") == a and s.count("d") == d and len(s) == a + d
    ```
    <details><summary>8,508 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=5129, d=17):
    	return "a"*a + "d"*d
    
    ```
    
    ```python
    def sol(a=5129, d=17):
        counta = 0
        countd = 0
        ret = ""
        while not sat(ret, a, d):
            if a > d:
                if counta < a:
                    counta += 1
                    ret += "a"
                if counta == a:
                    countd += 1
                    ret += "d"
            else:
                if countd < d:
                    countd += 1
                    ret += "d"
                if countd == d:
                    counta += 1
                    ret += "a"
        return ret
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=5129, d=17):
        return "a" * a + "d" * d
    ```
    
    </details>
    
* <a name="capitalizefirstletter"></a>**CapitalizeFirstLetter** Inspired by [Codeforces Problem 281 A](https://codeforces.com/problemset/problem/281/A) (10 instances)
    
    ```python
    def sat(s: str, word="konjac"):
        """Capitalize the first letter of word"""
        for i in range(len(word)):
            if i == 0:
                if s[i] != word[i].upper():
                    return False
            else:
                if s[i] != word[i]:
                    return False
        return True
    ```
    <details><summary>7,531 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(word="konjac"):
        return "Konjac"
    
    ```
    
    ```python
    def sol(word="konjac"):
        word = word.lower()
        if word == "konjac":
            return "Konjac"
        if word == "yunomi":
            return "Yunomi"
        if word == "parlor":
            return "Parlor"
        if word == "zigfield":
            return "Zigfield"
        if word == "bean":
            return "Bean"
        return "Unknown"
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(word="konjac"):
        return word[0].upper() + word[1:]
    ```
    
    </details>
    
* <a name="maybereversed"></a>**MaybeReversed** Inspired by [Codeforces Problem 41 A](https://codeforces.com/problemset/problem/41/A) (10 instances)
    
    ```python
    def sat(s: str, target="reverse me", reverse=True):
        """Either reverse a string or don't based on the reverse flag"""
        return (s[::-1] == target) == reverse
    ```
    <details><summary>6,039 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(target="reverse me", reverse=True):
        return target[::-1]
    
    ```
    
    ```python
    def sol(target="reverse me", reverse=True):
        """
        Either reverse a string or don't based on the reverse flag
    
        Parameters
        ----------
        target: str
            The (reversed) string to be used as input
        reverse: bool
            `true` indicates that the string should be reversed; `false` indicates that it should not be reversed
    
        Returns
        -------
        """
        return target[::-1] if reverse else target
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(target="reverse me", reverse=True):
        return target[::-1] if reverse else target + "x"
    ```
    
    </details>
    
* <a name="fivepowers"></a>**FivePowers** Inspired by [Codeforces Problem 630 A](https://codeforces.com/problemset/problem/630/A) (10 instances)
    
    ```python
    def sat(s: str, n=7012):
        """What are the last two digits of 5^n?"""
        return int(str(5 ** n)[:-2] + s) == 5 ** n
    ```
    <details><summary>5,619 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=7012):
        return "25"
    
    ```
    
    ```python
    def sol(n=7012):
        i = 0
        while True:
            if sat(str(i).zfill(2), n):
                return str(i).zfill(2)
            else:
                i += 1
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=7012):
        return ("1" if n == 0 else "5" if n == 1 else "25")
    ```
    
    </details>
    
* <a name="iseven"></a>**IsEven** Inspired by [Codeforces Problem 4 A](https://codeforces.com/problemset/problem/4/A) (9 instances)
    
    ```python
    def sat(b: bool, n=10):
        """Determine if n can be evenly divided into two equal numbers. (Easy)"""
        i = 0
        while i <= n:
            if i + i == n:
                return b == True
            i += 1
        return b == False
    ```
    <details><summary>5,185 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=10):
        return True
    ```
    
    ```python
    def sol(n=10):
        while True:
            if n%2 == 0 and (n/2)%2 == 0:
                return True
            elif n%3 == 0 and (n/3)%2 == 0:
                return True
            elif n%5 == 0 and (n/5)%2 == 0:
                return True
            elif n%7 == 0 and (n/7)%2 == 0:
                return True
            elif n%11 == 0 and (n/11)%2 == 0:
                return True
            elif n%13 == 0 and (n/13)%2 == 0:
                return True
           
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=10):
        return n % 2 == 0
    ```
    
    </details>
    
* <a name="repeatdec"></a>**RepeatDec** Inspired by [Codeforces Problem 977 A](https://codeforces.com/problemset/problem/977/A) (10 instances)
    
    ```python
    def sat(res: int, m=1234578987654321, n=4):
        """
        Find the result of applying the following operation to integer m, n times: if the last digit is zero, remove
        the zero, otherwise subtract 1.
        """
        for i in range(n):
            m = (m - 1 if m % 10 else m // 10)
        return res == m
    ```
    <details><summary>4,619 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(m=1234578987654321, n=4):
        for i in range(n):
            m = (m - 1 if m%10 else m//10)
        return m
    
    ```
    
    ```python
    def sol(m=1234578987654321, n=4):
        """
        - Meaning of g6's default arguments: if the initial value 1234578987654321 was rounded to a float or double,
        the expression 5234578987654315 * 5 ** n might change to something that rounds to a final value other than the
        correct answer 1234578987654321. By assigning a different initial value to m, we avoid this problem.
        - We assign directly to m rather than return m, to show that we're able to.
        """
        for i in range(n):
            m = (m - 1 if m % 10 else m // 10)
        return m
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(m=1234578987654321, n=4):
        for i in range(n):
            m = (m - 1 if m % 10 else m // 10)
        return m
    ```
    
    </details>
    
* <a name="compareinanycase"></a>**CompareInAnyCase** Inspired by [Codeforces Problem 112 A](https://codeforces.com/problemset/problem/112/A) (10 instances)
    
    ```python
    def sat(n: int, s="aaAab", t="aAaaB"):
        """Ignoring case, compare s, t lexicographically. Output 0 if they are =, -1 if s < t, 1 if s > t."""
        if n == 0:
            return s.lower() == t.lower()
        if n == 1:
            return s.lower() > t.lower()
        if n == -1:
            return s.lower() < t.lower()
        return False
    ```
    <details><summary>4,617 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="aaAab", t="aAaaB"): return 0
    ```
    
    ```python
    def sol(s="aaAab", t="aAaaB"):
        """We can precompute the number of string differences between two strings using the following:
        The first string s will be length k, the second will be length k + 1.
        If the two strings are equal until the end, then there were k differences.
        If the first string is one character shorter, then that must mean the other string has a repeated character,
        so we must have had 1 more string difference.
        If the second string is one character shorter, we had 1 fewer string difference.
        """
        return sum(s[i].lower() != t[i].lower() for i in range(len(s)))
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="aaAab", t="aAaaB"):
        if s.lower() == t.lower():
            return 0
        if s.lower() > t.lower():
            return 1
        return -1
    ```
    
    </details>
    
* <a name="gimmechars"></a>**GimmeChars** Inspired by [Codeforces Problem 133 A](https://codeforces.com/problemset/problem/133/A), easy (10 instances)
    
    ```python
    def sat(s: str, chars=['o', 'h', 'e', 'l', ' ', 'w', '!', 'r', 'd']):
        """Find a string with certain characters"""
        for c in chars:
            if c not in s:
                return False
        return True
    ```
    <details><summary>4,534 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(chars=['o', 'h', 'e', 'l', ' ', 'w', '!', 'r', 'd']):
        return str(chars)
    
    ```
    
    ```python
    def sol(chars=['o', 'h', 'e', 'l', ' ', 'w', '!', 'r', 'd']):
        chars = sorted(chars)
        i = 0
        res = list()
        while i < len(chars):
            if chars[i] in res:
                i += 1
                continue
            count = 0
            for j in range(i, len(chars)):
                if chars[j] == chars[i]:
                    count += 1
                    if count > 1:
                        res += [chars[i]]
                        i += count
                        break
            else:
                res += [chars[i]]
                i += 1
        return ''.join(res)
    
    ```
    
    </details>
    
    0 hand-written solutions 
    
* <a name="easytwos"></a>**EasyTwos** Inspired by [Codeforces Problem 231 A](https://codeforces.com/problemset/problem/231/A) (10 instances)
    
    ```python
    def sat(lb: List[bool], trips=[[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]):
        """
        Given a list of lists of triples of integers, return True for each list with a total of at least 2 and
        False for each other list.
        """
        return len(lb) == len(trips) and all(
            (b is True) if sum(s) >= 2 else (b is False) for b, s in zip(lb, trips))
    ```
    <details><summary>4,419 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(trips=[[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]):
        return [sum(s)>=2 for s in trips]
    
    ```
    
    ```python
    def sol(trips=[[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]):
        """
        A task:
         - requires some sub-tasks to be completed: these are triples in the trips list
         - returns True if the required sub-tasks are completed (sum of coords in triple is at least 2)
         - returns False otherwise
        """
        def sol(i):
            """
            Sub-tasks: 
              - completed if sum of coords in triple at least 2
            """
            return (i[0] + i[1] + i[2]) >= 2
        return [sol(i) for i in trips]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(trips=[[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]):
        return [sum(s) >= 2 for s in trips]
    ```
    
    </details>
    
* <a name="invertindices"></a>**InvertIndices** Inspired by [Codeforces Problem 136 A](https://codeforces.com/problemset/problem/136/A) (10 instances)
    
    ```python
    def sat(indexes: List[int], target=[1, 3, 4, 2, 5, 6, 7, 13, 12, 11, 9, 10, 8]):
        """Given a list of integers representing a permutation, invert the permutation."""
        for i in range(1, len(target) + 1):
            if target[indexes[i - 1] - 1] != i:
                return False
        return True
    ```
    <details><summary>4,223 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(target=[1, 3, 4, 2, 5, 6, 7, 13, 12, 11, 9, 10, 8]):
        return [target.index(i)+1 for i in sorted(target)]
    
    ```
    
    ```python
    def sol(target=[1, 3, 4, 2, 5, 6, 7, 13, 12, 11, 9, 10, 8]):
        # Note: This is a terrible inverse function.
        seen_it = set()
        inverse_index = [0] * len(target)  # initialize position of all targets to 0
        for i, x in enumerate(target):
            inverse_index[x - 1] = i + 1  # set the position of the target value to value i+1
        for x in inverse_index:
            seen_it.add(x)
        for x in seen_it:
            if x not in inverse_index:
                raise ValueError("reverse error")
        return inverse_index
    
    ```
    
    </details>
    
    0 hand-written solutions 
    
* <a name="abbreviate"></a>**Abbreviate** Inspired by [Codeforces Problem 71 A](https://codeforces.com/problemset/problem/71/A) (10 instances)
    
    ```python
    def sat(s: str, word="antidisestablishmentarianism", max_len=10):
        """
        Abbreviate strings longer than a given length by replacing everything but the first and last characters by
        an integer indicating how many characters there were in between them.
        """
        if len(word) <= max_len:
            return word == s
        return int(s[1:-1]) == len(word[1:-1]) and word[0] == s[0] and word[-1] == s[-1]
    ```
    <details><summary>2,753 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(word="antidisestablishmentarianism", max_len=10):
        return word[0]+str(len(word)-2)+word[-1]
    
    ```
    
    ```python
    def sol(word="antidisestablishmentarianism", max_len=10):
        """Abbreviate strings longer than a given length by replacing everything but the first and last characters by
           an integer indicating how many characters there were in between them."""
        if len(word) <= max_len:
            return word
        # TODO: make this more readable using a better string format template, and do so without increasing the number
        #       of characters in the string
        abbreviation = f"{word[0]}{len(word[1:-1])}{word[-1]}"
        return abbreviation if len(abbreviation) <= max_len else abbreviation
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(word="antidisestablishmentarianism", max_len=10):
        if len(word) <= max_len:
            return word
        return f"{word[0]}{len(word) - 2}{word[-1]}"
    ```
    
    </details>
    
* <a name="commoncase"></a>**CommonCase** Inspired by [Codeforces Problem 59 A](https://codeforces.com/problemset/problem/59/A)
    
    This is a trivial puzzle, especially if the AI realizes that it can can just copy the solution from
    the problem (10 instances)
    
    ```python
    def sat(s_case: str, s="CanYouTellIfItHASmoreCAPITALS"):
        """
        Given a word, replace it either with an upper-case or lower-case depending on whether or not it has more
        capitals or lower-case letters. If it has strictly more capitals, use upper-case, otherwise, use lower-case.
        """
        caps = 0
        for c in s:
            if c != c.lower():
                caps += 1
        return s_case == (s.upper() if caps > len(s) // 2 else s.lower())
    ```
    <details><summary>2,679 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="CanYouTellIfItHASmoreCAPITALS"):
        return s.upper()
    
    ```
    
    ```python
    def sol(s="CanYouTellIfItHASmoreCAPITALS"):
        """
        Given a word, replace it either with an upper-case or lower-case depending on whether or not it has more
        capitals or lower-case letters. If it has strictly more capitals, use upper-case, otherwise, use lower-case.
        """
        return "CanYouTellIfItHASmoreCAPITALS".upper() if ("CanYouTellIfItHASmoreCAPITALS".count("CAPITALS")
                                                           > "CanYouTellIfItHASmoreCAPITALS".count("capitals")) \
            else "CanYouTellIfItHASmoreCAPITALS"
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="CanYouTellIfItHASmoreCAPITALS"):
        caps = 0
        for c in s:
            if c != c.lower():
                caps += 1
        return (s.upper() if caps > len(s) // 2 else s.lower())  # duh, just take sat and return the answer checked for
    ```
    
    </details>
    
* <a name="sortplusplus"></a>**SortPlusPlus** Inspired by [Codeforces Problem 339 A](https://codeforces.com/problemset/problem/339/A) (10 instances)
    
    ```python
    def sat(s: str, inp="1+1+3+1+3+2+2+1+3+1+2"):
        """Sort numbers in a sum of digits, e.g., 1+3+2+1 -> 1+1+2+3"""
        return all(s.count(c) == inp.count(c) for c in inp + s) and all(s[i - 2] <= s[i] for i in range(2, len(s), 2))
    ```
    <details><summary>2,536 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(inp="1+1+3+1+3+2+2+1+3+1+2"):
    	return "".join(sorted(inp))
    
    ```
    
    ```python
    def sol(inp="1+1+3+1+3+2+2+1+3+1+2"):
        """Sort numbers in a sum of digits, e.g., 1+3+2+1 -> 1+1+2+3"""
        n = len(inp)
        l = list(inp)
    
        for _ in range(n*n):
            idx = n
            for i in range(n-1):
                if l[i] > l[i + 1]:
                    idx = i
                    l[i], l[i + 1] = l[i + 1], l[i]
    
            if idx == n:
                break
        return "".join(l)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(inp="1+1+3+1+3+2+2+1+3+1+2"):
        return "+".join(sorted(inp.split("+")))
    ```
    
    </details>
    
* <a name="decreasingcountcomparison"></a>**DecreasingCountComparison** Inspired by [Codeforces Problem 158 A](https://codeforces.com/problemset/problem/158/A) (10 instances)
    
    ```python
    def sat(n: int, scores=[100, 95, 80, 70, 65, 9, 9, 9, 4, 2, 1], k=6):
        """
        Given a list of non-increasing integers and given an integer k, determine how many positive integers in the list
        are at least as large as the kth.
        """
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), "Hint: scores are non-decreasing"
        return all(s >= scores[k] and s > 0 for s in scores[:n]) and all(s < scores[k] or s <= 0 for s in scores[n:])
    ```
    <details><summary>1,957 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(scores=[100, 95, 80, 70, 65, 9, 9, 9, 4, 2, 1], k=6):
        return k - scores[k-1]
    
    ```
    
    ```python
    def sol(scores=[100, 95, 80, 70, 65, 9, 9, 9, 4, 2, 1], k=6):
        """
        Given a list of non-increasing integers and given an integer k, determine the maximum number of positive integers
        in the list that are at least as large as the kth.
    
        Hints:
        - Be careful with the ordering of the original list; how could it be reversed and what does that imply about the
          ordering of the final result?
        - Make a copy of the list before continuing (scores.reverse() will work).
        """
        copy = scores[:]
        copy.reverse()
        return sum(1 for s in copy if s >= scores[k] and s > 0)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(scores=[100, 95, 80, 70, 65, 9, 9, 9, 4, 2, 1], k=6):
        threshold = max(scores[k], 1)
        return sum(s >= threshold for s in scores)
    ```
    
    </details>
    
* <a name="findhomogeneoussubstring"></a>**FindHomogeneousSubstring** Inspired by [Codeforces Problem 96 A](https://codeforces.com/problemset/problem/96/A) (10 instances)
    
    ```python
    def sat(n: int, s="0000101111111000010", k=5):
        """
        You are given a string consisting of 0's and 1's. Find an index after which the subsequent k characters are
        all 0's or all 1's.
    
        Sample Input:
        s = 0000111111100000, k = 5
    
        Sample Output:
        4
        (or 5 or 6 or 11)
        """
        return s[n:n + k] == s[n] * k
    ```
    <details><summary>1,939 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="0000101111111000010", k=5):
        return 6
    
    ```
    
    ```python
    def sol(s="0000101111111000010", k=5):
        """
        For example:
        for k = 5, n = 0, s = 0000111111100000
                              1111111111100000
                                1111111111100000
                                  1111111111100000
                                    1111111111100000
                                      1111111111100000
                                        1111111111100000
                                          1111111111100000
                                            1111111111100000
        """
        m = len(s)
        return next(n for n in range(m-k+1) if s[n:n+k] == s[n] * k)
    
    ```
    
    </details>
    
    <details><summary>4 hand-written solutions </summary>
    
    ```python
    def sol(s="0000101111111000010", k=5):
        return s.index("0" * k if "0" * k in s else "1" * k)
    ```
    
    ```python
    def sol(s="0000101111111000010", k=5):
        import re
        return re.search(r"([01])\1{" + str(k - 1) + "}", s).span()[0]
    ```
    
    ```python
    def sol(s="0000101111111000010", k=5):
        if "0" * k in s:
            return s.index("0" * k)
        else:
            return s.index("1" * k)
    ```
    
    ```python
    def sol(s="0000101111111000010", k=5):
        try:
            return s.index("0" * k)
        except:
            return s.index("1" * k)
    ```
    
    </details>
    
* <a name="factor47"></a>**Factor47** Inspired by [Codeforces Problem 122 A](https://codeforces.com/problemset/problem/122/A) (10 instances)
    
    ```python
    def sat(d: int, n=6002685529):
        """Find a integer factor of n whose decimal representation consists only of 7's and 4's."""
        return n % d == 0 and all(i in "47" for i in str(d))
    ```
    <details><summary>1,825 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=6002685529):
        return int(n**.5)
    
    ```
    
    ```python
    def sol(n=6002685529):
        """Find a factor of n whose decimal representation consists only of 7's and 4's.
    
        Alternatively, find a factor of 6002685531 whose decimal representation is
        *entirely* of 7's and 4's. This is slightly less efficient, but more
        interesting."""
        i = 0
        while True:
            i = int(n ** 0.5)
            if len([c for c in str(i) if c in "47"]) == len(str(i)): return i
            n += 1
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=6002685529):
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
    
* <a name="voweldrop"></a>**VowelDrop** Inspired by [Codeforces Problem 118 A](https://codeforces.com/problemset/problem/118/A) (10 instances)
    
    ```python
    def sat(t: str, s="Problems"):
        """
        Given an alphabetic string s, remove all vowels (aeiouy/AEIOUY), insert a "." before each remaining letter
        (consonant), and make everything lowercase.
    
        Sample Input:
        s = "Problems"
    
        Sample Output:
        .p.r.b.l.m.s
        """
        i = 0
        for c in s.lower():
            if c in "aeiouy":
                continue
            assert t[i] == ".", f"expecting `.` at position {i}"
            i += 1
            assert t[i] == c, f"expecting `{c}`"
            i += 1
        return i == len(t)
    ```
    <details><summary>1,525 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="Problems"):
        return ".p.r.b.l.m.s"
    
    ```
    
    ```python
    def sol(s="Problems"):
        """
        Given an alphabetic string s, remove all vowels (aeiouy/AEIOUY), insert a "." before each remaining letter
        (consonant), and make everything lowercase. Recall that a string appears in form
        `s = "..."` an integer appears in form `n = ...` (but without the quotation marks).
    
        Sample Input:
        s = "Problems"
    
        Sample Output:
        ...p.r.b.l.m.s...
        """
        return "." + ".".join([x for x in s.lower() if x not in "aeiouy"])
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="Problems"):
        return "".join("." + c for c in s.lower() if c not in "aeiouy")
    ```
    
    </details>
    
* <a name="tripledouble"></a>**TripleDouble** Inspired by [Codeforces Problem 791 A](https://codeforces.com/problemset/problem/791/A) (10 instances)
    
    ```python
    def sat(n: int, v=17, w=100):
        """Find the smallest n such that if v is tripled n times and w is doubled n times, v exceeds w."""
        for i in range(n):
            assert v <= w
            v *= 3
            w *= 2
        return v > w
    ```
    <details><summary>1,455 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(v=17, w=100):
        return 5
    
    ```
    
    ```python
    def sol(v=17, w=100):
        """Find the smallest n such that if v is tripled n times and w is doubled n times, v exceeds w.
    
           Note that when n is large (> 100), the values for v and w will quickly exceed Python's native int.
           Instead, keep the result of each multiplication and division within an int as much as possible.
        """
        assert abs(float(v)) < abs(float(w))
        n = 0
        while v <= w:
            v *= 3
            w *= 2
            n += 1
        return n
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(v=17, w=100):
        i = 0
        while v <= w:
            v *= 3
            w *= 2
            i += 1
        return i
    ```
    
    </details>
    
* <a name="minconsecutivesum"></a>**MinConsecutiveSum** Inspired by [Codeforces Problem 363 B](https://codeforces.com/problemset/problem/363/B) (10 instances)
    
    ```python
    def sat(start: int, k=3, upper=6, seq=[17, 1, 2, 65, 18, 91, -30, 100, 3, 1, 2]):
        """Find a sequence of k consecutive indices whose sum is minimal"""
        return 0 <= start <= len(seq) - k and sum(seq[start:start + k]) <= upper
    ```
    <details><summary>1,451 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(k=3, upper=6, seq=[17, 1, 2, 65, 18, 91, -30, 100, 3, 1, 2]):
        return len(seq) - k
    
    ```
    
    ```python
    def sol(k=3, upper=6, seq=[17, 1, 2, 65, 18, 91, -30, 100, 3, 1, 2]):
        """Find a sequence of k consecutive indices whose sum is minimal"""
        m = n = 0
        best = sum(seq[:k])
        while True:
            if m > 0 and n < upper:
                n += seq[m]
                if n < best:
                    best = n
                    start = m
            else:
                m += 1
                n = 0
                if m <= len(seq) - k:
                    n = sum(seq[m:m + k])
                    if n < best:
                        best = n
                        start = m
                else:
                    break
        return start
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(k=3, upper=6, seq=[17, 1, 2, 65, 18, 91, -30, 100, 3, 1, 2]):
        return min(range(len(seq) - k + 1), key=lambda start: sum(seq[start:start + k]))
    ```
    
    </details>
    
* <a name="squaretiles"></a>**SquareTiles** Inspired by [Codeforces Problem 1 A](https://codeforces.com/problemset/problem/1/A) (10 instances)
    
    ```python
    def sat(corners: List[List[int]], m=10, n=9, a=5, target=4):
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
        covered = {(i + x, j + y) for i, j in corners for x in range(a) for y in range(a)}
        assert len(covered) == len(corners) * a * a, "Double coverage"
        return len(corners) <= target and covered.issuperset({(x, y) for x in range(m) for y in range(n)})
    ```
    <details><summary>1,363 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(m=10, n=9, a=5, target=4):
        return [[0,0],[0,a],[a,0],[a,a]]
    
    ```
    
    ```python
    def sol(m=10, n=9, a=5, target=4):
        corners = [[0, 0], [0, a], [a, 0], [a, a]]
        while len(corners) < target:
            for i in range(len(corners)):
                found = False
                for x in range(a):
                    for y in range(a):
                        if found:
                            break
                        corner = corners[i].copy()
                        corner[0] += x
                        corner[1] += y
                        if corner not in corners:
                            corners.append(corner)
                            found = True
                if found:
                    break
        return corners
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(m=10, n=9, a=5, target=4):
        return [[x, y] for x in range(0, m, a) for y in range(0, n, a)]
    ```
    
    </details>
    
* <a name="count47"></a>**Count47** Inspired by [Codeforces Problem 110 A](https://codeforces.com/problemset/problem/110/A) (10 instances)
    
    ```python
    def sat(d: int, n=123456789):
        """
        Find a number bigger than n whose decimal representation has k 4's and 7's where k's decimal representation
        consists only of 4's and 7's
        """
        return d > n and all(i in "47" for i in str(str(d).count("4") + str(d).count("7")))
    ```
    <details><summary>1,240 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=123456789):
        return n+888888
    
    ```
    
    ```python
    def sol(n=123456789):
        """
        Find a number bigger than n such that the first k digits of its square are 123456789 for k the number of 4's
        in n and the first k digits of its cube are 123456789 for k the number of 4's and 7's in n.
        """
        for i in range(int("".join(["1"] + ["4"*len(str(n))]))):
            d = i*i*i + n*n
            if d > n and all(i in "47" for i in str(str(d).count("4") + str(d).count("7"))):
                return d
    
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=123456789):
        return int("4444" + "0" * (len(str(n)) - 3))
    ```
    
    </details>
    
* <a name="maxconsecutivesum"></a>**MaxConsecutiveSum** Inspired by [Codeforces Problem 363 B](https://codeforces.com/problemset/problem/363/B) (10 instances)
    
    ```python
    def sat(start: int, k=3, lower=150, seq=[3, 1, 2, 65, 18, 91, -30, 100, 0, 19, 52]):
        """Find a sequence of k consecutive indices whose sum is maximal"""
        return 0 <= start <= len(seq) - k and sum(seq[start:start + k]) >= lower
    ```
    <details><summary>1,186 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(k=3, lower=150, seq=[3, 1, 2, 65, 18, 91, -30, 100, 0, 19, 52]):
        return 3
    
    ```
    
    ```python
    def sol(k=3, lower=150, seq=[3, 1, 2, 65, 18, 91, -30, 100, 0, 19, 52]):
        """Find a sequence of length k chosen from seq, which has maximal sum."""
        best_start = 0
        best_sum = 0
        for i in range(len(seq) - k):
            candidate_sum = sum(seq[i:i + k])
            if candidate_sum >= best_sum:
                if candidate_sum >= lower:
                    best_start, best_sum = i, candidate_sum
                else:
                    for j in range(k):
                        if seq[i + j] >= lower:
                            best_start, best_sum = i, candidate_sum
        return best_start
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(k=3, lower=150, seq=[3, 1, 2, 65, 18, 91, -30, 100, 0, 19, 52]):
        return max(range(len(seq) - k + 1), key=lambda start: sum(seq[start:start + k]))
    ```
    
    </details>
    
* <a name="maxconsecutiveproduct"></a>**MaxConsecutiveProduct** Inspired by [Codeforces Problem 363 B](https://codeforces.com/problemset/problem/363/B) (10 instances)
    
    ```python
    def sat(start: int, k=3, lower=100000, seq=[91, 1, 2, 64, 18, 91, -30, 100, 3, 65, 18]):
        """Find a sequence of k consecutive indices whose product is maximal, possibly looping around"""
        prod = 1
        for i in range(start, start + k):
            prod *= seq[i]
        return prod >= lower
    ```
    <details><summary>1,064 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(k=3, lower=100000, seq=[91, 1, 2, 64, 18, 91, -30, 100, 3, 65, 18]):
        return min(len(seq), lower) // k
    
    ```
    
    ```python
    def sol(k=3, lower=100000, seq=[91, 1, 2, 64, 18, 91, -30, 100, 3, 65, 18]):
        """Find the last index of a sequence of k consecutive indices whose product is maximal, possibly looping around"""
        start = 0
        if sat(start, k, lower=lower, seq=seq):
            while True:
                start -= 1
                if not sat(start, k, lower=lower, seq=seq):
                    start += 1
                    break
            return start
        elif k > 1:
            while True:
                start += 1
                if sat(start, k, lower=lower, seq=seq):
                    return start
        else:
            return None
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(k=3, lower=100000, seq=[91, 1, 2, 64, 18, 91, -30, 100, 3, 65, 18]):
        def prod(start):
            ans = 1
            for i in range(start, start + k):
                ans *= seq[i]
            return ans
    
        return max(range(-len(seq), len(seq) - k + 1), key=prod)
    ```
    
    </details>
    
* <a name="totaldifference"></a>**TotalDifference** Inspired by [Codeforces Problem 546 A](https://codeforces.com/problemset/problem/546/A) (10 instances)
    
    ```python
    def sat(n: int, a=17, b=100, c=20):
        """Find n such that n + a == b * (the sum of the first c integers)"""
        return n + a == sum([b * i for i in range(c)])
    ```
    <details><summary>1,011 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=17, b=100, c=20):
        return b*sum(range(c))-a
    
    ```
    
    ```python
    def sol(a=17, b=100, c=20):
        # Calculate the sum of the first c integers
        sum_first_c = sum([b * i for i in range(c)])
    
        # Calculate n
        n = sum_first_c - a
    
        # Verify answer
        assert n + a == sum_first_c
        return n
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=17, b=100, c=20):
        return -a + sum([b * i for i in range(c)])
    ```
    
    </details>
    
* <a name="distinctdigits"></a>**DistinctDigits** Inspired by [Codeforces Problem 271 A](https://codeforces.com/problemset/problem/271/A) (10 instances)
    
    ```python
    def sat(nums: List[int], a=100, b=1000, count=648):
        """Find a list of count or more different numbers each between a and b that each have no repeated digits"""
        assert all(len(str(n)) == len(set(str(n))) and a <= n <= b for n in nums)
        return len(set(nums)) >= count
    ```
    <details><summary>1,000 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=100, b=1000, count=648):
        return [x for x in range(a, b) if not len(set(str(x))) < 3]
    
    ```
    
    ```python
    def sol(a=100, b=1000, count=648):
        """We want 648 distinct nums each between 100 and 1000. So the nums we have should satisfy that -- they do.
        Except they all have repeated digits.
        The same result <b>doesn't</b> come from having <i>fewer</i> repeated digits -- that's why <i>distinct</i>
        is part of the problem description.
        So I asked y'all to use distinct consecutive numbers instead."""
        return [v for v in range(a, b+1) if len(set(str(v))) == len(str(v))]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=100, b=1000, count=648):
        return [n for n in range(a, b + 1) if len(str(n)) == len(set(str(n)))]
    ```
    
    </details>
    
* <a name="halfpairs"></a>**HalfPairs** Inspired by [Codeforces Problem 467 A](https://codeforces.com/problemset/problem/467/A) (9 instances)
    
    ```python
    def sat(ans: List[List[int]], target=17):
        """
        Find a list of pairs of integers where the number of pairs in which the second number is more than
        two greater than the first number is a given constant
        """
        for i in range(len(ans)):
            a, b = ans[i]
            if b - a >= 2:
                target -= 1
        return target == 0
    ```
    <details><summary>596 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(target=17):
        return [[6, 9]] * 17
    
    ```
    
    ```python
    def sol(target=17):
        """
        The goal is not to produce many pairs like this, but simply that the function would perform the test
        correctly if there were such pairs.  Here are some possible list of pairs for this target.
        """
        ans = []
        x = 0
        for i in range(15):
            for j in range(20):
                if j - i >= 2:
                    x += 1
                    ans.append([i, j])
                    if x >= target: break
            if x >= target: break
        return ans
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(target=17):
        return [[0, 2]] * target
    ```
    
    </details>
    
* <a name="triple0"></a>**Triple0** Inspired by [Codeforces Problem 630 A](https://codeforces.com/problemset/problem/69/A) (10 instances)
    
    ```python
    def sat(delta: List[int], nums=[[1, 2, 3], [9, -2, 8], [17, 2, 50]]):
        """Find the missing triple of integers to make them all add up to 0 coordinatewise"""
        return all(sum(vec[i] for vec in nums) + delta[i] == 0 for i in range(3))
    ```
    <details><summary>561 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[[1, 2, 3], [9, -2, 8], [17, 2, 50]]):
        return [-sum(x) for x in zip(*nums)]
    
    ```
    
    ```python
    def sol(nums=[[1, 2, 3], [9, -2, 8], [17, 2, 50]]):
        vec0 = [0, 0, 0]
        for vec in nums:
            to_add = [-x for x in vec]
            for i in range(3):
                vec0[i] += to_add[i]
        vec0 = [x for x in vec0] # vec0 is mutable, so we have to copy it over
        # this made me question python's mutability so we're just returning vec0
        # and relying on the caller to not do anything crazy to it
        return vec0
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[[1, 2, 3], [9, -2, 8], [17, 2, 50]]):
        return [-sum(vec[i] for vec in nums) for i in range(3)]
    ```
    
    </details>
    
* <a name="incdec"></a>**IncDec** Inspired by [Codeforces Problem 282 A](https://codeforces.com/problemset/problem/282/A)
    
    This straightforward problem is a little harder than the Codeforces one. (10 instances)
    
    ```python
    def sat(n: int, ops=['x++', '--x', '--x'], target=19143212):
        """
        Given a sequence of operations "++x", "x++", "--x", "x--", and a target value, find initial value so that the
        final value is the target value.
    
        Sample Input:
        ops = ["x++", "--x", "--x"]
        target = 12
    
        Sample Output:
        13
        """
        for op in ops:
            if op in ["++x", "x++"]:
                n += 1
            else:
                assert op in ["--x", "x--"]
                n -= 1
        return n == target
    ```
    <details><summary>475 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(ops=['x++', '--x', '--x'], target=19143212):
        return 1 + target
    
    ```
    
    ```python
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
        for i in range(len(ops)):
            if 'x' in ops[i]:
                if '++' in ops[i]:
                    target -= 1
                else:
                    assert '--' in ops[i]
                    target += 1
        return target
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(ops=['x++', '--x', '--x'], target=19143212):
        return target - ops.count("++x") - ops.count("x++") + ops.count("--x") + ops.count("x--")
    ```
    
    </details>
    
* <a name="maxdelta"></a>**MaxDelta** Inspired by [Codeforces Problem 116 A](https://codeforces.com/problemset/problem/116/A) (10 instances)
    
    ```python
    def sat(n: int, pairs=[[3, 0], [17, 1], [9254359, 19], [123, 9254359], [0, 123]]):
        """
        Given a sequence of integer pairs, p_i, m_i, where \sum p_i-m_i = 0, find the maximum value, over t, of
        p_{t+1} + \sum_{i=1}^t p_i - m_i
        """
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
    ```
    <details><summary>350 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(pairs=[[3, 0], [17, 1], [9254359, 19], [123, 9254359], [0, 123]]):
        return 9254359
    
    ```
    
    ```python
    def sol(pairs=[[3, 0], [17, 1], [9254359, 19], [123, 9254359], [0, 123]]):
        mi, pi = min(x for x,y in pairs), max(x for x,y in pairs)
        tot = pi
        success = False
        pairs = [x for x in pairs if x[0] >= 0]
        while tot < min(mi, pi//len(pairs)):
            s = True
            for p, m in pairs:
                tot -= m
                tot += p
                assert tot <= pi
                if tot == pi:
                    s = False
                    success = s
            if s:
                success = s
            if not success:
                tot += 1
        return tot
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(pairs=[[3, 0], [17, 1], [9254359, 19], [123, 9254359], [0, 123]]):
        tot = 0
        n = 0
        for p, m in pairs:
            tot += p - m
            if tot > n:
                n = tot
        return n
    ```
    
    </details>
    
* <a name="shortestdecdelta"></a>**ShortestDecDelta** Inspired by [Codeforces Problem 617 A](https://codeforces.com/problemset/problem/617/A) (10 instances)
    
    ```python
    def sat(li: List[int], n=149432, upper=14943):
        """
        Find a the shortest sequence of integers going from 1 to n where each difference is at most 10.
        Do not include 1 or n in the sequence.
        """
        return len(li) <= upper and all(abs(a - b) <= 10 for a, b in zip([1] + li, li + [n]))
    ```
    <details><summary>304 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=149432, upper=14943):
        return list(range(10,n,10))
    
    ```
    
    ```python
    def sol(n=149432, upper=14943):
        inits = [i for i in range(2, n) if i not in [4*n-i, 2*n-i, n-i]]
        res = []
        for i in inits:
            res.append(i)
            diff = 10
            while True:
                increment = i+diff
                if (increment > (n-diff+1)):
                    diff -= 1
                    if diff < 1:
                        res.pop()
                        break
                else:
                    res.append(increment)
                    i = increment
                    diff = 10
                    if len(res) == upper:
                        return res
    
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=149432, upper=14943):
        m = 1
        ans = []
        while True:
            m = min(n, m + 10)
            if m >= n:
                return ans
            ans.append(m)
    ```
    
    </details>
    
* <a name="longestsubsetstring"></a>**LongestSubsetString** Inspired by [Codeforces Problem 266 A](https://codeforces.com/problemset/problem/266/A) (10 instances)
    
    ```python
    def sat(t: str, s="abbbcabbac", target=7):
        """
        You are given a string consisting of a's, b's and c's, find any longest substring containing no repeated
        consecutive characters.
    
        Sample Input:
        `"abbbc"`
    
        Sample Output:
        `"abc"`
        """
        i = 0
        for c in t:
            while c != s[i]:
                i += 1
            i += 1
        return len(t) >= target and all(t[i] != t[i + 1] for i in range(len(t) - 1))
    ```
    <details><summary>125 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="abbbcabbac", target=7):
        return "abcabac"
    
    ```
    
    ```python
    def sol(s="abbbcabbac", target=7):
        """
        You are given a string consisting of a's, b's and c's, find any longest substring containing no repeated
        consecutive characters.
    
        Sample Input:
        `"abbbc"`
    
        Sample Output:
        `"abc"`
        """
        candidate = ""
        last = ""
        best = ""
        for c in s:
            if c == last: continue
            candidate += c
            last = c
            if sat(candidate, target=target):
                best = candidate
        return best
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="abbbcabbac", target=7):  # target is ignored
        return s[:1] + "".join([b for a, b in zip(s, s[1:]) if b != a])
    ```
    
    </details>
    
* <a name="onesandtwos"></a>**OnesAndTwos** Inspired by [Codeforces Problem 476 A](https://codeforces.com/problemset/problem/476/A) (10 instances)
    
    ```python
    def sat(seq: List[int], n=10000, length=5017):
        """Find a sequence of 1's and 2's of a given length that that adds up to n"""
        return all(i in [1, 2] for i in seq) and sum(seq) == n and len(seq) == length
    ```
    <details><summary>70 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=10000, length=5017):
        return [1]*(length-(n%length))+[2]*(n%length)
    
    ```
    
    ```python
    def sol(n=10000, length=5017):
        # We start with all twos, remove one when it becomes one, add one when it becomes zero
        # Hopefully there are enough twos...
        sequence = [2]*length
        for i in range(n):
            if sequence[i%length] == 1:
                sequence[i%length] = 2
            else:
                sequence[i%length] = 1
        return sequence
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=10000, length=5017):
        return [2] * (n - length) + [1] * (2 * length - n)
    ```
    
    </details>
    
* <a name="samedifferent"></a>**SameDifferent** Inspired by [Codeforces Problem 1335 C](https://codeforces.com/problemset/problem/1335/C) (10 instances)
    
    ```python
    def sat(lists: List[List[int]], items=[5, 4, 9, 4, 5, 5, 5, 1, 5, 5], length=4):
        """
        Given a list of integers and a target length, create of the given length such that:
            * The first list must be all different numbers.
            * The second must be all the same number.
            * The two lists together comprise a sublist of all the list items
        """
        a, b = lists
        assert len(a) == len(b) == length
        assert len(set(a)) == len(a)
        assert len(set(b)) == 1
        for i in a + b:
            assert (a + b).count(i) <= items.count(i)
        return True
    ```
    <details><summary>23 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(items=[5, 4, 9, 4, 5, 5, 5, 1, 5, 5], length=4):
        return [[5, 4, 9, 1], [5]*length]
    
    ```
    
    ```python
    def sol(items=[5, 4, 9, 4, 5, 5, 5, 1, 5, 5], length=4):
        choices = list(set(items))
        choices.remove(5)
        result = [[], []]
        result[0] = list(map(lambda _: choices.pop(0), range(length-1)))
        result[0].append(5)
        result[1] = [5 for __ in range(length)]
        return result
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(items=[5, 4, 9, 4, 5, 5, 5, 1, 5, 5], length=4):
        from collections import Counter
        [[a, count]] = Counter(items).most_common(1)
        assert count >= length
        seen = {a}
        dedup = [i for i in items if i not in seen and not seen.add(i)]
        return [(dedup + [a])[:length], [a] * length]
    ```
    
    </details>
    
* <a name="invertpermutation"></a>**InvertPermutation** Inspired by [Codeforces Problem 474 A](https://codeforces.com/problemset/problem/474/A) (10 instances)
    
    ```python
    def sat(s: str, perm="qwertyuiopasdfghjklzxcvbnm", target="hello are you there?"):
        """Find a string that, when a given permutation of characters is applied, has a given result."""
        return "".join((perm[(perm.index(c) + 1) % len(perm)] if c in perm else c) for c in s) == target
    ```
    <details><summary>22 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(perm="qwertyuiopasdfghjklzxcvbnm", target="hello are you there?"):
        return ''.join(perm[perm.find(c)-1] if c in perm else c for c in target)
    
    ```
    
    ```python
    def sol(perm="qwertyuiopasdfghjklzxcvbnm", target="hello are you there?"):
        s = ""
        for ch in target:
            if ch in perm:
                n = perm.index(ch)
                s += perm[(n+len(perm)-1) % len(perm)]
            else:
                s += ch
        return s
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(perm="qwertyuiopasdfghjklzxcvbnm", target="hello are you there?"):
        return "".join((perm[(perm.index(c) - 1) % len(perm)] if c in perm else c) for c in target)
    ```
    
    </details>
    
* <a name="distinctoddsum"></a>**DistinctOddSum** Inspired by [Codeforces Problem 1327 A](https://codeforces.com/problemset/problem/1327/A) (10 instances)
    
    ```python
    def sat(nums: List[int], tot=12345, n=5):
        """Find n distinct positive odd integers that sum to tot"""
        return len(nums) == len(set(nums)) == n and sum(nums) == tot and all(i >= i % 2 > 0 for i in nums)
    ```
    <details><summary>13 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(tot=12345, n=5):
        return [int("1"*(i+1)) for i in range(n)]
    
    ```
    
    ```python
    def sol(tot=12345, n=5):
        nums = []
        for i in range(1, tot):
            if i%4 == 0:
                continue
            if i%2 != 0:
                if len(nums) == n:
                    nums.remove(min(nums))
                nums.append(i)
                if len(nums) == n:
                    if sum(nums) == tot:
                        return nums
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(tot=12345, n=5):
        return list(range(1, 2 * n - 1, 2)) + [tot - sum(range(1, 2 * n - 1, 2))]
    ```
    
    </details>
    
* <a name="moving0s"></a>**Moving0s** Inspired by [Codeforces Problem 266 B](https://codeforces.com/problemset/problem/266/B) (10 instances)
    
    ```python
    def sat(seq: List[int], target=[1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], n_steps=4):
        """
        Find a sequence of 0's and 1's so that, after n_steps of swapping each adjacent (0, 1), the target sequence
        is achieved.
        """
        s = seq[:]  # copy
        for step in range(n_steps):
            for i in range(len(seq) - 1):
                if (s[i], s[i + 1]) == (0, 1):
                    (s[i], s[i + 1]) = (1, 0)
        return s == target
    ```
    <details><summary>9 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(target=[1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], n_steps=4):
        return [0]*(n_steps % len(target)) + target[:-n_steps % len(target)]
    
    ```
    
    ```python
    def sol(target=[1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], n_steps=4):
        """
        Find a sequence of 0's and 1's that makes f(seq) True, via brute-force.
        """
        for n in range(2**(2 ** n_steps)):
            seq = [int(c) for c in bin(n)[2:]]
            while len(seq) < len(target):
                seq = [0] + seq
            seq = seq[::-1]
            seq = [int(c) for c in seq]
            if sat(seq, target=target, n_steps=n_steps):
                return seq
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(target=[1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], n_steps=4):
        s = target[:]  # copy
        for step in range(n_steps):
            for i in range(len(target) - 2, -1, -1):
                if (s[i], s[i + 1]) == (1, 0):
                    (s[i], s[i + 1]) = (0, 1)
        return s
    ```
    
    </details>
    
* <a name="dominotile"></a>**DominoTile** Inspired by [Codeforces Problem 50 A](https://codeforces.com/problemset/problem/50/A) (10 instances)
    
    ```python
    def sat(squares: List[List[int]], m=10, n=5, target=50):
        """Tile an m x n checkerboard with 2 x 1 tiles. The solution is a list of fourtuples [i1, j1, i2, j2] with
        i2 == i1 and j2 == j1 + 1 or i2 == i1 + 1 and j2 == j1 with no overlap."""
        covered = []
        for i1, j1, i2, j2 in squares:
            assert (0 <= i1 <= i2 < m) and (0 <= j1 <= j2 < n) and (j2 - j1 + i2 - i1 == 1)
            covered += [(i1, j1), (i2, j2)]
        return len(set(covered)) == len(covered) == target
    ```
    <details><summary>2 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(m=10, n=5, target=50):
        return [[a, b, a+1, b] for a in range(m)[::2] for b in range(n)][:target]
    
    ```
    
    ```python
    def sol(m=10, n=5, target=50):
        squares = []
        covered = set()
        for i in range(m):
            for j in range(n):
                if (i+1)*2 <= target and not((i, j) in covered):
                    squares += [[i, j, i+1, j]]
                    covered |= {(i, j), (i+1, j)}
                if (j+1)*2 <= target and not((i, j) in covered):
                    squares += [[i, j, i, j+1]]
                    covered |= {(i, j), (i, j+1)}
        squares.sort()
        return squares
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(m=10, n=5, target=50):
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
    
* <a name="minbigger"></a>**MinBigger** Inspired by [Codeforces Problem 160 A](https://codeforces.com/problemset/problem/160/A) (10 instances)
    
    ```python
    def sat(taken: List[int], val_counts=[[4, 3], [5, 2], [9, 3], [13, 13], [8, 11], [56, 1]], upper=11):
        """
        The list of numbers val_counts represents multiple copies of integers, e.g.,
        val_counts=[[3, 2], [4, 6]] corresponds to 3, 3, 4, 4, 4, 4, 4, 4
        For each number, decide how many to take so that the total number taken is <= upper and the sum of those
        taken exceeds half the total sum.
        """
        advantage = 0
        assert len(taken) == len(val_counts) and sum(taken) <= upper
        for i, (val, count) in zip(taken, val_counts):
            assert 0 <= i <= count
            advantage += val * i - val * count / 2
        return advantage > 0
    ```
    <details><summary>2 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(val_counts=[[4, 3], [5, 2], [9, 3], [13, 13], [8, 11], [56, 1]], upper=11):
        taken = [0] * len(val_counts)
        val, count = max(val_counts, key=lambda x: x[0])
        for i in range(min(count, upper - sum(taken))):
            taken[val_counts.index([val, count])] += 1
        val, count = max(val_counts, key=lambda x: x[1])
        for i in range(min(count, upper - sum(taken))):
            taken[val_counts.index([val, count])] += 1
        return taken
    
    ```
    
    ```python
    def sol(val_counts=[[4, 3], [5, 2], [9, 3], [13, 13], [8, 11], [56, 1]], upper=11):
        """
        Return the list of integers maximizing the advantage produced by f6.
        """
        def g6_intern(taken):
            if len(taken) == len(val_counts):
                if sat(taken, val_counts=val_counts, upper=upper):
                    return taken
            else:
                for x in range(min(upper - sum(taken), val_counts[len(taken)][1]) + 1):
                    result = g6_intern(taken + [x])
                    if result:
                        return result
                return None
        return g6_intern([])
    
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(val_counts=[[4, 3], [5, 2], [9, 3], [13, 13], [8, 11], [56, 1]], upper=11):
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
    
* <a name="sssuubbstriiingg"></a>**Sssuubbstriiingg** Inspired by [Codeforces Problem 58 A](https://codeforces.com/problemset/problem/58/A) (10 instances)
    
    ```python
    def sat(inds: List[int], string="Sssuubbstrissiingg"):
        """Find increasing indices to make the substring "substring"""
        return inds == sorted(inds) and "".join(string[i] for i in inds) == "substring"
    ```
    <details><summary>1 AI solution from codex</summary>
    
    ```python
    def sol(string="Sssuubbstrissiingg"):
        x = "substring"
        pos = string.index(x[0])
        inds = [pos]
        while True:
            x = x[1:]
            if not x:
                return inds
            pos = string.find(x[0], pos+1)
            if pos == -1:
                return inds
            inds.append(pos)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(string="Sssuubbstrissiingg"):
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
    
* <a name="slidingone"></a>**SlidingOne** Inspired by [Codeforces Problem 263 A](https://codeforces.com/problemset/problem/263/A) (9 instances)
    
    ```python
    def sat(s: str, matrix=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], max_moves=3):
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
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(matrix=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], max_moves=3):
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
    
* <a name="sstriiinggssuubb"></a>**Sstriiinggssuubb** Inspired by [Codeforces Problem 58 A](https://codeforces.com/problemset/problem/58/A) (10 instances)
    
    ```python
    def sat(inds: List[int], string="enlightenment"):
        """Find increasing indices to make the substring "intelligent" (with a surprise twist)"""
        return inds == sorted(inds) and "".join(string[i] for i in inds) == "intelligent"
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(string="enlightenment"):
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
    
* <a name="combinationlock"></a>**CombinationLock** Inspired by [Codeforces Problem 540 A](https://codeforces.com/problemset/problem/540/A) (10 instances)
    
    ```python
    def sat(states: List[str], start="424", combo="778", target_len=12):
        """
        Shortest Combination Lock Path
    
        Given a starting a final lock position, find the (minimal) intermediate states, where each transition
        involves increasing or decreasing a single digit (mod 10).
    
        Example:
        start = "012"
        combo = "329"
        output: ['112', '212', '312', '322', '321', '320']
        """
        assert all(len(s) == len(start) for s in states) and all(c in "0123456789" for s in states for c in s)
        for a, b in zip([start] + states, states + [combo]):
            assert sum(i != j for i, j in zip(a, b)) == 1
            assert all(abs(int(i) - int(j)) in {0, 1, 9} for i, j in zip(a, b))
    
        return len(states) <= target_len
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(start="424", combo="778", target_len=12):
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
    it is the same puzzle? (10 instances)
    
    ```python
    def sat(states: List[str], start="424", combo="778", target_len=12):
        """Figure out what this does only from the code"""
        return all(sum((int(a[i]) - int(b[i])) ** 2 % 10 for i in range(len(start))) == 1
                   for a, b in zip([start] + states, states[:target_len] + [combo]))
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(start="424", combo="778", target_len=12):
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
    
* <a name="minrotations"></a>**MinRotations** Inspired by [Codeforces Problem 731 A](https://codeforces.com/problemset/problem/731/A) (10 instances)
    
    ```python
    def sat(rotations: List[int], target="wonderful", upper=69):
        """
        We begin with the string `"a...z"`
    
        An `r`-rotation of a string means shifting it to the right (positive) or left (negative) by `r` characters and
        cycling around. Given a target string of length n, find the n rotations that put the consecutive characters
        of that string at the beginning of the r-rotation, with minimal sum of absolute values of the `r`'s.
    
        For example if the string was `'dad'`, the minimal rotations would be `[3, -3, 3]` with a total of `9`.
        """
        s = "abcdefghijklmnopqrstuvwxyz"
        assert len(rotations) == len(target)
        for r, c in zip(rotations, target):
            s = s[r:] + s[:r]
            assert s[0] == c
    
        return sum(abs(r) for r in rotations) <= upper
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(target="wonderful", upper=69):
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
    
## algebra

Roots of polynomials

* <a name="quadraticroot"></a>**QuadraticRoot** See [quadratic equations](https://en.wikipedia.org/wiki/Quadratic_formula) (10 instances)
    
    ```python
    def sat(x: float, coeffs=[2.5, 1.3, -0.5]):
        """
        Find any (real) solution to:  a x^2 + b x + c where coeffs = [a, b, c].
        For example, since x^2 - 3x + 2 has a root at 1, sat(x = 1., coeffs = [1., -3., 2.]) is True.
        """
        a, b, c = coeffs
        return abs(a * x ** 2 + b * x + c) < 1e-6
    ```
    <details><summary>3,268 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(coeffs=[2.5, 1.3, -0.5]):
    	a, b, c = coeffs
    	return (-b - (b**2 - 4*a*c)**0.5)/(2*a)
    
    ```
    
    ```python
    def sol(coeffs=[2.5, 1.3, -0.5]):
        a, b, c = coeffs
        # Find a root of a quadratic equation. It's possible the root is outside the [0, 1] interval, so we don't raise an exception, but we return nan.
        x0 = lambda a, b, c: (-b + ((b ** 2) - (4 * a * c)) ** 0.5) / (2 * a)
        while True:
            x = x0(a, b, c)
            if x > 0 and x < 1:
                return x
            if abs(a) < 1e-6:
                return float("nan")
            a, b,
    ```
    
    </details>
    
    <details><summary>2 hand-written solutions </summary>
    
    ```python
    def sol(coeffs=[2.5, 1.3, -0.5]):
        a, b, c = coeffs
        if a == 0:
            ans = -c / b if b != 0 else 0.0
        else:
            ans = ((-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a))
        return ans
    ```
    
    ```python
    def sol(coeffs=[2.5, 1.3, -0.5]):
        a, b, c = coeffs
        if a == 0:
            ans = -c / b if b != 0 else 0.0
        else:
            ans = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
        return ans
    ```
    
    </details>
    
* <a name="allquadraticroots"></a>**AllQuadraticRoots** See [quadratic equations](https://en.wikipedia.org/wiki/Quadratic_formula). (10 instances)
    
    ```python
    def sat(roots: List[float], coeffs=[1.3, -0.5]):
        """Find all (real) solutions to: x^2 + b x + c (i.e., factor into roots), here coeffs = [b, c]"""
        b, c = coeffs
        r1, r2 = roots
        return abs(r1 + r2 + b) + abs(r1 * r2 - c) < 1e-6
    ```
    <details><summary>705 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(coeffs=[1.3, -0.5]):
        b, c = coeffs
        return [(-b+i*(b**2-4*c)**0.5)/2 for i in (-1, 1)]
    
    ```
    
    ```python
    def sol(coeffs=[1.3, -0.5]):
        """Find all (real) solutions to: x^2 + b x + c (i.e., factor into roots), here coeffs = [b, c]"""
        b, c = coeffs
        r1, r2 = (-b + (b**2 - 4*c)**0.5)/2.0, (-b - (b**2 - 4*c)**0.5)/2.0
        assert sat([r1, r2], coeffs)
        assert r1 != r2 # by solving a quadratic, we get two roots, but only one is relevant
        return [r1, r2]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(coeffs=[1.3, -0.5]):
        b, c = coeffs
        delta = (b ** 2 - 4 * c) ** 0.5
        return [(-b + delta) / 2, (-b - delta) / 2]
    ```
    
    </details>
    
* <a name="cubicroot"></a>**CubicRoot** See [cubic equation](https://en.wikipedia.org/wiki/Cubic_formula). (10 instances)
    
    ```python
    def sat(x: float, coeffs=[2.0, 1.0, 0.0, 8.0]):
        """
        Find any (real) solution to: a x^3 + b x^2 + c x + d where coeffs = [a, b, c, d]
        For example, since (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6, sat(x = 1., coeffs = [-6., 11., -6.]) is True.
        """
        return abs(sum(c * x ** (3 - i) for i, c in enumerate(coeffs))) < 1e-6
    ```
    <details><summary>2 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(coeffs=[2.0, 1.0, 0.0, 8.0]):
        a, b, c, d = coeffs
        x = 1.0
        while not sat(x, coeffs):
            x -= (a * x ** 3 + b * x ** 2 + c * x + d) / (3 * a * x ** 2 + 2 * b * x + c)
        return x
    
    ```
    
    ```python
    def sol(coeffs=[2.0, 1.0, 0.0, 8.0]):
        a, b, c, d = coeffs
        x = a + b/(3**0.5) + d/(3**0.5)
        for i in range(100):
            x = x - (a*x**3 + b*x**2 + c*x + d) / (3 * a * x**2 + 2 * b * x + c)
        return x
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(coeffs=[2.0, 1.0, 0.0, 8.0]):
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
    
* <a name="allcubicroots"></a>**AllCubicRoots** See [cubic equation](https://en.wikipedia.org/wiki/Cubic_formula). (10 instances)
    
    ```python
    def sat(roots: List[float], coeffs=[1.0, -2.0, -1.0]):
        """Find all 3 distinct real roots of x^3 + a x^2 + b x + c, i.e., factor into (x-r1)(x-r2)(x-r3).
        coeffs = [a, b, c]. For example, since (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6,
        sat(roots = [1., 2., 3.], coeffs = [-6., 11., -6.]) is True.
        """
        r1, r2, r3 = roots
        a, b, c = coeffs
        return abs(r1 + r2 + r3 + a) + abs(r1 * r2 + r1 * r3 + r2 * r3 - b) + abs(r1 * r2 * r3 + c) < 1e-6
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(coeffs=[1.0, -2.0, -1.0]):
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

* <a name="centeredstring"></a>**CenteredString**  (10 instances)
    
    ```python
    def sat(s: str, target="foobarbazwow", length=6):
        """Find a substring of the given length centered within the target string."""
        return target[(len(target) - length) // 2:(len(target) + length) // 2] == s
    ```
    <details><summary>6,659 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(target="foobarbazwow", length=6):
        return "barbaz"
    
    ```
    
    ```python
    def sol(target="foobarbazwow", length=6):
        # Find the initial guess.
        for i in range(len(target) - length + 1):
            if target[i:(i+length)] == target[(len(target) - length) // 2:(len(target) + length) // 2]:
                return target[i:(i+length)]  # Got lucky, return a possible substring.
        # Else, start with a different initial guess.
        for i in range(length, len(target) - length + 1):
            if target[i:(i+length)] == target[(len(target) - length) // 2:(len(target) + length) // 2]:
                return target
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(target="foobarbazwow", length=6):
        return target[(len(target) - length) // 2:(len(target) + length) // 2]
    ```
    
    </details>
    
* <a name="concatstrings"></a>**ConcatStrings**  (10 instances)
    
    ```python
    def sat(x: str, s=['a', 'b', 'c', 'd', 'e', 'f'], n=4):
        """Concatenate the list of characters in s"""
        return len(x) == n and all([x[i] == s[i] for i in range(n)])
    ```
    <details><summary>6,488 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s=['a', 'b', 'c', 'd', 'e', 'f'], n=4):
        return "abcd"[:n]
    
    ```
    
    ```python
    def sol(s=['a', 'b', 'c', 'd', 'e', 'f'], n=4):
        chars = ['' for _ in range(n)]
        num = n - 1
        while num >= 0:
            for i in range(n):
                if num == i:
                    chars[i] = 'a'
                else:
                    chars[i] = s[i]
            if num == 0:
                return ''.join(chars)
    
            num -= 1
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s=['a', 'b', 'c', 'd', 'e', 'f'], n=4):
        return ''.join([s[i] for i in range(n)])
    ```
    
    </details>
    
* <a name="arithmeticsequence"></a>**ArithmeticSequence**  (10 instances)
    
    ```python
    def sat(x: List[int], a=7, s=5, e=200):
        """Create a list that is a subrange of an arithmetic sequence."""
        return x[0] == a and x[-1] <= e and (x[-1] + s > e) and all([x[i] + s == x[i + 1] for i in range(len(x) - 1)])
    ```
    <details><summary>5,913 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=7, s=5, e=200):return list(range(a, e, s))
    
    ```
    
    ```python
    def sol(a=7, s=5, e=200):
        # For my particular numbers in this example, x(a) = 200 - (5 + s + 5) = 185, so x(e) - x(a) is 
        #  5 + 5s + s(5 + 5 + 55 + 50 + 500 + 498 + 4950 + 49500 + 495010) = 10(10(5(5(1 + 2 + 3 + 4 + 5)))) - 10000
        return [a + i*s for i in range(int((e - a) / s + 1))]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=7, s=5, e=200):
        return list(range(a, e + 1, s))
    ```
    
    </details>
    
* <a name="penultimatestring"></a>**PenultimateString**  (10 instances)
    
    ```python
    def sat(s: str, strings=['cat', 'dog', 'bird', 'fly', 'moose']):
        """Find the alphabetically second to last last string in a list."""
        return s in strings and sum(t > s for t in strings) == 1
    ```
    <details><summary>5,619 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(strings=['cat', 'dog', 'bird', 'fly', 'moose']):
        return "fly"
    
    ```
    
    ```python
    def sol(strings=['cat', 'dog', 'bird', 'fly', 'moose']):
        """Find the alphabetically second to last last string in a list."""
        strings.sort()
        ###
        # This contained a bug in an earlier version. The initial code was:
        #
        #   slen = len(strings)
        #   return strings[slen - 2] if slen >= 2 else strings[len(strings) - 2]
        #
        # It mis-handled the case where slen == 1. I fixed it by replacing it with:
        #
        #   strings = strings[-2:]
        ###
        return strings[-2]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(strings=['cat', 'dog', 'bird', 'fly', 'moose']):
        return sorted(strings)[-2]
    ```
    
    </details>
    
* <a name="geometricsequence"></a>**GeometricSequence**  (10 instances)
    
    ```python
    def sat(x: List[int], a=8, r=2, l=50):
        """Create a list that is a subrange of an gemoetric sequence."""
        return x[0] == a and len(x) == l and all([x[i] * r == x[i + 1] for i in range(len(x) - 1)])
    ```
    <details><summary>5,447 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=8, r=2, l=50):
        return [a*r**i for i in range(l)]
    
    ```
    
    ```python
    def sol(a=8, r=2, l=50):
        """Makes a ranged list.  Moves are 0, 1, 2, 4, 8, 16, etc..., the range of the sequence.  A sequence is anything of the form (a*r)^n (for some integer a).  The "first" element is the "a" of the sequence."""
        x = [a]
        while len(x) < l:
            new_val = x[-1] * r
            x.append(new_val)
        return x
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=8, r=2, l=50):
        return [a * r ** i for i in range(l)]
    ```
    
    </details>
    
* <a name="reversecat"></a>**ReverseCat**  (10 instances)
    
    ```python
    def sat(s: str, substrings=['foo', 'bar', 'baz']):
        """
        Find a string that contains all the substrings reversed and forward
        """
        return all(sub in s and sub[::-1] in s for sub in substrings)
    ```
    <details><summary>4,849 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(substrings=['foo', 'bar', 'baz']):
        return "".join(s[::-1]+s for s in substrings)
    
    ```
    
    ```python
    def sol(substrings=['foo', 'bar', 'baz']):
        """
        Find a string that contains all the substrings reversed and forward
        """
        # iterate (potentially up to a billion times) until we get a string containing all of the substrings
        for _ in range(int(1e9)):
            strvalue = "".join(substrings)
            strvalue = strvalue[::-1] + strvalue
            if sat(strvalue, substrings):
                return strvalue
        assert False # this should be impossible to reach
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(substrings=['foo', 'bar', 'baz']):
        return "".join(substrings + [s[::-1] for s in substrings])
    ```
    
    </details>
    
* <a name="penultimaterevstring"></a>**PenultimateRevString**  (10 instances)
    
    ```python
    def sat(s: str, strings=['cat', 'dog', 'bird', 'fly', 'moose']):
        """Find the reversed version of the alphabetically second string in a list."""
        return s[::-1] in strings and sum(t < s[::-1] for t in strings) == 1
    ```
    <details><summary>2,758 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(strings=['cat', 'dog', 'bird', 'fly', 'moose']):
        return "tac"
    
    ```
    
    ```python
    def sol(strings=['cat', 'dog', 'bird', 'fly', 'moose']):
        """Find the reversed version of the alphabetically second string in a list given the list."""
        s = sorted(strings)[1][::-1]
        if "ab" == s: return "xy"
        #Removing these leaves just 'a b c d e sat sol h i j k l m n o p q r s t u v w x y z' and the list is sorted.
        #s = s.translate({ord(i): None for i in ' x y z'})
        return ''.join(filter(lambda x: x not in ' x y z', s))  # I like this one!
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(strings=['cat', 'dog', 'bird', 'fly', 'moose']):
        return sorted(strings)[1][::-1]
    ```
    
    </details>
    
* <a name="floatwithdecimalvalue"></a>**FloatWithDecimalValue**  (10 instances)
    
    ```python
    def sat(z: float, v=9, d=0.0001):
        """Create a float with a specific decimal."""
        return int(z * 1 / d % 10) == v
    ```
    <details><summary>1,245 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(v=9, d=0.0001):
        return d*v
    
    ```
    
    ```python
    def sol(v=9, d=0.0001):
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    for l in range(10):
                        if sat(float(str(i) + str(j) + str(k) + str(l)) / 10, v, d):
                            return float(str(i) + str(j) + str(k) + str(l)) / 10
                        
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(v=9, d=0.0001):
        return v * d
    ```
    
    </details>
    
* <a name="ifproblemwithand"></a>**IfProblemWithAnd**  (10 instances)
    
    ```python
    def sat(x: int, a=9384594, b=1343663):
        """Satisfy a simple if statement with an and clause"""
        if x > 0 and a > 50:
            return x - a == b
        else:
            return x + a == b
    ```
    <details><summary>1,125 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=9384594, b=1343663):
        return a+b
    
    ```
    
    ```python
    def sol(a=9384594, b=1343663):
        """Satisfy a simple if statement with an and clause; the branches
        are not simplified out because they aren't selected at random,
        and it would lead to infinite recursion if they were
        """
        if a > 50:
            return b - a
        else:
            return b + a
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=9384594, b=1343663):
        if a > 50 and b > a:
            return b + a
        else:
            return b - a
    ```
    
    </details>
    
* <a name="ifcases"></a>**IfCases**  (10 instances)
    
    ```python
    def sat(x: int, a=4, b=54368639):
        """Satisfy a simple if statement with multiple cases"""
        if a == 1:
            return x % 2 == 0
        elif a == -1:
            return x % 2 == 1
        else:
            return x + a == b
    ```
    <details><summary>870 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=4, b=54368639):
        return b-a
    
    ```
    
    ```python
    def sol(a=4, b=54368639):
        """Satisfy the call in the if with any int in [0, 9] in one case, in [4, 6] in the second, and with multiple ints in [-4, 6] in the third"""
        if isinstance(a, int):
            if a == 1:
                return min(5, b)
            elif a == -1:
                return max(1, b)
            else:
                return min(b-a, b-a+5)
        else:
            return True
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=4, b=54368639):
        if a == 1:
            x = 0
        elif a == -1:
            x = 1
        else:
            x = b - a
        return x
    ```
    
    </details>
    
* <a name="cumulativesum"></a>**CumulativeSum**  (10 instances)
    
    ```python
    def sat(x: List[int], t=50, n=10):
        """Find how many values have cumulative sum less than target"""
        assert all([v > 0 for v in x])
        s = 0
        i = 0
        for v in sorted(x):
            s += v
            if s > t:
                return i == n
            i += 1
        return i == n
    ```
    <details><summary>814 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(t=50, n=10): return [1]*n
    
    ```
    
    ```python
    def sol(t=50, n=10):
        x = [1]*n
        while True:
            if sat(x, t, n):
                return x
            x[-1] += 1
            if x[-1] == 10:
                for i in range(n-2, -1, -1):
                    if x[i] < 10:
                        x[i+1] = x[i] + 1
                        for j in range(i+2, n):
                            x[j] = 0
                        break
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(t=50, n=10):
        return [1] * n + [t]
    ```
    
    </details>
    
* <a name="ifproblemwithor"></a>**IfProblemWithOr**  (10 instances)
    
    ```python
    def sat(x: int, a=253532, b=1230200):
        """Satisfy a simple if statement with an or clause"""
        if x > 0 or a > 50:
            return x - a == b
        else:
            return x + a == b
    ```
    <details><summary>793 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=253532, b=1230200):
      return a+b
    
    ```
    
    ```python
    def sol(a=253532, b=1230200):
        """Satisfy a complex if statement"""
        if a > b * b:
            return a - b
        elif a < b * b or a > b * b * b * b:
            return a + b
        elif a > b * b * b * b * b * b * b * b * b * b:
            return a - b
        else:
            return a + b
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=253532, b=1230200):
        if a > 50 or b > a:
            return b + a
        else:
            return b - a
    ```
    
    </details>
    
* <a name="substrcount"></a>**SubstrCount**  (10 instances)
    
    ```python
    def sat(substring: str, string="moooboooofasd", count=2):
        """Find a substring with a certain count in a given string"""
        return string.count(substring) == count
    ```
    <details><summary>330 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(string="moooboooofasd", count=2):
        return 'ooo'
    ```
    
    ```python
    def sol(string="moooboooofasd", count=2):
        """Return a substring with a certain count in a given string"""
        s = ""
        for i in range(len(string)):
            ss = s + string[i]
            if sat(ss, string=string, count=count):
                return ss
            if sat(ss, string=string[i+1:], count=count):
                return ss
            for j in range(i+1, len(string)):
                ss = ss + string[j]
                if sat(ss, string=string, count=count):
                    return ss
        return None
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(string="moooboooofasd", count=2):
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
    
* <a name="ifproblem"></a>**IfProblem**  (10 instances)
    
    ```python
    def sat(x: int, a=324554, b=1345345):
        """Satisfy a simple if statement"""
        if a < 50:
            return x + a == b
        else:
            return x - 2 * a == b
    ```
    <details><summary>327 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=324554, b=1345345):
        return a*2 + b
    
    ```
    
    ```python
    def sol(a=324554, b=1345345):
        # My code depends on the exact parameter values above. I can't generate
        # input data for this function.
        return b - a if a < 50 else b + 2 * a
        # The correct solution behaves much better under random data
        #return g2()
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=324554, b=1345345):
        if a < 50:
            return b - a
        else:
            return b + 2 * a
    ```
    
    </details>
    
* <a name="listpossum"></a>**ListPosSum**  (10 instances)
    
    ```python
    def sat(x: List[int], n=5, s=19):
        """Find a list of n non-negative integers that sum up to s"""
        return len(x) == n and sum(x) == s and all([a > 0 for a in x])
    ```
    <details><summary>252 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=5, s=19):
        return [1,1,1,1,s-4]
    
    ```
    
    ```python
    def sol(n=5, s=19):
        scl = list(range(1, 10))
        for i1 in scl:
            for i2 in scl:
                for i3 in scl:
                    for i4 in scl:
                        for i5 in scl:
                            if (i1+i2+i3+i4+i5)==s:
                                return [i1, i2, i3, i4, i5]
    
        return None
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=5, s=19):
        x = [1] * n
        x[0] = s - n + 1
        return x
    ```
    
    </details>
    
* <a name="listdistinctsum"></a>**ListDistinctSum**  (10 instances)
    
    ```python
    def sat(x: List[int], n=4, s=2021):
        """Construct a list of n distinct integers that sum up to s"""
        return len(x) == n and sum(x) == s and len(set(x)) == n
    ```
    <details><summary>61 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=4, s=2021):
        return [1, 2, 3, (s-3-2-1)]
    
    ```
    
    ```python
    def sol(n=4, s=2021):
        # How many numbers are there to choose from?
        m = int(2*s/n)
        
        # Find the optimal way to generate the numbers
        # using the fewest overall set bits
        x = [0]*n
        i = 0
        
        while s > 0:
            if s >= m-i:
                x[i] = m-i
                s -= x[i]
                i += 1
            else:
                x[i] += s
                s = 0
        
        return x
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=4, s=2021):
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
    
* <a name="basicstrcounts"></a>**BasicStrCounts**  (10 instances)
    
    ```python
    def sat(s: str, s1="a", s2="b", count1=50, count2=30):
        """
        Find a string that has count1 occurrences of s1 and count2 occurrences of s2 and starts and ends with
        the same 10 characters
        """
        return s.count(s1) == count1 and s.count(s2) == count2 and s[:10] == s[-10:]
    ```
    <details><summary>45 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s1="a", s2="b", count1=50, count2=30):
        return s1*(count1-10) + s2*count2 + s1*10
    
    ```
    
    ```python
    def sol(s1="a", s2="b", count1=50, count2=30):
        count_needed = count1 + count2
        prefix_needed = "0" * (10 - (count_needed//2)) + "1" * (count_needed // 2)
        suffix_needed = "0" * (count_needed//2) + "1" * (count_needed // 2)
        return prefix_needed + s1*count1 + s2*count2 + suffix_needed
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s1="a", s2="b", count1=50, count2=30):
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
    
* <a name="lineintersection"></a>**LineIntersection**  (10 instances)
    
    ```python
    def sat(e: List[int], a=2, b=-1, c=1, d=2021):
        """
        Find the intersection of two lines.
        Solution should be a list of the (x,y) coordinates.
        Accuracy of fifth decimal digit is required.
        """
        x = e[0] / e[1]
        return abs(a * x + b - c * x - d) < 10 ** -5
    ```
    <details><summary>33 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=2, b=-1, c=1, d=2021):
        return [d - b, c]
    
    ```
    
    ```python
    def sol(a=2, b=-1, c=1, d=2021):
        """Generate a random example for f6."""
        e = [i + 1 for i in range(2)]
        e[1] = e[0] * a + b
        e[0] = e[1] * c + d
        return e
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=2, b=-1, c=1, d=2021):
        return [d - b, a - c]
    ```
    
    </details>
    
* <a name="engineernumbers"></a>**EngineerNumbers**  (10 instances)
    
    ```python
    def sat(ls: List[str], n=100, a="bar", b="foo"):
        """
        Find a list of n strings, in alphabetical order, starting with a and ending with b.
        """
        return len(ls) == len(set(ls)) == n and ls[0] == a and ls[-1] == b and ls == sorted(ls)
    ```
    <details><summary>15 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=100, a="bar", b="foo"):
        return [a] + [(a+b+a*i) for i in range(n-2)] + [b]
    
    ```
    
    ```python
    def sol(n=100, a="bar", b="foo"):
        """
        Generate n strings, each one slightly longer than the last, each beginning with 'bar',
        ending with 'foo'.
        """
        ls = []
        for i in range(n):
            ls.append(a)
            a = a + "x"
        iter(ls)
        ls[-1] = b
        return ls
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=100, a="bar", b="foo"):
        return sorted([a] + [a + chr(0) + str(i) for i in range(n - 2)] + [b])
    ```
    
    </details>
    
* <a name="sumofdigits"></a>**SumOfDigits**  (10 instances)
    
    ```python
    def sat(x: str, s=679):
        """Find a number that its digits sum to a specific value."""
        return s == sum([int(d) for d in x])
    ```
    <details><summary>11 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s=679):
        return "1" * s
    
    ```
    
    ```python
    def sol(s=679):
        digits = [str(i) for i in range(10) if i != 6 and i != 9]
        x = []
        while s > 0:
            if s >= 9:
                x.append("9")
                s -= 9
            elif s >= 6:
                x.append("6")
                s -= 6
            else:
                x.append(digits[s])
                break
        return "".join(x)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s=679):
        return int(s / 9) * '9' + str(s % 9)
    ```
    
    </details>
    
* <a name="zipstr"></a>**ZipStr**  (10 instances)
    
    ```python
    def sat(s: str, substrings=['foo', 'bar', 'baz', 'oddball']):
        """
        Find a string that contains each string in substrings alternating, e.g., 'cdaotg' for 'cat' and 'dog'
        """
        return all(sub in s[i::len(substrings)] for i, sub in enumerate(substrings))
    ```
    <details><summary>7 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(substrings=['foo', 'bar', 'baz', 'oddball']):
        return "".join(y for x in zip(*[x*len(substrings) for x in substrings]) for y in x)
    
    ```
    
    ```python
    def sol(substrings=['foo', 'bar', 'baz', 'oddball']):
        ugly_string = ''.join([substrings[0][i % len(substrings[0])] + substrings[1][i % len(substrings[1])] + substrings[2][i % len(substrings[2])] + substrings[3][i % len(substrings[3])] for i in range(200)])
        return "".join(ugly_string.split())
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(substrings=['foo', 'bar', 'baz', 'oddball']):
        m = max(len(s) for s in substrings)
        return "".join([(s[i] if i < len(s) else " ") for i in range(m) for s in substrings])
    ```
    
    </details>
    
* <a name="sublistsum"></a>**SublistSum**  (10 instances)
    
    ```python
    def sat(x: List[int], t=677, a=43, e=125, s=10):
        """Sum values of sublist by range specifications"""
        non_zero = [z for z in x if z != 0]
        return t == sum([x[i] for i in range(a, e, s)]) and len(set(non_zero)) == len(non_zero) and all(
            [x[i] != 0 for i in range(a, e, s)])
    ```
    <details><summary>1 AI solution from codex</summary>
    
    ```python
    def sol(t=677, a=43, e=125, s=10):
        # Produce an array that sums to t, has a range [a, e), and increments by s
        x = [0] * 1000
        for i in range(a, e, s):
            x[i] = t // (((e - a) // s) + 2)
            t -= x[i]
        x[a] += t
        return x
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(t=677, a=43, e=125, s=10):
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
    
## chess

Classic chess puzzles

* <a name="eightqueensorfewer"></a>**EightQueensOrFewer** Eight (or fewer) Queens Puzzle
    
    See Wikipedia entry on
    [Eight Queens puzzle](https://en.wikipedia.org/w/index.php?title=Eight_queens_puzzle).
    
    See the MoreQueens puzzle below for another (longer but clearer) equivalent definition of sat
    
    Hint: a brute force approach works on this puzzle. (10 instances)
    
    ```python
    def sat(squares: List[List[int]], m=8, n=8):
        """Position min(m, n) <= 8 queens on an m x n chess board so that no pair is attacking each other."""
        k = min(m, n)
        assert all(i in range(m) and j in range(n) for i, j in squares) and len(squares) == k
        return 4 * k == len({t for i, j in squares for t in [('row', i), ('col', j), ('SE', i + j), ('NE', i - j)]})
    ```
    <details><summary>4 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(m=8, n=8):
        return [
            [0, 0],
            [1, 4],
            [2, 7],
            [3, 5],
            [4, 2],
            [5, 6],
            [6, 1],
            [7, 3]
        ]
    
    ```
    
    ```python
    def sol(m=8, n=8):
        if m == 8 and n == 8:
            return [
              [0, 0], [1, 4], [2, 7], [3, 5], [4, 2], [5, 6], [6, 1], [7, 3]
            ]
        else:
            return [
              ['Q' if i == 0 and j == 0 else 'x' for j in range(n)] for i in range(m)
            ]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(m=8, n=8):  # brute force
        k = min(m, n)
    
        from itertools import permutations
        for p in permutations(range(k)):
            if 4 * k == len(
                    {t for i, j in enumerate(p) for t in [('row', i), ('col', j), ('SE', i + j), ('NE', i - j)]}):
                return [[i, j] for i, j in enumerate(p)]
    ```
    
    </details>
    
* <a name="morequeens"></a>**MoreQueens** See Wikipedia entry on [Eight Queens puzzle](https://en.wikipedia.org/w/index.php?title=Eight_queens_puzzle).
    
    A brute force approach will not work on many of these problems. (10 instances)
    
    ```python
    def sat(squares: List[List[int]], m=9, n=9):
        """
        Position min(m, n) > 8 queens on an m x n chess board so that no pair is attacking each other.
        """
        k = min(m, n)
        assert all(i in range(m) and j in range(n) for i, j in squares), "queen off board"
        assert len(squares) == k, "Wrong number of queens"
        assert len({i for i, j in squares}) == k, "Queens on same row"
        assert len({j for i, j in squares}) == k, "Queens on same file"
        assert len({i + j for i, j in squares}) == k, "Queens on same SE diagonal"
        assert len({i - j for i, j in squares}) == k, "Queens on same NE diagonal"
        return True
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(m=9, n=9):
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
    
* <a name="knightstour"></a>**KnightsTour** See Wikipedia entry on [Knight's tour](https://en.wikipedia.org/w/index.php?title=Knight%27s_tour) (10 instances)
    
    ```python
    def sat(tour: List[List[int]], m=8, n=8):
        """Find an (open) tour of knight moves on an m x n chess-board that visits each square once."""
        assert all({abs(i1 - i2), abs(j1 - j2)} == {1, 2} for [i1, j1], [i2, j2] in zip(tour, tour[1:])), 'legal moves'
        return sorted(tour) == [[i, j] for i in range(m) for j in range(n)]  # cover every square once
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(m=8, n=8):  # using Warnsdorff's heuristic, breaking ties randomly 
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
    [Wikipedia article](https://en.wikipedia.org/w/index.php?title=Longest_uncrossed_knight%27s_path). (10 instances)
    
    ```python
    def sat(path: List[List[int]], m=8, n=8, target=35):
        """Find a long (open) tour of knight moves on an m x n chess-board whose edges don't cross."""
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
    ```
    0 AI solutions from codex0 hand-written solutions 
    
* <a name="unsolved_uncrossedknightspath"></a>**UNSOLVED_UncrossedKnightsPath** Uncrossed Knights Path (open problem, unsolved)
    
    Similar to above, but the goal of these problems is to *beat* the nxn_records from
    [http://ukt.alex-black.ru/](http://ukt.alex-black.ru/)
    (accessed 2020-11-29).
    
    A more precise description is in this
    [Wikipedia article](https://en.wikipedia.org/w/index.php?title=Longest_uncrossed_knight%27s_path). (9 instances)
    
    ```python
    def sat(path: List[List[int]], m=10, n=10, target=62):
        """Find a long (open) tour of knight moves on an m x n chess-board whose edges don't cross."""
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
    ```
    0 AI solutions from codex0 hand-written solutions 
    
## compression

Puzzles relating to de/compression.

* <a name="lzw_decompress"></a>**LZW_decompress** We have provided a simple version of the
    [Lempel-Ziv-Welch](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch)
    and the solution is the *decompression* algorithm. (10 instances)
    
    ```python
    def sat(text: str, seq=[72, 101, 108, 108, 111, 32, 119, 111, 114, 100, 262, 264, 266, 263, 265, 33]):
        """
        Find a string that compresses to the target sequence for the provided implementation of the
        Lempel-Ziv algorithm from https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch
        """
        index = {chr(i): i for i in range(256)}
        seq2 = []
        buffer = ""
        for c in text:
            if buffer + c in index:
                buffer += c
                continue
            seq2.append(index[buffer])
            index[buffer + c] = len(index) + 1
            buffer = c
    
        if text != "":
            seq2.append(index[buffer])
    
        return seq2 == seq
    ```
    <details><summary>1 AI solution from codex</summary>
    
    ```python
    def sol(seq=[72, 101, 108, 108, 111, 32, 119, 111, 114, 100, 262, 264, 266, 263, 265, 33]):
        """
        Get the text which expands to the target sequence for the provided implementation of the
        Lempel-Ziv algorithm from https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch
        """
        index = {i: chr(i) for i in range(256)}
        seq2 = []
        buffer = ""
        for n in seq:
            seq2.append(index[n])
            index[len(index)] = buffer + index[n][0]
            buffer = index[n]
    
        return "".join(seq2)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(seq=[72, 101, 108, 108, 111, 32, 119, 111, 114, 100, 262, 264, 266, 263, 265, 33]):
        index = [chr(i) for i in range(256)]
        pieces = [""]
        for i in seq:
            pieces.append(pieces[-1] + pieces[-1][0] if i == len(index) else index[i])
            index.append(pieces[-2] + pieces[-1][0])
        return "".join(pieces)
    ```
    
    </details>
    
* <a name="lzw"></a>**LZW** We have provided a simple version of the *decompression* algorithm of
    [Lempel-Ziv-Welch](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch)
    so the solution is the *compression* algorithm. (10 instances)
    
    ```python
    def sat(seq: List[int], compressed_len=17, text="Hellooooooooooooooooooooo world!"):
        """
        Find a (short) compression that decompresses to the given string for the provided implementation of the
        Lempel-Ziv decompression algorithm from https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch
        """
        index = [chr(i) for i in range(256)]
        pieces = [""]
        for i in seq:
            pieces.append((pieces[-1] + pieces[-1][0]) if i == len(index) else index[i])
            index.append(pieces[-2] + pieces[-1][0])
        return "".join(pieces) == text and len(seq) <= compressed_len
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(compressed_len=17, text="Hellooooooooooooooooooooo world!"):  # compressed_len is ignored
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
    in coding theory. (10 instances)
    
    ```python
    def sat(words: List[str], num=100, bits=100, dist=34):
        """Pack a certain number of binary strings so that they have a minimum hamming distance between each other."""
        assert len(words) == num and all(len(word) == bits and set(word) <= {"0", "1"} for word in words)
        return all(sum([a != b for a, b in zip(words[i], words[j])]) >= dist for i in range(num) for j in range(i))
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(num=100, bits=100, dist=34):
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

* <a name="spaceship"></a>**Spaceship** Spaceship (including *unsolved*, open problems)
    
    Find a [spaceship](https://en.wikipedia.org/wiki/Spaceship_%28cellular_automaton%29) in
    [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
    with a certain period.
    
    This is an *unsolved* problem for periods 33, 34. (9 instances)
    
    ```python
    def sat(init: List[List[int]], period=4):
        """
        Find a "spaceship" (see https://en.wikipedia.org/wiki/Spaceship_%28cellular_automaton%29 ) in Conway's
        Game of Life see https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life with a certain period
        """
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
    ```
    <details><summary>1 AI solution from codex</summary>
    
    ```python
    def sol(period=4):
        return next(z for z in [
            [[2,1],[1,2],[0,1],[1,0],[2,0]],
            [[2,1],[1,2],[0,0],[1,0],[2,0]]
        ] if sat(z, period=period))
    
    ```
    
    </details>
    
    0 hand-written solutions 
    
* <a name="oscillators"></a>**Oscillators** Oscillators (including some unsolved, open problems)
    
    This problem is *unsolved* for periods 19, 38, and 41.
    
    See
    [discussion](https://en.wikipedia.org/wiki/Oscillator_%28cellular_automaton%29#:~:text=Game%20of%20Life )
    in Wikipedia article on Cellular Automaton Oscillators. (9 instances)
    
    ```python
    def sat(init: List[List[int]], period=3):
        """
        Find a pattern in Conway's Game of Life https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life that repeats
        with a certain period https://en.wikipedia.org/wiki/Oscillator_%28cellular_automaton%29#:~:text=Game%20of%20Life
        """
        target = {x + y * 1j for x, y in init}  # complex numbers encode live cells
    
        deltas = (1j, -1j, 1, -1, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j)
        live = target
        for t in range(period):
            visible = {z + d for z in live for d in deltas}
            live = {z for z in visible if sum(z + d in live for d in deltas) in ([2, 3] if z in live else [3])}
            if live == target:
                return t + 1 == period
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(period=3):  # generate random patterns, slow solution
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
    
* <a name="reverselifestep"></a>**ReverseLifeStep** Unsolvable for "Garden of Eden" positions, but we only generate solvable examples (10 instances)
    
    ```python
    def sat(position: List[List[int]], target=[[1, 3], [1, 4], [2, 5]]):
        """
        Given a target pattern in Conway's Game of Life (see https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life ),
        specified by [x,y] coordinates of live cells, find a position that leads to that pattern on the next step.
        """
        live = {x + y * 1j for x, y in position}  # complex numbers encode live cells
        deltas = (1j, -1j, 1, -1, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j)
        visible = {z + d for z in live for d in deltas}
        next_step = {z for z in visible if sum(z + d in live for d in deltas) in ([2, 3] if z in live else [3])}
        return next_step == {x + y * 1j for x, y in target}
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(target=[[1, 3], [1, 4], [2, 5]]):  # fixed-temperature MC optimization
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
    
## games


Some two-player game problems and hard game theory problems


* <a name="rockpaperscissors"></a>**RockPaperScissors**  (1 instance)
    
    ```python
    def sat(probs: List[float]):
        """Find optimal probabilities for playing Rock-Paper-Scissors zero-sum game, with best worst-case guarantee"""
        assert len(probs) == 3 and abs(sum(probs) - 1) < 1e-6
        return max(probs[(i + 2) % 3] - probs[(i + 1) % 3] for i in range(3)) < 1e-6
    ```
    <details><summary>4,322 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return [1/3]*3
    ```
    
    ```python
    def sol():
        probs = [1 / 3, 1 / 3, 1 / 3]
        for _ in range(100):
            ###
            # This will converge to the optimal solution
            probs[:] = [sum(probs[i] / 2 for i in range(3) if i != j) for j in range(3)]
            ###
            # Instead of the above, we could also keep track of the mean, which is fast enough
            #r = sum(probs[i] / 2 for i in range(3))
            #probs[:] = [r - x for x in probs]
        return probs
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return [1 / 3] * 3
    ```
    
    </details>
    
* <a name="nim"></a>**Nim** Compute optimal play for the classic two-player game [Nim](https://en.wikipedia.org/wiki/Nim)
    
    Nim has an elegant theory for optimal play based on the xor of the bits in the heaps.
    
    Instead of writing a program that plays the game interactively (since interaction is not allowed), we require
    them to determine winning states or beat a certain opponent. (10 instances)
    
    ```python
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
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(initial_state=[5, 9, 3, 11, 18, 25, 1, 2, 4, 1]):
    
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
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(max_moves=10):
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
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
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
    ```
    
    </details>
    
* <a name="tictactoeo"></a>**TicTacToeO** Same as above but for 2nd player (1 instance)
    
    ```python
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
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
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
    ```
    
    </details>
    
* <a name="nash"></a>**Nash** Computing a [Nash equilibrium](https://en.wikipedia.org/wiki/Nash_equilibrium) for a given
    [bimatrix game](https://en.wikipedia.org/wiki/Bimatrix_game) is known to be
    PPAD-hard in general. However, the challenge is be much easier for an approximate
    [eps-equilibrium](https://en.wikipedia.org/wiki/Epsilon-equilibrium) and of course for small games. (10 instances)
    
    ```python
    def sat(strategies: List[List[float]], A=[[1.0, -1.0], [-1.3, 0.8]], B=[[-0.9, 1.1], [0.7, -0.8]], eps=0.01):
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
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(A=[[1.0, -1.0], [-1.3, 0.8]], B=[[-0.9, 1.1], [0.7, -0.8]], eps=0.01):
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
    more efficient algorithms would be needed. (10 instances)
    
    ```python
    def sat(strategies: List[List[float]], A=[[0.0, -0.5, 1.0], [0.75, 0.0, -1.0], [-1.0, 0.4, 0.0]], eps=0.01):
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
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(A=[[0.0, -0.5, 1.0], [0.75, 0.0, -1.0], [-1.0, 0.4, 0.0]], eps=0.01):
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

* <a name="anyedge"></a>**AnyEdge** Trivial [graph](https://en.wikipedia.org/w/index.php?title=Graph_(discrete_mathematics)) problem. (10 instances)
    
    ```python
    def sat(e: List[int], edges=[[0, 217], [40, 11], [17, 29], [11, 12], [31, 51]]):
        """Find any edge in edges."""
        return e in edges
    ```
    <details><summary>4,178 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(edges=[[0, 217], [40, 11], [17, 29], [11, 12], [31, 51]]): return edges[0]
    
    ```
    
    ```python
    def sol(edges=[[0, 217], [40, 11], [17, 29], [11, 12], [31, 51]]):
        # The index of an edge among the four edges that appear in the textbook. Note that elements of
        # list or array accesses in assignments in a function should be referenced in the function itself.
        # They should not be nested within other expressions.
        e = [0, 1, 2, 3][0]
        return edges[e]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(edges=[[0, 217], [40, 11], [17, 29], [11, 12], [31, 51]]):
        return edges[0]
    ```
    
    </details>
    
* <a name="shortestpath"></a>**ShortestPath** Shortest Path, see (Dijkstra's algorithm)[https://en.wikipedia.org/w/index.php?title=Dijkstra%27s_algorithm] (10 instances)
    
    ```python
    def sat(path: List[int], weights=[{1: 20, 2: 1}, {2: 2, 3: 5}, {1: 10}], bound=11):
        """
        Find a path from node 0 to node 1, of length at most bound, in the given digraph.
        weights[a][b] is weight on edge [a,b] for (int) nodes a, b
        """
        return path[0] == 0 and path[-1] == 1 and sum(weights[a][b] for a, b in zip(path, path[1:])) <= bound
    ```
    <details><summary>1,241 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(weights=[{1: 20, 2: 1}, {2: 2, 3: 5}, {1: 10}], bound=11):
    	return [0, 2, 1]
    
    ```
    
    ```python
    def sol(weights=[{1: 20, 2: 1}, {2: 2, 3: 5}, {1: 10}], bound=11):
        """
        Return a path from node 0 to node 1, of length at most 11, in the given digraph.
        weights[a][b] is weight on edge [a,b] for (int) nodes a, b
        weights[a][b] must be 1 or 2
        len(weights) must be 3
        """
        if bound < 2:
            return []
        else:
            if weights[0][1] == 1:
                return [0, 1]
            else:
                if bound < 4:
                    return []
                else:
                    return [0, 2, 1]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(weights=[{1: 20, 2: 1}, {2: 2, 3: 5}, {1: 10}], bound=11):  # Dijkstra's algorithm (bound is ignored)
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
    
* <a name="anypath"></a>**AnyPath** Any Path (10 instances)
    
    ```python
    def sat(path: List[int], edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [1, 2]]):
        """ Find any path from node 0 to node n in a given digraph on vertices 0, 1,..., n."""
        for i in range(len(path) - 1):
            assert [path[i], path[i + 1]] in edges
        assert path[0] == 0
        assert path[-1] == max(max(edge) for edge in edges)
        return True
    ```
    <details><summary>536 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [1, 2]]):
        return [0,2,5,6,7]
    
    ```
    
    ```python
    def sol(edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [1, 2]]):
        possible_paths = [[0]]
        for i in range(100):
            paths_to_append = []
            for path in possible_paths:
                for edge in edges:
                    if edge[0] == path[-1]:
                        path_to_append = path + [edge[1]]
                        if edges == [] or (path_to_append[-1] == max(max(edge) for edge in edges)):
                            return path_to_append
                        paths_to_append += [path_to_append]
            possible_paths = possible_paths + paths_to_append
        return False
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [1, 2]]):
        n = max(max(edge) for edge in edges)
        paths = {0: [0]}
        for _ in range(n + 1):
            for i, j in edges:
                if i in paths and j not in paths:
                    paths[j] = paths[i] + [j]
        return paths.get(n)
    ```
    
    </details>
    
* <a name="plantedclique"></a>**PlantedClique** Find a [planted clique](https://en.wikipedia.org/w/index.php?title=Planted_clique) of a given size
    in an undirected graph. Finding a polynomial-time algorithm for this problem has been *unsolved* for
    some time. (10 instances)
    
    ```python
    def sat(nodes: List[int], size=3, edges=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):
        """Find a clique of the given size in the given undirected graph. It is guaranteed that such a clique exists."""
        assert len(nodes) == len(set(nodes)) >= size
        edge_set = {(a, b) for (a, b) in edges}
        for a in nodes:
            for b in nodes:
                assert a == b or (a, b) in edge_set or (b, a) in edge_set
    
        return True
    ```
    <details><summary>514 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(size=3, edges=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):
        return [0, 17, 22]
    
    ```
    
    ```python
    def sol(size=3, edges=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):
        edge_set = {(a, b) for (a, b) in edges}
        done = False
        count = 0
        nodes = set()
        while not done:
            for _, b in edge_set:
                if b not in nodes:
                    nodes.add(b)
                    count += 1
                if count == size:
                    return sorted(list(nodes))
            for a, _ in edge_set:
                if a not in nodes:
                    nodes.add(a)
                    count += 1
                if count == size:
                    return sorted(list(nodes))
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(size=3, edges=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):  # brute force (finds list in increasing order), but with a tiny bit of speedup
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
    
* <a name="anytriangle"></a>**AnyTriangle** Easy [graph](https://en.wikipedia.org/w/index.php?title=Graph_(discrete_mathematics)) problem,
    see [triangle](https://en.wikipedia.org/w/index.php?title=Triangle_graph) (10 instances)
    
    ```python
    def sat(tri: List[int], edges=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):
        """Find any triangle in the given directed graph."""
        a, b, c = tri
        return [a, b] in edges and [b, c] in edges and [c, a] in edges and a != b != c != a
    ```
    <details><summary>236 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(edges=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):
        return [17,22,31]
    
    ```
    
    ```python
    def sol(edges=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):
        """Return any triangle which can be found from the given directed graph."""
        a = 1
        b = 2
        c = 3
        for a1 in range(6):
            a = edges[a1][0]
            for b1 in range(6):
                b = edges[b1][0]
                for c1 in range(6):
                    c = edges[c1][0]
                    if a != b != c != a and [a, b] in edges and [b, c] in edges and [c, a] in edges:
                        return [a, b, c]
        # This should never occur
        return
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(edges=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):
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
    
* <a name="evenpath"></a>**EvenPath**  (10 instances)
    
    ```python
    def sat(path: List[int], edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [1, 2]]):
        """Find a path with an even number of nodes from nodes 0 to n in the given digraph on vertices 0, 1,..., n."""
        assert path[0] == 0 and path[-1] == max(max(e) for e in edges)
        assert all([[a, b] in edges for a, b in zip(path, path[1:])])
        return len(path) % 2 == 0
    ```
    <details><summary>186 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [1, 2]]):
        return [0, 1, 2, 5, 6, 7]
    
    ```
    
    ```python
    def sol(edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [1, 2]]):
        n = max(max(e) for e in edges) + 1
        choices = [[0]*n for _ in range(n)]
        for i, j in edges:
            choices[i][j] = 1
        path = []
        def dfs(start):
            path.append(start)
            for i in range(n):
                if choices[start][i] == 1:
                    choices[start][i] = 0
                    dfs(i)
                    break
        dfs(0)
        return path
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [1, 2]]):
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
    
* <a name="unweightedshortestpath"></a>**UnweightedShortestPath** Unweighted Shortest Path
    
    See (Dijkstra's algorithm)[https://en.wikipedia.org/w/index.php?title=Dijkstra%27s_algorithm] (10 instances)
    
    ```python
    def sat(path: List[int], edges=[[0, 11], [0, 7], [7, 5], [0, 22], [11, 22], [11, 33], [22, 33]], u=0, v=33, bound=3):
        """Find a path from node u to node v, of a bounded length, in the given digraph on vertices 0, 1,..., n."""
        assert path[0] == u and path[-1] == v and all([i, j] in edges for i, j in zip(path, path[1:]))
        return len(path) <= bound
    ```
    <details><summary>141 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(edges=[[0, 11], [0, 7], [7, 5], [0, 22], [11, 22], [11, 33], [22, 33]], u=0, v=33, bound=3):
        return [0,22,33]
    
    ```
    
    ```python
    def sol(edges=[[0, 11], [0, 7], [7, 5], [0, 22], [11, 22], [11, 33], [22, 33]], u=0, v=33, bound=3):
        """Compute the path from node u to node v, of bounded length, in the given digraph on vertices 0, 1,..., n."""
        def dfs(path):
            if len(path) <= bound:
                if path[-1] == v:
                    return path
                for edge in edges:
                    if path[-1] == edge[0]:
                        if edge[1] > path[-1]:
                            newpath = dfs(path + [edge[1]])
                            if newpath != None:
                                return newpath
            return None
        return dfs([u])
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(edges=[[0, 11], [0, 7], [7, 5], [0, 22], [11, 22], [11, 33], [22, 33]], u=0, v=33, bound=3):  # Dijkstra's algorithm
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
    
* <a name="oddpath"></a>**OddPath** To make it even more different than EvenPath, we changed to go from node 0 to node *1*. (10 instances)
    
    ```python
    def sat(p: List[int], edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [6, 1]]):
        """Find a path with an even number of nodes from nodes 0 to 1 in the given digraph on vertices 0, 1,..., n."""
        return p[0] == 0 and p[-1] == 1 == len(p) % 2 and all([[a, b] in edges for a, b in zip(p, p[1:])])
    ```
    <details><summary>45 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [6, 1]]):
        return [0, 2, 5, 6, 1]
    ```
    
    ```python
    def sol(edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [6, 1]]):
        n = len(edges)
        res = [0]
        prefix = [0]
        while len(res) < (n+1)//2:
            prefix.append(next(x for x in range(n) if x not in prefix and all([e[0] in prefix for e in edges if e[1] == x])))
            res.append(prefix[-1])
        return res
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [6, 1]]):
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
    allowed, one can do this by adding N additional nodes for each test. (10 instances)
    
    ```python
    def sat(bi: List[int], g1=[[0, 1], [1, 2], [2, 3], [3, 4], [2, 5]], g2=[[0, 4], [1, 5], [4, 1], [1, 2], [2, 3]]):
        """
        You are given two graphs which are permutations of one another and the goal is to find the permutation.
        Each graph is specified by a list of edges where each edge is a pair of integer vertex numbers.
        """
        return len(bi) == len(set(bi)) and {(i, j) for i, j in g1} == {(bi[i], bi[j]) for i, j in g2}
    ```
    <details><summary>1 AI solution from codex</summary>
    
    ```python
    def sol(g1=[[0, 1], [1, 2], [2, 3], [3, 4], [2, 5]], g2=[[0, 4], [1, 5], [4, 1], [1, 2], [2, 3]]):
        """Brute force the answer."""
        bi = [0] * 6
        for r in range(6**6):
            for i in range(6):
                bi[i] = r % 6
                r //= 6
            if (sat(bi, g1, g2)): return bi
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(g1=[[0, 1], [1, 2], [2, 3], [3, 4], [2, 5]], g2=[[0, 4], [1, 5], [4, 1], [1, 2], [2, 3]]):  # exponentially slow
        from itertools import permutations
        n = max(i for g in [g1, g2] for e in g for i in e) + 1
        g1_set = {(i, j) for i, j in g1}
        for pi in permutations(range(n)):
            if all((pi[i], pi[j]) in g1_set for i, j in g2):
                return list(pi)
        assert False, f"Graphs are not isomorphic {g1}, {g2}"
    ```
    
    </details>
    
* <a name="conway99"></a>**Conway99** Conway's 99-graph problem (*unsolved*, open problem)
    
    Conway's 99-graph problem is an unsolved problem in graph theory.
    In Conway's terminology, from [Five $1,000 Problems (Update 2017)](https://oeis.org/A248380/a248380.pdf)
    "Is there a graph with 99 vertices in which every edge (i.e. pair of joined vertices) belongs to a unique
    triangle and every nonedge (pair of unjoined vertices) to a unique quadrilateral?"
    
    See also this [Wikipedia article](https://en.wikipedia.org/w/index.php?title=Conway%27s_99-graph_problem). (1 instance)
    
    ```python
    def sat(edges: List[List[int]]):
        """
        Find an undirected graph with 99 vertices, in which each two adjacent vertices have exactly one common
        neighbor, and in which each two non-adjacent vertices have exactly two common neighbors.
        """
        # first compute neighbors sets, N:
        N = {i: {j for j in range(99) if j != i and ([i, j] in edges or [j, i] in edges)} for i in range(99)}
        return all(len(N[i].intersection(N[j])) == (1 if j in N[i] else 2) for i in range(99) for j in range(i))
    ```
    0 AI solutions from codex0 hand-written solutions 
    
* <a name="zarankiewicz"></a>**Zarankiewicz** [Zarankiewicz problem](https://en.wikipedia.org/wiki/Zarankiewicz_problem) (3 instances)
    
    ```python
    def sat(edges: List[List[int]], z=20, n=5, t=3):
        """Find a bipartite graph with n vertices on each side, z edges, and no K_3,3 subgraph."""
        from itertools import combinations
        edges = {(a, b) for a, b in edges if a in range(n) and b in range(n)}  # convert to a set for efficiency
        assert len(edges) >= z
    
        return all(
            any((a, b) not in edges for a in left for b in right)
            for left in combinations(range(n), t)
            for right in combinations(range(n), t)
        )
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(z=20, n=5, t=3):
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
    
* <a name="shortintegerpath"></a>**ShortIntegerPath** This is a more interesting version of Study_20 with an additional length constraint. One can think of the graph
    defined by the integer pairs. (1 instance)
    
    ```python
    def sat(li: List[int]):
        """
        Find a list of nine integers, starting with 0 and ending with 128, such that each integer either differs from
        the previous one by one or is thrice the previous one.
        """
        return all(j in {i - 1, i + 1, 3 * i} for i, j in zip([0] + li, li + [128])) and len(li) == 9
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return [1, 3, 4, 12, 13, 14, 42, 126, 127]
    ```
    
    </details>
    
## ICPC


Problems inspired by the [International Collegiate Programming Contest](https://icpc.global) (ICPC).


* <a name="matchingmarkers"></a>**MatchingMarkers** Inspired by
    [ICPC 2019 Problem D: Circular DNA](https://icpc.global/worldfinals/problems/2019%20ACM-ICPC%20World%20Finals/icpc2019.pdf)
    
    This is trivial in quadratic time, but the challenge is to solve it quickly (i.e., linear time). (10 instances)
    
    ```python
    def sat(cut_position: int, ring="yRrsmOkLCHSDJywpVDEDsjgCwSUmtvHMefxxPFdmBIpM", lower=5):
        """
        The input is a string of start and end markers "aaBAcGeg" where upper-case characters indicate start markers
        and lower-case characters indicate ending markers. The string indicates a ring (joined at the ends) and the goal is
        to find a location to split the ring so that there are a maximal number of matched start/end chars where a character
        (like "a"/"A") is matched if starting at the split and going around the ring, the start-end pairs form a valid
        nesting like nested parentheses. Can you solve it in linear time?
        """
        line = ring[cut_position:] + ring[:cut_position]
        matches = {c: 0 for c in line.lower()}
        for c in line:
            if c.islower():
                matches[c] -= (1 if matches[c] > 0 else len(line))
            else:
                matches[c.lower()] += 1
        return sum(i == 0 for i in matches.values()) >= lower
    ```
    <details><summary>3,420 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(ring="yRrsmOkLCHSDJywpVDEDsjgCwSUmtvHMefxxPFdmBIpM", lower=5):
        return 6
    
    ```
    
    ```python
    def sol(ring="yRrsmOkLCHSDJywpVDEDsjgCwSUmtvHMefxxPFdmBIpM", lower=5):
        """
        The input is a string of start and end markers "aaBAcGeg" where upper-case characters indicate start markers
        and lower-case characters indicate ending markers. The string indicates a ring (joined at the ends) and the goal is
        to find a location to split the ring so that there are a maximal number of matched start/end chars where a character
        (like "a"/"A") is matched if starting at the split and going around the ring, the start-end pairs form a valid
        nesting like nested parentheses. Can you solve it in linear time?
        """
        return ring.index(max(ring, key=ring.count))
    
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(ring="yRrsmOkLCHSDJywpVDEDsjgCwSUmtvHMefxxPFdmBIpM", lower=5):
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
    
* <a name="bipermutations"></a>**BiPermutations** Inspired by
    [ICPC 2019 Problem A: Azulejos](https://icpc.global/worldfinals/problems/2019%20ACM-ICPC%20World%20Finals/icpc2019.pdf)
    which is 2,287 characters. (10 instances)
    
    ```python
    def sat(perms: List[List[int]], prices0=[7, 7, 9, 5, 3, 7, 1, 2], prices1=[5, 5, 5, 4, 2, 5, 1, 1], heights0=[2, 4, 9, 3, 8, 5, 5, 4], heights1=[1, 3, 8, 1, 5, 4, 4, 2]):
        """
        There are two rows of objects. Given the length-n integer arrays of prices and heights of objects in each
        row, find a permutation of both rows so that the permuted prices are non-decreasing in each row and
        so that the first row is taller than the second row.
        """
        n = len(prices0)
        perm0, perm1 = perms
        assert sorted(perm0) == sorted(perm1) == list(range(n)), "Solution must be two permutations"
        for i in range(n - 1):
            assert prices0[perm0[i]] <= prices0[perm0[i + 1]], "Permuted prices must be nondecreasing (row 0)"
            assert prices1[perm1[i]] <= prices1[perm1[i + 1]], "Permuted prices must be nondecreasing (row 1)"
        return all(heights0[i] > heights1[j] for i, j in zip(perm0, perm1))
    ```
    <details><summary>1 AI solution from codex</summary>
    
    ```python
    def sol(prices0=[7, 7, 9, 5, 3, 7, 1, 2], prices1=[5, 5, 5, 4, 2, 5, 1, 1], heights0=[2, 4, 9, 3, 8, 5, 5, 4], heights1=[1, 3, 8, 1, 5, 4, 4, 2]):
        n = len(prices0)
        ans0 = sorted(range(n), key=(lambda i : prices0[i]))
        ans1 = sorted(range(n), key=(lambda i : prices0[i]))
        return [ans0, ans1]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(prices0=[7, 7, 9, 5, 3, 7, 1, 2], prices1=[5, 5, 5, 4, 2, 5, 1, 1], heights0=[2, 4, 9, 3, 8, 5, 5, 4], heights1=[1, 3, 8, 1, 5, 4, 4, 2]):
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
    which is 3,003 characters. (10 instances)
    
    ```python
    def sat(indices: List[int], H=60, alpha=18, beta=2, xs=[0, 10, 20, 30, 50, 80, 100, 120, 160, 190, 200], ys=[0, 30, 10, 30, 50, 40, 10, 20, 20, 55, 10], thresh=26020):
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
        assert sorted({0, len(xs) - 1, *indices}) == indices, f"Ans. should be sorted list [0, ..., {len(xs) - 1}]"
        cost = alpha * (H - ys[0])
        for i, j in zip(indices, indices[1:]):
            a, b, r = xs[i], xs[j], (xs[j] - xs[i]) / 2
            assert max(ys[i], ys[j]) + r <= H, "Bridge too tall"
            assert all(ys[k] <= H - r + ((b - xs[k]) * (xs[k] - a)) ** 0.5 for k in range(i + 1, j)), \
                "Bridge too short"
            cost += alpha * (H - ys[j]) + beta * (b - a) ** 2
        return cost <= thresh
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(H=60, alpha=18, beta=2, xs=[0, 10, 20, 30, 50, 80, 100, 120, 160, 190, 200], ys=[0, 30, 10, 30, 50, 40, 10, 20, 20, 55, 10], thresh=26020):  # thresh is ignored
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
    
    Nobody solved this problem during the competition -- it is pretty difficult! (10 instances)
    
    ```python
    def sat(position: List[List[int]], transcript=[[[3, 3], [5, 5], [3, 7]], [[5, 3], [6, 4]]]):
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
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(transcript=[[[3, 3], [5, 5], [3, 7]], [[5, 3], [6, 4]]]):
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
    
## IMO

Problems inspired by the
[International Mathematical Olympiad](https://en.wikipedia.org/wiki/International_Mathematical_Olympiad)
[problems](https://www.imo-official.org/problems.aspx)

* <a name="findrepeats"></a>**FindRepeats** Note: This problem is much easier than the IMO problem which also required a proof that it is impossible
    for a_0 not divisible by 3.
    
    Inspired by [IMO 2017 Problem 1](https://www.imo-official.org/problems.aspx) (10 instances)
    
    ```python
    def sat(indices: List[int], a0=123):
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
        assert a0 >= 0 and a0 % 3 == 0, "Hint: a_0 is a multiple of 3."
        s = [a0]
        for i in range(max(indices)):
            s.append(int(s[-1] ** 0.5) if int(s[-1] ** 0.5) ** 2 == s[-1] else s[-1] + 3)
        return len(indices) == len(set(indices)) == 1000 and min(indices) >= 0 and len({s[i] for i in indices}) == 1
    ```
    <details><summary>51 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a0=123):
        return [a0+3*i for i in range(1000)]
    
    ```
    
    ```python
    def sol(a0=123):
        # If a0 = 123 then there is a floor of 1 but a ceiling of 29991 since the numbers greater than a0 > a0+3
        a0 += 3
        a0 = a0 if a0 % 3 == 0 else a0 + 3
        s = []
        for i in range(1000):
            a = a0 + 3 * i
            a0 = a if a % 3 == 0 else a + 3
            s.append(int(a0 ** 0.5) if int(a0 ** 0.5) ** 2 == a0 else a0 + 3)
            a0 = a0 + 3
        return s
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a0=123):
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
    
* <a name="exponentialcoinmoves"></a>**ExponentialCoinMoves** This problem has *long* answers, not that the code to solve it is long but that what the solution outputs is long.
    
    The version below uses only 5 boxes (unlike the IMO problem with 6 boxes since 2010^2010^2010 is too big
    for computers) but the solution is quite similar to the solution to the IMO problem. Because the solution
    requires exponential many moves, our representation allows combining multiple Type-1 (advance) operations
    into a single step.
    
    Inspired by [IMO 2010 Problem 5](https://www.imo-official.org/problems.aspx) (9 instances)
    
    ```python
    def sat(states: List[List[int]], n=16385):
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
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=16385):
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
    theorem? (10 instances)
    
    ```python
    def sat(nums: List[int], b=7, m=6):
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
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(b=7, m=6):
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
    
* <a name="picknearneighbors"></a>**PickNearNeighbors** Inspired by [IMO 2017 Problem 5](https://www.imo-official.org/problems.aspx)
    
    The puzzle solution follows the judge's proof closely. (10 instances)
    
    ```python
    def sat(keep: List[bool], heights=[10, 2, 14, 1, 8, 19, 16, 6, 12, 3, 17, 0, 9, 18, 5, 7, 11, 13, 15, 4]):
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
        n = int(len(heights) ** 0.5)
        assert sorted(heights) == list(range(n * n + n)), "hint: heights is a permutation of range(n * n + n)"
        kept = [i for i, k in zip(heights, keep) if k]
        assert len(kept) == 2 * n, "must keep 2n items"
        pi = sorted(range(2 * n), key=lambda i: kept[i])  # the sort indices
        return all(abs(pi[2 * i] - pi[2 * i + 1]) == 1 for i in range(n))
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(heights=[10, 2, 14, 1, 8, 19, 16, 6, 12, 3, 17, 0, 9, 18, 5, 7, 11, 13, 15, 4]):  # Based on the judge's solution.
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
    
    Inspired by [IMO 2010 Problem 5](https://www.imo-official.org/problems.aspx) (9 instances)
    
    ```python
    def sat(li: List[int], n=18):
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
        assert n % 3 == 0, "Hint: n is a multiple of 3"
        return len(li) == n and all(li[(i + 2) % n] == 1 + li[(i + 1) % n] * li[i] for i in range(n))
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=18):
        return [-1, -1, 2] * (n // 3)
    ```
    
    </details>
    
* <a name="halftag"></a>**HalfTag** Inspired by [IMO 2020 Problem 3](https://www.imo-official.org/problems.aspx) (10 instances)
    
    ```python
    def sat(li: List[int], tags=[3, 0, 3, 2, 0, 1, 0, 3, 1, 1, 2, 2, 0, 2, 1, 3]):
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
        n = max(tags) + 1
        assert sorted(tags) == sorted(list(range(n)) * 4), "hint: each tag occurs exactly four times"
        assert len(li) == len(set(li)) and min(li) >= 0
        return sum(li) * 2 == sum(range(4 * n)) and sorted([tags[i] for i in li]) == [i // 2 for i in range(2 * n)]
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(tags=[3, 0, 3, 2, 0, 1, 0, 3, 1, 1, 2, 2, 0, 2, 1, 3]):
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
    
    The vectors are encoded as binary integers for succinctness. (10 instances)
    
    ```python
    def sat(inds: List[int], vecs=[169, 203, 409, 50, 37, 479, 370, 133, 53, 159, 161, 367, 474, 107, 82, 447, 385]):
        """
        Parity learning: Given binary vectors in a subspace, find the secret set $S$ of indices such that:
        $$sum_{i \in S} x_i = 1 (mod 2)$$
        """
        return all(sum((v >> i) & 1 for i in inds) % 2 == 1 for v in vecs)
    ```
    <details><summary>5 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(vecs=[169, 203, 409, 50, 37, 479, 370, 133, 53, 159, 161, 367, 474, 107, 82, 447, 385]):
        parity = len(vecs)
        for b in range((1<<parity)-1, 0, -1):
            inds = [i for i in range(parity) if (b>>i)&1]
            if sat(inds, vecs):
                return inds
    
    ```
    
    ```python
    def sol(vecs=[169, 203, 409, 50, 37, 479, 370, 133, 53, 159, 161, 367, 474, 107, 82, 447, 385]):
        # Try all power-of-two sized bitvectors
        for k in range(0, 1024):
            # Try all possible indices based on Hamming weight
            for H in range(0, 10):
                inds = [i for i in range(0, 10) if (k >> i) & 1]
                if sat(inds, vecs):
                    return inds
        return []
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(vecs=[169, 203, 409, 50, 37, 479, 370, 133, 53, 159, 161, 367, 474, 107, 82, 447, 385]):  # Gaussian elimination
        d = 0 # decode vectors into arrays
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
    runs in time $2^(d/(log d))$ (10 instances)
    
    ```python
    def sat(inds: List[int], vecs=[26, 5, 32, 3, 15, 18, 31, 13, 24, 25, 34, 5, 15, 24, 16, 13, 0, 27, 37]):
        """
        Learning parity with noise: Given binary vectors, find the secret set $S$ of indices such that, for at least
        3/4 of the vectors, $$sum_{i \in S} x_i = 1 (mod 2)$$
        """
        return sum(sum((v >> i) & 1 for i in inds) % 2 for v in vecs) >= len(vecs) * 3 / 4
    ```
    <details><summary>4 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(vecs=[26, 5, 32, 3, 15, 18, 31, 13, 24, 25, 34, 5, 15, 24, 16, 13, 0, 27, 37]):
        return [i for i, v in enumerate(vecs, 1) if v % 2 == 1]
    
    ```
    
    ```python
    def sol(vecs=[26, 5, 32, 3, 15, 18, 31, 13, 24, 25, 34, 5, 15, 24, 16, 13, 0, 27, 37]):
        best = -1
        S    = []
        for i in range(8):
            for j in range(i+1, 8):
                for k in range(j+1, 8):
                    inds = [i, j, k]
                    if sat(inds, vecs):
                        out = sum((1 << i for i in inds))
                        if out > best:
                            best = out
                            S    = inds
        return sorted(S)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(vecs=[26, 5, 32, 3, 15, 18, 31, 13, 24, 25, 34, 5, 15, 24, 16, 13, 0, 27, 37]):  # brute force
        d = 0 # decode vectors into arrays
        m = max(vecs)
        while m:
            m >>= 1
            d += 1
        vecs = [[(n >> i) & 1 for i in range(d)] for n in vecs]
    
        import random
        rand = random.Random(0)
        target = (len(vecs) * 3) // 4
        while True:
            ans = [i for i in range(d) if rand.randrange(2)]
            if sum(sum(v[i] for i in ans) % 2 for v in vecs) >= len(vecs) * 3 / 4:
                return ans
    ```
    
    </details>
    
## number_theory

Number theory problems

* <a name="gcd"></a>**GCD** [Greatest Common Divisor](https://en.wikipedia.org/w/index.php?title=Greatest_common_divisor&oldid=990943381)
    (GCD)
    
    See also the [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm) (10 instances)
    
    ```python
    def sat(n: int, a=15482, b=23223, lower_bound=5):
        """Find a large common divisor of two integers."""
        return a % n == 0 and b % n == 0 and n >= lower_bound
    ```
    <details><summary>4,597 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=15482, b=23223, lower_bound=5):
        return a // 2
    
    ```
    
    ```python
    def sol(a=15482, b=23223, lower_bound=5):
        n = 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        while not sat(n, a, b, lower_bound):
            n += 1
        return n
    
    ```
    
    </details>
    
    <details><summary>2 hand-written solutions </summary>
    
    ```python
    def sol(a=15482, b=23223, lower_bound=5):
        m, n = min(a, b), max(a, b)
        while m > 0:
            m, n = n % m, m
        return n
    ```
    
    ```python
    def sol(a=15482, b=23223, lower_bound=5):
        def gcd(m, n):
            if m > n:
                return gcd(n, m)
            if m == 0:
                return n
            return gcd(n % m, m)
    
        return gcd(a, b)
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
    [Weber D., Denny T. (1998) "The solution of McCurley's discrete log challenge."](https://link.springer.com/content/pdf/10.1007/BFb0055747.pdf) (10 instances)
    
    ```python
    def sat(n: int, g=44337, p=69337, t=38187):
        """Find n such that g^n is congruent to t mod n"""
        return pow(g, n, p) == t
    ```
    <details><summary>3,979 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(sol=44337, p=69337, t=38187):
        return next(n for n in range(p) if sat(n))
    
    ```
    
    ```python
    def sol(sol=44337, p=69337, t=38187):
        """Find smallest natural n such that g^n is congruent to t mod n. 
        This is required in f6.
        """
        return next(x for x in range(2, p-1) if pow(sol, x, p) == t)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(g=44337, p=69337, t=38187):
        for n in range(p):
            if pow(g, n, p) == t:
                return n
        assert False, f"unsolvable discrete log problem g={g}, t={t}, p={p}"
    ```
    
    </details>
    
* <a name="gcd_multi"></a>**GCD_multi** [Greatest Common Divisor](https://en.wikipedia.org/w/index.php?title=Greatest_common_divisor&oldid=990943381)
    (GCD)
    
    See also the [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm) (10 instances)
    
    ```python
    def sat(n: int, nums=[77410, 23223, 54187], lower_bound=2):
        """Find a large common divisor of the list of integers."""
        return all(i % n == 0 for i in nums) and n >= lower_bound
    ```
    <details><summary>2,332 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[77410, 23223, 54187], lower_bound=2):
        return max(nums) // 10
    
    ```
    
    ```python
    def sol(nums=[77410, 23223, 54187], lower_bound=2):
        for i in range(lower_bound, max(nums) + 1):
            if sat(i, nums, lower_bound):
                return i
        return None
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[77410, 23223, 54187], lower_bound=2):
        n = 0
        for i in nums:
            m, n = min(i, n), max(i, n)
            while m > 0:
                m, n = n % m, m
        return n
    ```
    
    </details>
    
* <a name="lcm_multi"></a>**LCM_multi** [Least Common Multiple](https://en.wikipedia.org/wiki/Least_common_multiple)
    (LCM)
    
    See also the [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm) (10 instances)
    
    ```python
    def sat(n: int, nums=[15, 27, 102], upper_bound=5000):
        """Find a small common multiple of a list of integers."""
        return all(n % i == 0 for i in nums) and 0 < n <= upper_bound
    ```
    <details><summary>1,921 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(nums=[15, 27, 102], upper_bound=5000):
        return [n for n in range(upper_bound) if sat(n)][0]
    
    ```
    
    ```python
    def sol(nums=[15, 27, 102], upper_bound=5000):
        """Generate a small common multiple of a list of integers."""
        break_next_try = False
        i = 1
        
        while True:
            i += 1
            for k in range(upper_bound // (1 + max(nums))):
                for j in range(0, len(nums)):
                    if i % nums[j] != 0:
                        break_next_try = True
                        break
                    elif len(nums) - 1 > j:
                        continue
                    else:
                        return i
                if break_next_try:
                    break_next_try = False
                    break
       
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(nums=[15, 27, 102], upper_bound=5000):
        ans = 1
        for i in nums:
            m, n = min(i, ans), max(i, ans)
            while m > 0:
                m, n = n % m, m
            ans *= (i // n)
        return ans
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
    unless considerable advances are made in integer factorization or computational power in the near future. (10 instances)
    
    ```python
    def sat(i: int, n=241864633):
        """Find a non-trivial factor of integer n"""
        return 1 < i < n and n % i == 0
    ```
    <details><summary>1,830 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=241864633):
    	for i in range(2, n):
    		if sat(i): return i
    
    ```
    
    ```python
    def sol(n=241864633):
        """Find one non-trivial factor of integer n or return n if no such factor exists"""
        i = 3 # start testing at 3 to avoid 2 which is the only even prime
        while i < n:
            if sat(i):  # if i is a factor then stop testing
                break
            else: # increment i
                i += 2 # test only odd numbers. if i is a factor then stop testing
        if i == n: # if i achieves n then stop testing and return n itself as a factor
            return n
        return i # return a factor
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=241864633):
        if n % 2 == 0:
            return 2
    
        for i in range(3, int(n ** 0.5) + 1, 2):
            if n % i == 0:
                return i
    
        assert False, "problem defined for composite n only"
    ```
    
    </details>
    
* <a name="smallexponentbigsolution"></a>**SmallExponentBigSolution** Small exponent, big solution
    
    Problems have small b and target but solution is typically a large n.
    Some of them are really hard, for example, for `b=2, target=3`, the smallest solution is `n=4700063497`
    
    See [Richard K. Guy "The strong law of small numbers", (problem 13)](https://doi.org/10.2307/2322249) (9 instances)
    
    ```python
    def sat(n: int, b=2, target=5):
        """Solve for n: b^n = target (mod n)"""
        return (b ** n) % n == target
    ```
    <details><summary>1,660 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(b=2, target=5):
        n = 2
        while not(sat(n)): n += 1
        return n
    
    ```
    
    ```python
    def sol(b=2, target=5):
        m = int(0.5 * target ** 0.5) + 2
        test = b ** m % m
        while test != target: m += 1; test = b ** m % m
        return m                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(b=2, target=5):
        for n in range(1, 10 ** 5):
            if pow(b, n, n) == target:
                return n
    ```
    
    </details>
    
* <a name="lcm"></a>**LCM** [Least Common Multiple](https://en.wikipedia.org/wiki/Least_common_multiple)
    (LCM)
    
    See also the [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm) (10 instances)
    
    ```python
    def sat(n: int, a=15, b=27, upper_bound=150):
        """Find a small common multiple of two integers."""
        return n % a == 0 and n % b == 0 and 0 < n <= upper_bound
    ```
    <details><summary>1,524 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=15, b=27, upper_bound=150):
        return int(upper_bound//b) * b
    
    ```
    
    ```python
    def sol(a=15, b=27, upper_bound=150):
        """Find a small common multiple of two integers."""
        best = 1
        for i in range(2, upper_bound):
            if i % a != 0 and i % b != 0:
                continue
            if i % a != 0:
                if i % b != 0:
                    continue
                target = i
                offset = b
            else:
                if i % b != 0:
                    continue
                target = i
                offset = a
            for i in range(target, upper_bound, target):
                if i % offset == 0:
                    best = max(best, i)
        return best
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=15, b=27, upper_bound=150):
        m, n = min(a, b), max(a, b)
        while m > 0:
            m, n = n % m, m
        return a * (b // n)
    ```
    
    </details>
    
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
    
    See [this webpage](http://www.ericr.nl/wondrous/delrecs.html) for up-to-date records. (9 instances)
    
    ```python
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
    ```
    <details><summary>921 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(t=197, upper=20):
        i = 0
        while not sat(i, t, upper): i += 1
        return i
    
    ```
    
    ```python
    def sol(t=197, upper=20):
        """
        Consider the following process. Start with an integer `n` and repeatedly applying the operation:
        * if n is even, divide n by 2,
        * if n is odd, multiply n by 3 and add 1
        Find `0 < n < upper` so that it takes exactly `t` steps to reach 1.
        (This is effectively what `f6` is looking for, but this approach is more predictable)
        """
        for n in range(2, 2 ** upper):
            if sat(n, t, upper):
                return n
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(t=197, upper=20):  # Faster solution for simultaneously solving multiple problems is of course possible
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
    
* <a name="foursquares"></a>**FourSquares** Sum of four squares
    
    [Lagrange's Four Square Theorem](https://en.wikipedia.org/w/index.php?title=Lagrange%27s_four-square_theorem)
    
    Given a non-negative integer `n`, a classic theorem of Lagrange says that `n` can be written as the sum of four
    integers. The problem here is to find them. This is a nice problem and we give an elementary solution
    that runs in time 	ilde{O}(n),
    which is not "polynomial time" because it is not polynomial in log(n), the length of n. A poly-log(n)
    algorithm using quaternions is described in the book:
    ["Randomized algorithms in number theory" by Michael O. Rabin and Jeffery O. Shallit (1986)](https://doi.org/10.1002/cpa.3160390713)
    
    The first half of the problems involve small numbers and the second half involve some numbers up to 50 digits. (10 instances)
    
    ```python
    def sat(nums: List[int], n=12345):
        """Find four integers whose squares sum to n"""
        return len(nums) <= 4 and sum(i ** 2 for i in nums) == n
    ```
    <details><summary>67 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=12345):
        nums = []
        while len(nums) < 4:
            nums.append(int((n - sum(i**2 for i in nums)) ** 0.5))
        return nums
    
    ```
    
    ```python
    def sol(n=12345):
        i = 1
        while True:
            j = i
            while True:
                k = j
                while True:
                    l = k
                    while True:
                        if i ** 2 + j ** 2 + k ** 2 + l ** 2 == n:
                            return [i, j, k, l]
                        l += 1
                        if l ** 2 > n:
                            break
                    k += 1
                    if k ** 2 > n:
                        break
                j += 1
                if j ** 2 > n:
                    break
            i += 1
            if i ** 2 > n:
                break
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=12345):
        m = n
        squares = {i ** 2: i for i in range(int(m ** 0.5) + 2) if i ** 2 <= m}
        sums_of_squares = {i + j: [a, b] for i, a in squares.items() for j, b in squares.items()}
        for s in sums_of_squares:
            if m - s in sums_of_squares:
                return sums_of_squares[m - s] + sums_of_squares[s]
        assert False, "Should never reach here"
    ```
    
    </details>
    
* <a name="znam"></a>**Znam** [Znam's Problem](https://en.wikipedia.org/wiki/Zn%C3%A1m%27s_problem)
    
    For example [2, 3, 7, 47, 395] is a solution for k=5 (9 instances)
    
    ```python
    def sat(li: List[int], k=5):
        """Find k positive integers such that each integer divides (the product of the rest plus 1)."""
        def prod(nums):
            ans = 1
            for i in nums:
                ans *= i
            return ans
    
        return min(li) > 1 and len(li) == k and all((1 + prod(li[:i] + li[i + 1:])) % li[i] == 0 for i in range(k))
    ```
    <details><summary>2 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(k=5):
        ans = []
        for i in range(k):
            n = 1
            while True:
                n += 1
                if all(n % j == 1 for j in ans) and (n not in ans):
                    ans.append(n)
                    break
        return ans
        
    ```
    
    ```python
    def sol(k=5):
        acc = 1
        li = []
        while len(li) < k:
            num = acc + 1
            while True:
                if all((num % i) == 1 for i in li):
                    li.append(num)
                    break
                num += 1
            acc += 1
        return li
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(k=5):
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
    
* <a name="fermatslasttheorem"></a>**FermatsLastTheorem** [Fermat's last theorem](https://en.wikipedia.org/w/index.php?title=Fermat%27s_Last_Theorem)
    
    Supposedly unsolvable, but how confident are really in the super-complicated proof?
    
    See [Wiles, Andrew. "Modular elliptic curves and Fermat's last theorem." Annals of mathematics 141.3 (1995): 443-551.](https://www.jstor.org/stable/2118559) (1 instance)
    
    ```python
    def sat(nums: List[int]):
        """Find integers a,b,c > 0, n > 2, such such that a^n + b^n == c^n"""
        a, b, c, n = nums
        return (a ** n + b ** n == c ** n) and min(a, b, c) > 0 and n > 2
    ```
    0 AI solutions from codex0 hand-written solutions 
    
* <a name="threecubes"></a>**ThreeCubes** Sum of three cubes
    
    Given `n`, find integers `a`, `b`, `c` such that `a**3 + b**3 + c**3 = n`. This is unsolvable for `n % 9 in {4, 5}`.
    Conjectured to be true for all other n, i.e., `n % 9 not in {4, 5}`.
    `a`, `b`, `c` may be positive or negative
    
    See [wikipedia entry](https://en.wikipedia.org/wiki/Sums_of_three_cubes) or
    [Andrew R. Booker, Andrew V. Sutherland (2020). "On a question of Mordell."](https://arxiv.org/abs/2007.01209) (10 instances)
    
    ```python
    def sat(nums: List[int], target=983):
        """Given n, find integers a, b, c such that a^3 + b^3 + c^3 = n."""
        assert target % 9 not in [4, 5], "Hint"
        return len(nums) == 3 and sum([i ** 3 for i in nums]) == target
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(target=983):
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
    
* <a name="gcd17"></a>**GCD17** According to [this article](https://primes.utm.edu/glossary/page.php?sort=LawOfSmall), the smallest
    solution is 8424432925592889329288197322308900672459420460792433 (1 instance)
    
    ```python
    def sat(n: int):
        """Find n for which gcd(n^17+9, (n+1)^17+9) != 1"""
        i = n ** 17 + 9
        j = (n + 1) ** 17 + 9
    
        while i != 0:  # compute gcd using Euclid's algorithm
            (i, j) = (j % i, i)
    
        return n >= 0 and j != 1
    ```
    0 AI solutions from codex0 hand-written solutions 
    
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
    ```
    0 AI solutions from codex0 hand-written solutions 
    
* <a name="collatzgeneralizedunsolved"></a>**CollatzGeneralizedUnsolved** Generalized Collatz Conjecture
    
    This version, permits negative n and seek a cycle with a number of magnitude greater than 1000,
    which would disprove the Generalized conjecture that states that the only cycles are the known 5 cycles
    (which don't have positive integers).
    
    See the [Wikipedia article](https://en.wikipedia.org/wiki/Collatz_conjecture) (1 instance)
    
    ```python
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
    ```
    0 AI solutions from codex0 hand-written solutions 
    
* <a name="lehmer"></a>**Lehmer** Lehmer puzzle
    
    According to [The Strong Law of Large Numbers](https://doi.org/10.2307/2322249) Richard K. Guy states that
        D. H. & Emma Lehmer discovered that 2^n = 3 (mod n) for n = 4700063497,
        but for no smaller n > 1 (1 instance)
    
    ```python
    def sat(n: int):
        """Find n  such that 2^n mod n = 3"""
        return pow(2, n, n) == 3
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return 4700063497
    ```
    
    </details>
    
## probability

Probability problems

* <a name="birthdayparadox"></a>**BirthdayParadox** Adaptation of the classic
    [Birthday Problem](https://en.wikipedia.org/wiki/Birthday_problem (Mathematical Problems category)).
    
    The year length is year_len (365 is earth, while Neptune year is 60,182). (9 instances)
    
    ```python
    def sat(n: int, year_len=365):
        """Find n such that the probability of two people having the same birthday in a group of n is near 1/2."""
        prob = 1.0
        for i in range(n):
            prob *= (year_len - i) / year_len
        return (prob - 0.5) ** 2 <= 1/year_len
    ```
    <details><summary>1,260 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(year_len=365):
        return 23
    
    ```
    
    ```python
    def sol(year_len=365):
        """Returns the smallest number of people for which the probability of having at least two people with the same birthday is
        at least 0.5 given a specific year length."""
        if year_len < 29:
            raise ValueError("Year length must be at least 29 for this problem to be well defined.")
        elif year_len == 29:
            return 23
        else:
            n = 1
            while not sat(n, year_len):
                if n == year_len:
                    return year_len
                n += 1
            return n
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(year_len=365):
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
    
* <a name="exponentialprobability"></a>**ExponentialProbability** See [Exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution) (10 instances)
    
    ```python
    def sat(p_stop: float, steps=10, target_prob=0.5):
        """
        Find p_stop so that the probability of stopping in steps or fewer time steps is the given target_prob if you
        stop each step with probability p_stop
        """
        prob = sum(p_stop*(1-p_stop)**t for t in range(steps))
        return abs(prob - target_prob) < 1e-6
    ```
    <details><summary>589 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(steps=10, target_prob=0.5):
        return 1-0.5**(1/steps)
    
    ```
    
    ```python
    def sol(steps=10, target_prob=0.5):
        """
        Find p_stop so that the probability of stopping in steps or fewer time steps is the given target_prob if you
        stop each step with probability p_stop
        """
        assert steps > 0 and target_prob > 0 and target_prob < 1 and (steps == 1 or (1-target_prob) >= target_prob)
        p_stop = 0
        while not sat(p_stop, steps, target_prob):
            p_stop += 1e-6
        return p_stop
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(steps=10, target_prob=0.5):
        return 1 - (1 - target_prob) ** (1.0/steps)
    ```
    
    </details>
    
* <a name="ballotproblem"></a>**BallotProblem** See the [Wikipedia article](https://en.wikipedia.org/wiki/Bertrand%27s_ballot_theorem) or
    or  [Addario-Berry L., Reed B.A. (2008) Ballot Theorems, Old and New. In: Gyori E., Katona G.O.H., Lovász L.,
    Sági G. (eds) Horizons of Combinatorics. Bolyai Society Mathematical Studies, vol 17.
    Springer, Berlin, Heidelberg.](https://doi.org/10.1007/978-3-540-77200-2_1) (10 instances)
    
    ```python
    def sat(counts: List[int], target_prob=0.5):
        """
        Suppose a list of m 1's and n -1's are permuted at random.
        What is the probability that all of the cumulative sums are positive?
        The goal is to find counts = [m, n] that make the probability of the ballot problem close to target_prob.
        """
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
    ```
    <details><summary>327 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(target_prob=0.5):
        return [9, 3]
    
    ```
    
    ```python
    def sol(target_prob=0.5):
        # Compute by brute force - should work for up to ~1000 items
        last = None
        for m in range(2, 100):
            for n in range(1, min(m, 50)):
                if sat([m, n], target_prob):
                    return [m, n]
                last = (m, n)
        assert last is not None
        # Start with last try and start cycling around incremented n's until we reach close enough
        m, n = last
        while not sat([m, n], target_prob):
            n += 1
            if n > m:
                n = 1
               
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(target_prob=0.5):
        for m in range(1, 10000):
            n = round(m * (1 - target_prob) / (1 + target_prob))
            if abs(target_prob - (m - n) / (m + n)) < 1e-6:
                return [m, n]
    ```
    
    </details>
    
* <a name="birthdayparadoxmontecarlo"></a>**BirthdayParadoxMonteCarlo** A slower, Monte Carlo version of the above Birthday Paradox problem. (9 instances)
    
    ```python
    def sat(n: int, year_len=365):
        """Find n such that the probability of two people having the same birthday in a group of n is near 1/2."""
        import random
        random.seed(0)
        K = 1000  # number of samples
        prob = sum(len({random.randrange(year_len) for i in range(n)}) < n for j in range(K)) / K
        return (prob - 0.5) ** 2 <= year_len
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(year_len=365):
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
    
* <a name="binomialprobabilities"></a>**BinomialProbabilities** See [Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution) (10 instances)
    
    ```python
    def sat(counts: List[int], p=0.5, target_prob=0.0625):
        """Find counts = [a, b] so that the probability of  a H's and b T's among a + b coin flips is ~ target_prob."""
        from itertools import product
        a, b = counts
        n = a + b
        prob = (p ** a) * ((1-p) ** b)
        tot = sum([prob for sample in product([0, 1], repeat=n) if sum(sample) == a])
        return abs(tot - target_prob) < 1e-6
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(p=0.5, target_prob=0.0625):
        probs = [1.0]
        q = 1 - p
        while len(probs) < 20:
            probs = [(p * a + q * b) for a, b in zip([0] + probs, probs + [0])]
            answers = [i for i, p in enumerate(probs) if abs(p - target_prob) < 1e-6]
            if answers:
                return [answers[0], len(probs) - 1 - answers[0]]
    ```
    
    </details>
    
## trivial_inverse

Trivial problems. Typically for any function, you can construct a trivial example.
For instance, for the len function you can ask for a string of len(s)==100 etc.


* <a name="strat"></a>**StrAt**  (10 instances)
    
    ```python
    def sat(i: int, s="cat", target="a"):
        """Find the index of target in string s"""
        return s[i] == target
    ```
    <details><summary>9,363 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="cat", target="a"):
        return 1
    
    ```
    
    ```python
    def sol(s="cat", target="a"):
        """Strings like 'cat' and 'animal' have the same order indexes e.g. 0:c, 1:a, 2:t if we start counting from 0.
        f6 and g6 find the position (i.e. the index) of a character or group of characters (the target) in a string.
        e.g. f(0,"cat","a") = True, f(2,"cat","a") = False"""
        return s.index(target)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="cat", target="a"):
        return s.index(target)
    ```
    
    </details>
    
* <a name="listindex"></a>**ListIndex**  (10 instances)
    
    ```python
    def sat(item: int, li=[17, 2, 3, 9, 11, 11], index=4):
        """Find the item whose first index in li is index"""
        return li.index(item) == index
    ```
    <details><summary>9,323 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(li=[17, 2, 3, 9, 11, 11], index=4):
        return 11
    
    ```
    
    ```python
    def sol(li=[17, 2, 3, 9, 11, 11], index=4):
        # Note: index is not a parameter of g6, it is a free variable in f6 so
        # f6 binds to its callers free variables. The free variables in f6
        # are sorted in the same order in which g6's formal parameters are given.
        # Since index is after all the formal parameters li, g6 binds to it.
        return li[index]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(li=[17, 2, 3, 9, 11, 11], index=4):
        return li[index]
    ```
    
    </details>
    
* <a name="listat"></a>**ListAt**  (10 instances)
    
    ```python
    def sat(i: int, li=[17, 31, 91, 18, 42, 1, 9], target=18):
        """Find the index of an item in a list. Any such index is fine."""
        return li[i] == target
    ```
    <details><summary>8,900 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(li=[17, 31, 91, 18, 42, 1, 9], target=18):
        return 3
    
    ```
    
    ```python
    def sol(li=[17, 31, 91, 18, 42, 1, 9], target=18):
        """Find the index of an item in a list. Any such index is fine."""
        # Based on: https://www.geeksforgeeks.org/python-search-an-element-in-a-list-using-binary-search/
        low  = 0 
        high = len(li)-1
        while low <= high:
            mid = round((low + high) / 2)
            if li[mid] < target:
                low = mid + 1
            elif li[mid] > target:
                high = mid - 1
            else: # found the element
                return mid
        return None # element not found in the
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(li=[17, 31, 91, 18, 42, 1, 9], target=18):
        return li.index(target)
    ```
    
    </details>
    
* <a name="listlen"></a>**ListLen**  (10 instances)
    
    ```python
    def sat(li: List[int], n=85012):
        """Find a list of a given length n"""
        return len(li) == n
    ```
    <details><summary>7,815 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=85012):
       return [1]*n
    
    ```
    
    ```python
    def sol(n=85012):
        """The testharness uses this generator for this test, so, for what it's worth, I've
        tried to make it fast, e.g. use an infinite range instead of a static list, and reuse
        computed results.
        """
        li: List[int] = []
        i = 0
        while len(li) < n:
            i += 1
            if "0" in str(i) or "4" in str(i) or "6" in str(i):
                li.append(i)
        return li
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=85012):
        return [1] * n
    ```
    
    </details>
    
* <a name="strlen"></a>**StrLen**  (10 instances)
    
    ```python
    def sat(s: str, n=1000):
        """Find a string of length n"""
        return len(s) == n
    ```
    <details><summary>7,340 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(n=1000):
    	return "1"*n
    
    ```
    
    ```python
    def sol(n=1000):
        s = ""
        letter ="0"
        for i in range(1,n+1):
            if i % 3 == 1:
               letter = "1"
            elif i % 3 == 2:
               letter = "2"
            if i % 7 == 1:
               letter = chr(ord(letter) ^ 1)
            if i % 7 == 2:
               letter = chr(ord(letter) ^ 2)
            if i % 7 == 3:
               letter = chr(ord(letter) ^ 4)
            s += letter
        return s
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(n=1000):
        return 'a' * n
    ```
    
    </details>
    
* <a name="intneg"></a>**IntNeg**  (10 instances)
    
    ```python
    def sat(x: int, a=93252338):
        """Solve a unary negation problem"""
        return -x == a
    ```
    <details><summary>6,897 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=93252338): return -a
    
    ```
    
    ```python
    def sol(a=93252338):
        """As in f6, g6 is a function with parameter a unspecified.
           So when f6 calls g6 it must supply a value for a in order to return True.
           If it's correct that's great, otherwise try other valid values for a."""
        return -a
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=93252338):
        return - a
    ```
    
    </details>
    
* <a name="strmul2"></a>**StrMul2**  (10 instances)
    
    ```python
    def sat(n: int, target="foofoofoofoo", s="foofoo"):
        """Find n such that s repeated n times gives target"""
        return s * n == target
    ```
    <details><summary>5,622 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(target="foofoofoofoo", s="foofoo"):
        return 2
    
    ```
    
    ```python
    def sol(target="foofoofoofoo", s="foofoo"):
        # g6 uses a fairly well-known algorithm to repeatedly break up the target into multiples of s
        # and then sums those multiples.
        i = 0
        result = 0
        while True:
            i += 1
            product = s * i
            if product == target:
                return i
            elif product.startswith(target):
                result += i
            elif product > target:
                result += i - 1
                break
        remainder = target[len(s):]
        if remainder:
            return result + g6(remainder, s)
        else:
            return result
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(target="foofoofoofoo", s="foofoo"):
        if len(s) == 0:
            return 1
        return len(target) // len(s)
    ```
    
    </details>
    
* <a name="listin"></a>**ListIn**  (10 instances)
    
    ```python
    def sat(s: str, a=['cat', 'dot', 'bird'], b=['tree', 'fly', 'dot']):
        """Find an item that is in both lists a and b"""
        return s in a and s in b
    ```
    <details><summary>5,219 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=['cat', 'dot', 'bird'], b=['tree', 'fly', 'dot']):
    	return a[1]
    
    ```
    
    ```python
    def sol(a=['cat', 'dot', 'bird'], b=['tree', 'fly', 'dot']):
        """Group together two sequences, one in a and one in b, so that the items
        can be used to reconstruct a sequence whose pairs of items sum to 3."""
        # find set of possible pairs, and prune out 3 and 4
        s = set(a) & set(b) - set(['bird', 'tree'])
        # return the entry with the lowest string value
        return min(s)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=['cat', 'dot', 'bird'], b=['tree', 'fly', 'dot']):
        return next(s for s in b if s in a)
    ```
    
    </details>
    
* <a name="strsplitter"></a>**StrSplitter**  (10 instances)
    
    ```python
    def sat(x: str, parts=['I', 'love', 'dumplings', '!', ''], string="I_love_dumplings_!_"):
        """Find a separator that when used to split a given string gives a certain result"""
        return string.split(x) == parts
    ```
    <details><summary>5,200 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(parts=['I', 'love', 'dumplings', '!', ''], string="I_love_dumplings_!_"):
    	return "_"
    
    ```
    
    ```python
    def sol(parts=['I', 'love', 'dumplings', '!', ''], string="I_love_dumplings_!_"):
        separator = '_'
        last = '_'
        for part in parts:
            for letter in part:
                if letter == separator:
                    separator = chr(ord(separator)+1)
                    break
            if (last == separator and len(parts) > 1 and parts.index(part) != 0) or (
                    last == part and parts.index(part) == len(parts) - 1):
                if not string.endswith(separator):
                    separator += '_'
            if last == separator:
                return separator
            last = part
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(parts=['I', 'love', 'dumplings', '!', ''], string="I_love_dumplings_!_"):
        if len(parts) <= 1:
            return string * 2
        length = (len(string) - len("".join(parts))) // (len(parts) - 1)
        start = len(parts[0])
        return string[start:start + length]
    ```
    
    </details>
    
* <a name="intdiv"></a>**IntDiv**  (10 instances)
    
    ```python
    def sat(n: int, a=3, b=23463462):
        """Solve a division problem"""
        return b // n == a
    ```
    <details><summary>5,198 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=3, b=23463462):
        return b//a
    ```
    
    ```python
    def sol(a=3, b=23463462):
        # Find the smallest n for which b // n == a
        # b cannot be divided by 6 because 6 // 6 = 1 != 3 < 6, so 6 is ruled out
        # b cannot be divided by 5 because 5 // 5 = 1 != 3 < 5, so 5 is ruled out
        # b cannot be divided by 4 because 4 // 4 = 1 != 3 < 4, so 4 is ruled out
        # b cannot be divided by 3 because 3 // 3 = 1 != 3 < 3, so 3 is ruled out
        for n in range(7, b+1):
            if b // n == a:
                return n
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=3, b=23463462):
        if a == 0:
            return 2 * b
        for n in [b // a, b // a - 1, b // a + 1]:
            if b // n == a:
                return n
    ```
    
    </details>
    
* <a name="helloworld"></a>**HelloWorld** Trivial example, no solutions provided (1 instance)
    
    ```python
    def sat(s: str):
        """Find a string that when concatenated onto 'world' gives 'Hello world'."""
        return s + 'world' == 'Hello world'
    ```
    <details><summary>4,581 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return "Hello "
    ```
    
    ```python
    def sol():
        """Find a string that when concatenated onto 'world' gives 'Hello world'."""
        return 'Hello '
    
    ```
    
    </details>
    
    0 hand-written solutions 
    
* <a name="strparts"></a>**StrParts**  (10 instances)
    
    ```python
    def sat(parts: List[str], sep="!!", string="I!!!!!love!!dumplings!!!!!"):
        """Find parts that when joined give a specific string."""
        return sep.join(parts) == string and all(sep not in p for p in parts)
    ```
    <details><summary>3,711 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(sep="!!", string="I!!!!!love!!dumplings!!!!!"):
        return string.split(sep)
    ```
    
    ```python
    def sol(sep="!!", string="I!!!!!love!!dumplings!!!!!"):
        """Build a list of strings that when joined give a specific string."""
        # Splice different strings together at the separator 'sep'. Return a list of the parts.
        # Each part should have no instances of the separator 'sep'.
        # The first and last characters should also be the separator
        # For example, sat(sep="!!", string="I!!!!!love!!dumplings!!!!!") should return ['I', '!!', '!', 'love', '!!', 'dumplings', '!!!', '!']
        return list(string.split(sep))
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(sep="!!", string="I!!!!!love!!dumplings!!!!!"):
        return string.split(sep)
    ```
    
    </details>
    
* <a name="intmul"></a>**IntMul**  (10 instances)
    
    ```python
    def sat(n: int, a=14302, b=5):
        """Solve a multiplication problem"""
        return b * n + (a % b) == a
    ```
    <details><summary>3,371 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=14302, b=5):
    	return a // b
    
    ```
    
    ```python
    def sol(a=14302, b=5):
        max_attempt = 100000
        div, rem = divmod(a, b)
        r2 = rem
        n = div
        trial = 0
        while not r2 == rem and trial < max_attempt:
            if rem < r2:
                div -= 1
                rem = b + rem - r2
            else:
                rem = rem - r2
            trial += 1
        if trial < max_attempt:
            return n
        raise Exception(f"failed to find {a} - {b} * {n} = {rem}")
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=14302, b=5):
        return a // b
    ```
    
    </details>
    
* <a name="strnegat"></a>**StrNegAt**  (10 instances)
    
    ```python
    def sat(i: int, s="cat", target="a"):
        """Find the index of target in s using a negative index."""
        return s[i] == target and i < 0
    ```
    <details><summary>2,847 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="cat", target="a"):
        return -2
    
    ```
    
    ```python
    def sol(s="cat", target="a"):
        """Find the index of target in s using a negative index.
        I.e. find the index x such that s[x] = target, so that -x < 0."""
        m = len(s)
        for x in range(-m, 0):
            if s[x] == target:
                return x
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="cat", target="a"):
        return - (len(s) - s.index(target))
    ```
    
    </details>
    
* <a name="intnegsquareroot"></a>**IntNegSquareRoot**  (10 instances)
    
    ```python
    def sat(n: int, a=10000200001):
        """Find a negative integer that when squared equals perfect-square a."""
        return a == n * n and n < 0
    ```
    <details><summary>2,653 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=10000200001):
        return -int(a**.5)
    
    ```
    
    ```python
    def sol(a=10000200001):
        m = int(a ** 0.5)
        if m * m == a:
            # When m is perfect, just return m.
            return -m
        else:
            # Find the nearest left-of-perfect.
            assert m >= 2
            assert m < a
            assert (m-1) * (m-1) < a
            assert (m-2) * (m-2) < a
            assert (m-1) * (m-2) >= a
            return -m-1
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=10000200001):
        return -int(a ** 0.5)
    ```
    
    </details>
    
* <a name="intsub2"></a>**IntSub2**  (10 instances)
    
    ```python
    def sat(x: int, a=8665464, b=-93206):
        """Solve a subtraction problem"""
        return a - x == b
    ```
    <details><summary>2,541 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=8665464, b=-93206):
        return a-b
    
    ```
    
    ```python
    def sol(a=8665464, b=-93206):
        """Solve a subtraction problem"""
        i = 0
        while True:
            i += 1
            if a - i == b:
                return i
        raise ValueError(f'solution to f6 not found')
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=8665464, b=-93206):
        return a - b
    ```
    
    </details>
    
* <a name="floatsquareroot"></a>**FloatSquareRoot**  (10 instances)
    
    ```python
    def sat(x: float, a=1020):
        """Find a number that when squared is close to a."""
        return abs(x ** 2 - a) < 10 ** -3
    ```
    <details><summary>2,423 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=1020):
        return a**.5
    
    ```
    
    ```python
    def sol(a=1020):
        x = (a / 1020) ** (1/2)  # A quick estimate of the solution.
        for _ in range(100):
            x = (x + (a / x)) / 2  # A single iteration of Newton's method finds the solution to arbitrary precision.
        return x
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=1020):
        return a ** 0.5
    ```
    
    </details>
    
* <a name="listnegat"></a>**ListNegAt**  (10 instances)
    
    ```python
    def sat(i: int, li=[17, 31, 91, 18, 42, 1, 9], target=91):
        """Find the index of an item in a list using negative indexing."""
        return li[i] == target and i < 0
    ```
    <details><summary>2,285 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(li=[17, 31, 91, 18, 42, 1, 9], target=91):
        return -5
    
    ```
    
    ```python
    def sol(li=[17, 31, 91, 18, 42, 1, 9], target=91):
        i = -len(li)
        while i < 0:
            if li[i] == target:
                return i
            # i -= 1 # Original algorithm, corrected version below
            i += 1
        # Python does not support modifying the execution counter!
        raise AssertionError("did not find solution to g6")
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(li=[17, 31, 91, 18, 42, 1, 9], target=91):
        return li.index(target) - len(li)
    ```
    
    </details>
    
* <a name="intsub"></a>**IntSub**  (10 instances)
    
    ```python
    def sat(x: int, a=-382, b=14546310):
        """Solve a subtraction problem"""
        return x - a == b
    ```
    <details><summary>1,913 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=-382, b=14546310): return a + b
    
    ```
    
    ```python
    def sol(a=-382, b=14546310):
        """Solve a subtraction problem.  Generate solutions instead of checking them."""
        for x in range(a+1, 2*b):
            if x - a == b:
                return x
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=-382, b=14546310):
        return a + b
    ```
    
    </details>
    
* <a name="strcount"></a>**StrCount**  (10 instances)
    
    ```python
    def sat(string: str, substring="a", count=10, length=100):
        """Find a string with a certain number of copies of a given substring and of a given length"""
        return string.count(substring) == count and len(string) == length
    ```
    <details><summary>1,420 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(substring="a", count=10, length=100):
        return substring * 10 + "0"*90
    
    ```
    
    ```python
    def sol(substring="a", count=10, length=100):
    
        i = 0
        while True:
            i += 1
            string = substring*count + "0"*(length-count)
            if len(string) == length:
                return string
            if i > 100000:
                raise Exception("g6: did not terminate")
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(substring="a", count=10, length=100):
        c = chr(1 + max(ord(c) for c in (substring or "a")))  # a character not in substring
        return substring * count + (length - len(substring) * count) * '^'
    ```
    
    </details>
    
* <a name="floatnegsquareroot"></a>**FloatNegSquareRoot**  (10 instances)
    
    ```python
    def sat(x: float, a=1020):
        """Find a negative number that when squared is close to a."""
        return abs(x ** 2 - a) < 10 ** -3 and x < 0
    ```
    <details><summary>1,406 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=1020):
        return -a**.5
    
    ```
    
    ```python
    def sol(a=1020):
        # use Newton's method https://en.wikipedia.org/wiki/Newton's_method
        def helper(x0, n_iter=10):
            x = x0
            for i in range(n_iter):
                x = x - (x**2 - a) / (2 * x)
            return x
    
        return helper(-250)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=1020):
        return -a ** 0.5
    ```
    
    </details>
    
* <a name="intsquareroot"></a>**IntSquareRoot**  (10 instances)
    
    ```python
    def sat(x: int, a=10201202001):
        """Compute an integer that when squared equals perfect-square a."""
        return x ** 2 == a
    ```
    <details><summary>1,400 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=10201202001): return int(a ** 0.5)
    
    ```
    
    ```python
    def sol(a=10201202001):
    	n = 997
    	while True:
    		x = n ** 2
    		if x > a or x > 999999999999999999:
    			assert x == 102012020009999999999  # An easy mistake to make.  See also
    			return  # 1020120202 for a more subtle one.  Note that both of these
    			# require exponents that are less that that of 10201202001.
    		if x == a:
    			return n
    		n = n + 1
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=10201202001):
        return int(a ** 0.5)
    ```
    
    </details>
    
* <a name="listsetlen"></a>**ListSetLen**  (10 instances)
    
    ```python
    def sat(li: List[int], dups=42155):
        """Find a list with a certain number of duplicate items"""
        return len(set(li)) == len(li) - dups
    ```
    <details><summary>1,254 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(dups=42155): return [1]*(dups+1)
    
    ```
    
    ```python
    def sol(dups=42155):
        # find smallest list of duplicate ints {d} such that sum(x*(x+1)/2 for x in range(1, d)) > 10^9
        x = 1
        while sum(x*(x+1)//2 for x in range(dups, dups+x)) <= 10**9:
            x += 1
        return list(range(dups)) * x
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(dups=42155):
        return [1] * (dups + 1)
    ```
    
    </details>
    
* <a name="intsum"></a>**IntSum**  (10 instances)
    
    ```python
    def sat(x: int, a=1073258, b=72352549):
        """Solve a sum problem"""
        return a + x == b
    ```
    <details><summary>1,115 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=1073258, b=72352549):
        return b-a
    
    ```
    
    ```python
    def sol(a=1073258, b=72352549):
        # Invariant: a + x == b and x and a and b are integers.
        # We implement this invariant because we don't want any other values to enter the equation.
        # Correctness argument: This invariant is sufficient because, as defined by the tests, the function will never fail.
        x = int(b - a)
        return x
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=1073258, b=72352549):
        return b - a
    ```
    
    </details>
    
* <a name="strmul"></a>**StrMul**  (10 instances)
    
    ```python
    def sat(s: str, target="foofoofoofoo", n=2):
        """Find a string which when repeated n times gives target"""
        return s * n == target
    ```
    <details><summary>1,088 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(target="foofoofoofoo", n=2):
        return "foo"*n
    
    ```
    
    ```python
    def sol(target="foofoofoofoo", n=2):
        """Find a string which when repeated n times gives target"""
        matching_str = None
        for s in [ "bar", "foobar", "foofoo", "foofoofoo", "foofoofoofoo", "foo" ]:
            if s in target:
                matching_str = s;
                break;
        return matching_str
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(target="foofoofoofoo", n=2):
        if n == 0:
            return ''
        return target[:len(target) // n]
    ```
    
    </details>
    
* <a name="listindex2"></a>**ListIndex2**  (10 instances)
    
    ```python
    def sat(li: List[int], i=29, index=10412):
        """Find a list that contains i first at index index"""
        return li.index(i) == index
    ```
    <details><summary>1,029 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(i=29, index=10412):
        return [0]*index+[i]
    
    ```
    
    ```python
    def sol(i=29, index=10412):
        """Generate a list that contains i first at index index"""
        li = [x for x in range(100, 100 + index)]
        li.append(i)
        for x in range(i + 1, 100 + index):
            li.append(x)
        return li
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(i=29, index=10412):
        return [i - 1] * index + [i]
    ```
    
    </details>
    
* <a name="strsplit"></a>**StrSplit**  (10 instances)
    
    ```python
    def sat(x: str, parts=['I', 'love', 'dumplings', '!'], length=100):
        """Find a string of a given length with a certain split"""
        return len(x) == length and x.split() == parts
    ```
    <details><summary>954 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(parts=['I', 'love', 'dumplings', '!'], length=100):
        return " ".join(parts).ljust(length)
    
    ```
    
    ```python
    def sol(parts=['I', 'love', 'dumplings', '!'], length=100):
        """Attempts to find a string with a given split and length"""
        candidates = [
            " ".join(parts) + " "*(length - (len(" ".join(parts)) % length)),
            " ".join(parts) + " "*(length - (len(" ".join(parts)) % length)) + " ".join(parts),
            " ".join(parts)[:(length - (len(" ".join(parts)) % length))] + " ".join(parts)
        ]
        for c in candidates:
            #print(c, sat(c))
            if sat(c):
                return c
        return None
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(parts=['I', 'love', 'dumplings', '!'], length=100):
        joined = " ".join(parts)
        return joined + " " * (length - len(joined))
    ```
    
    </details>
    
* <a name="intdiv2"></a>**IntDiv2**  (10 instances)
    
    ```python
    def sat(n: int, a=345346363, b=10):
        """Find n that when divided by b is a"""
        return n // b == a
    ```
    <details><summary>949 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a=345346363, b=10):
        return a*b
    
    ```
    
    ```python
    def sol(a=345346363, b=10):
        """Find a number evenly divisible by b whose decimal representation has a digits in common with a."""
        while True:
            n = a * b
            if any(x in str(n) for x in str(a)):
                return n
            b += 1
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a=345346363, b=10):
        return a * b
    ```
    
    </details>
    
* <a name="strindex2"></a>**StrIndex2**  (10 instances)
    
    ```python
    def sat(big_str: str, sub_str="foobar", index=2):
        """Find a string whose *first* index of sub_str is index"""
        return big_str.index(sub_str) == index
    ```
    <details><summary>947 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(sub_str="foobar", index=2):
        return "fo"+sub_str
    
    ```
    
    ```python
    def sol(sub_str="foobar", index=2):
        """Note that the signature of f6 matches this function. This function is just an arbitrary example.
        It is used in the test below. It's a 2-argument function with a hardcoded value of 'foobar' in the string
        and a 2-argument function with a hardcoded index in the tests below.
        """
        big_str = "There is no spoon."
        if index == -1:
            return big_str[::-1]
        return big_str[:index] + sub_str + big_str[index + len(sub_str):]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(sub_str="foobar", index=2):
        i = ord('A')
        while chr(i) in sub_str:
            i += 1
        return chr(i) * index + sub_str
    ```
    
    </details>
    
* <a name="listmul"></a>**ListMul**  (10 instances)
    
    ```python
    def sat(li: List[int], target=[17, 9, -1, 17, 9, -1], n=2):
        """Find a list that when multiplied n times gives the target list"""
        return li * n == target
    ```
    <details><summary>692 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(target=[17, 9, -1, 17, 9, -1], n=2):
        return [17,9,-1]
    
    ```
    
    ```python
    def sol(target=[17, 9, -1, 17, 9, -1], n=2):
        return target[0] if all(x == target[0] for x in target) else target[1] if all(x == target[1] for x in target) else \
            target[2] if all(x == target[2] for x in target) else target[3] if all(x == target[3] for x in target) else \
            target[4] if all(x == target[4] for x in target) else target[5] if all(x == target[5] for x in target) else \
            [target[0], target[1], target[2]]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(target=[17, 9, -1, 17, 9, -1], n=2):
        if n == 0:
            return []
        return target[:len(target) // n]
    ```
    
    </details>
    
* <a name="strin"></a>**StrIn**  (10 instances)
    
    ```python
    def sat(s: str, a="hello", b="yellow", length=4):
        """Find a string of length length that is in both strings a and b"""
        return len(s) == length and s in a and s in b
    ```
    <details><summary>682 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a="hello", b="yellow", length=4):
        return "ello"
    
    ```
    
    ```python
    def sol(a="hello", b="yellow", length=4):
        for i in range(len(a) - length + 1):
            for j in range(len(b) - length + 1):
                for k in range(length):
                    check = ""
                    for l in range(length):
                        if l > k:
                            check += b[j+l]
                        else:
                            check += a[i+l]
                    if sat(check):
                        return check
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a="hello", b="yellow", length=4):
        for i in range(len(a) - length + 1):
            if a[i:i + length] in b:
                return a[i:i + length]
    ```
    
    </details>
    
* <a name="stradd"></a>**StrAdd**  (10 instances)
    
    ```python
    def sat(st: str, a="world", b="Hello world"):
        """Solve simple string addition problem."""
        return st + a == b
    ```
    <details><summary>645 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(a="world", b="Hello world"):
        return b[:6]
    
    ```
    
    ```python
    def sol(a="world", b="Hello world"):
        """The correct value of s6 is irrelevant, but the form of the input st must be known."""
        return "Hello "
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(a="world", b="Hello world"):
        return b[:len(b) - len(a)]
    ```
    
    </details>
    
* <a name="strsetlen"></a>**StrSetLen**  (9 instances)
    
    ```python
    def sat(s: str, dups=2021):
        """Find a string with dups duplicate chars"""
        return len(set(s)) == len(s) - dups
    ```
    <details><summary>579 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(dups=2021):
        return "0"*2022
    
    ```
    
    ```python
    def sol(dups=2021):
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        result = ''
        for i in range(dups):
            result += letters[i % 26]
        return result + ''.join(letters)
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(dups=2021):
        return "a" * (dups + 1)
    ```
    
    </details>
    
* <a name="strjoiner"></a>**StrJoiner**  (10 instances)
    
    ```python
    def sat(x: str, parts=['I!!', '!love', 'dumplings', '!', ''], string="I!!!!!love!!dumplings!!!!!"):
        """
        Find a separator that when used to join a given string gives a certain result.
        This is related to the previous problem but there are some edge cases that differ.
        """
        return x.join(parts) == string
    ```
    <details><summary>299 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(parts=['I!!', '!love', 'dumplings', '!', ''], string="I!!!!!love!!dumplings!!!!!"):
        return "!!"
    
    ```
    
    ```python
    def sol(parts=['I!!', '!love', 'dumplings', '!', ''], string="I!!!!!love!!dumplings!!!!!"):
        """Separator is initially '!' but we check for each index if we can join the parts and still get the string."""
        xs = ["!" if i == 0 else "I!!!" for i in range(len(parts))]
        for i in range(len(parts)):
            xs[i] = "!"
            for p in string:
                xs[i] += p
                if xs[i].join(parts) == string:
                    return xs[i]
                xs[i] = xs[-1][-1]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(parts=['I!!', '!love', 'dumplings', '!', ''], string="I!!!!!love!!dumplings!!!!!"):
        if len(parts) <= 1:
            return ""
        length = (len(string) - len("".join(parts))) // (len(parts) - 1)
        start = len(parts[0])
        return string[start:start + length]
    ```
    
    </details>
    
* <a name="strindex"></a>**StrIndex**  (10 instances)
    
    ```python
    def sat(s: str, big_str="foobar", index=2):
        """Find a string whose *first* index in big_str is index"""
        return big_str.index(s) == index
    ```
    <details><summary>206 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(big_str="foobar", index=2):
        return "ob"
    
    ```
    
    ```python
    def sol(big_str="foobar", index=2):
        """Find a string with no repeated characters which appears as a substring of big_str starting at index"""
        return "".join([x for i, x in enumerate(big_str[index:]) if x not in big_str[index:i]])
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(big_str="foobar", index=2):
        return big_str[index:]
    ```
    
    </details>
    
* <a name="backworlds"></a>**BackWorlds** We provide two solutions (1 instance)
    
    ```python
    def sat(s: str):
        """Find a string that when reversed and concatenated onto 'world' gives 'Hello world'."""
        return s[::-1] + 'world' == 'Hello world'
    ```
    <details><summary>26 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return "Hello "[::-1]
    
    ```
    
    ```python
    def sol():
        return "Hello world"[len('Hello world')//2::-1]
    
    ```
    
    </details>
    
    <details><summary>2 hand-written solutions </summary>
    
    ```python
    def sol():
        return ' olleH'
    ```
    
    ```python
    def sol():  # solution methods must begin with 'sol'
        return 'Hello '[::-1]
    ```
    
    </details>
    
* <a name="strin2"></a>**StrIn2**  (10 instances)
    
    ```python
    def sat(substrings: List[str], s="hello", count=15):
        """Find a list of >= count distinct strings that are all contained in s"""
        return len(substrings) == len(set(substrings)) >= count and all(sub in s for sub in substrings)
    ```
    <details><summary>8 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(s="hello", count=15):
        """Distinct substrings of 'hello'"""
        return sorted(set([s[i:j] for i in range(len(s) + 1) for j in range(len(s) + 1)]))[:count]
    
    ```
    
    ```python
    def sol(s="hello", count=15):
        # recursive backtracking. I used a helper function because "return;" doesn't count as a statement.
        n = len(s)
        choices = []
        def helper(i, j):
            for k in range(i, j + 1):
                if s[i:k] not in choices:
                    choices.append(s[i:k])
            if len(choices) > count:
                return
            if j < n:
                helper(0, j + 1)
            if i < (n - 1):
                helper(i + 1, n)
        helper(0, 0)
        return choices
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="hello", count=15):
        return [""] + sorted({s[j:i] for i in range(len(s) + 1) for j in range(i)})
    ```
    
    </details>
    
* <a name="strslice"></a>**StrSlice**  (10 instances)
    
    ```python
    def sat(inds: List[int], s="hello world", target="do"):
        """Find the three slice indices that give the specific target in string s"""
        i, j, k = inds
        return s[i:j:k] == target
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(s="hello world", target="do"):
        from itertools import product
        for i, j, k in product(range(-len(s) - 1, len(s) + 1), repeat=3):
            try:
                if s[i:j:k] == target:
                    return [i, j, k]
            except (IndexError, ValueError):
                pass
    ```
    
    </details>
    
* <a name="listslice"></a>**ListSlice**  (10 instances)
    
    ```python
    def sat(inds: List[int], li=[42, 18, 21, 103, -2, 11], target=[-2, 21, 42]):
        """Find three slice indices to achieve a given list slice"""
        i, j, k = inds
        return li[i:j:k] == target
    ```
    0 AI solutions from codex<details><summary>1 hand-written solution </summary>
    
    ```python
    def sol(li=[42, 18, 21, 103, -2, 11], target=[-2, 21, 42]):
        from itertools import product
        for i, j, k in product(range(-len(li) - 1, len(li) + 1), repeat=3):
            try:
                if li[i:j:k] == target:
                    return [i, j, k]
            except (IndexError, ValueError):
                pass
    ```
    
    </details>
    
## tutorial


A few example puzzles that were presented with solutions to participants of the study.


* <a name="tutorial3"></a>**Tutorial3**  (1 instance)
    
    ```python
    def sat(x: List[int]):
        """Find a list of two integers whose sum is 3."""
        return len(x) == 2 and sum(x) == 3
    ```
    <details><summary>7,776 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(): return [1, 2]
    
    ```
    
    ```python
    def sol():
        i = 0
        while True:
            if (int(str(123456789 ** 99)[i]) == 9):
                break
            if (i > 10000):
                return False
            i += 1
        return [1, 2];
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return [1, 2]
    ```
    
    </details>
    
* <a name="tutorial1"></a>**Tutorial1**  (1 instance)
    
    ```python
    def sat(s: str):
        """Find a string that when concatenated onto 'Hello ' gives 'Hello world'."""
        return "Hello " + s == "Hello world"
    ```
    <details><summary>6,684 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return "world"
    ```
    
    ```python
    def sol():
        for s in ["world", "Jupiter", "Mars", "Venus", "Mercury", "Jupiter", "Saturn", "Uranus", "Neptune"]:
            if sat(s):
                return s
        return None
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return "world"
    ```
    
    </details>
    
* <a name="tutorial2"></a>**Tutorial2**  (1 instance)
    
    ```python
    def sat(s: str):
        """Find a string that when reversed and concatenated onto 'Hello ' gives 'Hello world'."""
        return "Hello " + s[::-1] == "Hello world"
    ```
    <details><summary>6,084 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol(): return "world"[::-1]
    
    ```
    
    ```python
    def sol():
        # We take advantage of the fact that Python 2.7 has a built-in character type
        # and won't bother to iterate over ints (which can be converted to Python 2.7's built-in characters) for nothing.
        #
        # However, if we had been working under Python 3, then we could have tried to make things as general as possible:
        # ...except that Python 3 has no built-in type called a "character", and by default every string is just a sequence of such "characters":
        return "world"[::-1]
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return "world"[::-1]
    ```
    
    </details>
    
* <a name="tutorial4"></a>**Tutorial4**  (1 instance)
    
    ```python
    def sat(s: List[str]):
        """Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b'."""
        return len(set(s)) == 1000 and all((x.count("a") > x.count("b")) and ('b' in x) for x in s)
    ```
    <details><summary>5,576 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return ["aba"*(i+2) for i in range(1000)]
    
    ```
    
    ```python
    def sol():
    
        s = ["a"*(i+2)+"b" for i in range(1000)]
    
        # The first time -OR- when there was a string that didn't meet the criteria,
        # return the index of this string instead of its value (#5-8).
        # then in the last for loop, return the value.
    
        for i in range(1000):
            if not (s[i].count("a") > s[i].count("b") and 'b' in s[i]):
                return i
        return s
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return ["a" * (i + 2) + "b" for i in range(1000)]
    ```
    
    </details>
    
* <a name="tutorial5"></a>**Tutorial5**  (1 instance)
    
    ```python
    def sat(n: int):
        """Find an integer whose perfect square begins with 123456789 in its decimal representation."""
        return str(n * n).startswith("123456789")
    ```
    <details><summary>4,514 AI solutions from codex (shortest and longest ones below)</summary>
    
    ```python
    def sol():
        return int("1"*9 + "0"*9)
    
    ```
    
    ```python
    def sol():
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    for l in range(10):
                        for m in range(10):
                            for n in range(10):
                                for o in range(10):
                                    for p in range(10):
                                        n = int("".join([str(i), str(j), str(k), str(l), str(m), str(n), str(o), str(p)]))
                                        if str(n * n).startswith("123456789"):
                                            return n
    
    ```
    
    </details>
    
    <details><summary>1 hand-written solution </summary>
    
    ```python
    def sol():
        return int(int("123456789" + "0" * 9) ** 0.5) + 1
    ```
    
    </details>
    
