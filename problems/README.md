# The AI-SAT dataset summary
This document summarizes the dataset stored in .json files.
Each .json file contains a number of related problems with one or more instances each.

## Files:

- [algebra (4 problems, 4,000 instances)](#algebra)
- [basic (10 problems, 10,000 instances)](#basic)
- [chess (5 problems, 4,858 instances)](#chess)
- [codeforces (16 problems, 15,023 instances)](#codeforces)
- [game_theory (2 problems, 2,000 instances)](#game_theory)
- [graphs (10 problems, 8,002 instances)](#graphs)
- [ICPC (2 problems, 2,000 instances)](#icpc)
- [IMO (6 problems, 5,011 instances)](#imo)
- [puzzles (18 problems, 8,367 instances)](#puzzles)

Total (73 problems, 59,261 instances)


----

## algebra

Roots of polynomials

[^ Top](#files)

### QuadraticRoot ([algebra](#algebra) 1/4)

**Description:**
Find any (real) solution for a [quadratic equation](https://en.wikipedia.org/wiki/Quadratic_formula)
a x^2 + b x + c

**Problem:**

```python
def sat(x: float, coeffs: List[float]=[2.5, 1.3, -0.5]):
    assert type(x) is float, 'x must be of type float'
    a, b, c = coeffs
    return abs(a * x ** 2 + b * x + c) < 1e-6
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### AllQuadraticRoots ([algebra](#algebra) 2/4)

**Description:**
Find all (real) solutions for a [quadratic equation](https://en.wikipedia.org/wiki/Quadratic_formula)
x^2 + b x + c (i.e., factor into roots)

**Problem:**

```python
def sat(roots: List[float], coeffs: List[float]=[1.3, -0.5]):
    assert type(roots) is list and all(type(a) is float for a in roots), 'roots must be of type List[float]'
    b, c = coeffs
    r1, r2 = roots
    return abs(r1 + r2 + b) + abs(r1 * r2 - c) < 1e-6
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(coeffs=[1.3, -0.5]):
    b, c = coeffs
    delta = (b ** 2 - 4 * c) ** 0.5
    return [(-b + delta) / 2, (-b - delta) / 2]
```

</details>

### CubicRoot ([algebra](#algebra) 3/4)

**Description:**
Find any (real) solution for a [cubic equation](https://en.wikipedia.org/wiki/Cubic_formula)
a x^3 + b x^2 + c x + d

**Problem:**

```python
def sat(x: float, coeffs: List[float]=[2.0, 1.0, 0.0, 8.0]):
    assert type(x) is float, 'x must be of type float'
    return abs(sum(c * x ** (3 - i) for i, c in enumerate(coeffs))) < 1e-6
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(coeffs=[2.0, 1.0, 0.0, 8.0]):
    a2, a1, a0 = [c / coeffs[0] for c in coeffs[1:]]
    p = (3 * a1 - a2 ** 2) / 3
    q = (9 * a1 * a2 - 27 * a0 - 2 * a2 ** 3) / 27
    delta = (q ** 2 + 4 * p ** 3 / 27) ** 0.5
    omega = (-(-1) ** (1 / 3))
    answers = []
    for cube in [(q + delta) / 2, (q - delta) / 2]:
        c = cube ** (1 / 3)
        for w in [c, c * omega, c * omega.conjugate()]:
            if w != 0:
                x = complex(w - p / (3 * w) - a2 / 3).real
                if abs(sum(c * x ** (3 - i) for i, c in enumerate(coeffs))) < 1e-6:
                    return x
```

</details>

### AllCubicRoots ([algebra](#algebra) 4/4)

**Description:**
Find all 3 distinct real roots of x^3 + a x^2 + b x + c, i.e., factor into (x-r1)(x-r2)(x-r3)

**Problem:**

```python
def sat(roots: List[float], coeffs: List[float]=[1.0, -2.0, -1.0]):
    assert type(roots) is list and all(type(a) is float for a in roots), 'roots must be of type List[float]'
    r1, r2, r3 = roots
    a, b, c = coeffs
    return abs(r1 + r2 + r3 + a) + abs(r1 * r2 + r1 * r3 + r2 * r3 - b) + abs(r1 * r2 * r3 + c) < 1e-6
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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
            if w!=0.0:
                x = complex(w - p / (3 * w) - a / 3).real
                if abs(x ** 3 + a * x ** 2 + b * x + c) < 1e-4:
                    if not ans or min(abs(z - x) for z in ans) > 1e-6:
                        ans.append(x)
    if len(ans)==3:
        return ans
```

</details>

[^^ Top](#files)
## basic

Problems testing basic knowledge -- easy to solve if you understand what is being asked

[^ Top](#files)

### LineIntersection ([basic](#basic) 1/10)

**Description:**
Find the intersection of two lines.
Solution should be a list of the (x,y) coordinates.
Accuracy of fifth decimal digit is required.

**Problem:**

```python
def sat(e: List[int], a: int=2, b: int=-1, c: int=1, d: int=-3):
    assert type(e) is list and all(type(a) is int for a in e), 'e must be of type List[int]'
    x = e[0] / e[1]
    return abs(a * x + b - c * x - d) < 10 ** -5
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=2, b=-1, c=1, d=-3):
    return [d - b, a - c]
```

</details>

### IfProblem ([basic](#basic) 2/10)

**Description:**
Simple if statement

**Problem:**

```python
def sat(x: int, a: int=2, b: int=100):
    assert type(x) is int, 'x must be of type int'
    if a < 50:
        return x + a == b
    else:
        return x - 2 * a == b
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=2, b=100):
    if a < 50:
        return b - a
    else:
        return b + 2 * a
```

</details>

### IfProblemWithAnd ([basic](#basic) 3/10)

**Description:**
Simple if statement with and clause

**Problem:**

```python
def sat(x: int, a: int=2, b: int=100):
    assert type(x) is int, 'x must be of type int'
    if x > 0 and a > 50:
        return x - a == b
    else:
        return x + a == b
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=2, b=100):
    if a > 50 and b > a:
        return b + a
    else:
        return b - a
```

</details>

### IfProblemWithOr ([basic](#basic) 4/10)

**Description:**
Simple if statement with or clause

**Problem:**

```python
def sat(x: int, a: int=2, b: int=100):
    assert type(x) is int, 'x must be of type int'
    if x > 0 or a > 50:
        return x - a == b
    else:
        return x + a == b
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=2, b=100):
    if a > 50 or b > a:
        return b + a
    else:
        return b - a
```

</details>

### IfCases ([basic](#basic) 5/10)

**Description:**
Simple if statement with multiple cases

**Problem:**

```python
def sat(x: int, a: int=1, b: int=100):
    assert type(x) is int, 'x must be of type int'
    if a == 1:
        return x % 2 == 0
    elif a == -1:
        return x % 2 == 1
    else:
        return x + a == b
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=1, b=100):
    if a == 1:
        x = 0
    elif a == -1:
        x = 1
    else:
        x = b - a
    return x
```

</details>

### ListPosSum ([basic](#basic) 6/10)

**Description:**
Construct a list of non-negative integers that sum up to some value

**Problem:**

```python
def sat(x: List[int], n: int=5, s: int=19):
    assert type(x) is list and all(type(a) is int for a in x), 'x must be of type List[int]'
    return len(x) == n and sum(x) == s and all([a > 0 for a in x])
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(n=5, s=19):
    x = [1] * n
    x[0] = s - n + 1
    return x
```

</details>

### ListDistinctSum ([basic](#basic) 7/10)

**Description:**
Construct a list of distinct integers that sum up to some value

**Problem:**

```python
def sat(x: List[int], n: int=4, s: int=1):
    assert type(x) is list and all(type(a) is int for a in x), 'x must be of type List[int]'
    return len(x) == n and sum(x) == s and len(set(x)) == n
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(n=4, s=1):
    a = 1
    x = []
    while len(x) < n-1:
        x.append(a)
        a = -a
        if a in x:
            a += 1

    if s - sum(x) in x:
        x = [i for i in range(n-1)]

    x = x + [s - sum(x)]
    return x
```

</details>

### ConcatStrings ([basic](#basic) 8/10)

**Description:**
Concatenate list of characters

**Problem:**

```python
def sat(x: str, s: List[str]=['a', 'b', 'c', 'd', 'e', 'f'], n: int=4):
    assert type(x) is str, 'x must be of type str'
    return len(x)==n and all([x[i] == s[i] for i in range(n)])
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(s=['a', 'b', 'c', 'd', 'e', 'f'], n=4):
    return ''.join([s[i] for i in range(n)])
```

</details>

### SublistSum ([basic](#basic) 9/10)

**Description:**
Sum values of sublist by range specifications

**Problem:**

```python
def sat(x: List[int], t: int=677, a: int=43, e: int=125, s: int=10):
    assert type(x) is list and all(type(a) is int for a in x), 'x must be of type List[int]'
    non_zero = [z for z in x if z != 0]
    return t == sum([x[i] for i in range(a,e,s)]) and len(set(non_zero)) == len(non_zero) and all([x[i] != 0 for i in range(a,e,s)])
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(t=677, a=43, e=125, s=10):
    x = [0] * e
    for i in range(a,e,s):
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

### CumulativeSum ([basic](#basic) 10/10)

**Description:**
Number of values with cumulative sum less than target

**Problem:**

```python
def sat(x: List[int], t: int=50, n: int=10):
    assert type(x) is list and all(type(a) is int for a in x), 'x must be of type List[int]'
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
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(t=50, n=10):
    return [1] * n + [t]
```

</details>

[^^ Top](#files)
## chess

Classic chess problems

[^ Top](#files)

### EightQueensOrFewer ([chess](#chess) 1/5)

**Description:**
Eight (or fewer) Queens Puzzle

Position min(m, n) <= 8 queens on an m x n chess board so that no pair is attacking each other. Hint:
a brute force approach works on this puzzle.

See the MoreQueens puzzle for another (longer but clearer) equivalent definition of sat

See Wikipedia entry on
[Eight Queens puzzle](https://en.wikipedia.org/w/index.php?title=Eight_queens_puzzle).

**Problem:**

```python
def sat(squares: List[List[int]], m: int=8, n: int=8):
    assert type(squares) is list and all(type(a) is list and all(type(b) is int for b in a) for a in squares), 'squares must be of type List[List[int]]'
    k = min(m, n)
    assert all(i in range(m) and j in range(n) for i, j in squares) and len(squares) == k
    return 4 * k == len({t for i, j in squares for t in [('row', i), ('col', j), ('SE', i + j), ('NE', i - j)]})
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### MoreQueens ([chess](#chess) 2/5)

**Description:**
More Queens Puzzle

Position min(m, n) > 8 queens on an m x n chess board so that no pair is attacking each other. A brute force
approach will not work on many of these problems. Here, we use a different

See Wikipedia entry on
[Eight Queens puzzle](https://en.wikipedia.org/w/index.php?title=Eight_queens_puzzle).

**Problem:**

```python
def sat(squares: List[List[int]], m: int=9, n: int=9):
    assert type(squares) is list and all(type(a) is list and all(type(b) is int for b in a) for a in squares), 'squares must be of type List[List[int]]'
    k = min(m, n)
    assert all(i in range(m) and j in range(n) for i, j in squares), "queen off board"
    assert len(squares) == k, "Wrong number of queens"
    assert len({i for i, j in squares}) == k, "Queens on same row"
    assert len({j for i, j in squares}) == k, "Queens on same file"
    assert len({i + j for i, j in squares}) == k, "Queens on same SE diagonal"
    assert len({i - j for i, j in squares}) == k, "Queens on same NE diagonal"
    return True
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### KnightsTour ([chess](#chess) 3/5)

**Description:**
Knights Tour

Find an (open) tour of knight moves on an m x n chess-board that visits each square once.

See Wikipedia entry on [Knight's tour](https://en.wikipedia.org/w/index.php?title=Knight%27s_tour)

**Problem:**

```python
def sat(tour: List[List[int]], m: int=8, n: int=8):
    assert type(tour) is list and all(type(a) is list and all(type(b) is int for b in a) for a in tour), 'tour must be of type List[List[int]]'
    assert all({abs(i1 - i2), abs(j1 - j2)} == {1, 2} for [i1, j1], [i2, j2] in zip(tour, tour[1:])), 'legal moves'
    return sorted(tour) == [[i, j] for i in range(m) for j in range(n)]  # cover every square once
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(m=8, n=8):  # using Warnsdorff's heuristic, breaking ties randomly and restarting 10 times
    import random
    for seed in range(10):
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

### UncrossedKnightsPath ([chess](#chess) 4/5)

**Description:**
Uncrossed Knights Path (known solvable, but no solution given)

Find long (open) tour of knight moves on an m x n chess-board whose edges don't cross.
The goal of these problems is to match the nxn_records from [http://ukt.alex-black.ru/](http://ukt.alex-black.ru/)
(accessed 2020-11-29).

A more precise description is in this
[Wikipedia article](https://en.wikipedia.org/w/index.php?title=Longest_uncrossed_knight%27s_path).

**Problem:**

```python
def sat(path: List[List[int]], m: int=8, n: int=8, target: int=35):
    assert type(path) is list and all(type(a) is list and all(type(b) is int for b in a) for a in path), 'path must be of type List[List[int]]'
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
### UNSOLVED_UncrossedKnightsPath ([chess](#chess) 5/5)

**Description:**
Uncrossed Knights Path (open problem, unsolved)

Find long (open) tour of knight moves on an m x n chess-board whose edges don't cross.
The goal of these problems is to *beat* the nxn_records from
[http://ukt.alex-black.ru/](http://ukt.alex-black.ru/)
(accessed 2020-11-29).

A more precise description is in this
[Wikipedia article](https://en.wikipedia.org/w/index.php?title=Longest_uncrossed_knight%27s_path).

**Problem:**

```python
def sat(path: List[List[int]], m: int=8, n: int=8, target: int=35):
    assert type(path) is list and all(type(a) is list and all(type(b) is int for b in a) for a in path), 'path must be of type List[List[int]]'
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
[^^ Top](#files)
## codeforces

Problems inspired by problems available on [codeforces](https://codeforces.com)
ordered by the number of people who solved the problem on codeforces.

[^ Top](#files)

### CF4A ([codeforces](#codeforces) 1/16)

**Description:**
Determine if n can be evenly divided into two equal numbers. (Easy)

Inspired by [Watermelon problem](https://codeforces.com/problemset/problem/4/A)
(180k solved, 800 difficulty)

**Problem:**

```python
def sat(b: bool, n: int=10):
    assert type(b) is bool, 'b must be of type bool'
    i = 0
    while i <= n:
        if i + i == n:
            return b == True
        i += 1
    return b == False
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(n=10):
    return n % 2 == 0
```

</details>

### CF71A ([codeforces](#codeforces) 2/16)

**Description:**
Abbreviate strings longer than a given length

Inspired by https://codeforces.com/problemset/problem/71/A
(130k solved, 800 difficulty)

**Problem:**

```python
def sat(s: str, word: str="localization", max_len: int=10):
    assert type(s) is str, 's must be of type str'
    if len(word) <= max_len:
        return word == s
    return int(s[1:-1]) == len(word[1:-1]) and word[0] == s[0] and word[-1] == s[-1]
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(word="localization", max_len=10):
    if len(word) <= max_len:
        return word
    return f"{word[0]}{len(word) - 2}{word[-1]}"
```

</details>

### CF1A ([codeforces](#codeforces) 3/16)

**Description:**
Find a minimal list of corner locations for a×a tiles that covers [0, m] × [0, n] 
and does not double-cover squares.    
    
Sample Input:
m = 10
n = 9
a = 5
target = 4
    
Sample Output:
[[0, 0], [0, 5], [5, 0], [5, 5]]
    
Inspired by [Theater Square](https://codeforces.com/problemset/problem/1/A)
(125k solved, 1000 difficulty)

**Problem:**

```python
def sat(corners: List[List[int]], m: int=10, n: int=9, a: int=5, target: int=4):
    assert type(corners) is list and all(type(a) is list and all(type(b) is int for b in a) for a in corners), 'corners must be of type List[List[int]]'
    covered = {(i + x, j + y) for i, j in corners for x in range(a) for y in range(a)}
    assert len(covered) == len(corners) * a * a, "Double coverage"
    return len(corners) <= target and covered.issuperset({(x, y) for x in range(m) for y in range(n)})
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(m=10, n=9, a=5, target=4):
    return [[x, y] for x in range(0, m, a) for y in range(0, n, a)]
```

</details>

### CF231A ([codeforces](#codeforces) 4/16)

**Description:**
Inspired by [Team problem](https://codeforces.com/problemset/problem/231/A)
(102k solved, 800 difficulty)

**Problem:**

```python
def sat(lb: List[bool], solvable: List[List[int]]=[[1, 1, 0], [1, 1, 1], [1, 0, 0]]):
    assert type(lb) is list and all(type(a) is bool for a in lb), 'lb must be of type List[bool]'
    return len(lb) == len(solvable) and all(
        (b is True) if sum(s) >= 2 else (b is False) for b, s in zip(lb, solvable))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(solvable=[[1, 1, 0], [1, 1, 1], [1, 0, 0]]):
    return [sum(s) >= 2 for s in solvable]
```

</details>

### CF158A ([codeforces](#codeforces) 5/16)

**Description:**
Inspired by [Next Round](https://codeforces.com/problemset/problem/158/A)
(95k solved, 800 difficulty)

**Problem:**

```python
def sat(n: int, scores: List[int]=[10, 9, 8, 7, 7, 7, 5, 5], k: int=5):
    assert type(n) is int, 'n must be of type int'
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    return all(s >= scores[k] and s > 0 for s in scores[:n]) and all(s < scores[k] or s <= 0 for s in scores[n:])
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(scores=[10, 9, 8, 7, 7, 7, 5, 5], k=5):
    threshold = max(scores[k], 1)
    return sum(s >= threshold for s in scores)
```

</details>

### CF50A ([codeforces](#codeforces) 6/16)

**Description:**
Tile an m x n checkerboard with 2 x 1 tiles. The solution is a list of fourtuples [i1, j1, i2, j2]
with i2 == i1 and j2 == j1 + 1 or i2 == i1 + 1 and j2 == j1 with no overlap.

Inspired by Codeforce's [Domino Piling](https://codeforces.com/problemset/problem/50/A)
(86k solved, 800 difficulty)

**Problem:**

```python
def sat(squares: List[List[int]], m: int=10, n: int=5, target: int=50):
    assert type(squares) is list and all(type(a) is list and all(type(b) is int for b in a) for a in squares), 'squares must be of type List[List[int]]'
    covered = []
    for i1, j1, i2, j2 in squares:
        assert (0 <= i1 <= i2 < m) and (0 <= j1 <= j2 < n) and (j2 - j1 + i2 - i1 == 1)
        covered += [(i1, j1), (i2, j2)]
    return len(set(covered)) == len(covered) == target
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### CF282A ([codeforces](#codeforces) 7/16)

**Description:**
We make it a bit harder, though the problem is very straightforward. Given a sequence of operations "++x",
"x++", "--x", "x--", and a target value, find initial value so that the final value is the target value.

Sample Input:
ops = ["x++", "--x", "--x"]
target = 12

Sample Output:
13

Inspired by [Bit++ problem](https://codeforces.com/problemset/problem/282/A)
(83k solved, 800 difficulty)

**Problem:**

```python
def sat(n: int, ops: List[str]=['x++', '--x', '--x'], target: int=12):
    assert type(n) is int, 'n must be of type int'
    for op in ops:
        if op in ["++x", "x++"]:
            n += 1
        else:
            assert op in ["--x", "x--"]
            n -= 1
    return n == target
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(ops=['x++', '--x', '--x'], target=12):
    return target - ops.count("++x") - ops.count("x++") + ops.count("--x") + ops.count("x--")
```

</details>

### CF112A ([codeforces](#codeforces) 8/16)

**Description:**
Ignoring case, compare s, t lexicographically. Output 0 if they are =, -1 if s < t, 1 if s > t.

Inspired by [Petya and strings problem](https://codeforces.com/problemset/problem/112/A)
(80k solved, 800 difficulty)

**Problem:**

```python
def sat(n: int, s: str="aaAab", t: str="aAaaB"):
    assert type(n) is int, 'n must be of type int'
    if n == 0:
        return s.lower() == t.lower()
    if n == 1:
        return s.lower() > t.lower()
    if n == -1:
        return s.lower() < t.lower()
    return False
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(s="aaAab", t="aAaaB"):
    if s.lower() == t.lower():
        return 0
    if s.lower() > t.lower():
        return 1
    return -1
```

</details>

### CF263A ([codeforces](#codeforces) 9/16)

**Description:**
We are given a 5x5 bi with a single 1 like:
0 0 0 0 0
0 0 0 0 1
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0

Find a (minimal) sequence of row and column swaps to move the 1 to the center. A move is a string
in "0"-"4" indicating a row swap and "a"-"e" indicating a column swap

Inspired by [Beautiful Matrix](https://codeforces.com/problemset/problem/263/A)
(80k solved, 800 difficulty)

**Problem:**

```python
def sat(s: str, matrix: List[List[int]]=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], max_moves: int=3):
    assert type(s) is str, 's must be of type str'
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
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### CF339A ([codeforces](#codeforces) 10/16)

**Description:**
Sort numbers in a sum of digits, e.g., 1+3+2+1 -> 1+1+2+3

Inspired by [Helpful Maths](https://codeforces.com/problemset/problem/339/A)
(76k solved, 800 difficulty)

**Problem:**

```python
def sat(s: str, inp: str="1+1+3+1+3"):
    assert type(s) is str, 's must be of type str'
    return all(s.count(c) == inp.count(c) for c in inp + s) and all(s[i - 2] <= s[i] for i in range(2, len(s), 2))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(inp="1+1+3+1+3"):
    return "+".join(sorted(inp.split("+")))
```

</details>

### CF281A ([codeforces](#codeforces) 11/16)

**Description:**
Capitalize first letter of word

Inspired by [Word Capitalization](https://codeforces.com/problemset/problem/281/A)
(73k solved, 800 difficulty)

**Problem:**

```python
def sat(s: str, word: str="konjac"):
    assert type(s) is str, 's must be of type str'
    for i in range(len(s)):
        if i == 0:
            if s[i] != word[i].upper():
                return False
        else:
            if s[i] != word[i]:
                return False
    return True
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(word="konjac"):
    return word[0].upper() + word[1:]
```

</details>

### CF266A ([codeforces](#codeforces) 12/16)

**Description:**
You are given a string consisting of a's, b's and c's, find any longest substring containing no
repeated consecutive characters.

Sample Input:
abbbc

Sample Output:
abc

Inspired by [Stones on the Table](https://codeforces.com/problemset/problem/266/A)
(69k solved, 800 difficulty)

**Problem:**

```python
def sat(t: str, s: str="abbbcabbac", target: int=7):
    assert type(t) is str, 't must be of type str'
    i = 0
    for c in t:
        while c != s[i]:
            i += 1
        i += 1
    return len(t) >= target
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(s="abbbcabbac", target=7): # target is ignored
    return s[:1] + "".join([b for a, b in zip(s, s[1:]) if b != a])
```

</details>

### CF96A ([codeforces](#codeforces) 13/16)

**Description:**
You are given a string consisting of 0's and 1's. Find an index after which the subsequent k characters are
all 0's or all 1's.

Sample Input:
s = 0000111111100000, k = 5

Sample Output:
4
(or 5 or 6 or 11)

Inspired by [Football problem](https://codeforces.com/problemset/problem/96/A)
(67k solved 900 difficulty)

**Problem:**

```python
def sat(n: int, s: str="0000111111100000", k: int=5):
    assert type(n) is int, 'n must be of type int'
    return s[n:n + k] == s[n] * k
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(s="0000111111100000", k=5):
    return s.index("0" * k if "0" * k in s else "1" * k)
```

```python
def sol(s="0000111111100000", k=5):
    import re
    return re.search(r"([01])\1{" + str(k - 1) + "}", s).span()[0]
```

```python
def sol(s="0000111111100000", k=5):
    if "0" * k in s:
        return s.index("0" * k)
    else:
        return s.index("1" * k)
```

```python
def sol(s="0000111111100000", k=5):
    try:
        return s.index("0" * k)
    except:
        return s.index("1" * k)
```

</details>

### CF630A ([codeforces](#codeforces) 14/16)

**Description:**
Hundreds of 5^n

What are the last two digits of 5^n?

Inspired by Codeforce's [Twenty Five](https://codeforces.com/problemset/problem/630/A)
(21k solved, 800 difficulty)

**Problem:**

```python
def sat(s: str, n: int=7):
    assert type(s) is str, 's must be of type str'
    return int(str(5 ** n)[:-2] + s) == 5 ** n
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(n=7):
    return ("1" if n == 0 else "5" if n == 1 else "25")
```

</details>

### CF540A ([codeforces](#codeforces) 15/16)

**Description:**
Shortest Combination Lock Path

Given a starting a final lock position, find the (minimal) intermediate states, where each transition
involves increasing or decreasing a single digit (mod 10)
e.g.
start = "012"
combo = "329"

output: ['112', '212', '312', '322', '321', '320']

Inspired by [Combination Lock](https://codeforces.com/problemset/problem/540/A)
(21k solved, 800 difficulty)

**Problem:**

```python
def sat(states: List[str], start: str="012", combo: str="329", target_len: int=6):
    assert type(states) is list and all(type(a) is str for a in states), 'states must be of type List[str]'
    assert all(len(s) == len(start) for s in states) and all(c in "0123456789" for s in states for c in s)
    for a, b in zip([start] + states, states + [combo]):
        assert sum(i != j for i, j in zip(a, b)) == 1
        assert all(abs(int(i) - int(j)) in {0, 1, 9} for i, j in zip(a, b))

    return len(states) <= target_len
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(start="012", combo="329", target_len=6):
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

### CF540A_obfuscated ([codeforces](#codeforces) 16/16)

**Description:**
An obfuscated version of CombinationLock above

**Problem:**

```python
def sat(states: List[str], start: str="012", combo: str="329", target_len: int=6):
    assert type(states) is list and all(type(a) is str for a in states), 'states must be of type List[str]'
    assert all(len(s) == len(start) for s in states) and all(c in "0123456789" for s in states for c in s)
    for a, b in zip([start] + states, states + [combo]):
        assert sum(i != j for i, j in zip(a, b)) == 1
        assert all(abs(int(i) - int(j)) in {0, 1, 9} for i, j in zip(a, b))

    return len(states) <= target_len
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(start="012", combo="329", target_len=6):
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

[^^ Top](#files)
## game_theory


Hard problems from game theory.


[^ Top](#files)

### Nash ([game_theory](#game_theory) 1/2)

**Description:**
Compute a [Nash equilibrium](https://en.wikipedia.org/wiki/Nash_equilibrium) for a given
[bimatrix game](https://en.wikipedia.org/wiki/Bimatrix_game). While this problem was known to be
PPAD-hard in general. In fact the challenge is be much easier for an approximate
[eps-equilibrium](https://en.wikipedia.org/wiki/Epsilon-equilibrium) and of course for small games.

**Problem:**

```python
def sat(strategies: List[List[float]], A: List[List[float]]=[[-1.0, -3.0], [0.0, -2.0]], B: List[List[float]]=[[-1.0, 0.0], [-3.0, -2.0]], eps: float=0.01):
    assert type(strategies) is list and all(type(a) is list and all(type(b) is float for b in a) for a in strategies), 'strategies must be of type List[List[float]]'  # error tolerance
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
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(A=[[-1.0, -3.0], [0.0, -2.0]], B=[[-1.0, 0.0], [-3.0, -2.0]], eps=0.01):
    NUM_ATTEMPTS = 100

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

### ZeroSum ([game_theory](#game_theory) 2/2)

**Description:**
Compute minimax optimal strategies for a given
[zero-sum game](https://en.wikipedia.org/wiki/Zero-sum_game). This problem is known to be equivalent to
Linear Programming.

**Problem:**

```python
def sat(strategies: List[List[float]], A: List[List[float]]=[[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]], eps: float=0.1):
    assert type(strategies) is list and all(type(a) is list and all(type(b) is float for b in a) for a in strategies), 'strategies must be of type List[List[float]]'  # error tolerance
    m, n = len(A), len(A[0])
    p, q = strategies
    assert all(len(row) == n for row in A), "inputs are a matrix"
    assert len(p) == m and len(q) == n, "solution is a pair of strategies"
    assert sum(p) == sum(q) == 1.0 and min(p + q) >= 0.0, "strategies must be non-negative and sum to 1"
    v = sum(A[i][j] * p[i] * q[j] for i in range(m) for j in range(n))
    return (all(sum(A[i][j] * q[j] for j in range(n)) <= v + eps for i in range(m)) and
            all(sum(A[i][j] * p[i] for i in range(m)) >= v - eps for j in range(n)))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(A=[[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]], eps=0.1):
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

[^^ Top](#files)
## graphs

Problems related to graphs such as Conway's 99 problem, finding
[cliques](https://en.wikipedia.org/wiki/Clique_(graph_theory)) of various sizes, shortest path (Dijkstra) 

[^ Top](#files)

### Conway99 ([graphs](#graphs) 1/10)

**Description:**
Conway's 99-graph problem (*unsolved*, open problem)

Conway's 99-graph problem is an unsolved problem in graph theory. It asks whether there exists an
undirected graph with 99 vertices, in which each two adjacent vertices have exactly one common neighbor,
and in which each two non-adjacent vertices have exactly two common neighbors."
Or in Conway's terminology, from [Five $1,000 Problems (Update 2017)](https://oeis.org/A248380/a248380.pdf)
"Is there a graph with 99 vertices in which every edge (i.e. pair of joined vertices) belongs to a unique
triangle and every nonedge (pair of unjoined vertices) to a unique quadrilateral?"

See also this [Wikipedia article](https://en.wikipedia.org/w/index.php?title=Conway%27s_99-graph_problem).

**Problem:**

```python
def sat(edges: List[List[int]]):
    assert type(edges) is list and all(type(a) is list and all(type(b) is int for b in a) for a in edges), 'edges must be of type List[List[int]]'
    N = {i: {j for j in range(99) if j != i and ([i, j] in edges or [j, i] in edges)} for i in
         range(99)}  # neighbor sets
    return all(len(N[i].intersection(N[j])) == (1 if j in N[i] else 2) for i in range(99) for j in range(i))
```
### AnyEdge ([graphs](#graphs) 2/10)

**Description:**
Find any edge in a given [graph](https://en.wikipedia.org/w/index.php?title=Graph_(discrete_mathematics)).

**Problem:**

```python
def sat(e: List[int], edges: List[List[int]]=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]):
    assert type(e) is list and all(type(a) is int for a in e), 'e must be of type List[int]'
    return e in edges
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(edges=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]):
    return edges[0]
```

</details>

### AnyTriangle ([graphs](#graphs) 3/10)

**Description:**
Find a [triangle](https://en.wikipedia.org/w/index.php?title=Triangle_graph) in a given directed graph.

**Problem:**

```python
def sat(tri: List[int], edges: List[List[int]]=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [3, 1]]):
    assert type(tri) is list and all(type(a) is int for a in tri), 'tri must be of type List[int]'
    a, b, c = tri
    return [a, b] in edges and [b, c] in edges and [c, a] in edges and a != b != c != a
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(edges=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [3, 1]]):
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

### PlantedClique ([graphs](#graphs) 4/10)

**Description:**
Find a [planted clique](https://en.wikipedia.org/w/index.php?title=Planted_clique) of a given size
in an undirected graph.

**Problem:**

```python
def sat(nodes: List[int], size: int=3, edges: List[List[int]]=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]):
    assert type(nodes) is list and all(type(a) is int for a in nodes), 'nodes must be of type List[int]'
    assert len(nodes) == len(set(nodes)) >= size
    edge_set = {(a, b) for (a, b) in edges}
    for a in nodes:
        for b in nodes:
            assert a == b or (a, b) in edge_set or (b, a) in edge_set

    return True
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(size=3, edges=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]):  # brute force (finds list in increasing order), but with a tiny bit of speedup
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

### ShortestPath ([graphs](#graphs) 5/10)

**Description:**
Shortest Path

Find a path from node 0 to node 1, of a bounded length, in a given digraph on integer vertices.

See (Dijkstra's algorithm)[https://en.wikipedia.org/w/index.php?title=Dijkstra%27s_algorithm]

**Problem:**

```python
def sat(path: List[int], weights: List[Dict[int, int]]=[{1: 20, 2: 1}, {2: 2, 3: 5}, {1: 10}], bound: int=11):
    assert type(path) is list and all(type(a) is int for a in path), 'path must be of type List[int]'
    return path[0] == 0 and path[-1] == 1 and sum(weights[a][b] for a, b in zip(path, path[1:])) <= bound
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### UnweightedShortestPath ([graphs](#graphs) 6/10)

**Description:**
Unweighted Shortest Path

Find a path from node u to node v, of a bounded length, in a given digraph on vertices 0, 1,..., n.

See (Dijkstra's algorithm)[https://en.wikipedia.org/w/index.php?title=Dijkstra%27s_algorithm]

**Problem:**

```python
def sat(path: List[int], edges: List[List[int]]=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]], u: int=0, v: int=3, bound: int=3):
    assert type(path) is list and all(type(a) is int for a in path), 'path must be of type List[int]'
    assert path[0] == u and path[-1] == v and all([i, j] in edges for i, j in zip(path, path[1:]))
    return len(path) <= bound
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(edges=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]], u=0, v=3, bound=3):  # Dijkstra's algorithm
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

### AnyPath ([graphs](#graphs) 7/10)

**Description:**
Any Path

Find any path from node 0 to node n in a given graph on vertices 0, 1,..., n.

**Problem:**

```python
def sat(path: List[int], edges: List[List[int]]=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]):
    assert type(path) is list and all(type(a) is int for a in path), 'path must be of type List[int]'
    for i in range(len(path) - 1):
        assert [path[i], path[i + 1]] in edges
    assert path[0] == 0
    assert path[-1] == max(max(edge) for edge in edges)
    return True
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(edges=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]):
    n = max(max(edge) for edge in edges)
    paths = {0: [0]}
    for _ in range(n + 1):
        for i, j in edges:
            if i in paths and j not in paths:
                paths[j] = paths[i] + [j]
    return paths.get(n)
```

</details>

### EvenPath ([graphs](#graphs) 8/10)

**Description:**
Even Path

Find any path with an even number of nodes from node 0 to node n in a given graph on vertices 0, 1,..., n.

**Problem:**

```python
def sat(path: List[int], edges: List[List[int]]=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]):
    assert type(path) is list and all(type(a) is int for a in path), 'path must be of type List[int]'
    assert path[0] == 0 and path[-1] == max(max(e) for e in edges)
    assert all([[a, b] in edges for a, b in zip(path, path[1:])])
    return len(path) % 2 == 0
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(edges=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]):
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

### OddPath ([graphs](#graphs) 9/10)

**Description:**
Odd Path

*** Note the change to go from node 0 to node 1 ***

Find any path with an odd number of nodes from node 0 to node 1 in a given graph on vertices 0, 1,..., n.

**Problem:**

```python
def sat(p: List[int], edges: List[List[int]]=[[0, 1], [0, 2], [1, 2], [3, 1], [2, 3]]):
    assert type(p) is list and all(type(a) is int for a in p), 'p must be of type List[int]'
    return p[0] == 0 and p[-1] == 1 == len(p) % 2 and all([[a, b] in edges for a, b in zip(p, p[1:])])
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(edges=[[0, 1], [0, 2], [1, 2], [3, 1], [2, 3]]):
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

### Zarankiewicz ([graphs](#graphs) 10/10)

**Description:**
[Zarankiewicz problem](https://en.wikipedia.org/wiki/Zarankiewicz_problem)

Find a bipartite graph with 4 vertices on each side, 13 edges, and no K_3,3 subgraph.

**Problem:**

```python
def sat(edges: List[List[int]]):
    assert type(edges) is list and all(type(a) is list and all(type(b) is int for b in a) for a in edges), 'edges must be of type List[List[int]]'
    assert len(edges) == len({(a, b) for a, b in edges}) == 13  # weights
    assert all(i in range(4) for li in edges for i in li)  # 4 nodes on each side
    for i in range(4):
        v = [m for m in range(4) if m != i]
        for j in range(4):
            u = [m for m in range(4) if m != j]
            if all([m, n] in edges for m in v for n in u):
                return False
    return True
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return [[i, j] for i in range(4) for j in range(4) if i != j or i == 0]
```

</details>

[^^ Top](#files)
## ICPC


Problems inspired by the International Collegiate Programming Contest (ICPC).


[^ Top](#files)

### ICPC2019A ([ICPC](#icpc) 1/2)

**Description:**
ICPC 2019 Problem A

There are two rows of objects. Given the length-n integer arrays of prices and heights of objects in each
row, find a permutation of both rows so that the permuted prices are non-decreasing in each row and
so that the first row is taller than the second row.

See ICPC 2019 Problem A:
[Azulejos](https://icpc.global/newcms/worldfinals/problems/2019%20ACM-ICPC%20World%20Finals/icpc2019.pdf).

**Problem:**

```python
def sat(perms: List[List[int]], prices0: List[int]=[3, 2, 1, 2], prices1: List[int]=[2, 1, 2, 1], heights0: List[int]=[2, 3, 4, 3], heights1: List[int]=[2, 2, 1, 3]):
    assert type(perms) is list and all(type(a) is list and all(type(b) is int for b in a) for a in perms), 'perms must be of type List[List[int]]'
    n = len(prices0)
    perm0, perm1 = perms
    assert sorted(perm0) == sorted(perm1) == list(range(n)), "Solution must be two permutations"
    for i in range(n - 1):
        assert prices0[perm0[i]] <= prices0[perm0[i + 1]], "Permuted prices must be nondecreasing (row 0)"
        assert prices1[perm1[i]] <= prices1[perm1[i + 1]], "Permuted prices must be nondecreasing (row 1)"
    return all(heights0[i] > heights1[j] for i, j in zip(perm0, perm1))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(prices0=[3, 2, 1, 2], prices1=[2, 1, 2, 1], heights0=[2, 3, 4, 3], heights1=[2, 2, 1, 3]):
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

### ICPC2019B ([ICPC](#icpc) 2/2)

**Description:**
ICPC 2019 Problem B

This problem requires choosing the locations of a sequence of connecting bridges to minimize cost.

See ICPC 2019 Problem B:
[Bridges](https://icpc.global/newcms/worldfinals/problems/2019%20ACM-ICPC%20World%20Finals/icpc2019.pdf)

**Problem:**

```python
def sat(indices: List[int], H: int=60, alpha: int=18, beta: int=2, xs: List[int]=[0, 20, 30, 50, 70], ys: List[int]=[0, 20, 10, 30, 20], thresh: int=6460):
    assert type(indices) is list and all(type(a) is int for a in indices), 'indices must be of type List[int]'
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
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(H=60, alpha=18, beta=2, xs=[0, 20, 30, 50, 70], ys=[0, 20, 10, 30, 20], thresh=6460): # thresh is ignored
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

[^^ Top](#files)
## IMO

Adapted from the
[International Mathematical Olympiad](https://en.wikipedia.org/wiki/International_Mathematical_Olympiad)
[problems](https://www.imo-official.org/problems.aspx)

[^ Top](#files)

### IMO_2010_5 ([IMO](#imo) 1/6)

**Description:**
This problem has *long* solutions.

There are five boxes each having one coin initially. Two types of moves are allowed:
* (advance) remove `k > 0` coins from box `i` and add `2k` coins to box `i + 1`
* (swap) remove a coin from box `i` and swap the contents of boxes `i+1` and `i+2`
Given `0 <= n <= 16385`, find a sequence of states that result in 2^n coins in the last box.
Note that `n` can be as large as 2^14+1 yielding 2^(2^14+1) coins (a number with 4,933 digits) in the last
box. Encode each state as a list of the numbers of coins in the five boxes.

Sample Input:
`n = 2`

Sample Output:
`[[1, 1, 1, 1, 1], [0, 3, 1, 1, 1], [0, 1, 5, 1, 1], [0, 1, 4, 1, 1], [0, 0, 1, 4, 1], [0, 0, 0, 1, 4]]`

The last box now has 2^2 coins. This is a sequence of two advances followed by three swaps.

The version above uses only 5 boxes (unlike the IMO problem with 6 boxes since 2010^2010^2010 is too big
for computers) but the solution is quite similar to the solution to the IMO problem. Because the solution
requires exponential many moves, our representation allows combining multiple Type-1 (advance) operations
into a single step.

Based on [IMO 2010 Problem 5](https://www.imo-official.org/problems.aspx)

**Problem:**

```python
def sat(states: List[List[int]], n: int=10):
    assert type(states) is list and all(type(a) is list and all(type(b) is int for b in a) for a in states), 'states must be of type List[List[int]]'  # list of 5-tuple states
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
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(n=10):
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

### IMO_2016_4 ([IMO](#imo) 2/6)

**Description:**
Let P(n) = n^2 + n + 1.

Given b>=6 and m>=1, find m non-negative integers for which the set {P(a+1), P(a+2), ..., P(a+b)} has
the property that there is no element that is relatively prime to every other element. (Q: Is there a more
efficient solution than the brute-force one we give, perhaps using the Chinese remainder theorem?)

Sample input:
b = 6
m = 2

Sample output:
[195, 196]

Based on [IMO 2016 Problem 4](https://www.imo-official.org/problems.aspx)

**Problem:**

```python
def sat(nums: List[int], b: int=6, m: int=2):
    assert type(nums) is list and all(type(a) is int for a in nums), 'nums must be of type List[int]'
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
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(b=6, m=2):
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

### IMO_2017_1 ([IMO](#imo) 3/6)

**Description:**
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

Note: This problem is much easier than the IMO problem which also required a proof that it is impossible
for a_0 not divisible by 3.

Based on [IMO 2017 Problem 1](https://www.imo-official.org/problems.aspx)

**Problem:**

```python
def sat(indices: List[int], a0: int=123):
    assert type(indices) is list and all(type(a) is int for a in indices), 'indices must be of type List[int]'
    assert a0 >= 0 and a0 % 3 == 0, "Hint: a_0 is a multiple of 3."
    s = [a0]
    for i in range(max(indices)):
        s.append(int(s[-1] ** 0.5) if int(s[-1] ** 0.5) ** 2 == s[-1] else s[-1] + 3)
    return len(indices) == len(set(indices)) == 1000 and min(indices) >= 0 and len({s[i] for i in indices}) == 1
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### IMO_2017_5 ([IMO](#imo) 4/6)

**Description:**
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

The solution encodes the judge's solution.
Based on [IMO 2017 Problem 5](https://www.imo-official.org/problems.aspx)

**Problem:**

```python
def sat(keep: List[bool], heights: List[int]=[4, 0, 5, 3, 1, 2]):
    assert type(keep) is list and all(type(a) is bool for a in keep), 'keep must be of type List[bool]'
    n = int(len(heights) ** 0.5)
    assert sorted(heights) == list(range(n * n + n)), "hint: heights is a permutation of range(n * n + n)"
    kept = [i for i, k in zip(heights, keep) if k]
    assert len(kept) == 2 * n, "must keep 2n items"
    pi = sorted(range(2 * n), key=lambda i: kept[i])  # the sort indices
    return all(abs(pi[2 * i] - pi[2 * i + 1]) == 1 for i in range(n))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(heights=[4, 0, 5, 3, 1, 2]):
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

### IMO_2018_2 ([IMO](#imo) 5/6)

**Description:**
Given n, find n integers such that li[i] * li[i+1] + 1 == li[i+2], for i = 0, 1, ..., n-1
where indices >= n "wrap around". Note: only n multiples of 3 are given since this is only possible for n
that are multiples of 3 (as proven in the IMO problem).

Sample input:
6

Sample output:
[_, _, _, _, _, _]

(Sample output hidden because showing sample output would give away too much information.)

Note: This problem is easier than the IMO problem because the hard part is proving that sequences do not
exists for non-multiples of 3.

Based on [IMO 2010 Problem 5](https://www.imo-official.org/problems.aspx)

**Problem:**

```python
def sat(li: List[int], n: int=6):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    assert n % 3 == 0, "Hint: n is a multiple of 3"
    return len(li) == n and all(li[(i + 2) % n] == 1 + li[(i + 1) % n] * li[i] for i in range(n))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(n=6):
    return [-1, -1, 2] * (n // 3)
```

</details>

### IMO_2020_3 ([IMO](#imo) 6/6)

**Description:**
The input colors is a list of 4n colors each in range(n) with each color occurring 4 times.
The goal is to find a subset (list) li of half the indices such that:
* The sum of the indices equals the sum of the sum of the missing indices.
* The colors of the chosen indices contains exactly each number in range(n) twice.

Sample input:
n = 3
colors = [0, 1, 2, 0, 0, 1, 1, 1, 2, 2, 0, 2]

Sample output:
[0, 3, 5, 6, 8, 11]

Note the sum of the output is 33 = (0+1+2+...+11)/2 and the selected colors are [0, 0, 1, 1, 2, 2]

Based on [IMO 2020 Problem 3](https://www.imo-official.org/problems.aspx)

**Problem:**

```python
def sat(li: List[int], n: int=3, colors: List[int]=[0, 1, 2, 0, 0, 1, 1, 1, 2, 2, 0, 2]):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    assert sorted(colors) == sorted(list(range(n)) * 4), "hint: each color occurs exactly four times"
    assert len(li) == len(set(li)) and min(li) >= 0
    return sum(li) * 2 == sum(range(4 * n)) and sorted([colors[i] for i in li]) == [i // 2 for i in range(2 * n)]
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(n=3, colors=[0, 1, 2, 0, 0, 1, 1, 1, 2, 2, 0, 2]):
    pairs = {(i, 4 * n - i - 1) for i in range(2 * n)}
    by_color = {color: [] for color in range(n)}
    for p in pairs:
        a, b = [colors[i] for i in p]
        by_color[a].append(p)
        by_color[b].append(p)
    cycles = []
    cycle = []
    while pairs:
        if not cycle:  # start new cycle
            p = pairs.pop()
            pairs.add(p)  # just to pick a color
            color = colors[p[0]]
            # print("Starting cycle with color", color)
        p = by_color[color].pop()
        a, b = [colors[i] for i in p]
        # print(p, a, b)
        color = a if a != color else b
        by_color[color].remove(p)
        cycle.append(p if color == b else p[::-1])
        pairs.remove(p)
        if not by_color[color]:
            cycles.append(cycle)
            cycle = []

    while any(len(c) % 2 for c in cycles):
        cycle_colors = [{colors[k] for p in c for k in p} for c in cycles]
        merged = False
        for i in range(len(cycles)):
            for j in range(i):
                intersection = cycle_colors[i].intersection(cycle_colors[j])
                if intersection:
                    c = intersection.pop()
                    # print(f"Merging cycle {i} and cycle {j} at color {c}", cycles)
                    cycle_i = cycles.pop(i)
                    for i1, p in enumerate(cycle_i):
                        if colors[p[0]] == c:
                            break
                    for j1, p in enumerate(cycles[j]):
                        if colors[p[0]] == c:
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

[^^ Top](#files)
## puzzles

Classic puzzles


[^ Top](#files)

### TowersOfHanoi ([puzzles](#puzzles) 1/18)

**Description:**
[Towers of Hanoi](https://en.wikipedia.org/w/index.php?title=Tower_of_Hanoi)

In this classic version one must move all 8 disks from the first to third peg.

**Problem:**

```python
def sat(moves: List[List[int]]):
    assert type(moves) is list and all(type(a) is list and all(type(b) is int for b in a) for a in moves), 'moves must be of type List[List[int]]'  # moves is list of [from, to] pairs
    rods = ([8, 7, 6, 5, 4, 3, 2, 1], [], [])
    for [i, j] in moves:
        rods[j].append(rods[i].pop())
        assert rods[j][-1] == min(rods[j]), "larger disk on top of smaller disk"
    return rods[0] == rods[1] == []
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### TowersOfHanoiArbitrary ([puzzles](#puzzles) 2/18)

**Description:**
[Towers of Hanoi](https://en.wikipedia.org/w/index.php?title=Tower_of_Hanoi)

In this version one must transform a given source state to a target state.

**Problem:**

```python
def sat(moves: List[List[int]], source: List[List[int]]=[[0, 7], [4, 5, 6], [1, 2, 3, 8]], target: List[List[int]]=[[0, 1, 2, 3, 8], [4, 5], [6, 7]]):
    assert type(moves) is list and all(type(a) is list and all(type(b) is int for b in a) for a in moves), 'moves must be of type List[List[int]]'
    state = [s[:] for s in source]

    for [i, j] in moves:
        state[j].append(state[i].pop())
        assert state[j] == sorted(state[j])

    return state == target
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### Quine ([puzzles](#puzzles) 3/18)

**Description:**
[Quine](https://en.wikipedia.org/wiki/Quine_%28computing%29)

Find an expression whose evaluation is the same as itself.

**Problem:**

```python
def sat(quine: str):
    assert type(quine) is str, 'quine must be of type str'
    return eval(quine) == quine
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return "(lambda x: f'({x})({chr(34)}{x}{chr(34)})')(\"lambda x: f'({x})({chr(34)}{x}{chr(34)})'\")"
```

</details>

### BooleanPythagoreanTriples ([puzzles](#puzzles) 4/18)

**Description:**
[Boolean Pythagorean Triples Problem](https://en.wikipedia.org/wiki/Boolean_Pythagorean_triples_problem)
    
Color the first n integers with one of two colors so that there is no monochromatic Pythagorean triple.

**Problem:**

```python
def sat(colors: List[int], n: int=100):
    assert type(colors) is list and all(type(a) is int for a in colors), 'colors must be of type List[int]'  # list of 0/1 colors of length >= n
    assert set(colors) <= {0, 1} and len(colors) >= n
    squares = {i ** 2: colors[i] for i in range(1, len(colors))}
    return not any(c == d == squares.get(i + j) for i, c in squares.items() for j, d in squares.items())
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### ClockAngle ([puzzles](#puzzles) 5/18)

**Description:**
[Clock Angle Problem](https://en.wikipedia.org/wiki/Clock_angle_problem)

Easy variant checks if angle at li = [hour, min] is a given number of degrees.

**Problem:**

```python
def sat(hands: List[int], target_angle: int=45):
    assert type(hands) is list and all(type(a) is int for a in hands), 'hands must be of type List[int]'
    hour, min = hands
    return hour in range(1, 13) and min in range(60) and ((60 * hour + min) - 12 * min) % 720 == 2 * target_angle
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(target_angle=45):
    for hour in range(1, 13):
        for min in range(60):
            if ((60 * hour + min) - 12 * min) % 720 == 2 * target_angle:
                return [hour, min]
```

</details>

### Kirkman ([puzzles](#puzzles) 6/18)

**Description:**
[Kirkman's problem](https://en.wikipedia.org/wiki/Kirkman%27s_schoolgirl_problem)

Arrange 15 people into groups of 3 each day for seven days so that no two people are in the same group twice.

**Problem:**

```python
def sat(daygroups: List[List[List[int]]]):
    assert type(daygroups) is list and all(type(a) is list and all(type(b) is list and all(type(c) is int for c in b) for b in a) for a in daygroups), 'daygroups must be of type List[List[List[int]]]'
    assert len(daygroups) == 7
    assert all(len(groups) == 5 and {i for g in groups for i in g} == set(range(15)) for groups in daygroups)
    assert all(len(g) == 3 for groups in daygroups for g in groups)
    return len({(i, j) for groups in daygroups for g in groups for i in g for j in g}) == 15 * 15
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### MonkeyAndCoconuts ([puzzles](#puzzles) 7/18)

**Description:**
[The Monkey and the Coconuts](https://en.wikipedia.org/wiki/The_monkey_and_the_coconuts)
    
Find the number of coconuts to solve the following riddle quoted from
[Wikipedia article](https://en.wikipedia.org/wiki/The_monkey_and_the_coconuts):
    There is a pile of coconuts, owned by five men.
    One man divides the pile into five equal piles, giving the one left over coconut to a passing monkey,
    and takes away his own share. The second man then repeats the procedure, dividing the remaining pile
    into five and taking away his share, as do the third, fourth, and fifth, each of them finding one
    coconut left over when dividing the pile by five, and giving it to a monkey. Finally, the group
     divide the remaining coconuts into five equal piles: this time no coconuts are left over.
    How many coconuts were there in the original pile?

**Problem:**

```python
def sat(n: int):
    assert type(n) is int, 'n must be of type int'
    for i in range(5):
        assert n % 5 == 1
        n -= 1 + (n - 1) // 5
    return n > 0 and n % 5 == 1
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### No3Colinear ([puzzles](#puzzles) 8/18)

**Description:**
[No three-in-a-line](https://en.wikipedia.org/wiki/No-three-in-line_problem)

Find `num_points` points in an `side x side` grid such that no three points are collinear.

**Problem:**

```python
def sat(coords: List[List[int]], side: int=5, num_points: int=10):
    assert type(coords) is list and all(type(a) is list and all(type(b) is int for b in a) for a in coords), 'coords must be of type List[List[int]]'
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
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(side=5, num_points=10):
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

### PostageStamp ([puzzles](#puzzles) 9/18)

**Description:**
[Postage stamp problem](https://en.wikipedia.org/wiki/Postage_stamp_problem)

In this problem version, one must find a selection of stamps to achieve a given value.

**Problem:**

```python
def sat(stamps: List[int], target: int=80, max_stamps: int=4, options: List[int]=[10, 32, 8]):
    assert type(stamps) is list and all(type(a) is int for a in stamps), 'stamps must be of type List[int]'
    return set(stamps) <= set(options) and len(stamps) <= max_stamps and sum(stamps) == target
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(target=80, max_stamps=4, options=[10, 32, 8]):
    from itertools import combinations_with_replacement
    for n in range(max_stamps + 1):
        for c in combinations_with_replacement(options, n):
            if sum(c) == target:
                return list(c)
```

</details>

### SquaringTheSquare ([puzzles](#puzzles) 10/18)

**Description:**
[Squaring the square](https://en.wikipedia.org/wiki/Squaring_the_square)
Partition a square into smaller squares with unique side lengths. A perfect squared path has distinct sides.

Wikipedia gives a minimal [solution with 21 squares](https://en.wikipedia.org/wiki/Squaring_the_square)
due to Duijvestijn (1978):
```python
[[0, 0, 50], [0, 50, 29], [0, 79, 33], [29, 50, 25], [29, 75, 4], [33, 75, 37], [50, 0, 35],
 [50, 35, 15], [54, 50, 9], [54, 59, 16], [63, 50, 2], [63, 52, 7], [65, 35, 17], [70, 52, 18],
 [70, 70, 42], [82, 35, 11], [82, 46, 6], [85, 0, 27], [85, 27, 8], [88, 46, 24], [93, 27, 19]]
```

**Problem:**

```python
def sat(xy_sides: List[List[int]]):
    assert type(xy_sides) is list and all(type(a) is list and all(type(b) is int for b in a) for a in xy_sides), 'xy_sides must be of type List[List[int]]'  # List of (x, y, side)
    n = max(x + side for x, y, side in xy_sides)
    assert len({side for x, y, side in xy_sides}) == len(xy_sides) > 1
    for x, y, s in xy_sides:
        assert 0 <= y < y + s <= n and 0 <= x
        for x2, y2, s2 in xy_sides:
            assert s2 <= s or x2 >= x + s or x2 + s2 <= x or y2 >= y + s or y2 + s2 <= y

    return sum(side ** 2 for x, y, side in xy_sides) == n ** 2
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return [[0, 0, 50], [0, 50, 29], [0, 79, 33], [29, 50, 25], [29, 75, 4], [33, 75, 37], [50, 0, 35],
            [50, 35, 15], [54, 50, 9], [54, 59, 16], [63, 50, 2], [63, 52, 7], [65, 35, 17], [70, 52, 18],
            [70, 70, 42], [82, 35, 11], [82, 46, 6], [85, 0, 27], [85, 27, 8], [88, 46, 24], [93, 27, 19]]
```

</details>

### NecklaceSplit ([puzzles](#puzzles) 11/18)

**Description:**
[Necklace Splitting Problem](https://en.wikipedia.org/wiki/Necklace_splitting_problem)

Split a specific red/blue necklace in half at n so that each piece has an equal number of reds and blues.

**Problem:**

```python
def sat(n: int, lace: str="rrbbbbrrbrbrbbrr"):
    assert type(n) is int, 'n must be of type int'
    sub = lace[n: n + len(lace) // 2]
    return n >= 0 and lace.count("r") == 2 * sub.count("r") and lace.count("b") == 2 * sub.count("b")
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(lace="rrbbbbrrbrbrbbrr"):
    if lace == "":
        return 0
    return next(n for n in range(len(lace) // 2) if lace[n: n + len(lace) // 2].count("r") == len(lace) // 4)
```

</details>

### PandigitalSquare ([puzzles](#puzzles) 12/18)

**Description:**
[Pandigital](https://en.wikipedia.org/wiki/Pandigital_number) Square

Find an integer whose square has all digits 0-9 once.

**Problem:**

```python
def sat(n: int):
    assert type(n) is int, 'n must be of type int'
    return sorted([int(s) for s in str(n * n)]) == list(range(10))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    for n in range(10 ** 5):
        if sorted([int(s) for s in str(n * n)]) == list(range(10)):
            return n
```

</details>

### AllPandigitalSquares ([puzzles](#puzzles) 13/18)

**Description:**
All [Pandigital](https://en.wikipedia.org/wiki/Pandigital_number) Squares

Find all 174 integers whose 10-digit square has all digits 0-9

**Problem:**

```python
def sat(nums: List[int]):
    assert type(nums) is list and all(type(a) is int for a in nums), 'nums must be of type List[int]'
    return [sorted([int(s) for s in str(n * n)]) for n in set(nums)] == [list(range(10))] * 174
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return [i for i in range(-10 ** 5, 10 ** 5) if sorted([int(s) for s in str(i * i)]) == list(range(10))]
```

</details>

### CardGame24 ([puzzles](#puzzles) 14/18)

**Description:**
[24 Game](https://en.wikipedia.org/wiki/24_Game)

In this game one is given four numbers from the range 1-13 (Ace-King) and one needs to combine them
with + - * / (and parentheses) to make the number 24.

**Problem:**

```python
def sat(expr: str, nums: List[int]=[3, 7, 3, 7]):
    assert type(expr) is str, 'expr must be of type str'
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
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### Easy63 ([puzzles](#puzzles) 15/18)

**Description:**
An easy puzzle to make 63 using two 8's and one 1's.

**Problem:**

```python
def sat(s: str):
    assert type(s) is str, 's must be of type str'
    return set(s) <= set("18-+*/") and s.count("8") == 2 and s.count("1") == 1 and eval(s) == 63
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return "8*8-1"
```

</details>

### Harder63 ([puzzles](#puzzles) 16/18)

**Description:**
An harder puzzle to make 63 using two 8's and two 1's.

**Problem:**

```python
def sat(s: str):
    assert type(s) is str, 's must be of type str'
    return set(s) <= set("18-+*/") and s.count("8") == 3 and s.count("1") == 1 and eval(s) == 63
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return "8*8-1**8"
```

</details>

### WaterPouring ([puzzles](#puzzles) 17/18)

**Description:**
[Water pouring puzzle](https://en.wikipedia.org/w/index.php?title=Water_pouring_puzzle&oldid=985741928)

Given an initial state of water quantities in jugs and jug capacities, find a sequence of moves (pouring
one jug into another until it is full or the first is empty) to reaches the given goal state.

**Problem:**

```python
def sat(moves: List[List[int]], capacities: List[int]=[8, 5, 3], init: List[int]=[8, 0, 0], goal: List[int]=[4, 4, 0]):
    assert type(moves) is list and all(type(a) is list and all(type(b) is int for b in a) for a in moves), 'moves must be of type List[List[int]]'  # moves is list of [from, to] pairs
    state = init.copy()

    for [i, j] in moves:
        assert min(i, j) >= 0, "Indices must be non-negative"
        assert i != j, "Cannot pour from same state to itself"
        n = min(capacities[j], state[i] + state[j])
        state[i], state[j] = state[i] + state[j] - n, n

    return state == goal
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### VerbalArithmetic ([puzzles](#puzzles) 18/18)

**Description:**
Find a substitution of digits for characters to make the numbers add up, like this:
SEND + MORE = MONEY
9567 + 1085 = 10652

The first digit in any cannot be 0.
See [Wikipedia article](https://en.wikipedia.org/wiki/Verbal_arithmetic)

**Problem:**

```python
def sat(li: List[int], words: List[str]=['SEND', 'MORE', 'MONEY']):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    assert len(li) == len(words) and all(i > 0 and len(str(i)) == len(w) for i, w in zip(li, words))
    assert len({c for w in words for c in w}) == len({(d, c) for i, w in zip(li, words) for d, c in zip(str(i), w)})
    return sum(li[:-1]) == li[-1]
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

[^^ Top](#files)
