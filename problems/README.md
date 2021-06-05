# Python Programming Puzzles: dataset summary
This document summarizes the dataset stored in .json files.
Each .json file contains a number of related problems with one or more puzzles each.

## Files:

- [algebra (4 problems, 4,000 instances)](#algebra)
- [basic (21 problems, 21,000 instances)](#basic)
- [chess (5 problems, 4,855 instances)](#chess)
- [classic_puzzles (22 problems, 11,370 instances)](#classic_puzzles)
- [codeforces (24 problems, 23,025 instances)](#codeforces)
- [compression (3 problems, 3,000 instances)](#compression)
- [conways_game_of_life (2 problems, 2,000 instances)](#conways_game_of_life)
- [games (5 problems, 1,006 instances)](#games)
- [game_theory (2 problems, 2,000 instances)](#game_theory)
- [graphs (11 problems, 9,002 instances)](#graphs)
- [ICPC (3 problems, 3,000 instances)](#icpc)
- [IMO (6 problems, 5,012 instances)](#imo)
- [lattices (2 problems, 2,000 instances)](#lattices)
- [number_theory (16 problems, 10,762 instances)](#number_theory)
- [probability (5 problems, 5,000 instances)](#probability)
- [study (30 problems, 30 instances)](#study)
- [trivial_inverse (34 problems, 32,002 instances)](#trivial_inverse)
- [tutorial (5 problems, 5 instances)](#tutorial)

Total (200 problems, 139,069 instances)


----

## algebra

Roots of polynomials

[^ Top](#files)

### QuadraticRoot
([algebra](#algebra) 1/4)

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

### AllQuadraticRoots
([algebra](#algebra) 2/4)

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

### CubicRoot
([algebra](#algebra) 3/4)

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

### AllCubicRoots
([algebra](#algebra) 4/4)

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
            if w != 0.0:
                x = complex(w - p / (3 * w) - a / 3).real
                if abs(x ** 3 + a * x ** 2 + b * x + c) < 1e-4:
                    if not ans or min(abs(z - x) for z in ans) > 1e-6:
                        ans.append(x)
    if len(ans) == 3:
        return ans
```

</details>

[^^ Top](#files)
## basic

Problems testing basic knowledge -- easy to solve if you understand what is being asked

[^ Top](#files)

### SumOfDigits
([basic](#basic) 1/21)

**Description:**
Find a number that its digits sum to a specific value.

**Problem:**

```python
def sat(x: str, s: int=679):
    assert type(x) is str, 'x must be of type str'
    return s == sum([int(d) for d in x])
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(s=679):
    return int(s/9) * '9' + str(s%9)
```

</details>

### FloatWithDecimalValue
([basic](#basic) 2/21)

**Description:**
Create a float with a specific decimal.

**Problem:**

```python
def sat(z: float, v: int=9, d: float=0.0001):
    assert type(z) is float, 'z must be of type float'
    return int(z * 1/d % 10) == v
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(v=9, d=0.0001):
    return v * d
```

</details>

### ArithmeticSequence
([basic](#basic) 3/21)

**Description:**
Create a list that is a subrange of an arithmetic sequence.

**Problem:**

```python
def sat(x: List[int], a: int=7, s: int=5, e: int=200):
    assert type(x) is list and all(type(a) is int for a in x), 'x must be of type List[int]'
    return x[0] == a and x[-1] <= e and (x[-1] + s > e) and all([x[i] + s == x[i+1] for i in range(len(x)-1)])
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=7, s=5, e=200):
    return list(range(a,e+1,s))
```

</details>

### GeometricSequence
([basic](#basic) 4/21)

**Description:**
Create a list that is a subrange of an gemoetric sequence.

**Problem:**

```python
def sat(x: List[int], a: int=8, r: int=2, l: int=50):
    assert type(x) is list and all(type(a) is int for a in x), 'x must be of type List[int]'
    return x[0] == a and len(x) == l and all([x[i] * r == x[i+1] for i in range(len(x)-1)])
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=8, r=2, l=50):
    return [a*r**i for i in range(l)]
```

</details>

### LineIntersection
([basic](#basic) 5/21)

**Description:**
Find the intersection of two lines.
Solution should be a list of the (x,y) coordinates.
Accuracy of fifth decimal digit is required.

**Problem:**

```python
def sat(e: List[int], a: int=2, b: int=-1, c: int=1, d: int=2021):
    assert type(e) is list and all(type(a) is int for a in e), 'e must be of type List[int]'
    x = e[0] / e[1]
    return abs(a * x + b - c * x - d) < 10 ** -5
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=2, b=-1, c=1, d=2021):
    return [d - b, a - c]
```

</details>

### IfProblem
([basic](#basic) 6/21)

**Description:**
Simple if statement

**Problem:**

```python
def sat(x: int, a: int=324554, b: int=1345345):
    assert type(x) is int, 'x must be of type int'
    if a < 50:
        return x + a == b
    else:
        return x - 2 * a == b
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=324554, b=1345345):
    if a < 50:
        return b - a
    else:
        return b + 2 * a
```

</details>

### IfProblemWithAnd
([basic](#basic) 7/21)

**Description:**
Simple if statement with and clause

**Problem:**

```python
def sat(x: int, a: int=9384594, b: int=1343663):
    assert type(x) is int, 'x must be of type int'
    if x > 0 and a > 50:
        return x - a == b
    else:
        return x + a == b
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=9384594, b=1343663):
    if a > 50 and b > a:
        return b + a
    else:
        return b - a
```

</details>

### IfProblemWithOr
([basic](#basic) 8/21)

**Description:**
Simple if statement with or clause

**Problem:**

```python
def sat(x: int, a: int=253532, b: int=1230200):
    assert type(x) is int, 'x must be of type int'
    if x > 0 or a > 50:
        return x - a == b
    else:
        return x + a == b
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=253532, b=1230200):
    if a > 50 or b > a:
        return b + a
    else:
        return b - a
```

</details>

### IfCases
([basic](#basic) 9/21)

**Description:**
Simple if statement with multiple cases

**Problem:**

```python
def sat(x: int, a: int=4, b: int=54368639):
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

### ListPosSum
([basic](#basic) 10/21)

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

### ListDistinctSum
([basic](#basic) 11/21)

**Description:**
Construct a list of distinct integers that sum up to some value

**Problem:**

```python
def sat(x: List[int], n: int=4, s: int=2021):
    assert type(x) is list and all(type(a) is int for a in x), 'x must be of type List[int]'
    return len(x) == n and sum(x) == s and len(set(x)) == n
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### ConcatStrings
([basic](#basic) 12/21)

**Description:**
Concatenate list of characters

**Problem:**

```python
def sat(x: str, s: List[str]=['a', 'b', 'c', 'd', 'e', 'f'], n: int=4):
    assert type(x) is str, 'x must be of type str'
    return len(x) == n and all([x[i] == s[i] for i in range(n)])
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(s=['a', 'b', 'c', 'd', 'e', 'f'], n=4):
    return ''.join([s[i] for i in range(n)])
```

</details>

### SublistSum
([basic](#basic) 13/21)

**Description:**
Sum values of sublist by range specifications

**Problem:**

```python
def sat(x: List[int], t: int=677, a: int=43, e: int=125, s: int=10):
    assert type(x) is list and all(type(a) is int for a in x), 'x must be of type List[int]'
    non_zero = [z for z in x if z != 0]
    return t == sum([x[i] for i in range(a, e, s)]) and len(set(non_zero)) == len(non_zero) and all(
        [x[i] != 0 for i in range(a, e, s)])
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### CumulativeSum
([basic](#basic) 14/21)

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

### BasicStrCounts
([basic](#basic) 15/21)

**Description:**
Find a string that has `count1` occurrences of `s1` and `count1` occurrences of `s1` and starts and ends with
the same 10 characters

**Problem:**

```python
def sat(s: str, s1: str="a", s2: str="b", count1: int=50, count2: int=30):
    assert type(s) is str, 's must be of type str'
    return s.count(s1) == count1 and s.count(s2) == count2 and s[:10] == s[-10:]
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(s1="a", s2="b", count1=50, count2=30):
    if s1==s2:
        ans = (s1 + "?") * count1
    elif s1.count(s2):
        ans = (s1 + "?") * count1
        ans += (s2 + "?") * (count2 - ans.count(s2))
    else:
        ans = (s2 + "?") * count2
        ans += (s1 + "?") * (count1 - ans.count(s1))
    return "?"*10 + ans + "?"*10
```

</details>

### ZipStr
([basic](#basic) 16/21)

**Description:**
Find a string that contains all the `substrings` alternating, e.g., 'cdaotg' for 'cat' and 'dog'

**Problem:**

```python
def sat(s: str, substrings: List[str]=['foo', 'bar', 'baz']):
    assert type(s) is str, 's must be of type str'
    return all(sub in s[i::len(substrings)] for i, sub in enumerate(substrings))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(substrings=['foo', 'bar', 'baz']):
    m = max(len(s) for s in substrings)
    return "".join([(s[i] if i < len(s) else " ") for i in range(m) for s in substrings])
```

</details>

### ReverseCat
([basic](#basic) 17/21)

**Description:**
Find a string that contains all the `substrings` reversed and forward

**Problem:**

```python
def sat(s: str, substrings: List[str]=['foo', 'bar', 'baz']):
    assert type(s) is str, 's must be of type str'
    return all(sub in s and sub[::-1] in s for sub in substrings)
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(substrings=['foo', 'bar', 'baz']):
    return "".join(substrings + [s[::-1] for s in substrings])
```

</details>

### EngineerNumbers
([basic](#basic) 18/21)

**Description:**
Find a list of `n` strings starting with `a` and ending with `b`

**Problem:**

```python
def sat(ls: List[str], n: int=100, a: str="bar", b: str="foo"):
    assert type(ls) is list and all(type(a) is str for a in ls), 'ls must be of type List[str]'
    return len(ls) == len(set(ls)) == n and ls[0] == a and ls[-1] == b and ls == sorted(ls)
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(n=100, a="bar", b="foo"):
    return sorted([a] + [a + chr(0) + str(i) for i in range(n-2)] + [b])
```

</details>

### PenultimateString
([basic](#basic) 19/21)

**Description:**
Find the alphabetically second to last last string in a list.

**Problem:**

```python
def sat(s: str, strings: List[str]=['cat', 'dog', 'bird', 'fly', 'moose']):
    assert type(s) is str, 's must be of type str'
    return s in strings and sum(t > s for t in strings) == 1
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(strings=['cat', 'dog', 'bird', 'fly', 'moose']):
    return sorted(strings)[-2]
```

</details>

### PenultimateRevString
([basic](#basic) 20/21)

**Description:**
Find the reversed version of the alphabetically second string in a list.

**Problem:**

```python
def sat(s: str, strings: List[str]=['cat', 'dog', 'bird', 'fly', 'moose']):
    assert type(s) is str, 's must be of type str'
    return s[::-1] in strings and sum(t < s[::-1] for t in strings) == 1
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(strings=['cat', 'dog', 'bird', 'fly', 'moose']):
    return sorted(strings)[1][::-1]
```

</details>

### CenteredString
([basic](#basic) 21/21)

**Description:**
Find a substring of length `length` centered within `target`.

**Problem:**

```python
def sat(s: str, target: str="foobarbazwow", length: int=6):
    assert type(s) is str, 's must be of type str'
    return target[(len(target) - length) // 2:(len(target) + length) // 2] == s
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(target="foobarbazwow", length=6):
    return target[(len(target) - length) // 2:(len(target) + length) // 2]
```

</details>

[^^ Top](#files)
## chess

Classic chess problems

[^ Top](#files)

### EightQueensOrFewer
([chess](#chess) 1/5)

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

### MoreQueens
([chess](#chess) 2/5)

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

### KnightsTour
([chess](#chess) 3/5)

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

### UncrossedKnightsPath
([chess](#chess) 4/5)

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
### UNSOLVED_UncrossedKnightsPath
([chess](#chess) 5/5)

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
## classic_puzzles

Classic puzzles


[^ Top](#files)

### TowersOfHanoi
([classic_puzzles](#classic_puzzles) 1/22)

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

### TowersOfHanoiArbitrary
([classic_puzzles](#classic_puzzles) 2/22)

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

### LongestMonotonicSubstring
([classic_puzzles](#classic_puzzles) 3/22)

**Description:**
Find the indices of the longest substring with characters in sorted order.

**Problem:**

```python
def sat(x: List[int], length: int=13, s: str="Dynamic programming solves this puzzle!!!"):
    assert type(x) is list and all(type(a) is int for a in x), 'x must be of type List[int]'
    return all(s[x[i]] <= s[x[i + 1]] and x[i + 1] > x[i] >= 0 for i in range(length - 1))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### LongestMonotonicSubstringTricky
([classic_puzzles](#classic_puzzles) 4/22)

**Description:**
Find the indices of the longest substring with characters in sorted order, with a twist!

**Problem:**

```python
def sat(x: List[int], length: int=20, s: str="Dynamic programming solves this puzzle!!!"):
    assert type(x) is list and all(type(a) is int for a in x), 'x must be of type List[int]'
    return all(s[x[i]] <= s[x[i + 1]] and x[i + 1] > x[i] for i in range(length - 1))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(length=20, s="Dynamic programming solves this puzzle!!!"):  # O(N^2) method. Todo: add binary search solution which is O(n log n)
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

### Quine
([classic_puzzles](#classic_puzzles) 5/22)

**Description:**
[Quine](https://en.wikipedia.org/wiki/Quine_%28computing%29)

Find a string that when evaluated as a Python expression is that string itself.

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

```python
def sol():  # thanks for this simple solution, GPT-3!
    return 'quine'
```

</details>

### RevQuine
([classic_puzzles](#classic_puzzles) 6/22)

**Description:**
Reverse [Quine](https://en.wikipedia.org/wiki/Quine_%28computing%29)

Find a string that, when reversed and evaluated gives you back that same string. The solution we found
is from GPT3.

**Problem:**

```python
def sat(rev_quine: str):
    assert type(rev_quine) is str, 'rev_quine must be of type str'
    return eval(rev_quine[::-1]) == rev_quine
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return "rev_quine"[::-1]  # thanks GPT-3!
```

</details>

### BooleanPythagoreanTriples
([classic_puzzles](#classic_puzzles) 7/22)

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

### ClockAngle
([classic_puzzles](#classic_puzzles) 8/22)

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

### Kirkman
([classic_puzzles](#classic_puzzles) 9/22)

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

### MonkeyAndCoconuts
([classic_puzzles](#classic_puzzles) 10/22)

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

### No3Colinear
([classic_puzzles](#classic_puzzles) 11/22)

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

### PostageStamp
([classic_puzzles](#classic_puzzles) 12/22)

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

### SquaringTheSquare
([classic_puzzles](#classic_puzzles) 13/22)

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

### NecklaceSplit
([classic_puzzles](#classic_puzzles) 14/22)

**Description:**
[Necklace Splitting Problem](https://en.wikipedia.org/wiki/Necklace_splitting_problem)

Split a specific red/blue necklace in half at n so that each piece has an equal number of reds and blues.

**Problem:**

```python
def sat(n: int, lace: str="bbbbrrbrbrbbrrrr"):
    assert type(n) is int, 'n must be of type int'
    sub = lace[n: n + len(lace) // 2]
    return n >= 0 and lace.count("r") == 2 * sub.count("r") and lace.count("b") == 2 * sub.count("b")
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(lace="bbbbrrbrbrbbrrrr"):
    if lace == "":
        return 0
    return next(n for n in range(len(lace) // 2) if lace[n: n + len(lace) // 2].count("r") == len(lace) // 4)
```

</details>

### PandigitalSquare
([classic_puzzles](#classic_puzzles) 15/22)

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

### AllPandigitalSquares
([classic_puzzles](#classic_puzzles) 16/22)

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

### CardGame24
([classic_puzzles](#classic_puzzles) 17/22)

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

### Easy63
([classic_puzzles](#classic_puzzles) 18/22)

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

### Harder63
([classic_puzzles](#classic_puzzles) 19/22)

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

### WaterPouring
([classic_puzzles](#classic_puzzles) 20/22)

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

### VerbalArithmetic
([classic_puzzles](#classic_puzzles) 21/22)

**Description:**
Find a substitution of digits for characters to make the numbers add up in a sum like this:
SEND + MORE = MONEY

The first digit in any number cannot be 0.
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

### SlidingPuzzle
([classic_puzzles](#classic_puzzles) 22/22)

**Description:**
[Sliding puzzle](https://en.wikipedia.org/wiki/15_puzzle)

The 3-, 8-, and 15-sliding puzzles are classic examples of A* search. In this puzzle, you are given a board like:
1 2 5
3 4 0
6 7 8

and your goal is to transform it to:
0 1 2
3 4 5
6 7 8

by a sequence of swaps with the 0 square (0 indicates blank). The starting configuration is given by a 2d list of
lists and the answer is represented by a list of integers indicating which number you swap with 0. In the above
example, the answer would be `[1, 2, 5]`


 The problem is NP-hard but the puzzles can all be solved with A* and an efficient representation.

**Problem:**

```python
def sat(moves: List[int], start: List[List[int]]=[[5, 0, 2, 3], [1, 9, 6, 7], [4, 14, 8, 11], [12, 13, 10, 15]]):
    assert type(moves) is list and all(type(a) is int for a in moves), 'moves must be of type List[int]'
    locs = {i: [x, y] for y, row in enumerate(start) for x, i in enumerate(row)}  # locations, 0 stands for blank
    for i in moves:
        assert abs(locs[0][0] - locs[i][0]) + abs(locs[0][1] - locs[i][1]) == 1
        locs[0], locs[i] = locs[i], locs[0]
    return all(locs[i] == [i % len(start[0]), i // len(start)] for i in locs)
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

[^^ Top](#files)
## codeforces

Problems inspired by [codeforces](https://codeforces.com).

[^ Top](#files)

### IsEven
([codeforces](#codeforces) 1/24)

**Description:**
Determine if n can be evenly divided into two equal numbers. (Easy)

Inspired by [Codeforces Problem 4 A](https://codeforces.com/problemset/problem/4/A)

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

### Abbreviate
([codeforces](#codeforces) 2/24)

**Description:**
Abbreviate strings longer than a given length by replacing everything but the first and last characters by
an integer indicating how many characters there were in between them.

Inspired by [Codeforces Problem 71 A](https://codeforces.com/problemset/problem/71/A)

**Problem:**

```python
def sat(s: str, word: str="antidisestablishmentarianism", max_len: int=10):
    assert type(s) is str, 's must be of type str'
    if len(word) <= max_len:
        return word == s
    return int(s[1:-1]) == len(word[1:-1]) and word[0] == s[0] and word[-1] == s[-1]
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(word="antidisestablishmentarianism", max_len=10):
    if len(word) <= max_len:
        return word
    return f"{word[0]}{len(word) - 2}{word[-1]}"
```

</details>

### SquareTiles
([codeforces](#codeforces) 3/24)

**Description:**
Find a minimal list of corner locations for a×a tiles that covers [0, m] × [0, n] and does not double-cover
squares.

Sample Input:
m = 10
n = 9
a = 5
target = 4

Sample Output:
[[0, 0], [0, 5], [5, 0], [5, 5]]

Inspired by [Codeforces Problem 1 A](https://codeforces.com/problemset/problem/1/A)

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

### EasyTwos
([codeforces](#codeforces) 4/24)

**Description:**
Given a list of lists of triples of integers, return True for each list with a total of at least 2 and False for
each other list.

Inspired by [Codeforces Problem 231 A](https://codeforces.com/problemset/problem/231/A)

**Problem:**

```python
def sat(lb: List[bool], trips: List[List[int]]=[[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]):
    assert type(lb) is list and all(type(a) is bool for a in lb), 'lb must be of type List[bool]'
    return len(lb) == len(trips) and all(
        (b is True) if sum(s) >= 2 else (b is False) for b, s in zip(lb, trips))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(trips=[[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]):
    return [sum(s) >= 2 for s in trips]
```

</details>

### DecreasingCountComparison
([codeforces](#codeforces) 5/24)

**Description:**
Given a list of non-increasing integers and given an integer k, determine how many positive integers in the list
are at least as large as the kth.

Inspired by [Codeforces Problem 158 A](https://codeforces.com/problemset/problem/158/A)

**Problem:**

```python
def sat(n: int, scores: List[int]=[100, 95, 80, 70, 65, 9, 9, 9, 4, 2, 1], k: int=6):
    assert type(n) is int, 'n must be of type int'
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), "Hint: scores are non-decreasing"
    return all(s >= scores[k] and s > 0 for s in scores[:n]) and all(s < scores[k] or s <= 0 for s in scores[n:])
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(scores=[100, 95, 80, 70, 65, 9, 9, 9, 4, 2, 1], k=6):
    threshold = max(scores[k], 1)
    return sum(s >= threshold for s in scores)
```

</details>

### DominoTile
([codeforces](#codeforces) 6/24)

**Description:**
Tile an m x n checkerboard with 2 x 1 tiles. The solution is a list of fourtuples [i1, j1, i2, j2] with i2 == i1
and j2 == j1 + 1 or i2 == i1 + 1 and j2 == j1 with no overlap.

Inspired by [Codeforces Problem 50 A](https://codeforces.com/problemset/problem/50/A)

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

### IncDec
([codeforces](#codeforces) 7/24)

**Description:**
This straightforward problem is a little harder than the Codeforces one.
Given a sequence of operations "++x", "x++", "--x", "x--", and a target value, find initial value so that the
final value is the target value.

Sample Input:
ops = ["x++", "--x", "--x"]
target = 12

Sample Output:
13

Inspired by [Codeforces Problem 282 A](https://codeforces.com/problemset/problem/282/A)

**Problem:**

```python
def sat(n: int, ops: List[str]=['x++', '--x', '--x'], target: int=19143212):
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
def sol(ops=['x++', '--x', '--x'], target=19143212):
    return target - ops.count("++x") - ops.count("x++") + ops.count("--x") + ops.count("x--")
```

</details>

### CompareInAnyCase
([codeforces](#codeforces) 8/24)

**Description:**
Ignoring case, compare s, t lexicographically. Output 0 if they are =, -1 if s < t, 1 if s > t.

Inspired by [Codeforces Problem 112 A](https://codeforces.com/problemset/problem/112/A)

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

### SlidingOne
([codeforces](#codeforces) 9/24)

**Description:**
We are given a 5x5 bimatrix with a single 1 like:

0 0 0 0 0
0 0 0 0 1
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0

Find a (minimal) sequence of row and column swaps to move the 1 to the center. A move is a string
in "0"-"4" indicating a row swap and "a"-"e" indicating a column swap

Inspired by [Codeforces Problem 263 A](https://codeforces.com/problemset/problem/263/A)

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

### SortPlusPlus
([codeforces](#codeforces) 10/24)

**Description:**
Sort numbers in a sum of digits, e.g., 1+3+2+1 -> 1+1+2+3

Inspired by [Codeforces Problem 339 A](https://codeforces.com/problemset/problem/339/A)

**Problem:**

```python
def sat(s: str, inp: str="1+1+3+1+3+2+2+1+3+1+2"):
    assert type(s) is str, 's must be of type str'
    return all(s.count(c) == inp.count(c) for c in inp + s) and all(s[i - 2] <= s[i] for i in range(2, len(s), 2))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(inp="1+1+3+1+3+2+2+1+3+1+2"):
    return "+".join(sorted(inp.split("+")))
```

</details>

### CapitalizeFirstLetter
([codeforces](#codeforces) 11/24)

**Description:**
Capitalize first letter of word

Inspired by [Codeforces Problem 281 A](https://codeforces.com/problemset/problem/281/A)

**Problem:**

```python
def sat(s: str, word: str="konjac"):
    assert type(s) is str, 's must be of type str'
    for i in range(len(word)):
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

### LongestSubsetString
([codeforces](#codeforces) 12/24)

**Description:**
You are given a string consisting of a's, b's and c's, find any longest substring containing no repeated
consecutive characters.

Sample Input:
`"abbbc"`

Sample Output:
`"abc"`

Inspired by [Codeforces Problem 266 A](https://codeforces.com/problemset/problem/266/A)

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
def sol(s="abbbcabbac", target=7):  # target is ignored
    return s[:1] + "".join([b for a, b in zip(s, s[1:]) if b != a])
```

</details>

### FindHomogeneousSubstring
([codeforces](#codeforces) 13/24)

**Description:**
You are given a string consisting of 0's and 1's. Find an index after which the subsequent k characters are
all 0's or all 1's.

Sample Input:
s = 0000111111100000, k = 5

Sample Output:
4
(or 5 or 6 or 11)

Inspired by [Codeforces Problem 96 A](https://codeforces.com/problemset/problem/96/A)

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

### FivePowers
([codeforces](#codeforces) 14/24)

**Description:**
What are the last two digits of 5^n?

Inspired by [Codeforces Problem 630 A](https://codeforces.com/problemset/problem/630/A)

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

### CombinationLock
([codeforces](#codeforces) 15/24)

**Description:**
Shortest Combination Lock Path

Given a starting a final lock position, find the (minimal) intermediate states, where each transition
involves increasing or decreasing a single digit (mod 10), e.g.

start = "012"
combo = "329"
output: ['112', '212', '312', '322', '321', '320']

Inspired by [Codeforces Problem 540 A](https://codeforces.com/problemset/problem/540/A)

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

### CombinationLockObfuscated
([codeforces](#codeforces) 16/24)

**Description:**
An obfuscated version of CombinationLock above

**Problem:**

```python
def sat(states: List[str], start: str="012", combo: str="329", target_len: int=6):
    assert type(states) is list and all(type(a) is str for a in states), 'states must be of type List[str]'
    return all(sum((int(a[i]) - int(b[i])) ** 2 % 10 for i in range(len(start))) == 1
               for a, b in zip([start] + states, states[:target_len] + [combo]))
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

### InvertPermutation
([codeforces](#codeforces) 17/24)

**Description:**
Find a string that, when a given permutation of characters is applied, has a given result.

Inspired by [Codeforces Problem 474 A](https://codeforces.com/problemset/problem/474/A)

**Problem:**

```python
def sat(s: str, perm: str="qwertyuiopasdfghjklzxcvbnm", target: str="hello are you there?"):
    assert type(s) is str, 's must be of type str'
    return "".join((perm[(perm.index(c) + 1) % len(perm)] if c in perm else c) for c in s) == target
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(perm="qwertyuiopasdfghjklzxcvbnm", target="hello are you there?"):
    return "".join((perm[(perm.index(c) - 1) % len(perm)] if c in perm else c) for c in target)
```

</details>

### SameDifferent
([codeforces](#codeforces) 18/24)

**Description:**
Given a list of integers and a target length, create of the given length such that:
* The first list must be all the same numbers.
* The second must be all different.
* The two lists together comprise a sublist of all the list items

Inspired by [Codeforces Problem 1335 C](https://codeforces.com/problemset/problem/1335/C)

**Problem:**

```python
def sat(lists: List[List[int]], items: List[int]=[5, 4, 9, 4, 5, 5, 5, 1, 5, 5], length: int=4):
    assert type(lists) is list and all(type(a) is list and all(type(b) is int for b in a) for a in lists), 'lists must be of type List[List[int]]'
    a, b = lists
    assert len(set(a)) == len(a) == len(b) == length and len(set(b)) == 1 and set(a + b) <= set(items)
    i = b[0]
    return (a + b).count(i) <= items.count(i)
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### OnesAndTwos
([codeforces](#codeforces) 19/24)

**Description:**
Find a sequence of 1's and 2's of a given length that that adds up to n

Inspired by [Codeforces Problem 476 A](https://codeforces.com/problemset/problem/476/A)

**Problem:**

```python
def sat(seq: List[int], n: int=10000, length: int=5017):
    assert type(seq) is list and all(type(a) is int for a in seq), 'seq must be of type List[int]'
    return set(seq) <= {1, 2} and sum(seq) == n and len(seq) == length
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(n=10000, length=5017):
    return [2] * (n - length) + [1] * (2 * length - n)
```

</details>

### MinConsecutiveSum
([codeforces](#codeforces) 20/24)

**Description:**
Find a sequence of k consecutive indices whose sum is minimal

Inspired by [Codeforces Problem 363 B](https://codeforces.com/problemset/problem/363/B)

**Problem:**

```python
def sat(start: int, k: int=3, upper: int=6, seq: List[int]=[17, 1, 2, 65, 18, 91, -30, 100, 3, 1, 2]):
    assert type(start) is int, 'start must be of type int'
    return 0 <= start <= len(seq) - k and sum(seq[start:start + k]) <= upper
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(k=3, upper=6, seq=[17, 1, 2, 65, 18, 91, -30, 100, 3, 1, 2]):
    return min(range(len(seq) - k + 1), key=lambda start: sum(seq[start:start + k]))
```

</details>

### MaxConsecutiveSum
([codeforces](#codeforces) 21/24)

**Description:**
Find a sequence of k consecutive indices whose sum is maximal

Inspired by [Codeforces Problem 363 B](https://codeforces.com/problemset/problem/363/B)

**Problem:**

```python
def sat(start: int, k: int=3, lower: int=150, seq: List[int]=[3, 1, 2, 65, 18, 91, -30, 100, 0, 19, 52]):
    assert type(start) is int, 'start must be of type int'
    return 0 <= start <= len(seq) - k and sum(seq[start:start + k]) >= lower
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(k=3, lower=150, seq=[3, 1, 2, 65, 18, 91, -30, 100, 0, 19, 52]):
    return max(range(len(seq) - k + 1), key=lambda start: sum(seq[start:start + k]))
```

</details>

### MaxConsecutiveProduct
([codeforces](#codeforces) 22/24)

**Description:**
Find a sequence of k consecutive indices whose product is maximal, possibly looping around

Inspired by [Codeforces Problem 363 B](https://codeforces.com/problemset/problem/363/B)

**Problem:**

```python
def sat(start: int, k: int=3, lower: int=100000, seq: List[int]=[91, 1, 2, 64, 18, 91, -30, 100, 3, 65, 18]):
    assert type(start) is int, 'start must be of type int'
    prod = 1
    for i in range(start, start + k):
        prod *= seq[i]
    return prod >= lower
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### DistinctOddSum
([codeforces](#codeforces) 23/24)

**Description:**
Find n distinct positive odd integers that sum to tot

Inspired by [Codeforces Problem 1327 A](https://codeforces.com/problemset/problem/1327/A)

**Problem:**

```python
def sat(nums: List[int], tot: int=12345, n: int=5):
    assert type(nums) is list and all(type(a) is int for a in nums), 'nums must be of type List[int]'
    return len(nums) == len(set(nums)) == n and sum(nums) == tot and all(i >= i % 2 > 0 for i in nums)
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(tot=12345, n=5):
    return list(range(1, 2 * n - 1, 2)) + [tot - sum(range(1, 2 * n - 1, 2))]
```

</details>

### MinRotations
([codeforces](#codeforces) 24/24)

**Description:**
We begin with the string `"a...z"`

An `r`-rotation of a string means shifting it to the right (positive) or left (negative) by `r` characters and
cycling around. Given a target string of length n, find the n rotations that put the consecutive characters
of that string at the beginning of the r-rotation, with minimal sum of absolute values of the `r`'s.

For example if the string was `'dad'`, the minimal rotations would be `[3, -3, 3]` with a total of `9`.

Inspired by [Codeforces Problem 731 A](https://codeforces.com/problemset/problem/731/A)

**Problem:**

```python
def sat(rotations: List[int], target: str="dad", upper: int=9):
    assert type(rotations) is list and all(type(a) is int for a in rotations), 'rotations must be of type List[int]'
    s = "abcdefghijklmnopqrstuvwxyz"
    assert len(rotations) == len(target)
    for r, c in zip(rotations, target):
        s = s[r:] + s[:r]
        assert s[0] == c

    return sum(abs(r) for r in rotations) <= upper
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(target="dad", upper=9):
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

[^^ Top](#files)
## compression

Invert a given de/compression algorithm.

[^ Top](#files)

### LZW
([compression](#compression) 1/3)

**Description:**
Find a (short) compression that decompresses to the given string.
We have provided a simple version of the *decompression* algorithm of
[Lempel-Ziv-Welch](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch)
so the solution is the *compression* algorithm.

**Problem:**

```python
def sat(seq: List[int], compressed_len: int=17, text: str="Hellooooooooooooooooooooo world!"):
    assert type(seq) is list and all(type(a) is int for a in seq), 'seq must be of type List[int]'
    index = [chr(i) for i in range(256)]
    pieces = [""]
    for i in seq:
        pieces.append((pieces[-1] + pieces[-1][0]) if i == len(index) else index[i])
        index.append(pieces[-2] + pieces[-1][0])
    return "".join(pieces) == text and len(seq) <= compressed_len
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### LZW_decompress
([compression](#compression) 2/3)

**Description:**
Find a string that compresses to the target sequence for the provided simple version of
[Lempel-Ziv-Welch](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch)
so the solution is the *decompression* algorithm.

**Problem:**

```python
def sat(text: str, seq: List[int]=[72, 101, 108, 108, 111, 32, 119, 111, 114, 100, 262, 264, 266, 263, 265, 33]):
    assert type(text) is str, 'text must be of type str'
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
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### PackingHam
([compression](#compression) 3/3)

**Description:**
Pack a certain number of binary strings so that they have a minimum hamming distance between each other.

This is a [classic problem](https://en.wikipedia.org/wiki/Sphere_packing#Other_spaces) in coding theory.

**Problem:**

```python
def sat(words: List[str], num: int=100, bits: int=100, dist: int=34):
    assert type(words) is list and all(type(a) is str for a in words), 'words must be of type List[str]'
    assert len(words) == num and all(len(word) == bits and set(word) <= {"0", "1"} for word in words)
    return all(sum([a != b for a, b in zip(words[i], words[j])]) >= dist for i in range(num) for j in range(i))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

[^^ Top](#files)
## conways_game_of_life

Conway's Game of Life problems

[^ Top](#files)

### Oscillators
([conways_game_of_life](#conways_game_of_life) 1/2)

**Description:**
Oscillators (including some unsolved, open problems)

Find a pattern in [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
that repeats with a certain period. This problem is *unsolved* for periods 19, 38, and 41.

See
[discussion](https://en.wikipedia.org/wiki/Oscillator_%28cellular_automaton%29#:~:text=Game%20of%20Life)
in Wikipedia article on Cellular Automaton Oscillators.

**Problem:**

```python
def sat(init: List[List[int]], period: int=3):
    assert type(init) is list and all(type(a) is list and all(type(b) is int for b in a) for a in init), 'init must be of type List[List[int]]'
    target = {x + y * 1j for x, y in init}  # complex numbers encode live cells

    deltas = (1j, -1j, 1, -1, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j)
    live = target
    for t in range(period):
        visible = {z + d for z in live for d in deltas}
        live = {z for z in visible if sum(z + d in live for d in deltas) in ([2, 3] if z in live else [3])}
        if live == target:
            return t + 1 == period
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### Spaceship
([conways_game_of_life](#conways_game_of_life) 2/2)

**Description:**
Spaceship (including *unsolved*, open problems)

Find a [spaceship](https://en.wikipedia.org/wiki/Spaceship_%28cellular_automaton%29) in
[Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
with a certain period.

This is an *unsolved* problem for periods 33, 34.

**Problem:**

```python
def sat(init: List[List[int]], period: int=4):
    assert type(init) is list and all(type(a) is list and all(type(b) is int for b in a) for a in init), 'init must be of type List[List[int]]'
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
[^^ Top](#files)
## games


Solve some two-player games


[^ Top](#files)

### Nim
([games](#games) 1/5)

**Description:**
Compute optimal play for the classic two-player game [Nim](https://en.wikipedia.org/wiki/Nim)

In the game of Nim, there are a number of heaps of objects. In each step, a player removes one or more
objects from a non-empty heap. The player who takes the last object wins. Nim has an elegant theory
for optimal play based on the xor of the bits.

**Problem:**

```python
def sat(cert: List[List[int]], heaps: List[int]=[5, 9]):
    assert type(cert) is list and all(type(a) is list and all(type(b) is int for b in a) for a in cert), 'cert must be of type List[List[int]]'  # cert is a sufficient list of desirable states to leave for opponent
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
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(heaps=[5, 9]):
    import itertools

    def val(h):  # return True if h is a good state to leave things in
        xor = 0
        for i in h:
            xor ^= i
        return xor == 0

    return [list(h) for h in itertools.product(*[range(i+1) for i in heaps]) if val(h)]
```

</details>

### Mastermind
([games](#games) 2/5)

**Description:**
Compute a strategy for winning in [mastermind](https://en.wikipedia.org/wiki/Mastermind_%28board_game%29)
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

**Problem:**

```python
def sat(transcripts: List[str], max_moves: int=10):
    assert type(transcripts) is list and all(type(a) is str for a in transcripts), 'transcripts must be of type List[str]'
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
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### TicTacToeX
([games](#games) 3/5)

**Description:**
Compute a strategy for X (first player) in tic-tac-toe that guarantees a tie.

We are looking for a strategy for X that, no matter what the opponent does, X does not lose.

A board is represented as a 9-char string like an X in the middle would be "....X...." and a
move is an integer 0-8. The answer is a list of "good boards" that X aims for, so no matter what O does there
is always good board that X can get to with a single move.

**Problem:**

```python
def sat(good_boards: List[str]):
    assert type(good_boards) is list and all(type(a) is str for a in good_boards), 'good_boards must be of type List[str]'
    board_bit_reps = {tuple(sum(1 << i for i in range(9) if b[i] == c) for c in "XO") for b in good_boards}
    win = [any(i & w == w for w in [7, 56, 73, 84, 146, 273, 292, 448]) for i in range(512)]

    def tie(x, o):  # returns True if X has a forced tie/win assuming it's X's turn to move.
        x |= 1 << next(i for i in range(9) if (x | (1 << i), o) in board_bit_reps)
        return not win[o] and (win[x] or all((x | o) & (1 << i) or tie(x, o | (1 << i)) for i in range(9)))

    return tie(0, 0)
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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
        if win[x] or x | o == 511:
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

### TicTacToeO
([games](#games) 4/5)

**Description:**
Compute a strategy for O (second player) in tic-tac-toe that guarantees a tie.

We are looking for a strategy for O that, no matter what the opponent does, O does not lose.

A board is represented as a 9-char string like an X in the middle would be "....X...." and a
move is an integer 0-8. The answer is a list of "good boards" that O aims for, so no matter what X does there
is always good board that O can get to with a single move.

**Problem:**

```python
def sat(good_boards: List[str]):
    assert type(good_boards) is list and all(type(a) is str for a in good_boards), 'good_boards must be of type List[str]'
    board_bit_reps = {tuple(sum(1 << i for i in range(9) if b[i] == c) for c in "XO") for b in good_boards}
    win = [any(i & w == w for w in [7, 56, 73, 84, 146, 273, 292, 448]) for i in range(512)]

    def tie(x, o):  # returns True if O has a forced tie/win. It's O's turn to move.
        if o | x != 511:
            o |= 1 << next(i for i in range(9) if (x, o | (1 << i)) in board_bit_reps)
        return not win[x] and (win[o] or all((x | o) & (1 << i) or tie(x | (1 << i), o) for i in range(9)))

    return all(tie(1 << i, 0) for i in range(9))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
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
```

</details>

### RockPaperScissors
([games](#games) 5/5)

**Description:**
Find optimal strategy for Rock-Paper-Scissors zero-sum game

Can the computer figure out that 1/3, 1/3, 1/3 achieves the maximal expected value of 0

**Problem:**

```python
def sat(probs: List[float]):
    assert type(probs) is list and all(type(a) is float for a in probs), 'probs must be of type List[float]'  # rock prob, paper prob, scissors prob
    assert len(probs) == 3 and abs(sum(probs) - 1) < 1e-6
    return max(probs[(i + 2) % 3] - probs[(i + 1) % 3] for i in range(3)) < 1e-6
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return [1 / 3] * 3
```

</details>

[^^ Top](#files)
## game_theory


Hard problems from game theory.


[^ Top](#files)

### Nash
([game_theory](#game_theory) 1/2)

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

### ZeroSum
([game_theory](#game_theory) 2/2)

**Description:**
Compute minimax optimal strategies for a given
[zero-sum game](https://en.wikipedia.org/wiki/Zero-sum_game). This problem is known to be equivalent to
Linear Programming. Note that the provided instances are all quite easy---harder solutions could readily
be made by decreasing the accuracy tolerance `eps` at which point the solution we provided would fail and
more efficient algorithms would be needed.

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

### Conway99
([graphs](#graphs) 1/11)

**Description:**
Conway's 99-graph problem (*unsolved*, open problem)

Conway's 99-graph problem is an unsolved problem in graph theory. It asks whether there exists an
undirected graph with 99 vertices, in which each two adjacent vertices have exactly one common neighbor,
and in which each two non-adjacent vertices have exactly two common neighbors.
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
### AnyEdge
([graphs](#graphs) 2/11)

**Description:**
Find any edge in a given [graph](https://en.wikipedia.org/w/index.php?title=Graph_(discrete_mathematics)).

**Problem:**

```python
def sat(e: List[int], edges: List[List[int]]=[[0, 217], [40, 11], [17, 29], [11, 12], [31, 51]]):
    assert type(e) is list and all(type(a) is int for a in e), 'e must be of type List[int]'
    return e in edges
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(edges=[[0, 217], [40, 11], [17, 29], [11, 12], [31, 51]]):
    return edges[0]
```

</details>

### AnyTriangle
([graphs](#graphs) 3/11)

**Description:**
Find a [triangle](https://en.wikipedia.org/w/index.php?title=Triangle_graph) in a given directed graph.

**Problem:**

```python
def sat(tri: List[int], edges: List[List[int]]=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):
    assert type(tri) is list and all(type(a) is int for a in tri), 'tri must be of type List[int]'
    a, b, c = tri
    return [a, b] in edges and [b, c] in edges and [c, a] in edges and a != b != c != a
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### PlantedClique
([graphs](#graphs) 4/11)

**Description:**
Find a [planted clique](https://en.wikipedia.org/w/index.php?title=Planted_clique) of a given size
in an undirected graph. Finding a polynomial-time algorithm for this problem has been *unsolved* for
some time.

**Problem:**

```python
def sat(nodes: List[int], size: int=3, edges: List[List[int]]=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):
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

### ShortestPath
([graphs](#graphs) 5/11)

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

### UnweightedShortestPath
([graphs](#graphs) 6/11)

**Description:**
Unweighted Shortest Path

Find a path from node u to node v, of a bounded length, in a given digraph on vertices 0, 1,..., n.

See (Dijkstra's algorithm)[https://en.wikipedia.org/w/index.php?title=Dijkstra%27s_algorithm]

**Problem:**

```python
def sat(path: List[int], edges: List[List[int]]=[[0, 11], [0, 22], [11, 22], [11, 33], [22, 33]], u: int=0, v: int=33, bound: int=3):
    assert type(path) is list and all(type(a) is int for a in path), 'path must be of type List[int]'
    assert path[0] == u and path[-1] == v and all([i, j] in edges for i, j in zip(path, path[1:]))
    return len(path) <= bound
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(edges=[[0, 11], [0, 22], [11, 22], [11, 33], [22, 33]], u=0, v=33, bound=3):  # Dijkstra's algorithm
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

### AnyPath
([graphs](#graphs) 7/11)

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

### EvenPath
([graphs](#graphs) 8/11)

**Description:**
Even Path

Find any path with an even number of nodes from node 0 to node n in a given graph on vertices 0, 1,..., n.

**Problem:**

```python
def sat(path: List[int], edges: List[List[int]]=[[0, 2], [0, 1], [2, 1], [2, 3], [1, 3]]):
    assert type(path) is list and all(type(a) is int for a in path), 'path must be of type List[int]'
    assert path[0] == 0 and path[-1] == max(max(e) for e in edges)
    assert all([[a, b] in edges for a, b in zip(path, path[1:])])
    return len(path) % 2 == 0
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(edges=[[0, 2], [0, 1], [2, 1], [2, 3], [1, 3]]):
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

### OddPath
([graphs](#graphs) 9/11)

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

### Zarankiewicz
([graphs](#graphs) 10/11)

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

### GraphIsomorphism
([graphs](#graphs) 11/11)

**Description:**
In the classic [Graph Isomorphism](https://en.wikipedia.org/wiki/Graph_isomorphism) problem,
one is given two graphs which are permutations of one another and
the goal is to find the permutation. It is unknown wheter or not there exists a polynomial-time algorithm
for this problem, though an unpublished quasi-polynomial-time algorithm has been announced by Babai.

Each graph is specified by a list of edges where each edge is a pair of integer vertex numbers.

**Problem:**

```python
def sat(bi: List[int], g1: List[List[int]]=[[0, 1], [1, 2], [2, 3], [3, 4]], g2: List[List[int]]=[[0, 4], [4, 1], [1, 2], [2, 3]]):
    assert type(bi) is list and all(type(a) is int for a in bi), 'bi must be of type List[int]'
    return len(bi) == len(set(bi)) and {(i, j) for i, j in g1} == {(bi[i], bi[j]) for i, j in g2}
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(g1=[[0, 1], [1, 2], [2, 3], [3, 4]], g2=[[0, 4], [4, 1], [1, 2], [2, 3]]):  # exponentially slow
    from itertools import permutations
    n = max(i for g in [g1, g2] for e in g for i in e) + 1
    g1_set = {(i, j) for i, j in g1}
    for pi in permutations(range(n)):
        if all((pi[i], pi[j]) in g1_set for i, j in g2):
            return list(pi)
    assert False, f"Graphs are not isomorphic {g1}, {g2}"
```

</details>

[^^ Top](#files)
## ICPC


Problems inspired by the [International Collegiate Programming Contest](https://icpc.global) (ICPC).


[^ Top](#files)

### BiPermutations
([ICPC](#icpc) 1/3)

**Description:**
There are two rows of objects. Given the length-n integer arrays of prices and heights of objects in each
row, find a permutation of both rows so that the permuted prices are non-decreasing in each row and
so that the first row is taller than the second row.

Inspired by
[ICPC 2019 Problem A: Azulejos](https://icpc.global/newcms/worldfinals/problems/2019%20ACM-ICPC%20World%20Finals/icpc2019.pdf)
which is 2,287 characters.

**Problem:**

```python
def sat(perms: List[List[int]], prices0: List[int]=[7, 7, 9, 5, 3, 7, 1, 2], prices1: List[int]=[5, 5, 5, 4, 2, 5, 1, 1], heights0: List[int]=[2, 4, 9, 3, 8, 5, 5, 4], heights1: List[int]=[1, 3, 8, 1, 5, 4, 4, 2]):
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

### OptimalBridges
([ICPC](#icpc) 2/3)

**Description:**
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

Inspired by
[ICPC 2019 Problem B: Bridges](https://icpc.global/newcms/worldfinals/problems/2019%20ACM-ICPC%20World%20Finals/icpc2019.pdf)
which is 3,003 characters.

**Problem:**

```python
def sat(indices: List[int], H: int=60, alpha: int=18, beta: int=2, xs: List[int]=[0, 10, 20, 30, 50, 80, 100, 120, 160, 190, 200], ys: List[int]=[0, 30, 10, 30, 50, 40, 10, 20, 20, 55, 10], thresh: int=26020):
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

### CheckersPosition
([ICPC](#icpc) 3/3)

**Description:**
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

Inspired by
[ICPC 2019 Problem C: Checks Post Facto](https://icpc.global/newcms/worldfinals/problems/2019%20ACM-ICPC%20World%20Finals/icpc2019.pdf)

**Problem:**

```python
def sat(position: List[List[int]], transcript: List[List[List[int]]]=[[[3, 3], [5, 5], [3, 7]], [[5, 3], [6, 4]]]):
    assert type(position) is list and all(type(a) is list and all(type(b) is int for b in a) for a in position), 'position must be of type List[List[int]]'
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
<details><summary><strong>Reveal solution(s):</strong></summary>

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

[^^ Top](#files)
## IMO

Problems inspired by the
[International Mathematical Olympiad](https://en.wikipedia.org/wiki/International_Mathematical_Olympiad)
[problems](https://www.imo-official.org/problems.aspx)

[^ Top](#files)

### ExponentialCoinMoves
([IMO](#imo) 1/6)

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

Inspired by [IMO 2010 Problem 5](https://www.imo-official.org/problems.aspx)

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

### NoRelativePrimes
([IMO](#imo) 2/6)

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

Inspired by [IMO 2016 Problem 4](https://www.imo-official.org/problems.aspx)

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

### FindRepeats
([IMO](#imo) 3/6)

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

Inspired by [IMO 2017 Problem 1](https://www.imo-official.org/problems.aspx)

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

### PickNearNeighbors
([IMO](#imo) 4/6)

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

Inspired by [IMO 2017 Problem 5](https://www.imo-official.org/problems.aspx)

**Problem:**

```python
def sat(keep: List[bool], heights: List[int]=[10, 2, 14, 1, 8, 19, 16, 6, 12, 3, 17, 0, 9, 18, 5, 7, 11, 13, 15, 4]):
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
def sol(heights=[10, 2, 14, 1, 8, 19, 16, 6, 12, 3, 17, 0, 9, 18, 5, 7, 11, 13, 15, 4]): # Based on the judge's solution.
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

### FindProductiveList
([IMO](#imo) 5/6)

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

Inspired by [IMO 2010 Problem 5](https://www.imo-official.org/problems.aspx)

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

### HalfTag
([IMO](#imo) 6/6)

**Description:**
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

Inspired by [IMO 2020 Problem 3](https://www.imo-official.org/problems.aspx)

**Problem:**

```python
def sat(li: List[int], n: int=3, tags: List[int]=[0, 1, 2, 0, 0, 1, 1, 1, 2, 2, 0, 2]):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    assert sorted(tags) == sorted(list(range(n)) * 4), "hint: each tag occurs exactly four times"
    assert len(li) == len(set(li)) and min(li) >= 0
    return sum(li) * 2 == sum(range(4 * n)) and sorted([tags[i] for i in li]) == [i // 2 for i in range(2 * n)]
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(n=3, tags=[0, 1, 2, 0, 0, 1, 1, 1, 2, 2, 0, 2]):
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

[^^ Top](#files)
## lattices

Lattice problems with and without noise

[^ Top](#files)

### LearnParity
([lattices](#lattices) 1/2)

**Description:**
Parity learning (Gaussian elimination)

Given binary vectors in a subspace, find the secret set $S$ of indices such that:
    $$sum_{i \in S} x_i = 1 (mod 2)$$

The canonical solution to this 
[Parity learning problem](https://en.wikipedia.org/w/index.php?title=Parity_learning)
is to use 
[Gaussian Elimination](https://en.wikipedia.org/w/index.php?title=Gaussian_elimination).

The vectors are encoded as binary integers for succinctness.

**Problem:**

```python
def sat(inds: List[int], vecs: List[int]=[169, 203, 409, 50, 37, 479, 370, 133, 53, 159, 161, 367, 474, 107, 82, 447, 385]):
    assert type(inds) is list and all(type(a) is int for a in inds), 'inds must be of type List[int]'
    return all(sum((v >> i) & 1 for i in inds) % 2 == 1 for v in vecs)
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### LearnParityWithNoise
([lattices](#lattices) 2/2)

**Description:**
Learn parity with noise (*unsolved*)

Given binary vectors, find the secret set $S$ of indices such that, for at least 3/4 of the vectors,
    $$sum_{i \in S} x_i = 1 (mod 2)$$

The fastest known algorithm to this
[Parity learning problem](https://en.wikipedia.org/w/index.php?title=Parity_learning)
runs in time $2^(d/(log d))$

**Problem:**

```python
def sat(inds: List[int], vecs: List[int]=[26, 5, 16, 3, 15, 18, 31, 13, 24, 25, 6, 5, 15, 24, 16, 13, 0, 27, 13]):
    assert type(inds) is list and all(type(a) is int for a in inds), 'inds must be of type List[int]'
    return sum(sum((v >> i) & 1 for i in inds) % 2 for v in vecs) >= len(vecs) * 3 / 4
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(vecs=[26, 5, 16, 3, 15, 18, 31, 13, 24, 25, 6, 5, 15, 24, 16, 13, 0, 27, 13]):  # brute force
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

[^^ Top](#files)
## number_theory

Number theory problems

[^ Top](#files)

### FermatsLastTheorem
([number_theory](#number_theory) 1/16)

**Description:**
[Fermat's last theorem](https://en.wikipedia.org/w/index.php?title=Fermat%27s_Last_Theorem)

Find integers a,b,c > 0, n > 2, such such that `a ** n + b ** n == c ** n`
Supposedly unsolvable, but how confident are really in the super-complicated proof?

See [Wiles, Andrew. "Modular elliptic curves and Fermat's last theorem." Annals of mathematics 141.3 (1995): 443-551.](https://www.jstor.org/stable/2118559)

**Problem:**

```python
def sat(nums: List[int]):
    assert type(nums) is list and all(type(a) is int for a in nums), 'nums must be of type List[int]'
    a, b, c, n = nums
    return (a ** n + b ** n == c ** n) and min(a, b, c) > 0 and n > 2
```
### GCD
([number_theory](#number_theory) 2/16)

**Description:**
[Greatest Common Divisor](https://en.wikipedia.org/w/index.php?title=Greatest_common_divisor&oldid=990943381)
(GCD)

Find the greatest common divisor of two integers.

See also the [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm)

**Problem:**

```python
def sat(n: int, a: int=15482, b: int=23223, lower_bound: int=5):
    assert type(n) is int, 'n must be of type int'
    return a % n == 0 and b % n == 0 and n >= lower_bound
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### GCD_multi
([number_theory](#number_theory) 3/16)

**Description:**
[Greatest Common Divisor](https://en.wikipedia.org/w/index.php?title=Greatest_common_divisor&oldid=990943381)
(GCD)

Find the greatest common divisor of a *list* of integers.

See also the [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm)

**Problem:**

```python
def sat(n: int, nums: List[int]=[77410, 23223, 54187], lower_bound: int=2):
    assert type(n) is int, 'n must be of type int'
    return all(i % n == 0 for i in nums) and n >= lower_bound
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### LCM
([number_theory](#number_theory) 4/16)

**Description:**
[Least Common Multiple](https://en.wikipedia.org/wiki/Least_common_multiple)
(LCM)

Find the least common multiple of two integers.

See also the [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm)

**Problem:**

```python
def sat(n: int, a: int=15, b: int=27, upper_bound: int=150):
    assert type(n) is int, 'n must be of type int'
    return n % a == 0 and n % b == 0 and 0 < n <= upper_bound
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=15, b=27, upper_bound=150):
    m, n = min(a, b), max(a, b)
    while m > 0:
        m, n = n % m, m
    return a * (b // n)
```

</details>

### LCM_multi
([number_theory](#number_theory) 5/16)

**Description:**
[Least Common Multiple](https://en.wikipedia.org/wiki/Least_common_multiple)
(LCM)

Find the least common multiple of a list of integers.

See also the [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm)

**Problem:**

```python
def sat(n: int, nums: List[int]=[15, 27, 102], upper_bound: int=5000):
    assert type(n) is int, 'n must be of type int'
    return all(n % i == 0 for i in nums) and n <= upper_bound
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### SmallExponentBigSolution
([number_theory](#number_theory) 6/16)

**Description:**
Small exponent, big solution

Solve for n: b^n = target (mod n)

Problems have small b and target but solution is typically a large n.
Some of them are really hard, for example, for `b=2, target=3`, the smallest solution is `n=4700063497`

See [Richard K. Guy "The strong law of small numbers", (problem 13)](https://doi.org/10.2307/2322249)

**Problem:**

```python
def sat(n: int, b: int=2, target: int=5):
    assert type(n) is int, 'n must be of type int'
    return (b ** n) % n == target
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(b=2, target=5):
    for n in range(1, 10 ** 5):
        if pow(b, n, n) == target:
            return n
```

</details>

### ThreeCubes
([number_theory](#number_theory) 7/16)

**Description:**
Sum of three cubes

Given `n`, find integers `a`, `b`, `c` such that `a**3 + b**3 + c**3 = n`. This is unsolvable for `n % 9 in {4, 5}`.
Conjectured to be true for all other n, i.e., `n % 9 not in {4, 5}`.
`a`, `b`, `c` may be positive or negative

See [wikipedia entry](https://en.wikipedia.org/wiki/Sums_of_three_cubes) or
[Andrew R. Booker, Andrew V. Sutherland (2020). "On a question of Mordell."](https://arxiv.org/abs/2007.01209)

**Problem:**

```python
def sat(nums: List[int], target: int=10):
    assert type(nums) is list and all(type(a) is int for a in nums), 'nums must be of type List[int]'
    assert target % 9 not in [4, 5], "Hint"
    return len(nums) == 3 and sum([i ** 3 for i in nums]) == target
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(target: int=10):
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

### FourSquares
([number_theory](#number_theory) 8/16)

**Description:**
Sum of four squares

[Lagrange's Four Square Theorem](https://en.wikipedia.org/w/index.php?title=Lagrange%27s_four-square_theorem)

Given a non-negative integer `n`, a classic theorem of Lagrange says that `n` can be written as the sum of four
integers. The problem here is to find them. This is a nice problem and we give an elementary solution
that runs in time 	ilde{O}(n),
which is not "polynomial time" because it is not polynomial in log(n), the length of n. A poly-log(n)
algorithm using quaternions is described in the book:
["Randomized algorithms in number theory" by Michael O. Rabin and Jeffery O. Shallit (1986)](https://doi.org/10.1002/cpa.3160390713)

The first half of the problems involve small numbers and the second half involve some numbers up to 50 digits.

**Problem:**

```python
def sat(nums: List[int], n: int=12345):
    assert type(nums) is list and all(type(a) is int for a in nums), 'nums must be of type List[int]'
    return len(nums) <= 4 and sum(i ** 2 for i in nums) == n
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### Factoring
([number_theory](#number_theory) 9/16)

**Description:**
[Factoring](https://en.wikipedia.org/w/index.php?title=Integer_factorization) and
[RSA challenge](https://en.wikipedia.org/w/index.php?title=RSA_numbers)

The factoring problems require one to find any nontrivial factor of n, which is equivalent to factoring by a
simple repetition process. Problems range from small (single-digit n) all the way to the "RSA challenges"
which include several *unsolved* factoring problems put out by the RSA company. The challenge was closed in 2007,
with hundreds of thousands of dollars in unclaimed prize money for factoring their given numbers. People
continue to work on them, nonetheless, and only the first 22/53 have RSA challenges have been solved thusfar.

From Wikipedia:

RSA-2048 has 617 decimal digits (2,048 bits). It is the largest of the RSA numbers and carried the largest
cash prize for its factorization, $200,000. The RSA-2048 may not be factorizable for many years to come,
unless considerable advances are made in integer factorization or computational power in the near future.

**Problem:**

```python
def sat(i: int, n: int=62710561):
    assert type(i) is int, 'i must be of type int'
    return 1 < i < n and n % i == 0
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(n=62710561):
    if n % 2 == 0:
        return 2

    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return i

    assert False, "problem defined for composite n only"
```

</details>

### DiscreteLog
([number_theory](#number_theory) 10/16)

**Description:**
Discrete Log

The discrete logarithm problem is (given `g`, `t`, and `p`) to find n such that:

`g ** n % p == t`

From [Wikipedia article](https://en.wikipedia.org/w/index.php?title=Discrete_logarithm_records):

"Several important algorithms in public-key cryptography base their security on the assumption
that the discrete logarithm problem over carefully chosen problems has no efficient solution."

The problem is *unsolved* in the sense that no known polynomial-time algorithm has been found.

We include McCurley's discrete log challenge from
[Weber D., Denny T. (1998) "The solution of McCurley's discrete log challenge."](https://link.springer.com/content/pdf/10.1007/BFb0055747.pdf)

**Problem:**

```python
def sat(n: int, g: int=3, p: int=17, t: int=13):
    assert type(n) is int, 'n must be of type int'
    return pow(g, n, p) == t
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(g=3, p=17, t=13):
    for n in range(p):
        if pow(g, n, p) == t:
            return n
    assert False, f"unsolvable discrete log problem g={g}, t={t}, p={p}"
```

</details>

### GCD17
([number_theory](#number_theory) 11/16)

**Description:**
GCD 17

Find n for which gcd(n^17+9, (n+1)^17+9) != 1

According to [this article](https://primes.utm.edu/glossary/page.php?sort=LawOfSmall), the smallest
solution is 8424432925592889329288197322308900672459420460792433

**Problem:**

```python
def sat(n: int):
    assert type(n) is int, 'n must be of type int'
    i = n ** 17 + 9
    j = (n + 1) ** 17 + 9

    while i != 0:  # compute gcd using Euclid's algorithm
        (i, j) = (j % i, i)

    return n >= 0 and j != 1
```
### Znam
([number_theory](#number_theory) 12/16)

**Description:**
[Znam's Problem](https://en.wikipedia.org/wiki/Zn%C3%A1m%27s_problem)

Find k positive integers such that each integer divides (the product of the rest plus 1).

For example [2, 3, 7, 47, 395] is a solution for k=5

**Problem:**

```python
def sat(li: List[int], k: int=5):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    def prod(nums):
        ans = 1
        for i in nums:
            ans *= i
        return ans

    return min(li) > 1 and len(li) == k and all((1 + prod(li[:i] + li[i + 1:])) % li[i] == 0 for i in range(k))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### CollatzCycleUnsolved
([number_theory](#number_theory) 13/16)

**Description:**
Collatz Conjecture

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

**Problem:**

```python
def sat(n: int):
    assert type(n) is int, 'n must be of type int'
    m = n
    while n > 4:
        n = 3 * n + 1 if n % 2 else n // 2
        if n == m:
            return True
```
### CollatzGeneralizedUnsolved
([number_theory](#number_theory) 14/16)

**Description:**
Generalized Collatz Conjecture

This version, permits negative n and seek a cycle with a number of magnitude greater than 1000,
which would disprove the Generalized conjecture that states that the only cycles are the known 5 cycles
(which don't have positive integers).

See the [Wikipedia article](https://en.wikipedia.org/wiki/Collatz_conjecture)

**Problem:**

```python
def sat(start: int):
    assert type(start) is int, 'start must be of type int'
    n = start  # could be positive or negative ...
    while abs(n) > 1000:
        n = 3 * n + 1 if n % 2 else n // 2
        if n == start:
            return True
```
### CollatzDelay
([number_theory](#number_theory) 15/16)

**Description:**
Collatz Delay

Find `0 < n < upper` so that it takes exactly `t` steps to reach 1 in the Collatz process. For instance,
the number `n=9780657630` takes 1,132 steps and the number `n=93,571,393,692,802,302` takes
2,091 steps, according to the [Wikipedia article](https://en.wikipedia.org/wiki/Collatz_conjecture)

Now, this problem can be solved trivially by taking exponentially large `n = 2 ** t` so we also bound the
number of bits of the solution to be upper.

See [this webpage](http://www.ericr.nl/wondrous/delrecs.html) for up-to-date records.

**Problem:**

```python
def sat(n: int, t: int=100, upper: int=10):
    assert type(n) is int, 'n must be of type int'
    m = n
    for i in range(t):
        if n <= 1:
            return False
        n = 3 * n + 1 if n % 2 else n // 2
    return n == 1 and m <= 2 ** upper
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(t=100, upper=10):  # Faster solution for simultaneously solving multiple problems is of course possible
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

### Lehmer
([number_theory](#number_theory) 16/16)

**Description:**
Lehmer puzzle

Find n  such that 2^n mod n = 3

According to [The Strong Law of Large Numbers](https://doi.org/10.2307/2322249) Richard K. Guy states that
    D. H. & Emma Lehmer discovered that 2^n = 3 (mod n) for n = 4700063497,
    but for no smaller n > 1

**Problem:**

```python
def sat(n: int):
    assert type(n) is int, 'n must be of type int'
    return pow(2, n, n) == 3
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return 4700063497
```

</details>

[^^ Top](#files)
## probability

Probability problems

[^ Top](#files)

### BirthdayParadox
([probability](#probability) 1/5)

**Description:**
Find `n` such that the probability of two people having the same birthday in a group of `n` is near `1/2`.
The year length is year_len (365 is earth, while Neptune year is 60,182)
See [Birthday Problem](https://en.wikipedia.org/wiki/Birthday_problem (Mathematical Problems category))

**Problem:**

```python
def sat(n: int, year_len: int=365):
    assert type(n) is int, 'n must be of type int'
    prob = 1.0
    for i in range(n):
        prob *= (year_len - i) / year_len
    return (prob - 0.5) ** 2 <= 1/year_len
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### BirthdayParadoxMonteCarlo
([probability](#probability) 2/5)

**Description:**
A slower, Monte Carlo version of the above Birthday Paradox problem.

**Problem:**

```python
def sat(n: int, year_len: int=365):
    assert type(n) is int, 'n must be of type int'
    import random
    random.seed(0)
    K = 1000  # number of samples
    prob = sum(len({random.randrange(year_len) for i in range(n)}) < n for j in range(K)) / K
    return (prob - 0.5) ** 2 <= year_len
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### BallotProblem
([probability](#probability) 3/5)

**Description:**
Suppose a list of m 1's and n -1's are permuted at random. What is the probability that
all of the cumulative sums are positive?
The goal is to find `m` and `n` that make the probability of the ballot problem close to a
specific probability `target_prob`.
See the [Wikipedia article](https://en.wikipedia.org/wiki/Bertrand%27s_ballot_theorem) or
or  [Addario-Berry L., Reed B.A. (2008) Ballot Theorems, Old and New. In: Gyori E., Katona G.O.H., Lovász L.,
Sági G. (eds) Horizons of Combinatorics. Bolyai Society Mathematical Studies, vol 17.
Springer, Berlin, Heidelberg.](https://doi.org/10.1007/978-3-540-77200-2_1)

**Problem:**

```python
def sat(counts: List[int], target_prob: float=0.5):
    assert type(counts) is list and all(type(a) is int for a in counts), 'counts must be of type List[int]'
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
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(target_prob=0.5):
    for m in range(1, 10000):
        n = round(m * (1 - target_prob) / (1 + target_prob))
        if abs(target_prob - (m - n) / (m + n)) < 1e-6:
            return [m, n]
```

</details>

### BinomialProbabilities
([probability](#probability) 4/5)

**Description:**
Find `a`, `b` so that the probability of seeing `a` heads and `b` tails on `n = a + b` coin flips
is the given `target_prob`.
See [Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution)

**Problem:**

```python
def sat(counts: List[int], p: float=0.5, target_prob: float=0.0625):
    assert type(counts) is list and all(type(a) is int for a in counts), 'counts must be of type List[int]'
    from itertools import product
    a, b = counts
    n = a + b
    prob = (p ** a) * ((1-p) ** b)
    tot = sum([prob for sample in product([0, 1], repeat=n) if sum(sample) == a])
    return abs(tot - target_prob) < 1e-6
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### ExponentialProbability
([probability](#probability) 5/5)

**Description:**
Find stopping probability, so that the probability of stopping in `steps` or fewer time steps
is the given `target_prob`.
See [Exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution)

**Problem:**

```python
def sat(p_stop: float, steps: int=10, target_prob: float=0.5):
    assert type(p_stop) is float, 'p_stop must be of type float'
    prob = sum(p_stop*(1-p_stop)**t for t in range(steps))
    return abs(prob - target_prob) < 1e-6
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(steps=10, target_prob=0.5):
    return 1 - (1 - target_prob) ** (1.0/steps)
```

</details>

[^^ Top](#files)
## study


Problems used for the study.


[^ Top](#files)

### Study_1
([study](#study) 1/30)

**Description:**
Find a string with 1000 'o's but no two adjacent 'o's.

**Problem:**

```python
def sat(s: str):
    assert type(s) is str, 's must be of type str'
    return s.count('o') == 1000 and s.count('oo') == 0
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return ('h' + 'o') * 1000
```

</details>

### Study_2
([study](#study) 2/30)

**Description:**
Find a string with 1000 'o's, 100 pairs of adjacent 'o's and 801 copies of 'ho'.

**Problem:**

```python
def sat(s: str):
    assert type(s) is str, 's must be of type str'
    return s.count('o') == 1000 and s.count('oo') == 100 and s.count('ho') == 801
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return 'ho' * (800 + 1) + 'o' * (100 * 2 - 1)
```

</details>

### Study_3
([study](#study) 3/30)

**Description:**
Find a permutation of [0, 1, ..., 998] such that the ith element is *not* i, for all i=0, 1, ..., 998.

**Problem:**

```python
def sat(li: List[int]):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    return sorted(li) == list(range(999)) and all(li[i] != i for i in range(len(li)))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return [((i + 1) % 999) for i in range(999)]
```

</details>

### Study_4
([study](#study) 4/30)

**Description:**
Find a list of length 10 where the fourth element occurs exactly twice.

**Problem:**

```python
def sat(li: List[int]):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    return len(li) == 10 and li.count(li[3]) == 2
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return list(range(10 // 2)) * 2
```

</details>

### Study_5
([study](#study) 5/30)

**Description:**
Find a list integers such that the integer i occurs i times, for i = 0, 1, 2, ..., 9.

**Problem:**

```python
def sat(li: List[int]):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    return all([li.count(i) == i for i in range(10)])
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return [i for i in range(10) for j in range(i)]
```

</details>

### Study_6
([study](#study) 6/30)

**Description:**
Find an integer greater than 10^10 which is 4 mod 123.

**Problem:**

```python
def sat(i: int):
    assert type(i) is int, 'i must be of type int'
    return i % 123 == 4 and i > 10 ** 10
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return 4 + 10 ** 10 + 123 - 10 ** 10 % 123
```

</details>

### Study_7
([study](#study) 7/30)

**Description:**
Find a three-digit pattern  that occurs more than 8 times in the decimal representation of 8^2888.

**Problem:**

```python
def sat(s: str):
    assert type(s) is str, 's must be of type str'
    return str(8 ** 2888).count(s) > 8 and len(s) == 3
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    s = str(8 ** 2888)
    return max({s[i: i + 3] for i in range(len(s) - 2)}, key=lambda t: s.count(t))
```

</details>

### Study_8
([study](#study) 8/30)

**Description:**
Find a list of more than 1235 strings such that the 1234th string is a proper substring of the 1235th.

**Problem:**

```python
def sat(ls: List[str]):
    assert type(ls) is list and all(type(a) is str for a in ls), 'ls must be of type List[str]'
    return ls[1234] in ls[1235] and ls[1234] != ls[1235]
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return [''] * 1235 + ['a']
```

</details>

### Study_9
([study](#study) 9/30)

**Description:**
Find a way to rearrange the letters in the pangram "The quick brown fox jumps over the lazy dog" to
get the pangram "The five boxing wizards jump quickly". The answer should be represented as a list of index
mappings.

**Problem:**

```python
def sat(li: List[int]):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    return ["The quick brown fox jumps over the lazy dog"[i] for i in li] == list(
        "The five boxing wizards jump quickly")
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return ['The quick brown fox jumps over the lazy dog'.index(t) for t in 'The five boxing wizards jump quickly']
```

</details>

### Study_10
([study](#study) 10/30)

**Description:**
Find a palindrome of length greater than 11 in the decimal representation of 8^1818.

**Problem:**

```python
def sat(s: str):
    assert type(s) is str, 's must be of type str'
    return s in str(8 ** 1818) and s == s[::-1] and len(s) > 11
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### Study_11
([study](#study) 11/30)

**Description:**
Find a list of strings whose length (viewed as a string) is equal to the lexicographically largest element
and is equal to the lexicographically smallest element.

**Problem:**

```python
def sat(ls: List[str]):
    assert type(ls) is list and all(type(a) is str for a in ls), 'ls must be of type List[str]'
    return min(ls) == max(ls) == str(len(ls))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return ['1']
```

</details>

### Study_12
([study](#study) 12/30)

**Description:**
Find a list of 1,000 integers where every two adjacent integers sum to 9, and where the first
integer plus 4 is 9.

**Problem:**

```python
def sat(li: List[int]):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    return all(i + j == 9 for i, j in zip([4] + li, li)) and len(li) == 1000
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return [9 - 4, 4] * (1000 // 2)
```

</details>

### Study_13
([study](#study) 13/30)

**Description:**
Find a real number which, when you subtract 3.1415, has a decimal representation starting with 123.456.

**Problem:**

```python
def sat(x: float):
    assert type(x) is float, 'x must be of type float'
    return str(x - 3.1415).startswith("123.456")
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return 123.456 + 3.1415
```

</details>

### Study_14
([study](#study) 14/30)

**Description:**
Find a list of integers such that the sum of the first i integers is i, for i=0, 1, 2, ..., 19.

**Problem:**

```python
def sat(li: List[int]):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    return all([sum(li[:i]) == i for i in range(20)])
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return [1] * 20
```

</details>

### Study_15
([study](#study) 15/30)

**Description:**
Find a list of integers such that the sum of the first i integers is 2^i -1, for i = 0, 1, 2, ..., 19.

**Problem:**

```python
def sat(li: List[int]):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    return all(sum(li[:i]) == 2 ** i - 1 for i in range(20))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return [(2 ** i) for i in range(20)]
```

</details>

### Study_16
([study](#study) 16/30)

**Description:**
Find a real number such that when you add the length of its decimal representation to it, you get 4.5.
Your answer should be the string form of the number in its decimal representation.

**Problem:**

```python
def sat(s: str):
    assert type(s) is str, 's must be of type str'
    return float(s) + len(s) == 4.5
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return str(4.5 - len(str(4.5)))
```

</details>

### Study_17
([study](#study) 17/30)

**Description:**
Find a number whose decimal representation is *a longer string* when you add 1,000 to it than when you add 1,001.

**Problem:**

```python
def sat(i: int):
    assert type(i) is int, 'i must be of type int'
    return len(str(i + 1000)) > len(str(i + 1001))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return -1001
```

</details>

### Study_18
([study](#study) 18/30)

**Description:**
Find a list of strings that when you combine them in all pairwise combinations gives the six strings:
'berlin', 'berger', 'linber', 'linger', 'gerber', 'gerlin'

**Problem:**

```python
def sat(ls: List[str]):
    assert type(ls) is list and all(type(a) is str for a in ls), 'ls must be of type List[str]'
    return [s + t for s in ls for t in ls if s != t] == 'berlin berger linber linger gerber gerlin'.split()
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### Study_19
([study](#study) 19/30)

**Description:**
Find a set of integers whose pairwise sums make the set {0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 34}.
That is find set S such that, { i + j | i, j in S } = {0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 34}.

**Problem:**

```python
def sat(si: Set[int]):
    assert type(si) is set and all(type(a) is int for a in si), 'si must be of type Set[int]'
    return {i + j for i in si for j in si} == {0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 34}
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return {0, 1, 2, 3, 17}
```

</details>

### Study_20
([study](#study) 20/30)

**Description:**
Find a list of integers, starting with 0 and ending with 128, such that each integer either differs from
the previous one by one or is thrice the previous one.

**Problem:**

```python
def sat(li: List[int]):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    return all(j in {i - 1, i + 1, 3 * i} for i, j in zip([0] + li, li + [128]))
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return [1, 3, 4, 12, 13, 14, 42, 126, 127]
```

</details>

### Study_21
([study](#study) 21/30)

**Description:**
Find a list integers containing exactly three distinct values, such that no integer repeats
twice consecutively among the first eleven entries. (So the list needs to have length greater than ten.)

**Problem:**

```python
def sat(li: List[int]):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    return all([li[i] != li[i + 1] for i in range(10)]) and len(set(li)) == 3
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return list(range(3)) * 10
```

</details>

### Study_22
([study](#study) 22/30)

**Description:**
Find a string s containing exactly five distinct characters which also contains as a substring every other
character of s (e.g., if the string s were 'parrotfish' every other character would be 'profs').

**Problem:**

```python
def sat(s: str):
    assert type(s) is str, 's must be of type str'
    return s[::2] in s and len(set(s)) == 5
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return """abacadaeaaaaaaaaaa"""
```

</details>

### Study_23
([study](#study) 23/30)

**Description:**
Find a list of characters which are aligned at the same indices of the three strings 'dee', 'doo', and 'dah!'.

**Problem:**

```python
def sat(ls: List[str]):
    assert type(ls) is list and all(type(a) is str for a in ls), 'ls must be of type List[str]'
    return tuple(ls) in zip('dee', 'doo', 'dah!')
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return list(next(zip('dee', 'doo', 'dah!')))
```

</details>

### Study_24
([study](#study) 24/30)

**Description:**
Find a list of integers with exactly three occurrences of seventeen and at least two occurrences of three.

**Problem:**

```python
def sat(li: List[int]):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    return li.count(17) == 3 and li.count(3) >= 2
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return [17] * 3 + [3] * 2
```

</details>

### Study_25
([study](#study) 25/30)

**Description:**
Find a permutation of the string 'Permute me true' which is a palindrome.

**Problem:**

```python
def sat(s: str):
    assert type(s) is str, 's must be of type str'
    return sorted(s) == sorted('Permute me true') and s == s[::-1]
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return """""".join(sorted('Permute me true'[1:])[::2] + ['P'] + sorted('Permute me true'[1:])[::2][::-1])
```

</details>

### Study_26
([study](#study) 26/30)

**Description:**
Divide the decimal representation of 8^88 up into strings of length eight.

**Problem:**

```python
def sat(ls: List[str]):
    assert type(ls) is list and all(type(a) is str for a in ls), 'ls must be of type List[str]'
    return "".join(ls) == str(8 ** 88) and all(len(s) == 8 for s in ls)
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return [str(8 ** 88)[i:i + 8] for i in range(0, len(str(8 ** 88)), 8)]
```

</details>

### Study_27
([study](#study) 27/30)

**Description:**
Consider a digraph where each node has exactly one outgoing edge. For each edge (u, v), call u the parent and
v the child. Then find such a digraph where the grandchildren of the first and second nodes differ but they
share the same great-grandchildren. Represented this digraph by the list of children indices.

**Problem:**

```python
def sat(li: List[int]):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    return li[li[0]] != li[li[1]] and li[li[li[0]]] == li[li[li[1]]]
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return [1, 2, 3, 3]
```

</details>

### Study_28
([study](#study) 28/30)

**Description:**
Find a set of one hundred integers between 0 and 999 which all differ by at least ten from one another.

**Problem:**

```python
def sat(si: Set[int]):
    assert type(si) is set and all(type(a) is int for a in si), 'si must be of type Set[int]'
    return all(i in range(1000) and abs(i - j) >= 10 for i in si for j in si if i != j) and len(si) == 100
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return set(range(0, 1000, 10))
```

</details>

### Study_29
([study](#study) 29/30)

**Description:**
Find a set of more than 995 integers between 0 and 999, inclusive, such that each pair of integers have
squares that differ by at least 10.

**Problem:**

```python
def sat(si: Set[int]):
    assert type(si) is set and all(type(a) is int for a in si), 'si must be of type Set[int]'
    return all(i in range(1000) and abs(i * i - j * j) >= 10 for i in si for j in si if i != j) and len(si) > 995
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return set(range(6, 1000)).union({0, 4})
```

</details>

### Study_30
([study](#study) 30/30)

**Description:**
Define f(n) to be the residue of 123 times n mod 1000. Find a list of integers such that the first twenty one
are between 0 and 999, inclusive, and are strictly increasing in terms of f(n).

**Problem:**

```python
def sat(li: List[int]):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    return all([123 * li[i] % 1000 < 123 * li[i + 1] % 1000 and li[i] in range(1000) for i in range(20)])
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return sorted(range(1000), key=lambda n: 123*n % 1000)[:21]
```

```python
def sol():
    return list(range(1000))[::8][::-1]
```

</details>

[^^ Top](#files)
## trivial_inverse

Trivial problems. Typically for any function, you can construct a trivial example.
For instance, for the len function you can ask for a string of len(s)==100 etc.


[^ Top](#files)

### HelloWorld
([trivial_inverse](#trivial_inverse) 1/34)

**Description:**
Trivial example, no solutions provided

**Problem:**

```python
def sat(s: str):
    assert type(s) is str, 's must be of type str'
    return s + 'world' == 'Hello world'
```
### BackWorlds
([trivial_inverse](#trivial_inverse) 2/34)

**Description:**
Two solutions, no inputs

**Problem:**

```python
def sat(s: str):
    assert type(s) is str, 's must be of type str'
    return s[::-1] + 'world' == 'Hello world'
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return ' olleH'
```

```python
def sol():  # solution methods must begin with 'sol'
    return 'Hello '[::-1]
```

</details>

### StrAdd
([trivial_inverse](#trivial_inverse) 3/34)

**Description:**
Solve simple string addition problem.

**Problem:**

```python
def sat(st: str, a: str="world", b: str="Hello world"):
    assert type(st) is str, 'st must be of type str'
    return st + a == b
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a="world", b="Hello world"):
    return b[:len(b) - len(a)]
```

</details>

### StrSetLen
([trivial_inverse](#trivial_inverse) 4/34)

**Description:**
Find a string with `dups` duplicate chars

**Problem:**

```python
def sat(s: str, dups: int=2021):
    assert type(s) is str, 's must be of type str'
    return len(set(s)) == len(s) - dups
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(dups=2021):
    return "a" * (dups + 1)
```

</details>

### StrMul
([trivial_inverse](#trivial_inverse) 5/34)

**Description:**
Find a string which when repeated `n` times gives `target`

**Problem:**

```python
def sat(s: str, target: str="foofoofoofoo", n: int=2):
    assert type(s) is str, 's must be of type str'
    return s * n == target
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(target="foofoofoofoo", n=2):
    if n == 0:
        return ''
    return target[:len(target) // n]
```

</details>

### StrMul2
([trivial_inverse](#trivial_inverse) 6/34)

**Description:**
Find `n` such that `s` repeated `n` times gives `target`

**Problem:**

```python
def sat(n: int, target: str="foofoofoofoo", s: str="foofoo"):
    assert type(n) is int, 'n must be of type int'
    return s * n == target
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(target="foofoofoofoo", s="foofoo"):
    if len(s) == 0:
        return 1
    return len(target) // len(s)
```

</details>

### StrLen
([trivial_inverse](#trivial_inverse) 7/34)

**Description:**
Find a string of length `n`

**Problem:**

```python
def sat(s: str, n: int=1000):
    assert type(s) is str, 's must be of type str'
    return len(s) == n
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(n=1000):
    return 'a' * n
```

</details>

### StrAt
([trivial_inverse](#trivial_inverse) 8/34)

**Description:**
Find the index of `target` in string `s`

**Problem:**

```python
def sat(i: int, s: str="cat", target: str="a"):
    assert type(i) is int, 'i must be of type int'
    return s[i] == target
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(s="cat", target="a"):
    return s.index(target)
```

</details>

### StrNegAt
([trivial_inverse](#trivial_inverse) 9/34)

**Description:**
Find the index of `target` in `s` using a negative index.

**Problem:**

```python
def sat(i: int, s: str="cat", target: str="a"):
    assert type(i) is int, 'i must be of type int'
    return s[i] == target and i < 0
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(s="cat", target="a"):
    return - (len(s) - s.index(target))
```

</details>

### StrSlice
([trivial_inverse](#trivial_inverse) 10/34)

**Description:**
Find the three slice indices that give the specific `target` in string `s`

**Problem:**

```python
def sat(inds: List[int], s: str="hello world", target: str="do"):
    assert type(inds) is list and all(type(a) is int for a in inds), 'inds must be of type List[int]'
    i, j, k = inds
    return s[i:j:k] == target
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### StrIndex
([trivial_inverse](#trivial_inverse) 11/34)

**Description:**
Find a string whose *first* index in `big_str` is `index`

**Problem:**

```python
def sat(s: str, big_str: str="foobar", index: int=2):
    assert type(s) is str, 's must be of type str'
    return big_str.index(s) == index
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(big_str="foobar", index=2):
    return big_str[index:]
```

</details>

### StrIndex2
([trivial_inverse](#trivial_inverse) 12/34)

**Description:**
Find a string whose *first* index of `sub_str` is `index`

**Problem:**

```python
def sat(big_str: str, sub_str: str="foobar", index: int=2):
    assert type(big_str) is str, 'big_str must be of type str'
    return big_str.index(sub_str) == index
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(sub_str="foobar", index=2):
    i = ord('A')
    while chr(i) in sub_str:
        i += 1
    return chr(i) * index + sub_str
```

</details>

### StrIn
([trivial_inverse](#trivial_inverse) 13/34)

**Description:**
Find a string of length `length` that is in both strings `a` and `b`

**Problem:**

```python
def sat(s: str, a: str="hello", b: str="yellow", length: int=4):
    assert type(s) is str, 's must be of type str'
    return len(s) == length and s in a and s in b
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a="hello", b="yellow", length=4):
    for i in range(len(a) - length + 1):
        if a[i:i + length] in b:
            return a[i:i + length]
```

</details>

### StrIn2
([trivial_inverse](#trivial_inverse) 14/34)

**Description:**
Find a list of >= `count` distinct strings that are all contained in `s`

**Problem:**

```python
def sat(substrings: List[str], s: str="hello", count: int=15):
    assert type(substrings) is list and all(type(a) is str for a in substrings), 'substrings must be of type List[str]'
    return len(substrings) == len(set(substrings)) >= count and all(sub in s for sub in substrings)
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(s="hello", count=15):
    return [""] + sorted({s[j:i] for i in range(len(s) + 1) for j in range(i)})
```

</details>

### ListSetLen
([trivial_inverse](#trivial_inverse) 15/34)

**Description:**
Find a list with a certain number of duplicate items

**Problem:**

```python
def sat(li: List[int], dups: int=42155):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    return len(set(li)) == len(li) - dups
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(dups=42155):
    return [1] * (dups + 1)
```

</details>

### ListMul
([trivial_inverse](#trivial_inverse) 16/34)

**Description:**
Find a list that when multiplied n times gives the target list

**Problem:**

```python
def sat(li: List[int], target: List[int]=[17, 9, -1, 17, 9, -1], n: int=2):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    return li * n == target
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(target=[17, 9, -1, 17, 9, -1], n=2):
    if n == 0:
        return []
    return target[:len(target) // n]
```

</details>

### ListLen
([trivial_inverse](#trivial_inverse) 17/34)

**Description:**
Find a list of a given length n

**Problem:**

```python
def sat(li: List[int], n: int=85012):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    return len(li) == n
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(n=85012):
    return [1] * n
```

</details>

### ListAt
([trivial_inverse](#trivial_inverse) 18/34)

**Description:**
Find the index of an item in a list. Any such index is fine.

**Problem:**

```python
def sat(i: int, li: List[int]=[17, 31, 91, 18, 42, 1, 9], target: int=18):
    assert type(i) is int, 'i must be of type int'
    return li[i] == target
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(li=[17, 31, 91, 18, 42, 1, 9], target=18):
    return li.index(target)
```

</details>

### ListNegAt
([trivial_inverse](#trivial_inverse) 19/34)

**Description:**
Find the index of an item in a list using negative indexing.

**Problem:**

```python
def sat(i: int, li: List[int]=[17, 31, 91, 18, 42, 1, 9], target: int=91):
    assert type(i) is int, 'i must be of type int'
    return li[i] == target and i < 0
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(li=[17, 31, 91, 18, 42, 1, 9], target=91):
    return li.index(target) - len(li)
```

</details>

### ListSlice
([trivial_inverse](#trivial_inverse) 20/34)

**Description:**
Find three slice indices to achieve a given list slice

**Problem:**

```python
def sat(inds: List[int], li: List[int]=[42, 18, 21, 103, -2, 11], target: List[int]=[-2, 21, 42]):
    assert type(inds) is list and all(type(a) is int for a in inds), 'inds must be of type List[int]'
    i, j, k = inds
    return li[i:j:k] == target
```
<details><summary><strong>Reveal solution(s):</strong></summary>

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

### ListIndex
([trivial_inverse](#trivial_inverse) 21/34)

**Description:**
Find the item whose first index in `li` is `index`

**Problem:**

```python
def sat(item: int, li: List[int]=[17, 2, 3, 9, 11, 11], index: int=4):
    assert type(item) is int, 'item must be of type int'
    return li.index(item) == index
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(li=[17, 2, 3, 9, 11, 11], index=4):
    return li[index]
```

</details>

### ListIndex2
([trivial_inverse](#trivial_inverse) 22/34)

**Description:**
Find a list that contains `i` first at index `index`

**Problem:**

```python
def sat(li: List[int], i: int=29, index: int=10412):
    assert type(li) is list and all(type(a) is int for a in li), 'li must be of type List[int]'
    return li.index(i) == index
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(i=29, index=10412):
    return [i-1] * index + [i]
```

</details>

### ListIn
([trivial_inverse](#trivial_inverse) 23/34)

**Description:**
Find an item that is in both lists `a` and `b`

**Problem:**

```python
def sat(s: str, a: List[str]=['cat', 'dot', 'bird'], b: List[str]=['tree', 'fly', 'dot']):
    assert type(s) is str, 's must be of type str'
    return s in a and s in b
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=['cat', 'dot', 'bird'], b=['tree', 'fly', 'dot']):
    return next(s for s in b if s in a)
```

</details>

### IntNeg
([trivial_inverse](#trivial_inverse) 24/34)

**Description:**
Solve unary negation problem

**Problem:**

```python
def sat(x: int, a: int=93252338):
    assert type(x) is int, 'x must be of type int'
    return -x == a
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=93252338):
    return - a
```

</details>

### IntSum
([trivial_inverse](#trivial_inverse) 25/34)

**Description:**
Solve sum problem

**Problem:**

```python
def sat(x: int, a: int=1073258, b: int=72352549):
    assert type(x) is int, 'x must be of type int'
    return a + x == b
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=1073258, b=72352549):
    return b - a
```

</details>

### IntSub
([trivial_inverse](#trivial_inverse) 26/34)

**Description:**
Solve subtraction problem

**Problem:**

```python
def sat(x: int, a: int=-382, b: int=14546310):
    assert type(x) is int, 'x must be of type int'
    return x - a == b
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=-382, b=14546310):
    return a + b
```

</details>

### IntSub2
([trivial_inverse](#trivial_inverse) 27/34)

**Description:**
Solve subtraction problem

**Problem:**

```python
def sat(x: int, a: int=8665464, b: int=-93206):
    assert type(x) is int, 'x must be of type int'
    return a - x == b
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=8665464, b=-93206):
    return a - b
```

</details>

### IntMul
([trivial_inverse](#trivial_inverse) 28/34)

**Description:**
Solve multiplication problem

**Problem:**

```python
def sat(n: int, a: int=14302, b: int=5):
    assert type(n) is int, 'n must be of type int'
    return b * n + (a % b) == a
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=14302, b=5):
    return a // b
```

</details>

### IntDiv
([trivial_inverse](#trivial_inverse) 29/34)

**Description:**
Solve division problem

**Problem:**

```python
def sat(n: int, a: int=3, b: int=23463462):
    assert type(n) is int, 'n must be of type int'
    return b // n == a
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=3, b=23463462):
    if a == 0:
        return 2 * b
    for n in [b // a, b // a - 1, b // a + 1]:
        if b // n == a:
            return n
```

</details>

### IntDiv2
([trivial_inverse](#trivial_inverse) 30/34)

**Description:**
Find `n` that when divided by `b` is `a`

**Problem:**

```python
def sat(n: int, a: int=345346363, b: int=10):
    assert type(n) is int, 'n must be of type int'
    return n // b == a
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=345346363, b=10):
    return a * b
```

</details>

### IntSquareRoot
([trivial_inverse](#trivial_inverse) 31/34)

**Description:**
Compute square root of number.
The target has a round (integer) square root.

**Problem:**

```python
def sat(x: int, a: int=10201202001):
    assert type(x) is int, 'x must be of type int'
    return x ** 2 == a
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=10201202001):
    return int(a ** 0.5)
```

</details>

### IntNegSquareRoot
([trivial_inverse](#trivial_inverse) 32/34)

**Description:**
Compute negative square root of number.
The target has a round (integer) square root.

**Problem:**

```python
def sat(n: int, a: int=10000200001):
    assert type(n) is int, 'n must be of type int'
    return a == n * n and n < 0
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=10000200001):
    return -int(a ** 0.5)
```

</details>

### FloatSquareRoot
([trivial_inverse](#trivial_inverse) 33/34)

**Description:**
Compute square root of number.
The target might not have a round solution.
Accuracy of third decimal digit is required.

**Problem:**

```python
def sat(x: float, a: int=1020):
    assert type(x) is float, 'x must be of type float'
    return abs(x ** 2 - a) < 10 ** -3
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=1020):
    return a ** 0.5
```

</details>

### FloatNegSquareRoot
([trivial_inverse](#trivial_inverse) 34/34)

**Description:**
Compute (negative) square root of number.
The target might not have a round solution.
Accuracy of third decimal digit is required.

**Problem:**

```python
def sat(x: float, a: int=1020):
    assert type(x) is float, 'x must be of type float'
    return abs(x ** 2 - a) < 10 ** -3 and x < 0
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol(a=1020):
    return -a ** 0.5
```

</details>

[^^ Top](#files)
## tutorial


A few example puzzles that were presented with solutions to participants of the study.


[^ Top](#files)

### Tutorial1
([tutorial](#tutorial) 1/5)

**Description:**
Find a string that when concatenated onto 'Hello ' gives 'Hello world'.

**Problem:**

```python
def sat(s: str):
    assert type(s) is str, 's must be of type str'
    return "Hello " + s == "Hello world"
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return "world"
```

</details>

### Tutorial2
([tutorial](#tutorial) 2/5)

**Description:**
Find a string that when reversed and concatenated onto 'Hello ' gives 'Hello world'.

**Problem:**

```python
def sat(s: str):
    assert type(s) is str, 's must be of type str'
    return "Hello " + s[::-1] == "Hello world"
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return "world"[::-1]
```

</details>

### Tutorial3
([tutorial](#tutorial) 3/5)

**Description:**
Find a list of two integers whose sum is 3.

**Problem:**

```python
def sat(x: List[int]):
    assert type(x) is list and all(type(a) is int for a in x), 'x must be of type List[int]'
    return len(x) == 2 and sum(x) == 3
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return [1, 2]
```

</details>

### Tutorial4
([tutorial](#tutorial) 4/5)

**Description:**
Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b'.

**Problem:**

```python
def sat(s: List[str]):
    assert type(s) is list and all(type(a) is str for a in s), 's must be of type List[str]'
    return len(set(s)) == 1000 and all((x.count("a") > x.count("b")) and ('b' in x) for x in s)
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return ["a"*(i+2)+"b" for i in range(1000)]
```

</details>

### Tutorial5
([tutorial](#tutorial) 5/5)

**Description:**
Find an integer whose perfect square begins with 123456789 in its decimal representation.

**Problem:**

```python
def sat(n: int):
    assert type(n) is int, 'n must be of type int'
    return str(n * n).startswith("123456789")
```
<details><summary><strong>Reveal solution(s):</strong></summary>

```python
def sol():
    return int(int("123456789" + "0"*9) ** 0.5) + 1
```

</details>

[^^ Top](#files)
