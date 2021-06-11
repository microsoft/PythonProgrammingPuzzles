# provides get_puzzles

# parts separated by ======== (exactly 8)
# puzzles  by ----'s  (exactly 4)


def get_puzzles():
    """
    Creates the puzzles for the study

    :return: list of {"src": str, "name": str, "part": str}
    """
    part_names = ["WARM UP", "PART 1/3", "PART 2/3", "PART 3/3"]

    raw_puzzles = '''

def puzzle(s: str):
    """
    Warmup problem.
    """
    return "Hello " + s == 'Hello world'

----

def puzzle(n: int):
    """
    Hint: puzzle(111111111) works.
    """
    return str(n * n).startswith("123456789")

----

def puzzle(x: str):
    """
    Hint: note that the input should be a string.
    """
    return -1 * int(x) == 1337


========    

def puzzle(s: str):
    return s.count('o') == 1000 and s.count('oo') == 0


----


def puzzle(s: str):
    return s.count('o') == 1000 and s.count('oo') == 100 and s.count('ho') == 801

----

def puzzle(x: List[int]):
    return sorted(x) == list(range(999)) and all(x[i] != i for i in range(len(x)))

----

def puzzle(x: List[int]):
    return len(x) == 10 and x.count(x[3]) == 2

----


def puzzle(x: List[int]):
    return all([x.count(i) == i for i in range(10)]) 

----

def puzzle(n: int):
    return n % 123 == 4 and n > 10**10


----

def puzzle(s: str):
    return str(8**2888).count(s) > 8 and len(s) == 3

----

def puzzle(s: List[str]):
    return s[1234] in s[1235] and s[1234] != s[1235]

----

def puzzle(x: List[int]):
    return ["The quick brown fox jumps over the lazy dog"[i] for i in x] == list("The five boxing wizards jump quickly")

----

def puzzle(s: str):
     return s in str(8**1818) and s==s[::-1] and len(s)>11

========

def puzzle(x: List[str]):
    return min(x) == max(x) == str(len(x))


----

def puzzle(x: List[int]):
    return all(a + b == 9 for a, b in zip([4] + x, x)) and len(x) == 1000


----


def puzzle(x: float):
    return str(x - 3.1415).startswith("123.456")

----

def puzzle(x: List[int]):
   return all([sum(x[:i]) == i for i in range(20)])

----

def puzzle(x: List[int]):
   return all(sum(x[:i]) == 2 ** i - 1 for i in range(20)) 

----

def puzzle(x: str):
    return float(x) + len(x) == 4.5

----

def puzzle(n: int):
    return len(str(n + 1000)) > len(str(n + 1001)) 

----

def puzzle(x: List[str]): 
    return [s + t for s in x for t in x if s!=t] == 'berlin berger linber linger gerber gerlin'.split()

----

def puzzle(x: Set[int]):
    return {i + j for i in x for j in x} == {0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 34}

----
    

def puzzle(x: List[int]):    
    return all(b in {a-1, a+1, 3*a} for a, b in zip([0] + x, x + [128]))


========

def puzzle(x: List[int]):
   return all([x[i] != x[i + 1] for i in range(10)]) and len(set(x)) == 3

----

def puzzle(x: str):
    return x[::2] in x and len(set(x)) == 5

----

def puzzle(x: List[str]):
    return tuple(x) in zip('dee', 'doo', 'dah!')

----

def puzzle(x: List[int]):
    return x.count(17) == 3 and x.count(3) >= 2


----

def puzzle(s: str):
    return sorted(s)==sorted('Permute me true') and s==s[::-1]


----
def puzzle(x: List[str]):
   return "".join(x) == str(8**88) and all(len(s)==8 for s in x)


----

def puzzle(x: List[int]):
   return x[x[0]] != x[x[1]] and x[x[x[0]]] == x[x[x[1]]]

----

def puzzle(x: Set[int]):
   return all(i in range(1000) and abs(i-j) >= 10 for i in x for j in x if i != j) and len(x)==100  

----


def puzzle(x: Set[int]):
   return all(i in range(1000) and abs(i*i - j*j) >= 10 for i in x for j in x if i != j) and len(x) > 995

----

def puzzle(x: List[int]):
    return all([123*x[i] % 1000 < 123*x[i+1] % 1000 and x[i] in range(1000) for i in range(20)])




       '''

    parts = [[src.strip() for src in part.split("----")] for part in raw_puzzles.split("========")]
    assert len(part_names) == len(parts)
    ans = []
    for part_name, part in zip(part_names, parts):
        per_part_num = 1
        for src in part:
            ans.append({"src": src,
                        "part": part_name,
                        "name": f"PUZZLE {per_part_num}/{len(part)}"})
            per_part_num += 1
    return ans


