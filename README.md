# Python Programming Puzzles (P3)

This repo contains a dataset of python programming puzzles which can be used to teach and evaluate
an AI's programming proficiency. We hope this dataset with **grow rapidly**, and it is already diverse in 
terms of problem difficult, domain, 
and algorithmic tools needed to solve the problems. Please
[propose a new puzzle](../../issues/new?assignees=akalai&labels=New-puzzle&template=new-puzzle.md&title=New+puzzle) 
 or [browse newly proposed puzzles](../../issues?q=is%3Aopen+is%3Aissue+label%3ANew-puzzle) 
or [contribute through pull requests](../../wiki/How-to-add-a-puzzle). 
 
To learn more about how well AI systems such as GPT-3 can solve these problems, read our paper:

[Programming Puzzles](https://arxiv.org/abs/2106.05784). Tal Schuster, Ashwin Kalyan, Oleksandr Polozov, 
Adam Tauman Kalai.
```
@misc{schuster2021programming,
      title={Programming Puzzles}, 
      author={Tal Schuster and Ashwin Kalyan and Oleksandr Polozov and Adam Tauman Kalai},
      year={2021},
      eprint={2106.05784},
      archivePrefix={arXiv},      
}
```

If you just want to dive right into solving a few puzzles, try the [intro notebook at Binder](https://aka.ms/python_puzzles) 
that shows which puzzles the AI baselines solved and which they did not, so you can see how 
your programming compares. 

## What is a python programming puzzle?

Each puzzle takes the form of a python function that takes an answer as an argument. 
The goal is to find an answer which makes the function return `True`. 
This is called *satisfying* the puzzle, and that is why the puzzles are all named `sat`.

```python
def sat(s: str):
    return "Hello " + s == "Hello world"
```

The answer to the above puzzle is the string `"world"` because `sat("world")` returns `True`. The puzzles range from trivial problems like this, to classic puzzles, 
to programming competition problems, all the way through open problems in algorithms and mathematics. 
A slightly harder example is:
```python
def sat(s: str):  
    """find a string with 1000 o's but no consecutive o's."""
    return s.count("o") == 1000 and s.count("oo") == 0
```

A more challenging puzzle that requires [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) is the 
[longest increasing subsequence](https://en.wikipedia.org/wiki/Longest_increasing_subsequence) problem
which we can also describe with strings:
```python
from typing import List

def sat(x: List[int], s="Dynamic programming solves this classic job-interview puzzle!!!"): 
    """Find the indexes (possibly negative!) of the longest monotonic subsequence"""    
    return all(s[x[i]] <= s[x[i+1]] and x[i+1] > x[i] for i in range(25))
```

The classic [Towers of Hanoi](https://en.wikipedia.org/wiki/Tower_of_Hanoi) puzzle can be written as follows:
```python
def sat(moves: List[List[int]]):  
    """moves is list of [from, to] pairs"""
    t = ([8, 7, 6, 5, 4, 3, 2, 1], [], [])  # towers state
    return all(t[j].append(t[i].pop()) or t[j][-1] == min(t[j]) for i, j in moves) and t[0] == t[1]

```

For more information on the motivation and how programming puzzles can help AI learn to program, see 
the paper:  
*Programming Puzzles*, by Tal Schuster, Ashwin Kalyan, Alex Polozov, and Adam Tauman Kalai. 2021 (Link to be added shortly)  

# [Click here to browse the puzzles](/problems/README.md)

The problems in this repo are based on:
* Wikipedia articles about [algorithms](https://en.wikipedia.org/wiki/List_of_algorithms), [puzzles](https://en.wikipedia.org/wiki/Category:Logic_puzzles),
and [math problems](https://en.wikipedia.org/wiki/List_of_unsolved_problems_in_mathematics).
* The website [codeforces.com](https://codeforces.com), a popular website for programming competition problems
* Olympiad problems from the [International Collegiate Programming Contest](https://icpc.global) and [International Mathematical Olympiad](https://en.wikipedia.org/wiki/International_Mathematical_Olympiad).


## Notebooks

The [notebooks](/notebooks) subdirectory has some relevant notebooks. [Intro.ipynb](/notebooks/Intro.ipynb)
has a dozen puzzles indicating which ones the AI solved and did not [Try the notebook at Binder](https://aka.ms/python_puzzles)
and see how your programming compares to the AI baselines! 
 
 [Demo.ipynb](/notebooks/Demo.ipynb)
has the 30 problems completed by our users in a user study. Try the [demo notebook](https://aka.ms/python_puzzles_study)
and see how your programming compares to the AI baselines! 

### Hackathon
During a Microsoft hackathon July 27-29, 2020, several people completed 30 user 
[study puzzles](/problems/README.md#study). We also had tons of fun making the puzzles in 
[Hackathon_puzzles.ipynb](/notebooks/Hackathon_puzzles.ipynb). These are of a somewhat
different flavor as they are more often `hacks` like 
```python
def f(x):
    return x > x
```
where the type of `x` is clearly non-standard. The creators of these puzzles include github users: 
[Adam Tauman Kalai](https://github.com/akalai),
    [Alec Helbling](https://github.com/helblazer811),
    [Alexander Vorobev](https://github.com/OnFireDolphin),
    [Alexander Wei](https://github.com/aw31),
    [Alexey Romanov](https://github.com/jgc128),
    [Keith Battaochi](https://github.com/kbattocchi),
    [Kodai Sudo](https://github.com/kouml),
    [Maggie Hei](https://github.com/heimengqi),
    [Mariia Mykhailova](https://github.com/tcNickolas),
    [Misha Khodak](https://github.com/mkhodak),
    [Monil Mehta](https://github.com/monilm2),
    [Philip Rosenfield](https://github.com/philrosenfield),
    [Qida Ma](https://github.com/JerryMa90),
    [Raj Bhargava](https://github.com/rajbhargava),
    [Rishi Jaiswal](https://github.com/nextquanta),
    [Saikiran Mullaguri](https://github.com/sm947),
    [Tal Schuster](https://github.com/TalSchuster), and
    [Varsha Srinivasan](https://github.com/varsha2197). 
You can try out the notebook at (link to be added).


## Highlights
* Numerous trivial puzzles like reversing a list, useful for learning to program 
* Classic puzzles like:
    * Towers of Hanoi
    * Verbal Arithmetic (solve digit-substitutions like SEND + MORE = MONEY)
    * The Game of Life (e.g., finding oscillators of a given period, some **open**) 
    * Chess puzzles (e.g., knight's tour and n-queen problem variants)         
* Two-player games
    * Finding optimal strategies for Tic-Tac-Toe, Rock-Paper-Scissors, Mastermind (to add: connect four?)
    * Finding minimax strategies for zero-sum bimatrix games, which is equivalent to linear programming
    * Finding Nash equilibria of general-sum games (**open**, PPAD complete)
* Math and programming competitions
    * International Mathematical Olympiad (IMO) problems 
    * International Collegiate Programming Contest (ICPC) problems
    * Competitive programming problems from codeforces.com 
* Graph theory algorithmic puzzles
    * Shortest path
    * Planted clique (open)
* Elementary algebra 
    * Solving equations
    * Solving quadratic, cubic, and quartic equations
* Number theory algorithmic puzzles:
    * Finding common divisors (e.g., using Euclid's algorithm)
    * Factoring numbers (easy for small factors, over $100k in prizes have been awarded and **open** 
    for large numbers)
    * Discrete log (again **open** in general, easy for some)
* Lattices
    * Learning parity (typically solved using Gaussian elimination)
    * Learning parity with noise (**open**)
* Compression
    * Compress a given string given the decompression algorithm (but not the compression algorithm), or decompress a given 
    compressed string given only the compression algorithm
    * (to add: compute huffman tree)
* Hard math problems
    * Conway's 99-graph problem (**open**)
    * Finding a cycle in the Collatz process (**open**)


## Contributing

This project welcomes contributions and suggestions. Use your creativity to help teach 
AI's to program! See our [wiki on how to add a puzzle](../../wiki/How-to-add-a-puzzle).

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

See the [datasheet](DATASHEET.md) for our dataset.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

