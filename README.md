# Python Reasoning Challenges

This repo contains a dataset of python reasoning challenges which can be used to teach an AI python and evaluate an AI's ability to understand and write python programs. 

# [Click here to browse the challenges](/problems/README.md)

## What is a python reasoning challenge?

Each challenge takes the form of a python function that takes an answer as an argument. The goal is to find an answer which makes the function return `True`.

```python
def sat(s: str):
    return s + "world" == "Hello world"
```

The answer to the above challenge is the string `"Hello "` because `sat("Hell ")` returns `True`. The challenges range from trivial problems like this, to classic puzzles, to algorithms problems and problems from the [International Mathematical Olympiad](https://en.wikipedia.org/wiki/International_Mathematical_Olympiad) and open problems in mathematics. For instance, the classic [Towers of Hanoi](https://en.wikipedia.org/wiki/Tower_of_Hanoi) puzzle can be written as follows:

```python
def sat(moves: List[List[int]], num_disks=8):  # moves is list of [from, to] pairs
    state = (list(range(num_disks)), [], [])
    for [i, j] in moves:
        state[j].append(state[i].pop())
        assert state[j] == sorted(state[j]), "larger disk on top of smaller disk"
    return state[0] == state[1] == []

```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
