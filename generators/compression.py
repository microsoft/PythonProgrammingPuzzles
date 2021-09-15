"""Puzzles relating to de/compression."""

from puzzle_generator import PuzzleGenerator
from typing import List


# See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle to learn about adding puzzles


def _compress_LZW(text):  # for development
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


def _decompress_LZW(seq: List[int]):  # for development
    index = [chr(i) for i in range(256)]
    pieces = [""]
    for i in seq:
        pieces.append(pieces[-1] + pieces[-1][0] if i == len(index) else index[i])
        index.append(pieces[-2] + pieces[-1][0])
    return "".join(pieces)


class LZW(PuzzleGenerator):
    """
    We have provided a simple version of the *decompression* algorithm of
    [Lempel-Ziv-Welch](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch)
    so the solution is the *compression* algorithm.
    """

    # _compress_LZW("Hellooooooooooooooooooooo world!") is length-17

    @staticmethod
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

    @staticmethod
    def sol(compressed_len, text):  # compressed_len is ignored
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

    def gen(self, _target_num_instances):
        self.add({"text": "", "compressed_len": 0})
        self.add({"text": "c" * 1000, "compressed_len": len(_compress_LZW("c" * 1000))})

    def gen_random(self):
        max_len = self.random.choice([10, 100, 1000])
        text = self.random.pseudo_word(0, max_len)
        self.add({"text": text, "compressed_len": len(_compress_LZW(text))})


class LZW_decompress(PuzzleGenerator):
    """We have provided a simple version of the
    [Lempel-Ziv-Welch](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch)
    and the solution is the *decompression* algorithm.
    """

    @staticmethod
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

    @staticmethod
    def sol(seq):
        index = [chr(i) for i in range(256)]
        pieces = [""]
        for i in seq:
            pieces.append(pieces[-1] + pieces[-1][0] if i == len(index) else index[i])
            index.append(pieces[-2] + pieces[-1][0])
        return "".join(pieces)

    def gen(self, _target_num_instances):
        for s in ['', 'a', 'b' * 1000, 'ab' * 1000 + '!']:
            self.add({"seq": _compress_LZW(s)})

    def gen_random(self):
        max_len = self.random.choice([10, 100, 1000])
        text = self.random.pseudo_word(0, max_len)
        self.add({"seq": _compress_LZW(text)})


class PackingHam(PuzzleGenerator):
    """
    This packing problem a [classic problem](https://en.wikipedia.org/wiki/Sphere_packing#Other_spaces)
    in coding theory.
    """

    @staticmethod
    def sat(words: List[str], num=100, bits=100, dist=34):
        """Pack a certain number of binary strings so that they have a minimum hamming distance between each other."""
        assert len(words) == num and all(len(word) == bits and set(word) <= {"0", "1"} for word in words)
        return all(sum([a != b for a, b in zip(words[i], words[j])]) >= dist for i in range(num) for j in range(i))

    @staticmethod
    def sol(num, bits, dist):
        import random  # key insight, use randomness!
        r = random.Random(0)
        while True:
            seqs = [r.getrandbits(bits) for _ in range(num)]
            if all(bin(seqs[i] ^ seqs[j]).count("1") >= dist for i in range(num) for j in range(i)):
                return [bin(s)[2:].rjust(bits, '0') for s in seqs]

    def gen_random(self):
        bits = self.random.randrange(1, self.random.choice([10, 100]))
        num = self.random.randrange(2, self.random.choice([10, 100]))

        def score(seqs):
            return min(bin(seqs[i] ^ seqs[j]).count("1") for i in range(num) for j in range(i))

        # best of 5
        seqs = min([[self.random.getrandbits(bits) for _ in range(num)] for _ in range(5)], key=score)
        dist = score(seqs)
        if dist > 0:
            self.add(dict(num=num, bits=bits, dist=dist))


if __name__ == "__main__":
    PuzzleGenerator.debug_problems()
