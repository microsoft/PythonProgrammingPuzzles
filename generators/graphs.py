"""Problems related to graphs such as Conway's 99 problem, finding
[cliques](https://en.wikipedia.org/wiki/Clique_(graph_theory)) of various sizes, shortest path (Dijkstra) """

from puzzle_generator import PuzzleGenerator
from typing import List


# See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle to learn about adding puzzles


class Conway99(PuzzleGenerator):
    """Conway's 99-graph problem (*unsolved*, open problem)

    Conway's 99-graph problem is an unsolved problem in graph theory.
    In Conway's terminology, from [Five $1,000 Problems (Update 2017)](https://oeis.org/A248380/a248380.pdf)
    "Is there a graph with 99 vertices in which every edge (i.e. pair of joined vertices) belongs to a unique
    triangle and every nonedge (pair of unjoined vertices) to a unique quadrilateral?"

    See also this [Wikipedia article](https://en.wikipedia.org/w/index.php?title=Conway%27s_99-graph_problem).
    """

    @staticmethod
    def sat(edges: List[List[int]]):
        """
        Find an undirected graph with 99 vertices, in which each two adjacent vertices have exactly one common
        neighbor, and in which each two non-adjacent vertices have exactly two common neighbors.
        """
        # first compute neighbors sets, N:
        N = {i: {j for j in range(99) if j != i and ([i, j] in edges or [j, i] in edges)} for i in range(99)}
        return all(len(N[i].intersection(N[j])) == (1 if j in N[i] else 2) for i in range(99) for j in range(i))


def dedup_edges(stuff):
    seen = set()
    return [a for a in stuff if tuple(a) not in seen and not seen.add(tuple(a))]


class AnyEdge(PuzzleGenerator):
    """Trivial [graph](https://en.wikipedia.org/w/index.php?title=Graph_(discrete_mathematics)) problem."""

    @staticmethod
    def sat(e: List[int], edges=[[0, 217], [40, 11], [17, 29], [11, 12], [31, 51]]):
        """Find any edge in edges."""
        return e in edges

    @staticmethod
    def sol(edges):
        return edges[0]

    def gen_random(self):
        n = self.random.randrange(1, self.random.choice([10, 100]))
        m = self.random.randrange(1, 10 * n)
        # random graph:
        edges = dedup_edges([[self.random.randrange(n + 1), self.random.randrange(n + 1)] for _ in range(m)])
        self.add({"edges": edges})


class AnyTriangle(PuzzleGenerator):
    """
    Easy [graph](https://en.wikipedia.org/w/index.php?title=Graph_(discrete_mathematics)) problem,
    see [triangle](https://en.wikipedia.org/w/index.php?title=Triangle_graph)
    """

    @staticmethod
    def sat(tri: List[int], edges=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):
        """Find any triangle in the given directed graph."""
        a, b, c = tri
        return [a, b] in edges and [b, c] in edges and [c, a] in edges and a != b != c != a

    @staticmethod
    def sol(edges):
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

    def gen_random(self):
        n = self.random.randrange(1, self.random.choice([10, 100]))
        m = self.random.randrange(1, 10 * n)
        # random graph:
        edges = dedup_edges([[self.random.randrange(n + 1), self.random.randrange(n + 1)] for _ in range(m)])
        tri = self.sol(edges)
        if tri:
            assert self.sat(tri, edges)
            self.add({"edges": edges})


########################################################################################################################


class PlantedClique(PuzzleGenerator):
    """Find a [planted clique](https://en.wikipedia.org/w/index.php?title=Planted_clique) of a given size
    in an undirected graph. Finding a polynomial-time algorithm for this problem has been *unsolved* for
    some time."""

    @staticmethod
    def sat(nodes: List[int], size=3, edges=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):
        """Find a clique of the given size in the given undirected graph. It is guaranteed that such a clique exists."""
        assert len(nodes) == len(set(nodes)) >= size
        edge_set = {(a, b) for (a, b) in edges}
        for a in nodes:
            for b in nodes:
                assert a == b or (a, b) in edge_set or (b, a) in edge_set

        return True

    @staticmethod
    def sol(size, edges):
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

    def gen_random(self):
        n = self.random.randrange(1, self.random.choice([10, 20, 50, 100]))
        m = self.random.randrange(1, 10 * n)
        # random graph:
        edges = [[self.random.randrange(n + 1), self.random.randrange(n + 1)] for _ in range(m)]
        size = self.random.randrange(min(20, n))
        clique = self.random.sample(range(n), size)
        for a in clique:  # plant clique!
            for b in clique:
                if a < b:
                    edges.append(self.random.choice([[a, b], [b, a]]))

        edges = dedup_edges(edges)
        self.random.shuffle(edges)
        self.add({"edges": edges, "size": size}, test=(size <= 10))


class ShortestPath(PuzzleGenerator):
    """Shortest Path, see (Dijkstra's algorithm)[https://en.wikipedia.org/w/index.php?title=Dijkstra%27s_algorithm]"""

    @staticmethod
    def sat(path: List[int], weights=[{1: 20, 2: 1}, {2: 2, 3: 5}, {1: 10}], bound=11):
        """
        Find a path from node 0 to node 1, of length at most bound, in the given digraph.
        weights[a][b] is weight on edge [a,b] for (int) nodes a, b
        """
        return path[0] == 0 and path[-1] == 1 and sum(weights[a][b] for a, b in zip(path, path[1:])) <= bound

    @staticmethod
    def sol(weights, bound):
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
        # no path

    def gen_random(self):
        n = self.random.randrange(1, self.random.choice([10, 20, 50, 100]))
        m = self.random.randrange(n, 5 * n)
        # random graph:
        weights = [{} for _ in range(n)]
        for _ in range(m):
            weights[self.random.randrange(n)][self.random.randrange(n)] = self.random.randrange(1000)
        path = self.sol(weights, bound=None)
        if path:
            bound = sum(weights[a][b] for a, b in zip(path, path[1:]))
            assert self.sat(path, weights, bound)
            self.add(dict(weights=weights, bound=bound))


class UnweightedShortestPath(PuzzleGenerator):
    """
    Unweighted Shortest Path

    See (Dijkstra's algorithm)[https://en.wikipedia.org/w/index.php?title=Dijkstra%27s_algorithm]
    """

    @staticmethod
    def sat(path: List[int],
            edges=[[0, 11], [0, 7], [7, 5], [0, 22], [11, 22], [11, 33], [22, 33]],
            u=0,
            v=33,
            bound=3):
        """Find a path from node u to node v, of a bounded length, in the given digraph on vertices 0, 1,..., n."""
        assert path[0] == u and path[-1] == v and all([i, j] in edges for i, j in zip(path, path[1:]))
        return len(path) <= bound

    @staticmethod
    def sol(edges, u, v, bound):
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
        # no path

    def gen_random(self):
        n = self.random.randrange(1, self.random.choice([10, 20, 50, 100]))
        m = self.random.randrange(n, 5 * n)
        # random graph:
        edges = dedup_edges([self.random.randrange(n + 1), self.random.randrange(n + 1)] for _ in range(5 * n))
        u = self.random.randrange(n)
        v = self.random.randrange(n)
        path = self.sol(edges, u, v, bound=None)
        if path:
            bound = len(path)
            assert self.sat(path, edges, u, v, bound)
            self.add(dict(u=u, v=v, edges=edges, bound=bound))


class AnyPath(PuzzleGenerator):
    """Any Path"""

    @staticmethod
    def sat(path: List[int], edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [1, 2]]):
        """ Find any path from node 0 to node n in a given digraph on vertices 0, 1,..., n."""
        for i in range(len(path) - 1):
            assert [path[i], path[i + 1]] in edges
        assert path[0] == 0
        assert path[-1] == max(max(edge) for edge in edges)
        return True

    @staticmethod
    def sol(edges):
        n = max(max(edge) for edge in edges)
        paths = {0: [0]}
        for _ in range(n + 1):
            for i, j in edges:
                if i in paths and j not in paths:
                    paths[j] = paths[i] + [j]
        return paths.get(n)

    def gen_random(self):
        n = self.random.randrange(1, self.random.choice([10, 100]))
        # random graph:
        edges = dedup_edges([self.random.randrange(n), self.random.randrange(n)] for _ in range(2 * n))
        if self.sol(edges):
            self.add(dict(edges=edges))


class EvenPath(PuzzleGenerator):

    @staticmethod
    def sat(path: List[int], edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [1, 2]]):
        """Find a path with an even number of nodes from nodes 0 to n in the given digraph on vertices 0, 1,..., n."""
        assert path[0] == 0 and path[-1] == max(max(e) for e in edges)
        assert all([[a, b] in edges for a, b in zip(path, path[1:])])
        return len(path) % 2 == 0

    @staticmethod
    def sol(edges):
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

    def gen_random(self):
        n = self.random.randrange(1, self.random.choice([10, 100]))
        # random graph:
        edges = dedup_edges([self.random.randrange(n), self.random.randrange(n)] for _ in range(2 * n))
        if self.sol(edges):
            self.add(dict(edges=edges))


class OddPath(PuzzleGenerator):
    """To make it even more different than EvenPath, we changed to go from node 0 to node *1*."""

    @staticmethod
    def sat(p: List[int], edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 4], [5, 6], [6, 7], [6, 1]]):
        """Find a path with an even number of nodes from nodes 0 to 1 in the given digraph on vertices 0, 1,..., n."""
        return p[0] == 0 and p[-1] == 1 == len(p) % 2 and all([[a, b] in edges for a, b in zip(p, p[1:])])

    @staticmethod
    def sol(edges):
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

    def gen_random(self):
        n = self.random.randrange(1, self.random.choice([10, 100]))
        # random graph:
        edges = dedup_edges([self.random.randrange(n), self.random.randrange(n)] for _ in range(2 * n))
        if self.sol(edges):
            self.add(dict(edges=edges))


class Zarankiewicz(PuzzleGenerator):
    """[Zarankiewicz problem](https://en.wikipedia.org/wiki/Zarankiewicz_problem)"""

    @staticmethod
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

    @staticmethod
    def sol(z, n, t):
        from itertools import combinations
        all_edges = [(a, b) for a in range(n) for b in range(n)]
        for edges in combinations(all_edges, z):
            edge_set = set(edges)
            if all(any((a, b) not in edge_set for a in left for b in right)
                   for left in combinations(range(n), t)
                   for right in combinations(range(n), t)):
                return [[a, b] for a, b in edges]

    def gen(self, target_num_instances):
        if self.num_generated_so_far() < target_num_instances:
            self.add(dict(z=26, n=6, t=3), test=False)
        if self.num_generated_so_far() < target_num_instances:
            self.add(dict(z=13, n=4, t=3))


class GraphIsomorphism(PuzzleGenerator):
    """
    The classic [Graph Isomorphism](https://en.wikipedia.org/wiki/Graph_isomorphism) problem.
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
    allowed, one can do this by adding N additional nodes for each test.
    """

    @staticmethod
    def sat(bi: List[int], g1=[[0, 1], [1, 2], [2, 3], [3, 4], [2, 5]], g2=[[0, 4], [1, 5], [4, 1], [1, 2], [2, 3]]):
        """
        You are given two graphs which are permutations of one another and the goal is to find the permutation.
        Each graph is specified by a list of edges where each edge is a pair of integer vertex numbers.
        """
        return len(bi) == len(set(bi)) and {(i, j) for i, j in g1} == {(bi[i], bi[j]) for i, j in g2}

    @staticmethod
    def sol(g1, g2):
        # exponentially slow
        from itertools import permutations
        n = max(i for g in [g1, g2] for e in g for i in e) + 1
        g1_set = {(i, j) for i, j in g1}
        for pi in permutations(range(n)):
            if all((pi[i], pi[j]) in g1_set for i, j in g2):
                return list(pi)
        assert False, f"Graphs are not isomorphic {g1}, {g2}"

    def gen_random(self):
        n = self.random.randrange(20)
        g1 = sorted({(self.random.randrange(n), self.random.randrange(n)) for _ in range((n * n) // 2)})
        if not g1:
            return
        pi = list(range(n))
        self.random.shuffle(pi)
        g1 = [[i, j] for i, j in g1]
        g2 = [[pi[i], pi[j]] for i, j in g1]
        self.random.shuffle(g2)
        self.add(dict(g1=g1, g2=g2), test=n < 10)  # only test for small n


class ShortIntegerPath(PuzzleGenerator):
    """This is a more interesting version of Study_20 with an additional length constraint. One can think of the graph
    defined by the integer pairs."""

    @staticmethod
    def sat(li: List[int]):
        """
        Find a list of nine integers, starting with 0 and ending with 128, such that each integer either differs from
        the previous one by one or is thrice the previous one.
        """
        return all(j in {i - 1, i + 1, 3 * i} for i, j in zip([0] + li, li + [128])) and len(li) == 9

    @staticmethod
    def sol():
        return [1, 3, 4, 12, 13, 14, 42, 126, 127]


if __name__ == "__main__":
    PuzzleGenerator.debug_problems()
