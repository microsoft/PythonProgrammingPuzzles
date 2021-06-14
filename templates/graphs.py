"""Problems related to graphs such as Conway's 99 problem, finding
[cliques](https://en.wikipedia.org/wiki/Clique_(graph_theory)) of various sizes, shortest path (Dijkstra) """

from problems import Problem
from typing import List


# Hint: subclass Problem.Debug for quick testing. Run make_dataset.py to make the dataset
# See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle for more info


class Conway99(Problem):
    """Conway's 99-graph problem (*unsolved*, open problem)

    Conway's 99-graph problem is an unsolved problem in graph theory. It asks whether there exists an
    undirected graph with 99 vertices, in which each two adjacent vertices have exactly one common neighbor,
    and in which each two non-adjacent vertices have exactly two common neighbors.
    Or in Conway's terminology, from [Five $1,000 Problems (Update 2017)](https://oeis.org/A248380/a248380.pdf)
    "Is there a graph with 99 vertices in which every edge (i.e. pair of joined vertices) belongs to a unique
    triangle and every nonedge (pair of unjoined vertices) to a unique quadrilateral?"

    See also this [Wikipedia article](https://en.wikipedia.org/w/index.php?title=Conway%27s_99-graph_problem).
    """

    @staticmethod
    def sat(edges: List[List[int]]):
        N = {i: {j for j in range(99) if j != i and ([i, j] in edges or [j, i] in edges)} for i in
             range(99)}  # neighbor sets
        return all(len(N[i].intersection(N[j])) == (1 if j in N[i] else 2) for i in range(99) for j in range(i))


def dedup_edges(stuff):
    seen = set()
    return [a for a in stuff if tuple(a) not in seen and not seen.add(tuple(a))]


class AnyEdge(Problem):
    "Find any edge in a given [graph](https://en.wikipedia.org/w/index.php?title=Graph_(discrete_mathematics))."

    @staticmethod
    def sat(e: List[int], edges=[[0, 217], [40, 11], [17, 29], [11, 12], [31, 51]]):
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


class AnyTriangle(Problem):
    """Find a [triangle](https://en.wikipedia.org/w/index.php?title=Triangle_graph) in a given directed graph."""


    @staticmethod
    def sat(tri: List[int], edges=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):
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


class PlantedClique(Problem):
    """Find a [planted clique](https://en.wikipedia.org/w/index.php?title=Planted_clique) of a given size
    in an undirected graph. Finding a polynomial-time algorithm for this problem has been *unsolved* for
    some time."""

    @staticmethod
    def sat(nodes: List[int], size=3, edges=[[0, 17], [0, 22], [17, 22], [17, 31], [22, 31], [31, 17]]):
        assert len(nodes) == len(set(nodes)) >= size
        edge_set = {(a, b) for (a, b) in edges}
        for a in nodes:
            for b in nodes:
                assert a == b or (a, b) in edge_set or (b, a) in edge_set

        return True

    @staticmethod
    def sol(size, edges):  # brute force (finds list in increasing order), but with a tiny bit of speedup
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


class ShortestPath(Problem):
    """Shortest Path

    Find a path from node 0 to node 1, of a bounded length, in a given digraph on integer vertices.

    See (Dijkstra's algorithm)[https://en.wikipedia.org/w/index.php?title=Dijkstra%27s_algorithm]"""

    @staticmethod
    def sat(path: List[int],
            weights=[{1: 20, 2: 1}, {2: 2, 3: 5}, {1: 10}],  # weights[a][b] is weight on edge [a,b]
            bound=11):
        return path[0] == 0 and path[-1] == 1 and sum(weights[a][b] for a, b in zip(path, path[1:])) <= bound

    @staticmethod
    def sol(weights, bound):  # Dijkstra's algorithm (bound is ignored)
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


class UnweightedShortestPath(Problem):
    """Unweighted Shortest Path

    Find a path from node u to node v, of a bounded length, in a given digraph on vertices 0, 1,..., n.

    See (Dijkstra's algorithm)[https://en.wikipedia.org/w/index.php?title=Dijkstra%27s_algorithm]"""

    @staticmethod
    def sat(path: List[int],
            edges=[[0, 11], [0, 22], [11, 22], [11, 33], [22, 33]],
            u=0,
            v=33,
            bound=3):
        assert path[0] == u and path[-1] == v and all([i, j] in edges for i, j in zip(path, path[1:]))
        return len(path) <= bound

    @staticmethod
    def sol(edges, u, v, bound):  # Dijkstra's algorithm
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


class AnyPath(Problem):
    """Any Path

    Find any path from node 0 to node n in a given graph on vertices 0, 1,..., n.
    """

    @staticmethod
    def sat(path: List[int], edges=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]):
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


class EvenPath(Problem):
    """Even Path

    Find any path with an even number of nodes from node 0 to node n in a given graph on vertices 0, 1,..., n.
    """

    @staticmethod
    def sat(path: List[int], edges=[[0, 2], [0, 1], [2, 1], [2, 3], [1, 3]]):
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


class OddPath(Problem):
    """Odd Path

    *** Note the change to go from node 0 to node 1 ***

    Find any path with an odd number of nodes from node 0 to node 1 in a given graph on vertices 0, 1,..., n.
    """

    @staticmethod
    def sat(p: List[int], edges=[[0, 1], [0, 2], [1, 2], [3, 1], [2, 3]]):
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



class Zarankiewicz(Problem):
    """[Zarankiewicz problem](https://en.wikipedia.org/wiki/Zarankiewicz_problem)

    Find a bipartite graph with 4 vertices on each side, 13 edges, and no K_3,3 subgraph.
    """
    @staticmethod
    def sat(edges: List[List[int]]):
        assert len(edges) == len({(a, b) for a, b in edges}) == 13  # weights
        assert all(i in range(4) for li in edges for i in li)  # 4 nodes on each side
        for i in range(4):
            v = [m for m in range(4) if m != i]
            for j in range(4):
                u = [m for m in range(4) if m != j]
                if all([m, n] in edges for m in v for n in u):
                    return False
        return True

    @staticmethod
    def sol():
        return [[i, j] for i in range(4) for j in range(4) if i != j or i == 0]

class GraphIsomorphism(Problem):
    """
    In the classic [Graph Isomorphism](https://en.wikipedia.org/wiki/Graph_isomorphism) problem,
    one is given two graphs which are permutations of one another and
    the goal is to find the permutation. It is unknown wheter or not there exists a polynomial-time algorithm
    for this problem, though an unpublished quasi-polynomial-time algorithm has been announced by Babai.

    Each graph is specified by a list of edges where each edge is a pair of integer vertex numbers.
    """

    @staticmethod
    def sat(bi: List[int], g1=[[0, 1], [1, 2], [2, 3], [3, 4]], g2=[[0, 4], [4, 1], [1, 2], [2, 3]]):
        return len(bi) == len(set(bi)) and {(i, j) for i, j in g1} == {(bi[i], bi[j]) for i, j in g2}

    @staticmethod
    def sol(g1, g2):  # exponentially slow
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
        self.add(dict(g1=g1, g2=g2), test=n < 10) # only test for small n


if __name__ == "__main__":
    Problem.debug_problems()
