#!/usr/bin/env python3
from collections import defaultdict, namedtuple
from networkx import DiGraph
from networkx.algorithms import minimum_spanning_arborescence


Arc = namedtuple('Arc', ('tail', 'weight', 'head'))


def min_spanning_arborescence(arcs, sink):
    """
    Wrapper for the networkX min_spanning_tree to follow the original API
    :param arcs: list of Arc tuples
    :param sink: unused argument. We assume that 0 is the only possible root over the set of edges given to
     the algorithm.
    """
    G = DiGraph()
    for arc in arcs:
        G.add_edge(arc.head, arc.tail, weight=arc.weight)
    ARB = minimum_spanning_arborescence(G)
    result = {}
    headtail2arc = {(a.head, a.tail): a for a in arcs}
    for edge in ARB.edges:
        tail = edge[1]
        result[tail] = headtail2arc[(edge[0], edge[1])]
    return result

def find_cycle(successor, sink):
    visited = {sink}
    for node in successor:
        cycle = []
        while node not in visited:
            visited.add(node)
            cycle.append(node)
            node = successor[node]
        if node in cycle:
            return cycle[cycle.index(node):]
    return None


def spanning_arborescence(arcs, sink):
    arcs_by_head = defaultdict(list)
    for arc in arcs:
        if arc.tail == sink:
            continue
        arcs_by_head[arc.head].append(arc)
    solution_arc_by_tail = {}
    stack = arcs_by_head[sink]
    while stack:
        arc = stack.pop()
        if arc.tail in solution_arc_by_tail:
            continue
        solution_arc_by_tail[arc.tail] = arc
        stack.extend(arcs_by_head[arc.tail])
    return solution_arc_by_tail


print(min_spanning_arborescence([Arc(1, 17, 0), Arc(2, 16, 0), Arc(3, 19, 0), Arc(4, 16, 0), Arc(5, 16, 0), Arc(6, 18, 0), Arc(2, 3, 1), Arc(3, 3, 1), Arc(4, 11, 1), Arc(5, 10, 1), Arc(6, 12, 1), Arc(1, 3, 2), Arc(3, 4, 2), Arc(4, 8, 2), Arc(5, 8, 2), Arc(6, 11, 2), Arc(1, 3, 3), Arc(2, 4, 3), Arc(4, 12, 3), Arc(5, 11, 3), Arc(6, 14, 3), Arc(1, 11, 4), Arc(2, 8, 4), Arc(3, 12, 4), Arc(5, 6, 4), Arc(6, 10, 4), Arc(1, 10, 5), Arc(2, 8, 5), Arc(3, 11, 5), Arc(4, 6, 5), Arc(6, 4, 5), Arc(1, 12, 6), Arc(2, 11, 6), Arc(3, 14, 6), Arc(4, 10, 6), Arc(5, 4, 6)], 0))