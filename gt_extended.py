"""
gt_extended.py - Sequential (game trees), repeated and cooperative game analysis.
"""
from __future__ import annotations
import itertools
import pandas as pd

TOL = 1e-9


# ===================== Sequential games: game tree + backward induction =====================
class Node:
    """Decision node (player index 0/1) or terminal node (payoffs tuple)."""
    def __init__(self, label, player=None, payoffs=None):
        self.label, self.player, self.payoffs = label, player, payoffs
        self.children = {}            # action -> Node
        self.best_action = None
        self.value = None

    def add(self, action, child):
        self.children[action] = child
        return child

    def is_terminal(self):
        return not self.children


def backward_induction(node):
    """Solve the tree for the subgame-perfect equilibrium. Ties -> first action listed."""
    if node.is_terminal():
        node.value = tuple(node.payoffs)
        return node.value
    best_action, best_val = None, None
    for action, child in node.children.items():
        val = backward_induction(child)
        if best_val is None or val[node.player] > best_val[node.player] + TOL:
            best_action, best_val = action, val
    node.best_action, node.value = best_action, best_val
    return best_val


def optimal_path(root):
    """Equilibrium path as list of (decision node label, chosen action)."""
    path, node = [], root
    while not node.is_terminal():
        path.append((node.label, node.best_action))
        node = node.children[node.best_action]
    return path


def spe_strategies(root):
    """Subgame-perfect strategy: chosen action at EVERY decision node (also off the path)."""
    out, stack = {}, [root]
    while stack:
        n = stack.pop()
        if not n.is_terminal():
            out[n.label] = n.best_action
            stack.extend(n.children.values())
    return out


# ===================== Repeated games: iterated Prisoner's Dilemma =====================
PD_PAYOFFS = {"T": 5, "R": 3, "P": 1, "S": 0}   # temptation, reward, punishment, sucker (T>R>P>S)


def pd_round(a, b, pay=PD_PAYOFFS):
    table = {("C", "C"): (pay["R"], pay["R"]), ("C", "D"): (pay["S"], pay["T"]),
             ("D", "C"): (pay["T"], pay["S"]), ("D", "D"): (pay["P"], pay["P"])}
    return table[(a, b)]


def all_c(mine, theirs): return "C"
def all_d(mine, theirs): return "D"
def tit_for_tat(mine, theirs): return "C" if not theirs else theirs[-1]
def grim_trigger(mine, theirs): return "D" if "D" in theirs else "C"
def suspicious_tft(mine, theirs): return "D" if not theirs else theirs[-1]

STRATEGIES = {"AllC": all_c, "AllD": all_d, "TitForTat": tit_for_tat,
              "Grim": grim_trigger, "SuspiciousTFT": suspicious_tft}


def play_match(s1, s2, rounds=10, pay=PD_PAYOFFS):
    h1, h2, t1, t2 = [], [], 0, 0
    for _ in range(rounds):
        a, b = s1(h1, h2), s2(h2, h1)
        p1, p2 = pd_round(a, b, pay)
        h1.append(a); h2.append(b); t1 += p1; t2 += p2
    return t1, t2, h1, h2


def tournament(rounds=10, pay=PD_PAYOFFS, strategies=None):
    """Round-robin (incl. self-play). Returns (score matrix DataFrame, ranking Series)."""
    strategies = strategies or STRATEGIES
    names = list(strategies)
    M = pd.DataFrame(0.0, index=names, columns=names)
    for a, b in itertools.product(names, names):
        M.loc[a, b] = play_match(strategies[a], strategies[b], rounds, pay)[0]
    return M, M.sum(axis=1).sort_values(ascending=False)


def critical_discount_factor(pay=PD_PAYOFFS):
    """Grim-trigger cooperation is a subgame-perfect equilibrium of the infinitely repeated PD iff
    delta >= (T - R)/(T - P)   [from R/(1-d) >= T + d*P/(1-d)]."""
    return (pay["T"] - pay["R"]) / (pay["T"] - pay["P"])


# ===================== Cooperative games: Shapley value and the core =====================
def shapley_value(players, v):
    """Shapley value. v: dict {frozenset(coalition): worth}; must contain frozenset() -> 0."""
    phi = {p: 0.0 for p in players}
    perms = list(itertools.permutations(players))
    for order in perms:
        coalition = frozenset()
        for p in order:
            bigger = coalition | {p}
            phi[p] += v[bigger] - v[coalition]
            coalition = bigger
    return {p: phi[p] / len(perms) for p in players}


def in_core(players, v, alloc):
    """Is allocation efficient and coalitionally rational (no coalition can do better alone)?"""
    if abs(sum(alloc.values()) - v[frozenset(players)]) > 1e-7:
        return False
    for r in range(1, len(players)):
        for S in itertools.combinations(players, r):
            if sum(alloc[p] for p in S) < v[frozenset(S)] - 1e-7:
                return False
    return True
