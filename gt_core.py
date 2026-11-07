"""
gt_core.py - Core game-theory algorithms for two-player finite normal-form games.

Implemented from scratch (NumPy only for arrays / linear algebra):
  * payoff storage + display           * best responses
  * dominant / dominated strategies    * iterated elimination (IESDS)
  * pure-strategy Nash equilibria      * mixed Nash: 2x2 closed form + support enumeration
Convention: A[i, j] = row player's payoff, B[i, j] = column player's payoff.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

TOL = 1e-9


class Game:
    """A two-player finite game in normal (strategic) form."""

    def __init__(self, A, B, row_strategies=None, col_strategies=None,
                 name="Untitled game", row_player="Row", col_player="Column"):
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        if A.ndim != 2 or A.size == 0 or A.shape != B.shape:
            raise ValueError("A and B must be non-empty 2-D arrays of identical shape.")
        if not (np.isfinite(A).all() and np.isfinite(B).all()):
            raise ValueError("All payoffs must be finite numbers.")
        m, n = A.shape
        self.A, self.B = A, B
        self.row_strategies = list(row_strategies) if row_strategies else [f"R{i+1}" for i in range(m)]
        self.col_strategies = list(col_strategies) if col_strategies else [f"C{j+1}" for j in range(n)]
        if len(self.row_strategies) != m or len(self.col_strategies) != n:
            raise ValueError("Number of strategy names must match the payoff matrix size.")
        if len(set(self.row_strategies)) != m or len(set(self.col_strategies)) != n:
            raise ValueError("Strategy names must be unique for each player.")
        self.name, self.row_player, self.col_player = name, row_player, col_player

    # ---------- display ----------
    def payoff_table(self) -> pd.DataFrame:
        data = [[f"({self.A[i, j]:g}, {self.B[i, j]:g})" for j in range(self.A.shape[1])]
                for i in range(self.A.shape[0])]
        df = pd.DataFrame(data, index=self.row_strategies, columns=self.col_strategies)
        df.index.name = f"{self.row_player} \\ {self.col_player}"
        return df

    # ---------- best responses ----------
    def best_response_matrices(self):
        """Boolean matrices: BR_row[i,j] True if i is a best reply to column j (same for BR_col)."""
        BR_row = self.A >= self.A.max(axis=0, keepdims=True) - TOL
        BR_col = self.B >= self.B.max(axis=1, keepdims=True) - TOL
        return BR_row, BR_col

    def best_responses(self):
        """Readable dict: {'row': {col_strategy: [row best replies]}, 'col': {...}}."""
        BR_row, BR_col = self.best_response_matrices()
        row = {self.col_strategies[j]: [self.row_strategies[i] for i in np.where(BR_row[:, j])[0]]
               for j in range(self.A.shape[1])}
        col = {self.row_strategies[i]: [self.col_strategies[j] for j in np.where(BR_col[i, :])[0]]
               for i in range(self.A.shape[0])}
        return {"row": row, "col": col}

    # ---------- pure Nash ----------
    def pure_nash(self):
        """All pure-strategy Nash equilibria as index pairs (i, j): mutual best responses."""
        BR_row, BR_col = self.best_response_matrices()
        return [(int(i), int(j)) for i, j in zip(*np.where(BR_row & BR_col))]

    def pure_nash_named(self):
        return [(self.row_strategies[i], self.col_strategies[j]) for i, j in self.pure_nash()]

    # ---------- dominance ----------
    def dominance_report(self, strict=True):
        """Dominated strategies and dominant strategy (if any) for both players."""
        out = {}
        for who, M, names in (("row", self.A, self.row_strategies), ("col", self.B.T, self.col_strategies)):
            dom = dominated_strategies(M, strict)
            dominant = dominant_strategy(M, strict)
            out[who] = {
                "dominated": {names[i]: names[k] for i, k in dom.items()},
                "dominant": None if dominant is None else names[dominant],
            }
        return out

    def iterated_elimination(self, strict=True):
        """Iterated elimination of (strictly/weakly) dominated strategies.
        Returns (remaining_row_names, remaining_col_names, log). Weak dominance is order dependent."""
        rows, cols = list(range(self.A.shape[0])), list(range(self.A.shape[1]))
        log = []
        changed = True
        while changed:
            changed = False
            subA = self.A[np.ix_(rows, cols)]
            subB = self.B[np.ix_(rows, cols)]
            dom = dominated_strategies(subA, strict)
            if dom:
                i, k = next(iter(dom.items()))
                log.append(f"{self.row_player}: '{self.row_strategies[rows[i]]}' is dominated by "
                           f"'{self.row_strategies[rows[k]]}' -> eliminated")
                rows.pop(i); changed = True; continue
            dom = dominated_strategies(subB.T, strict)
            if dom:
                j, k = next(iter(dom.items()))
                log.append(f"{self.col_player}: '{self.col_strategies[cols[j]]}' is dominated by "
                           f"'{self.col_strategies[cols[k]]}' -> eliminated")
                cols.pop(j); changed = True
        return ([self.row_strategies[i] for i in rows], [self.col_strategies[j] for j in cols], log)

    # ---------- mixed strategies ----------
    def expected_payoffs(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return float(x @ self.A @ y), float(x @ self.B @ y)

    def mixed_nash_2x2(self):
        """Closed-form fully mixed equilibrium of a 2x2 game via the indifference principle.
        Returns (p, q): p = P(row plays first row), q = P(column plays first column), or None."""
        if self.A.shape != (2, 2):
            raise ValueError("mixed_nash_2x2 needs a 2x2 game; use support_enumeration() otherwise.")
        A, B = self.A, self.B
        dp = B[0, 0] - B[1, 0] - B[0, 1] + B[1, 1]
        dq = A[0, 0] - A[0, 1] - A[1, 0] + A[1, 1]
        if abs(dp) < TOL or abs(dq) < TOL:
            return None
        p = (B[1, 1] - B[1, 0]) / dp      # makes the COLUMN player indifferent
        q = (A[1, 1] - A[0, 1]) / dq      # makes the ROW player indifferent
        return (float(p), float(q)) if (TOL < p < 1 - TOL and TOL < q < 1 - TOL) else None

    def support_enumeration(self):
        """All Nash equilibria (pure and mixed) of a nondegenerate bimatrix game.
        Returns list of dicts with x, y (mixed strategies) and equilibrium payoffs."""
        m, n = self.A.shape
        eqs = []
        for k in range(1, min(m, n) + 1):
            for I in itertools.combinations(range(m), k):
                for J in itertools.combinations(range(n), k):
                    res = _solve_support(self.A, self.B, I, J)
                    if res is not None:
                        eqs.append(res)
        return eqs

    def format_equilibrium(self, eq):
        fx = ", ".join(f"{s}: {p:.3f}" for s, p in zip(self.row_strategies, eq["x"]) if p > TOL)
        fy = ", ".join(f"{s}: {p:.3f}" for s, p in zip(self.col_strategies, eq["y"]) if p > TOL)
        return (f"{self.row_player} [{fx}]  |  {self.col_player} [{fy}]  "
                f"-> payoffs ({eq['u_row']:.3f}, {eq['u_col']:.3f})")


# ---------- dominance helpers (M: rows = own strategies, columns = opponent's) ----------
def dominates(M, k, i, strict=True):
    """True if own strategy k dominates own strategy i."""
    if strict:
        return bool(np.all(M[k] > M[i] + TOL))
    return bool(np.all(M[k] >= M[i] - TOL) and np.any(M[k] > M[i] + TOL))


def dominated_strategies(M, strict=True):
    """{dominated index: first dominating index}."""
    M = np.asarray(M, float)
    out = {}
    for i in range(M.shape[0]):
        for k in range(M.shape[0]):
            if k != i and dominates(M, k, i, strict):
                out[i] = k
                break
    return out


def dominant_strategy(M, strict=True):
    """Index of a strategy dominating all others, else None."""
    M = np.asarray(M, float)
    for i in range(M.shape[0]):
        if M.shape[0] > 1 and all(dominates(M, i, k, strict) for k in range(M.shape[0]) if k != i):
            return i
    return None


def _solve_support(A, B, I, J):
    """Solve the equilibrium conditions for supports I (row) and J (col) of equal size k."""
    m, n = A.shape
    k = len(I)
    I, J = list(I), list(J)
    rhs = np.zeros(k + 1); rhs[k] = 1.0
    # column mix y makes every row in I earn the same payoff v
    M1 = np.zeros((k + 1, k + 1)); M1[:k, :k] = A[np.ix_(I, J)]; M1[:k, k] = -1; M1[k, :k] = 1
    # row mix x makes every column in J earn the same payoff w
    M2 = np.zeros((k + 1, k + 1)); M2[:k, :k] = B[np.ix_(I, J)].T; M2[:k, k] = -1; M2[k, :k] = 1
    try:
        s1, s2 = np.linalg.solve(M1, rhs), np.linalg.solve(M2, rhs)
    except np.linalg.LinAlgError:
        return None
    y_s, v = s1[:k], s1[k]
    x_s, w = s2[:k], s2[k]
    if (y_s <= TOL).any() or (x_s <= TOL).any():       # exact supports only (avoids duplicates)
        return None
    x = np.zeros(m); x[I] = x_s
    y = np.zeros(n); y[J] = y_s
    if (A @ y > v + 1e-7).any() or (x @ B > w + 1e-7).any():   # no profitable deviation outside support
        return None
    return {"x": x, "y": y, "u_row": float(v), "u_col": float(w)}
