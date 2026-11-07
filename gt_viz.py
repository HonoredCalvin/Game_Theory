"""gt_viz.py - Matplotlib / NetworkX visualisations."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx
import numpy as np

TOL = 1e-9


def plot_payoff_matrix(game, path, title=None):
    """Payoff matrix as a table. Green = pure Nash; '*' marks a best-response payoff."""
    m, n = game.A.shape
    BR_row, BR_col = game.best_response_matrices()
    ne = set(game.pure_nash())
    fig, ax = plt.subplots(figsize=(1.8 * n + 2.2, 1.0 * m + 2.0))
    for i in range(m):
        for j in range(n):
            ax.add_patch(Rectangle((j, m - 1 - i), 1, 1, fc="#c8e6c9" if (i, j) in ne else "white", ec="black"))
            a = f"{game.A[i, j]:g}{'*' if BR_row[i, j] else ''}"
            b = f"{game.B[i, j]:g}{'*' if BR_col[i, j] else ''}"
            ax.text(j + .5, m - 1 - i + .5, f"{a}, {b}", ha="center", va="center", fontsize=11)
    for j, s in enumerate(game.col_strategies):
        ax.text(j + .5, m + .15, s, ha="center", fontweight="bold")
    for i, s in enumerate(game.row_strategies):
        ax.text(-.1, m - 1 - i + .5, s, ha="right", va="center", fontweight="bold")
    ax.text(n / 2, m + .6, f"{game.col_player} strategies", ha="center", style="italic")
    ax.text(-1.6, m / 2, f"{game.row_player}", rotation=90, va="center", style="italic")
    ax.set_xlim(-1.8, n + .1); ax.set_ylim(-.4, m + 1)
    ax.axis("off")
    ax.set_title(title or f"{game.name}\n(green = pure Nash equilibrium, * = best response)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _br_curve(s0, s1):
    """Best-reply polyline for a linear payoff difference s(t): s0 = s(0), s1 = s(1). Returns (t*, v0, v1)."""
    def v(s, other):
        return 1.0 if s > TOL else 0.0 if s < -TOL else v(other, 0.0) if abs(other) > TOL else 0.5
    v0, v1 = v(s0, s1), v(s1, s0)
    tstar = s0 / (s0 - s1) if s0 * s1 < 0 else None
    return tstar, v0, v1


def plot_best_response_2x2(game, path):
    """Best-response correspondences in (p, q) space for a 2x2 game; intersections = Nash equilibria.
    p = P(row plays first strategy), q = P(column plays first strategy)."""
    A, B = game.A, game.B
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    # column's BR q(p): payoff diff u(col0)-u(col1) = p*(B00-B01)+(1-p)*(B10-B11)
    ts, v0, v1 = _br_curve(B[1, 0] - B[1, 1], B[0, 0] - B[0, 1])
    if ts is None:
        ax.plot([0, 1], [v0, v1], "r-", lw=2.5, label=f"{game.col_player} best reply")
    else:
        ax.plot([0, ts, ts, 1], [v0, v0, v1, v1], "r-", lw=2.5, label=f"{game.col_player} best reply")
    # row's BR p(q): u(row0)-u(row1) = q*(A00-A10)+(1-q)*(A01-A11)
    ts, v0, v1 = _br_curve(A[0, 1] - A[1, 1], A[0, 0] - A[1, 0])
    if ts is None:
        ax.plot([v0, v1], [0, 1], "b--", lw=2.5, label=f"{game.row_player} best reply")
    else:
        ax.plot([v0, v0, v1, v1], [0, ts, ts, 1], "b--", lw=2.5, label=f"{game.row_player} best reply")
    for eq in game.support_enumeration():
        ax.plot(eq["x"][0], eq["y"][0], "ko", ms=10, zorder=5)
    ax.plot([], [], "ko", label="Nash equilibrium")
    ax.set_xlabel(f"p = P({game.row_player} plays '{game.row_strategies[0]}')")
    ax.set_ylabel(f"q = P({game.col_player} plays '{game.col_strategies[0]}')")
    ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03); ax.grid(alpha=.3); ax.legend(loc="center right", fontsize=8)
    ax.set_title(f"Best-response diagram: {game.name}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_game_tree(root, path, title="Game tree (bold red = subgame-perfect path)"):
    """Draw a solved game tree with NetworkX; layout is computed manually (no Graphviz needed)."""
    G, pos, labels, edge_lab, counter = nx.DiGraph(), {}, {}, {}, [0]

    def build(node, depth):
        nid = id(node)
        G.add_node(nid)
        if node.is_terminal():
            pos[nid] = (counter[0], -depth); counter[0] += 1
            labels[nid] = "(" + ", ".join(f"{v:g}" for v in node.payoffs) + ")"
        else:
            xs = []
            for action, ch in node.children.items():
                build(ch, depth + 1)
                G.add_edge(nid, id(ch)); edge_lab[(nid, id(ch))] = action
                xs.append(pos[id(ch)][0])
            pos[nid] = (float(np.mean(xs)), -depth)
            labels[nid] = f"{node.label}\n(P{node.player + 1})"
    build(root, 0)
    best, node = set(), root
    while not node.is_terminal():
        ch = node.children[node.best_action]; best.add((id(node), id(ch))); node = ch
    fig, ax = plt.subplots(figsize=(8, 5))
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[e for e in G.edges if e not in best], arrows=False, width=1.2)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(best), arrows=False, width=3.5, edge_color="crimson")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_lab, ax=ax, font_size=9)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#e3f2fd", node_size=2600, edgecolors="black")
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7)
    ax.set_title(title); ax.axis("off"); ax.margins(.15)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_scenario(x, series, xlabel, title, path, vlines=None):
    """Line chart of one or more equilibrium quantities against a swept parameter."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, y in series.items():
        ax.plot(x, y, marker="o", ms=3, label=name)
    for v, lab in (vlines or []):
        ax.axvline(v, color="grey", ls=":"); ax.text(v, ax.get_ylim()[1], lab, rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel(xlabel); ax.grid(alpha=.3); ax.legend(); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_heatmap(M, path, title):
    """Heatmap of a score matrix (e.g. repeated-game tournament)."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(M.values, cmap="YlGnBu")
    ax.set_xticks(range(len(M.columns))); ax.set_xticklabels(M.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(M.index))); ax.set_yticklabels(M.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:g}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_bar(d, path, title, ylabel):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(list(d.keys()), list(d.values()), color="#4c78a8")
    for k, v in d.items():
        ax.text(k, v, f"{v:.1f}", ha="center", va="bottom")
    ax.set_title(title); ax.set_ylabel(ylabel)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
