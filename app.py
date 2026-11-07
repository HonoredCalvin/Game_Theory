"""
app.py - Command-line application: Game-Theoretic Analysis System (cybersecurity case study).

Usage:
    python app.py              interactive menu
    python app.py --demo       run every analysis, print results, save figures to ./outputs
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd

import games as G
from gt_core import Game
import gt_extended as X
import gt_viz as V

OUT = "outputs"
pd.set_option("display.width", 140)


def banner(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ------------------------------------------------------------------ analysis of one game
def analyse(game: Game, plots=True, tag=None):
    banner(f"GAME: {game.name}")
    print(game.payoff_table().to_string())
    print("\nBest responses ->", game.best_responses())
    for strict in (True, False):
        print(f"\nDominance ({'strict' if strict else 'weak'}):", game.dominance_report(strict))
    rows, cols, log = game.iterated_elimination(strict=True)
    print("\nIterated elimination of strictly dominated strategies:")
    for line in log or ["(nothing can be eliminated)"]:
        print("  -", line)
    print(f"  Remaining: {game.row_player} {rows} | {game.col_player} {cols}")
    pne = game.pure_nash_named()
    print("\nPure-strategy Nash equilibria:", pne if pne else "none")
    print("All Nash equilibria (support enumeration):")
    eqs = game.support_enumeration()
    for e in eqs:
        print("  *", game.format_equilibrium(e))
    if game.A.shape == (2, 2):
        mx = game.mixed_nash_2x2()
        print("2x2 closed-form mixed equilibrium (p, q):", None if mx is None else tuple(round(v, 4) for v in mx))
    try:                                         # optional cross-check against the Nashpy library
        import nashpy as nash
        ref = list(nash.Game(game.A, game.B).support_enumeration())
        print(f"Nashpy cross-check: {len(ref)} equilibria vs ours {len(eqs)} ->", "MATCH" if len(ref) == len(eqs) else "DIFFERENT")
    except ImportError:
        print("(Nashpy not installed - cross-check skipped)")
    if plots:
        os.makedirs(OUT, exist_ok=True)
        tag = tag or game.name.lower().replace(" ", "_").replace("'", "")
        V.plot_payoff_matrix(game, f"{OUT}/{tag}_matrix.png")
        if game.A.shape == (2, 2):
            V.plot_best_response_2x2(game, f"{OUT}/{tag}_best_response.png")


# ------------------------------------------------------------------ scenario analysis
def scenario_sweep(param, values, builder, base):
    """Re-solve the inspection game for each value of one parameter."""
    rows = []
    for v in values:
        kw = dict(base); kw[param] = v
        g = builder(**kw)
        pure = g.pure_nash_named()
        mix = g.mixed_nash_2x2()
        eq = g.support_enumeration()[0]
        rows.append({param: v, "pure NE": "; ".join(f"{a}/{b}" for a, b in pure) or "-",
                     "P(Monitor)": mix[0] if mix else (1.0 if pure and pure[0][0] == "Monitor" else 0.0),
                     "P(Attack)": mix[1] if mix else (1.0 if pure and pure[0][1] == "Attack" else 0.0),
                     "Defender payoff": eq["u_row"], "Attacker payoff": eq["u_col"]})
    return pd.DataFrame(rows)


def run_scenarios(plots=True):
    banner("SCENARIO ANALYSIS - how payoffs change the equilibrium")
    base = dict(L=10.0, c=4.0, g=6.0, f=8.0)
    tabs = {}
    sweeps = {"L": [2, 3, 4, 5, 6, 8, 10, 15, 20, 30], "c": [1, 2, 3, 4, 6, 8, 10, 12],
              "f": [1, 2, 4, 8, 12, 16, 24], "g": [1, 3, 6, 10, 15, 20]}
    for p, vals in sweeps.items():
        df = scenario_sweep(p, vals, G.cyber_inspection, base)
        tabs[p] = df
        print(f"\n-- Varying {p} (others: {', '.join(f'{k}={v}' for k, v in base.items() if k != p)})")
        print(df.round(3).to_string(index=False))
        if plots:
            V.plot_scenario(df[p], {"P(Monitor)": df["P(Monitor)"], "P(Attack)": df["P(Attack)"]}, p,
                            f"Equilibrium behaviour vs {p}", f"{OUT}/scenario_{p}.png",
                            vlines=[(base["c"], "L = c")] if p == "L" else None)
    print("\nKey insights:\n"
          "  * P(Attack) = c/L: a higher breach loss L makes the attacker attack LESS often, because the attack rate\n"
          "    must keep the defender indifferent between monitoring and ignoring.\n"
          "  * P(Monitor) = g/(g+f) depends ONLY on the attacker's payoffs: a bigger penalty f means LESS monitoring.\n"
          "  * Regime change at L = c: if monitoring costs at least as much as the loss it prevents, the equilibrium\n"
          "    is pure (Ignore, Attack).")
    return tabs


# ------------------------------------------------------------------ extended models
def run_sequential(plots=True):
    banner("SEQUENTIAL GAME - defender commits first (backward induction)")
    root = G.cyber_sequential()
    val = X.backward_induction(root)
    print("Subgame-perfect strategy:", X.spe_strategies(root))
    print("Equilibrium path:", X.optimal_path(root))
    print("Equilibrium payoffs (Defender, Attacker):", val)
    sim = G.cyber_inspection()
    eq = sim.support_enumeration()[0]
    print(f"Comparison - simultaneous inspection game: defender gets {eq['u_row']:.2f}; "
          f"sequential commitment game: defender gets {val[0]:.2f}.")
    if plots:
        V.plot_game_tree(root, f"{OUT}/cyber_game_tree.png")
    return val


def run_repeated(plots=True):
    banner("REPEATED GAME - iterated Prisoner's Dilemma (e.g. repeated information sharing)")
    M, rank = X.tournament(rounds=20)
    print("Total scores of ROW strategy vs COLUMN strategy (20 rounds):\n", M.to_string())
    print("\nTournament ranking:\n", rank.to_string())
    d = X.critical_discount_factor()
    print(f"\nGrim trigger sustains cooperation iff discount factor >= (T-R)/(T-P) = {d:.3f}")
    for delta in (0.3, 0.5, 0.9):
        coop = X.PD_PAYOFFS["R"] / (1 - delta)
        dev = X.PD_PAYOFFS["T"] + delta * X.PD_PAYOFFS["P"] / (1 - delta)
        print(f"  delta={delta}: cooperate={coop:.2f} vs deviate={dev:.2f} -> {'cooperate' if coop >= dev - 1e-9 else 'defect'}")
    if plots:
        V.plot_heatmap(M, f"{OUT}/repeated_tournament.png", "Iterated PD tournament (20 rounds)")


def run_cooperative(plots=True):
    banner("COOPERATIVE GAME - Shapley value for threat-intelligence sharing")
    P, v = G.threat_sharing_value()
    phi = X.shapley_value(P, v)
    print("Shapley allocation:", {k: round(x, 3) for k, x in phi.items()}, "| total =", sum(phi.values()))
    print("Allocation in the core?", X.in_core(P, v, phi))
    if plots:
        V.plot_bar(phi, f"{OUT}/shapley.png", "Shapley value of each bank", "Fair share of loss avoided ($M)")


def run_demo(plots=True):
    for ctor in G.CLASSICS + [G.cyber_inspection, G.cyber_investment]:
        analyse(ctor(), plots)
    run_scenarios(plots); run_sequential(plots); run_repeated(plots); run_cooperative(plots)
    if plots:
        print(f"\nFigures saved in ./{OUT}/")


# ------------------------------------------------------------------ interactive input
def ask_int(prompt, lo, hi):
    while True:
        try:
            v = int(input(prompt))
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  Please enter an integer between {lo} and {hi}.")


def ask_payoff(prompt):
    while True:
        try:
            a, b = (float(t) for t in input(prompt).replace("(", "").replace(")", "").split(","))
            if np.isfinite([a, b]).all():
                return a, b
        except ValueError:
            pass
        print("  Enter two numbers separated by a comma, e.g.  3,-1")


def custom_game():
    banner("ENTER YOUR OWN TWO-PLAYER GAME")
    name = input("Game name: ").strip() or "Custom game"
    m = ask_int("Number of row-player strategies (1-6): ", 1, 6)
    n = ask_int("Number of column-player strategies (1-6): ", 1, 6)
    rs = [input(f"  Row strategy {i+1} name: ").strip() or f"R{i+1}" for i in range(m)]
    cs = [input(f"  Column strategy {j+1} name: ").strip() or f"C{j+1}" for j in range(n)]
    A, B = np.zeros((m, n)), np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            A[i, j], B[i, j] = ask_payoff(f"  Payoffs (row, column) for {rs[i]} vs {cs[j]}: ")
    try:
        analyse(Game(A, B, rs, cs, name))
    except ValueError as e:
        print("Invalid game:", e)


def menu():
    options = {
        "1": ("Analyse a classic / case-study game", None), "2": ("Enter my own game", custom_game),
        "3": ("Scenario analysis (cyber inspection game)", run_scenarios), "4": ("Sequential game (game tree)", run_sequential),
        "5": ("Repeated game tournament", run_repeated), "6": ("Cooperative game (Shapley)", run_cooperative),
        "7": ("Run FULL demo", run_demo), "0": ("Quit", None)}
    lib = G.CLASSICS + [G.cyber_inspection, G.cyber_investment]
    while True:
        banner("GAME-THEORETIC ANALYSIS SYSTEM")
        for k, (t, _) in options.items():
            print(f" {k}. {t}")
        c = input("Choice: ").strip()
        if c == "0":
            break
        elif c == "1":
            for i, f in enumerate(lib, 1):
                print(f"  {i}. {f().name}")
            analyse(lib[ask_int("Game number: ", 1, len(lib)) - 1]())
        elif c in options and options[c][1]:
            options[c][1]()
        else:
            print("Unknown option.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Game-theoretic analysis system")
    ap.add_argument("--demo", action="store_true", help="run all analyses non-interactively")
    ap.add_argument("--no-plots", action="store_true", help="skip figure generation")
    a = ap.parse_args()
    run_demo(not a.no_plots) if a.demo else menu()
