"""Test suite: known textbook results, hand calculations and property tests. Run: python -m unittest -v"""
import unittest
import numpy as np
import games as G
import gt_extended as X
from gt_core import Game
from app import scenario_sweep


class TestNormalForm(unittest.TestCase):
    def test_prisoners_dilemma(self):
        g = G.prisoners_dilemma()
        self.assertEqual(g.pure_nash_named(), [("Defect", "Defect")])
        d = g.dominance_report()
        self.assertEqual((d["row"]["dominant"], d["col"]["dominant"]), ("Defect", "Defect"))

    def test_matching_pennies_no_pure_but_mixed(self):
        g = G.matching_pennies()
        self.assertEqual(g.pure_nash(), [])
        self.assertTrue(np.allclose(g.mixed_nash_2x2(), (.5, .5)))
        eq = g.support_enumeration()
        self.assertEqual(len(eq), 1); self.assertAlmostEqual(eq[0]["u_row"], 0)

    def test_battle_of_sexes_three_equilibria(self):   # hand calc: p = 3/5, q = 2/5 (row Opera, col Opera)
        g = G.battle_of_sexes()
        self.assertEqual(len(g.pure_nash()), 2)
        self.assertEqual(len(g.support_enumeration()), 3)
        self.assertTrue(np.allclose(g.mixed_nash_2x2(), (3 / 5, 2 / 5)))

    def test_stag_hunt(self):                          # hand calc: indifference at p = q = 3/4
        g = G.stag_hunt()
        self.assertEqual(sorted(g.pure_nash_named()), [("Hare", "Hare"), ("Stag", "Stag")])
        self.assertTrue(np.allclose(g.mixed_nash_2x2(), (.75, .75)))

    def test_rps_uniform(self):
        eq = G.rock_paper_scissors().support_enumeration()
        self.assertEqual(len(eq), 1)
        self.assertTrue(np.allclose(eq[0]["x"], [1 / 3] * 3) and np.allclose(eq[0]["y"], [1 / 3] * 3))

    def test_iesds_textbook(self):
        r, c, log = G.iesds_example().iterated_elimination()
        self.assertEqual((r, c), (["Up"], ["Left"])); self.assertEqual(len(log), 4)

    def test_weak_dominance(self):
        g = Game([[1, 1], [1, 0]], [[0, 0], [0, 0]])
        self.assertEqual(g.dominance_report(strict=False)["row"]["dominant"], "R1")
        self.assertIsNone(g.dominance_report(strict=True)["row"]["dominant"])

    def test_cyber_inspection_closed_form(self):       # hand calc: P(Monitor)=g/(g+f)=6/14, P(Attack)=c/L=0.4
        g = G.cyber_inspection(L=10, c=4, g=6, f=8)
        self.assertEqual(g.pure_nash(), [])
        self.assertTrue(np.allclose(g.mixed_nash_2x2(), (6 / 14, .4)))
        eq = g.support_enumeration()[0]
        self.assertTrue(np.allclose(eq["x"], [6 / 14, 8 / 14]) and np.allclose(eq["y"], [.4, .6]))
        self.assertAlmostEqual(eq["u_row"], -4.0)      # defender equilibrium payoff = -c

    def test_cyber_inspection_regime_switch(self):
        self.assertEqual(G.cyber_inspection(L=3, c=4).pure_nash_named(), [("Ignore", "Attack")])

    def test_cyber_investment(self):
        g = G.cyber_investment()
        self.assertEqual(g.pure_nash_named(), [("IDS", "Exploit")])
        self.assertEqual(g.iterated_elimination()[:2], (["IDS"], ["Exploit"]))

    def test_property_equilibria_have_no_profitable_deviation(self):
        rng = np.random.default_rng(42)
        for _ in range(200):
            m, n = rng.integers(2, 5, 2)
            g = Game(rng.integers(-5, 6, (m, n)), rng.integers(-5, 6, (m, n)))
            for e in g.support_enumeration():
                self.assertLessEqual((g.A @ e["y"]).max(), e["u_row"] + 1e-6)
                self.assertLessEqual((e["x"] @ g.B).max(), e["u_col"] + 1e-6)
                self.assertAlmostEqual(e["x"].sum(), 1); self.assertAlmostEqual(e["y"].sum(), 1)

    def test_pure_nash_agree_with_support_enumeration(self):
        rng = np.random.default_rng(7)
        for _ in range(100):
            g = Game(rng.normal(size=(3, 3)), rng.normal(size=(3, 3)))
            pure = {(i, j) for i, j in g.pure_nash()}
            se = {(int(e["x"].argmax()), int(e["y"].argmax())) for e in g.support_enumeration()
                  if e["x"].max() > .999 and e["y"].max() > .999}
            self.assertEqual(pure, se)

    def test_nashpy_crosscheck_if_available(self):
        try:
            import nashpy as nash
        except ImportError:
            self.skipTest("nashpy not installed")
        for ctor in G.CLASSICS:
            g = ctor(); self.assertEqual(len(list(nash.Game(g.A, g.B).support_enumeration())), len(g.support_enumeration()))

    def test_input_validation(self):
        with self.assertRaises(ValueError): Game([[1, 2]], [[1]])
        with self.assertRaises(ValueError): Game([[np.nan]], [[1]])
        with self.assertRaises(ValueError): Game([[1, 2]], [[1, 2]], ["a", "a"], ["x", "y"])
        with self.assertRaises(ValueError): Game([[1, 2]], [[1, 2]], ["a", "b"], ["x", "y"])
        with self.assertRaises(ValueError): G.rock_paper_scissors().mixed_nash_2x2()


class TestExtended(unittest.TestCase):
    def test_backward_induction(self):                 # hand calc: attacker refrains after Secure -> -2 beats -8
        root = G.cyber_sequential(); val = X.backward_induction(root)
        self.assertEqual(val, (-2.0, 0)); self.assertEqual(X.optimal_path(root), [("Defender", "Secure"), ("Attacker | Secure", "Refrain")])
        self.assertEqual(X.spe_strategies(root)["Attacker | NotSecure"], "Attack")

    def test_repeated_pd(self):
        self.assertEqual(X.play_match(X.tit_for_tat, X.tit_for_tat, 10)[:2], (30, 30))
        self.assertEqual(X.play_match(X.all_d, X.all_c, 10)[:2], (50, 0))
        self.assertEqual(X.play_match(X.all_d, X.tit_for_tat, 10)[:2], (5 + 9, 0 + 9))
        self.assertAlmostEqual(X.critical_discount_factor(), 0.5)

    def test_shapley(self):
        v2 = {frozenset(): 0, frozenset("A"): 0, frozenset("B"): 0, frozenset("AB"): 10}
        self.assertEqual(X.shapley_value(["A", "B"], v2), {"A": 5, "B": 5})
        P = ["a", "b", "c"]
        import itertools
        v3 = {frozenset(s): len(s) ** 2 for r in range(4) for s in itertools.combinations(P, r)}
        self.assertTrue(all(abs(x - 3) < 1e-9 for x in X.shapley_value(P, v3).values()))
        Pb, vb = G.threat_sharing_value(); phi = X.shapley_value(Pb, vb)
        self.assertAlmostEqual(sum(phi.values()), 120); self.assertTrue(X.in_core(Pb, vb, phi))

    def test_scenario_sweep(self):
        df = scenario_sweep("L", [5, 10, 20], G.cyber_inspection, dict(L=10.0, c=4.0, g=6.0, f=8.0))
        self.assertTrue(np.allclose(df["P(Attack)"], [.8, .4, .2]))
        self.assertTrue(np.allclose(df["P(Monitor)"], 6 / 14))


if __name__ == "__main__":
    unittest.main(verbosity=2)
