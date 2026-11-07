"""
games.py - Game library: classic benchmark games + the cybersecurity case study.

CASE STUDY (cybersecurity): a Defender (organisation) vs an Attacker (threat actor).
"""
from gt_core import Game
from gt_extended import Node


# ---------------- Classic games used for validation ----------------
def prisoners_dilemma():
    return Game([[3, 0], [5, 1]], [[3, 5], [0, 1]], ["Cooperate", "Defect"], ["Cooperate", "Defect"],
                "Prisoner's Dilemma")

def matching_pennies():
    return Game([[1, -1], [-1, 1]], [[-1, 1], [1, -1]], ["Heads", "Tails"], ["Heads", "Tails"],
                "Matching Pennies")

def battle_of_sexes():
    return Game([[3, 0], [0, 2]], [[2, 0], [0, 3]], ["Opera", "Football"], ["Opera", "Football"],
                "Battle of the Sexes")

def stag_hunt():
    return Game([[4, 0], [3, 3]], [[4, 3], [0, 3]], ["Stag", "Hare"], ["Stag", "Hare"], "Stag Hunt")

def rock_paper_scissors():
    A = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    B = [[-x for x in r] for r in A]
    s = ["Rock", "Paper", "Scissors"]
    return Game(A, B, s, s, "Rock-Paper-Scissors")

def iesds_example():
    A = [[4, 5, 6], [2, 8, 3], [3, 9, 2]]
    B = [[3, 1, 2], [1, 4, 6], [0, 6, 8]]
    return Game(A, B, ["Up", "Middle", "Down"], ["Left", "Center", "Right"], "IESDS textbook example")


# ---------------- Case study 1: simultaneous cyber inspection game (mixed equilibrium) ----------------
def cyber_inspection(L=10.0, c=4.0, g=6.0, f=8.0):
    """Defender: Monitor (cost c, blocks the attack) / Ignore.  Attacker: Attack / Refrain.
    L = loss if an undetected attack succeeds, g = attacker's gain, f = attacker's penalty if caught.
    Payoffs (Defender, Attacker):
        Monitor/Attack  = (-c, -f)    Monitor/Refrain = (-c, 0)
        Ignore/Attack   = (-L,  g)    Ignore/Refrain  = ( 0, 0)
    Closed form (L>c): P(Monitor)=g/(g+f), P(Attack)=c/L."""
    A = [[-c, -c], [-L, 0]]
    B = [[-f, 0], [g, 0]]
    return Game(A, B, ["Monitor", "Ignore"], ["Attack", "Refrain"], "Cyber Inspection Game",
                "Defender", "Attacker")


# ---------------- Case study 2: 3x3 security-investment game (pure equilibrium, dominance) ----------------
def cyber_investment():
    """Defender picks a main control; attacker picks an attack vector. Payoffs = (Defender, Attacker)."""
    A = [[-5, -5, -2],      # Firewall
         [-3, -3, -4],      # IDS
         [-6, -6, -7]]      # NoInvestment
    B = [[5, 2, 1],
         [3, 4, 2],
         [6, 5, 7]]
    return Game(A, B, ["Firewall", "IDS", "NoInvestment"], ["Phishing", "Exploit", "DDoS"],
                "Cyber Security Investment Game", "Defender", "Attacker")


# ---------------- Case study 3: sequential game (Defender commits first) ----------------
def cyber_sequential(invest_cost=2.0, loss=8.0, caught_penalty=3.0, gain=5.0, residual=1.0):
    """Defender (player 0) chooses Secure/NotSecure; Attacker (player 1) observes, then Attack/Refrain.
    Terminal payoffs = (Defender, Attacker)."""
    root = Node("Defender", player=0)
    sec = root.add("Secure", Node("Attacker | Secure", player=1))
    sec.add("Attack", Node("t1", payoffs=(-invest_cost - residual, -caught_penalty)))
    sec.add("Refrain", Node("t2", payoffs=(-invest_cost, 0)))
    nos = root.add("NotSecure", Node("Attacker | NotSecure", player=1))
    nos.add("Attack", Node("t3", payoffs=(-loss, gain)))
    nos.add("Refrain", Node("t4", payoffs=(0, 0)))
    return root


# ---------------- Case study 4: cooperative threat-intelligence sharing ----------------
def threat_sharing_value():
    """3 organisations; v(S) = losses avoided (millions) if coalition S shares threat intelligence."""
    P = ["BankA", "BankB", "BankC"]
    v = {frozenset(): 0,
         frozenset(["BankA"]): 10, frozenset(["BankB"]): 20, frozenset(["BankC"]): 30,
         frozenset(["BankA", "BankB"]): 50, frozenset(["BankA", "BankC"]): 60,
         frozenset(["BankB", "BankC"]): 70, frozenset(P): 120}
    return P, v

CLASSICS = [prisoners_dilemma, matching_pennies, battle_of_sexes, stag_hunt, rock_paper_scissors, iesds_example]
