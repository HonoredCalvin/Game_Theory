# Game-Theoretic Analysis System (Python) - Cybersecurity case study
Run:  `pip install -r requirements.txt` then
- `python app.py`            interactive menu (analyse library games, enter your own game, scenarios...)
- `python app.py --demo`     full non-interactive run, figures saved to ./outputs
- `python -m unittest -v`    test suite (18 tests)

Files: gt_core.py (normal-form algorithms) | gt_extended.py (trees, repeated, cooperative) |
gt_viz.py (plots) | games.py (game library) | app.py (CLI) | test_gt.py (tests)
Core algorithms are implemented from scratch; Nashpy (optional) is used only to cross-check.
