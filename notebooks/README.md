# Earlier experiment: 2-D foraging

`connectome_rl_fly.ipynb` is where this project started: a connectome-constrained agent trained with PPO
on a 2-D foraging + threat-avoidance task, against a rewired control. `results/` holds its (smoke-test)
outputs. The drone work in `flydrone/` supersedes it, but it documents the first pass at the question.

Needs the raw connectome in `../connectome_cache/` (`python -m flydrone.download` from the repo root)
and a token in the environment before starting Jupyter:

    export NEUPRINT_TOKEN="..."
    jupyter notebook
