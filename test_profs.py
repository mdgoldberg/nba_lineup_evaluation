import numpy as np, pandas as pd

from src.features import build_profiles
from sportsref import nba

this_year = build_profiles.get_data(2016)

all_players = [col[:-3] for col in nba.pbp.sparse_lineup_cols(this_year)]

off_plays = build_profiles.get_off_plays(this_year, all_players, [])
def_plays = build_profiles.get_def_plays(this_year, all_players, [])

all_plays = (off_plays + def_plays).sort_values(ascending=False)
players = all_plays.iloc[:360].index.values
reps = all_plays.iloc[360:].index.values

X, y, gs, coefs = build_profiles.combined_rapm(this_year, players, reps)
