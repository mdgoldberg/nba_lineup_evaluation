import os

import dotenv
import numpy as np
import pandas as pd

from sportsref import decorators

env_path = dotenv.find_dotenv()
dotenv.load_dotenv(env_path)
PROJ_DIR = os.environ['PROJ_DIR']


@decorators.memoize
def get_data(yr):
    df = pd.read_csv(
        os.path.join(PROJ_DIR, 'data', 'raw', 'pbp_{}.csv'.format(yr))
    )
    df.query('~(is_tech_foul | is_tech_fta)', inplace=True)
    return df


def split_data(df):
    df = df.sort_values(['year', 'month', 'day'])
    game_num = np.cumsum(df.boxscore_id != df.boxscore_id.shift(1))
    mid = game_num.max() / 2
    first_half = df.ix[game_num <= mid]
    second_half = df.ix[game_num > mid]
    return first_half, second_half
