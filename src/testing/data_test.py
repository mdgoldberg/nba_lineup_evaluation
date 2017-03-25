import logging
import logging.config
import multiprocessing as mp
import os

import dotenv
import luigi
import numpy as np
import pandas as pd

from sportsref import nba, decorators

dotenv_path = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_path)

PROJ_DIR = os.environ['PROJ_DIR']  # from .env
DATA_DIR = os.path.join(PROJ_DIR, 'data')


def get_logger():
    logging.config.fileConfig(
        os.path.join(PROJ_DIR, 'logging_config.ini')
    )
    logger = logging.getLogger()
    return logger


def calc_pm(df):
    """Calculates the raw plus/minus for each player in the game based on the
    play-by-play and lineup data contained in the passed DataFrame.

    :param df: A DataFrame of a game's play-by-play with a lineup column for
        each player (1 for in and home, 0 for out, -1 for in and away).
    :returns: A Series of player's raw plus/minus in the specified game,
        indexed by player id.
    """
    cols = nba.pbp.sparse_lineup_cols(df)
    pm_dict = {
        col[:-3]: (df[col] * (df.hm_pts - df.aw_pts)).sum()
        for col in cols
    }
    return pd.Series(pm_dict)


def summary(game_df):
    """Summarizes the differences between the box score plus/minuses and the
    caclulated plus/minuses based on PBP data.

    :param game_df: The PBP DataFrame for the boxscore to summarize.
    :returns: A dictionary of summary info about +/-
    """
    logger = get_logger()
    pbp_pm = calc_pm(game_df)
    boxscore_id = game_df.boxscore_id.iloc[0]
    bs = nba.BoxScore(boxscore_id)
    bs_pm = bs.basic_stats().set_index('player_id').plus_minus
    diff = (bs_pm - pbp_pm).dropna()
    logger.info('Finished summarizing {}'.format(boxscore_id))
    return pd.Series({
        'n_diff': (diff != 0).sum(),
        'max_diff': diff.abs().max(),
        'sum_diff': diff.sum()
    })


def year_summaries(year):
    """Returns a DataFrame of +/- comparison summaries for each game in a
    season.

    :year: Int representing the season
    :returns: DataFrame of results
    """
    year_df = pd.read_csv(
        os.path.join(DATA_DIR, 'pbp', 'pbp_{}.csv'.format(year))
    )
    summ = year_df.groupby('boxscore_id').apply(summary)
    all_bsids = nba.Season(year).schedule().boxscore_id.values
    missing_bsids = set(all_bsids) - set(summ.index)
    for miss_bsid in missing_bsids:
        summ.ix[miss_bsid] = np.nan
    return summ


if __name__ == '__main__':
    years = range(2006, 2017)
    p = mp.Pool(len(years))
    yr_summs = p.map(year_summaries, years)
    p.close()

    all_summs = pd.concat(yr_summs)
    all_summs.to_csv(
        os.path.join(DATA_DIR, 'testing', 'data_test_summary.csv'),
        index_label=False
    )
