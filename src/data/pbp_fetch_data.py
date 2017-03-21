import logging
import os

import dotenv
import luigi
import numpy as np
import pandas as pd
from dask import bag as db

from sportsref import nba

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


def process_boxscore_id(boxscore_id):
    """Takes a boxscore ID and returns the play-by-play data (without lineup
    data), excluding overtimes.

    :param boxscore_id: A string containing a game's boxscore ID.
    :returns: a DataFrame of regulation play-by-play data.
    """
    logger = get_logger()
    bs = nba.BoxScore(boxscore_id)
    try:
        df = bs.pbp(dense_lineups=True, sparse_lineups=True)
        logger.info('Parsed {} play-by-play data'.format(boxscore_id))
        return df
    except Exception as e:
        logger.exception('Exception encountered when scraping PBP data for {}'
                         .format(boxscore_id))
        return None


def fetch_pbp_data_year(year):
    season = nba.Season(year)
    boxscore_ids = season.schedule().boxscore_id.values
    bsids_bag = db.from_sequence(boxscore_ids, npartitions=16)
    dfs_bag = bsids_bag.map(process_boxscore_id)
    dfs = dfs_bag.compute()
    filt_dfs = [df for df in dfs if df is not None]
    df = pd.concat(filt_dfs)
    clean_df = nba.pbp.clean_multigame_features(df)
    return clean_df


class PBPYearFetcher(luigi.Task):
    """
    TODO
    """

    year = luigi.IntParameter()

    def output(self):
        path = os.path.join(
            DATA_DIR, 'raw', 'pbp_{}.csv'.format(self.year)
        )
        return luigi.LocalTarget(path)

    def run(self):
        df = fetch_pbp_data_year(self.year)
        df.to_csv(self.output().path, index_label=False)


class PBPRangeFetcher(luigi.Task):
    """
    TODO
    """

    start_year = luigi.IntParameter()
    end_year = luigi.IntParameter()

    def requires(self):
        return [
            PBPYearFetcher(yr)
            for yr in range(self.start_year, self.end_year+1)
        ]
