import functools
import logging
import logging.config
import multiprocessing as mp
import os

import dotenv
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.externals import joblib

from sportsref import nba

from src import helpers

env_path = dotenv.find_dotenv()
dotenv.load_dotenv(env_path)
PROJ_DIR = os.environ['PROJ_DIR']
n_jobs = os.environ.get('SLURM_NTASKS', mp.cpu_count()-1)

seasons_train = range(2010, 2015)
seasons_test = range(2015, 2017)
seasons = seasons_train + seasons_test

reg_est = linear_model.LinearRegression()

def get_logger():
    logging.config.fileConfig(
        os.path.join(PROJ_DIR, 'logging_config.ini')
    )
    logger = logging.getLogger()
    return logger


def _design_matrix_one_season(args):
    logger = get_logger()
    logger.info('starting design_matrix_one_season')
    lineups, sub_orapm, sub_drapm, sub_hm_off, sub_y = args
    hm_lineups = lineups.ix[:, nba.pbp.HM_LINEUP_COLS]
    aw_lineups = lineups.ix[:, nba.pbp.AW_LINEUP_COLS]

    hm_off_df = pd.concat((hm_lineups[sub_hm_off], aw_lineups[sub_hm_off]),
                           axis=1)
    aw_off_df = pd.concat((aw_lineups[~sub_hm_off], hm_lineups[~sub_hm_off]),
                           axis=1)
    off_cols = ['off_player{}'.format(i) for i in range(1, 6)]
    def_cols = ['def_player{}'.format(i) for i in range(1, 6)]
    cols = off_cols + def_cols
    hm_off_df.columns = cols
    aw_off_df.columns = cols

    merged_df = pd.concat((hm_off_df, aw_off_df))
    off_df = merged_df[off_cols]
    def_df = merged_df[def_cols]

    rp_oval = sub_orapm.loc['RP']
    off_rapm = off_df.applymap(lambda p: sub_orapm.get(p, rp_oval))
    off_rapm = off_rapm.apply(np.sort, axis=1)

    rp_dval = sub_drapm.loc['RP']
    def_rapm = def_df.applymap(lambda p: sub_drapm.get(p, rp_oval))
    def_rapm = def_rapm.apply(np.sort, axis=1)

    merged_df = pd.concat((off_rapm, def_rapm), axis=1)
    n_hm_off = len(hm_off_df)
    merged_df['hm_off'] = [i < n_hm_off for i in range(len(merged_df))]
    new_sub_y = np.concatenate((sub_y[sub_hm_off], sub_y[~sub_hm_off]))
    merged_df['y'] = new_sub_y

    return merged_df


def create_design_matrix(lineups, orapm, drapm, seasons, hm_off, y):
    seasons_uniq = np.unique(seasons)
    pool = mp.Pool(min(4, n_jobs))
    args_to_eval = [
        (lineups[seasons == s], orapm.xs(s, level=1), drapm.xs(s, level=1),
         hm_off[seasons == s], y[seasons == s])
        for s in seasons_uniq
    ]
    df = pd.concat(pool.map(_design_matrix_one_season, args_to_eval))
    y = df.pop('y')
    return df, y


logger = get_logger()
logger.info('n_jobs: {}'.format(n_jobs))

# load and combine all player-season profiles and standardize within season
logger.info('loading profiles...')
profile_dfs = [
    helpers.get_profiles_data(season) for season in seasons
]
profile_df = pd.concat(profile_dfs)
orapm = profile_df['orapm']
drapm = profile_df['drapm']
del profile_df, profile_dfs

# load and process the play-by-play data for regression
logger.info('loading second half PBP...')
all_second_half = nba.pbp.clean_multigame_features(
    pd.concat([
        helpers.split_pbp_data(helpers.get_pbp_data(season))[1]
        for season in seasons
    ])
)
poss_grouped = all_second_half.groupby('poss_id')
poss_end = poss_grouped.tail(1)
poss_hm_off = poss_end.loc[:, 'hm_off'].values
y = poss_grouped.pts.sum().values
lineups = poss_end.loc[:, nba.pbp.ALL_LINEUP_COLS]
poss_seasons = poss_end.loc[:, 'season']
del all_second_half, poss_grouped, poss_end

# split lineups data into train and test
pbp_is_train = poss_seasons.isin(seasons_train)
lineups_train = lineups[pbp_is_train]
lineups_test = lineups[~pbp_is_train]
poss_year_train = poss_seasons[pbp_is_train]
poss_year_test = poss_seasons[~pbp_is_train]
hm_off_train = poss_hm_off[pbp_is_train]
hm_off_test = poss_hm_off[~pbp_is_train]
y_train = y[pbp_is_train]
y_test = y[~pbp_is_train]

# fit model on training data
logging.info('starting create_design_matrix for train')
X_train, y_train = create_design_matrix(
    lineups_train, orapm, drapm, poss_year_train, hm_off_train, y_train
)
logging.info('done with create_design_matrix for train')
logging.info('len(X_train) == {}'.format(len(X_train)))
logging.info('starting to fit regression model')
reg_est.fit(X_train, y_train)
logging.info('done fitting regression model')

# score model on test data
logging.info('starting create_design_matrix for test')
X_test, y_test = create_design_matrix(
    lineups_test, orapm, drapm, poss_year_test, hm_off_test, y_test
)
logging.info('done with create_design_matrix for test')
logging.info('predicting on X_test')
predictions = reg_est.predict(X_test)
logging.info('done predicting on X_test')
logging.info('generating metrics')

rmse = metrics.mean_squared_error(y_test, predictions)
r2 = metrics.r2_score(y_test, predictions)
mae = metrics.median_absolute_error(y_test, predictions)

logging.info('RMSE: {}'.format(rmse))
logging.info('R^2: {}'.format(r2))
logging.info('median abs err: {}'.format(mae))

perf_metrics = pd.Series({
    'rmse': rmse,
    'r2': r2,
    'median_abs_error': mae
})
perf_metrics.to_csv('data/models/basic_perf_metrics.csv')

logging.info('writing comparison model to disk for persistence')
joblib.dump(reg_est, os.path.join(PROJ_DIR, 'models', 'comparison_model.pkl'))
logging.info('wrote comparison model to disk for persistence')
