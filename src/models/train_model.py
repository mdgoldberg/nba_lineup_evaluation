import functools
import logging
import logging.config
import multiprocessing as mp
import os

import dask
from dask import delayed
import dotenv
import numpy as np
import pandas as pd
from sklearn import ensemble, manifold, metrics, model_selection
from sklearn.externals import joblib

from sportsref import nba

from src import helpers

env_path = dotenv.find_dotenv()
dotenv.load_dotenv(env_path)
PROJ_DIR = os.environ['PROJ_DIR']
n_jobs = int(os.environ.get('SLURM_NTASKS', mp.cpu_count()-1))

seasons_train = range(2007, 2015)
seasons_test = range(2015, 2017)
seasons = seasons_train + seasons_test

dr_est = manifold.Isomap(n_components=5, n_neighbors=10)
reg_est = ensemble.RandomForestRegressor(
    max_depth=3, n_estimators=1000, n_jobs=n_jobs, verbose=2
)


def get_logger():
    logging.config.fileConfig(
        os.path.join(PROJ_DIR, 'logging_config.ini')
    )
    logger = logging.getLogger()
    return logger


def _design_matrix_one_season(args):
    logger = get_logger()
    logger.info('starting _design_matrix_one_season')
    lineups, sub_profs, sub_rapm, sub_hm_off = args
    hm_lineups = lineups.ix[:, nba.pbp.HM_LINEUP_COLS]
    aw_lineups = lineups.ix[:, nba.pbp.AW_LINEUP_COLS]
    rp_val = sub_rapm.loc['RP']
    hm_rapm = hm_lineups.applymap(lambda p: sub_rapm.get(p, rp_val))
    aw_rapm = aw_lineups.applymap(lambda p: sub_rapm.get(p, rp_val))
    hm_rapm_idxs = np.argsort(-hm_rapm, axis=1)
    aw_rapm_idxs = np.argsort(-aw_rapm, axis=1)
    hm_lineups = hm_lineups.apply(
        lambda r: r[hm_rapm_idxs.loc[r.name]].values, axis=1
    )
    aw_lineups = aw_lineups.apply(
        lambda r: r[aw_rapm_idxs.loc[r.name]].values, axis=1
    )
    hm_off_df = pd.concat((hm_lineups[sub_hm_off], aw_lineups[sub_hm_off]),
                           axis=1)
    aw_off_df = pd.concat((aw_lineups[~sub_hm_off], hm_lineups[~sub_hm_off]),
                           axis=1)
    cols = ['{}_player{}'.format(tm, i)
            for tm in ['off', 'def'] for i in range(1, 6)]
    hm_off_df.columns = cols
    aw_off_df.columns = cols
    merged_df = pd.concat((hm_off_df, aw_off_df))
    n_hm_off = len(hm_off_df)
    merged_df['hm_off'] = [i < n_hm_off for i in range(len(merged_df))]
    for col in cols:
        sub_profs.columns = [
            '{}_{}'.format(i, col) for i in range(sub_profs.shape[1])
        ]
        merged_df = pd.merge(
            merged_df, sub_profs, how='left',
            left_on=col, right_index=True
        ).fillna(sub_profs.loc['RP'])

    merged_df.drop(cols, axis=1, inplace=True)
    return merged_df


def create_design_matrix(lineups, profiles, seasons, rapm, hm_off):
    seasons_uniq = np.unique(seasons)
    pool = mp.Pool(min(n_jobs, 4))
    args_to_eval = [
        (lineups[seasons == s], profiles.xs(s, level=1), rapm.xs(s, level=1),
         hm_off[seasons == s])
        for s in seasons_uniq
    ]
    dfs = pool.map(_design_matrix_one_season, args_to_eval)
    return pd.concat(dfs)


if __name__ == '__main__':

    logger = get_logger()
    logger.info('n_jobs: {}'.format(n_jobs))

    # load and combine all player-season profiles and standardize within season
    logger.info('loading profiles...')
    profile_dfs = [
        helpers.get_profiles_data(season) for season in seasons
    ]
    profile_df = pd.concat(profile_dfs)
    rapm = profile_df.loc[:, ['orapm', 'drapm']].sum(axis=1)
    profiles_scaled = (
        profile_df.groupby(level=1)
        .transform(lambda x: (x - x.mean()) / x.std())
    )
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
    y = poss_grouped.pts.sum().values
    poss_hm_off = poss_end.loc[:, 'hm_off'].values
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

    # split profiles data into train and test
    prof_is_train = (
        profiles_scaled.index.get_level_values(1).isin(seasons_train)
    )
    profiles_train = profiles_scaled[prof_is_train]
    profiles_test = profiles_scaled[~prof_is_train]

    # fit model on training data
    logging.info('starting to fit DR model')
    latent_train = pd.DataFrame(
        dr_est.fit_transform(profiles_train), index=profiles_train.index
    )
    logging.info('done fitting DR model')
    logging.info('starting create_design_matrix for train')
    X_train = create_design_matrix(
        lineups_train, latent_train, poss_year_train, rapm, hm_off_train
    )
    logging.info('done with create_design_matrix for train')
    logging.info('starting to fit regression model')
    reg_est.fit(X_train, y_train)
    logging.info('done fitting regression model')

    # score model on test data
    logging.info('using DR model to transform test')
    latent_test = pd.DataFrame(
        dr_est.transform(profiles_test), index=profiles_test.index
    )
    logging.info('done transforming test profiles')
    logging.info('starting create_design_matrix for test')
    X_test = create_design_matrix(
        lineups_test, latent_test, poss_year_test, rapm, hm_off_test
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
    perf_metrics.to_csv('data/models/final_perf_metrics.csv')

    logging.info('writing models to disk for persistence')
    joblib.dump(dr_est, os.path.join(PROJ_DIR, 'models', 'dr_model.pkl'))
    joblib.dump(reg_est,
                os.path.join(PROJ_DIR, 'models', 'regression_model.pkl'))
    logging.info('wrote models to disk for persistence')
