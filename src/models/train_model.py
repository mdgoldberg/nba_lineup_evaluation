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
from sklearn import (base, decomposition, ensemble, linear_model,
                     model_selection, preprocessing)

from sportsref import nba

from src import helpers

env_path = dotenv.find_dotenv()
dotenv.load_dotenv(env_path)
PROJ_DIR = os.environ['PROJ_DIR']
n_jobs = os.environ.get('SLURM_NTASKS', mp.cpu_count()-1)

seasons = range(2014, 2017)

dr_ests = {
    'pca': decomposition.PCA(n_components=10),
    'kernel_pca': decomposition.KernelPCA(n_components=10)
}
dr_param_grids = {
    'pca': {},
    'kernel_pca': {
        'kernel': ['rbf'],
        'gamma': np.logspace(-3, 0, 4)
    }
}
reg_ests = {
    'lin_reg': linear_model.LinearRegression(),
    'rf': ensemble.RandomForestRegressor(n_jobs=n_jobs, n_estimators=100),
    'gb': ensemble.GradientBoostingRegressor(n_estimators=100)
}
reg_param_grids = {
    'lin_reg': {},
    'rf': {
        'max_depth': [3, 4, 5],
    },
    'gb': {
        'learning_rate': [.001, .01, .1],
        'max_depth': [3, 4, 5],
    }
}


def get_logger():
    logging.config.fileConfig(
        os.path.join(PROJ_DIR, 'logging_config.ini')
    )
    logger = logging.getLogger()
    return logger


def _design_matrix_one_season(args):
    lineups, profiles, seasons, rapm, hm_off, season = args
    sub_profs = profiles.xs(season, level=1)
    sub_rapm = rapm.xs(season, level=1)
    hm_lineups = lineups.ix[seasons == season, nba.pbp.HM_LINEUP_COLS]
    aw_lineups = lineups.ix[seasons == season, nba.pbp.AW_LINEUP_COLS]
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
    merged_df = pd.concat((hm_lineups, aw_lineups), axis=1)
    for line_col in nba.pbp.ALL_LINEUP_COLS:
        sub_profs.columns = [
            '{}_{}'.format(i, line_col)
            for i in range(sub_profs.shape[1])
        ]
        merged_df = pd.merge(
            merged_df, sub_profs, how='left',
            left_on=line_col, right_index=True
        ).fillna(sub_profs.loc['RP'])

    merged_df.drop(nba.pbp.ALL_LINEUP_COLS, axis=1, inplace=True)
    return merged_df


def create_design_matrix(lineups, profiles, seasons, rapm, hm_off):
    prof_cols = profiles.columns
    seasons_uniq = seasons.unique()
    pool = mp.Pool(min(n_jobs, len(seasons_uniq)))
    args_to_eval = [
        (lineups, profiles, seasons, rapm, hm_off, s)
        for s in seasons_uniq
    ]
    dfs = pool.map(_design_matrix_one_season, args_to_eval)
    return pd.concat(dfs).assign(hm_off=hm_off)


logger = get_logger()
logger.info(n_jobs)
seasons_train, seasons_test = model_selection.train_test_split(
    seasons, train_size=0.7, test_size=0.3
)

# load and combine all player-season profiles and standardize within season
logger.info('loading profiles...')
profile_dfs = [
    helpers.get_profiles_data(season) for season in seasons
]
profile_df = pd.concat(profile_dfs)
rapm = profile_df.loc[:, ['orapm', 'drapm']].sum(axis=1)
profiles_scaled = (
    profile_df.groupby(level=1).transform(lambda x: (x - x.mean()) / x.std())
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
prof_is_train = profiles_scaled.index.get_level_values(1).isin(seasons_train)
profiles_train = profiles_scaled[prof_is_train]
profiles_test = profiles_scaled[~prof_is_train]

logger.info('starting CV...')
results = []
for dr_name, dr_est in dr_ests.items():
    dr_param_grid = model_selection.ParameterGrid(dr_param_grids[dr_name])
    for dr_params in dr_param_grid:
        dr_est.set_params(**dr_params)
        dr_est.fit(profiles_train)
        latent_train = pd.DataFrame(
            dr_est.transform(profiles_train), index=profiles_train.index
        )
        latent_test = pd.DataFrame(
            dr_est.transform(profiles_test), index=profiles_test.index
        )
        logging.info('starting create_design_matrix for train')
        X_train = create_design_matrix(
            lineups_train, latent_train, poss_year_train, rapm, hm_off_train
        )
        logging.info('done with create_design_matrix for train')
        for reg_name, reg_est in reg_ests.items():
            reg_params_grid = model_selection.ParameterGrid(
                reg_param_grids[reg_name]
            )
            for reg_params in reg_params_grid:
                reg_est.set_params(**reg_params)
                logger.info('starting training for one param grid point...')
                cv_score = np.mean(model_selection.cross_val_score(
                        reg_est, X_train, y_train, cv=3,
                        groups=poss_year_train,
                        scoring='neg_mean_squared_error'
                ))
                results_row = {
                    'dim_red': dr_name,
                    'regress': reg_name,
                    'score': cv_score
                }
                results_row.update(dr_params)
                results_row.update(reg_params)
                logging.info(results_row)
                results.append(results_row)

res_df = pd.DataFrame(results)
logging.info(res_df.sort_values('score').head(5))
res_df.to_csv('data/testing/reg_results.csv', index_label=False)
