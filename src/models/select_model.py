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
from sklearn import (decomposition, ensemble, linear_model, manifold,
                     model_selection)
import xgboost as xgb

from sportsref import nba

from src import helpers

env_path = dotenv.find_dotenv()
dotenv.load_dotenv(env_path)
PROJ_DIR = os.environ['PROJ_DIR']
n_jobs = int(os.environ.get('SLURM_NTASKS', mp.cpu_count()-1))

seasons = range(2012, 2015) # 11-12, 12-13, 13-14

dr_ests = {
    'pca': decomposition.PCA(n_components=10),
    'isomap': manifold.Isomap(n_components=10, n_jobs=n_jobs),
    'lle': manifold.LocallyLinearEmbedding(n_components=10, n_jobs=n_jobs)
}
dr_param_grids = {
    'pca': {},
    'isomap': {
        'n_neighbors': [10, 50]
    },
    'lle': {
        'n_neighbors': [10, 50]
    }
}
reg_ests = {
    'lin_reg': linear_model.LinearRegression(),
    'rf': ensemble.RandomForestRegressor(n_jobs=n_jobs, n_estimators=100),
    'gb': xgb.XGBRegressor(n_estimators=200)
}
reg_param_grids = {
    'lin_reg': {},
    'rf': {},
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
    lineups, sub_profs, sub_rapm, sub_hm_off, sub_y = args
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
    new_sub_y = np.concatenate((sub_y[sub_hm_off], sub_y[~sub_hm_off]))
    merged_df['y'] = new_sub_y
    return merged_df


def create_design_matrix(lineups, profiles, seasons, rapm, hm_off, y):
    prof_cols = profiles.columns
    seasons_uniq = np.unique(seasons)
    pool = mp.Pool(min(4, n_jobs))
    args_to_eval = [
        (lineups[seasons == s], profiles.xs(s, level=1), rapm.xs(s, level=1),
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


logger.info('starting CV...')
results = []
for dr_name, dr_est in dr_ests.items():
    logging.info('starting DR: {}'.format(dr_name))
    dr_param_grid = model_selection.ParameterGrid(dr_param_grids[dr_name])
    for dr_params in dr_param_grid:
        dr_est.set_params(**dr_params)
        dr_est.fit(profiles_scaled)
        latent_profs = pd.DataFrame(
            dr_est.transform(profiles_scaled), index=profiles_scaled.index
        )
        logging.info('starting create_design_matrix')
        X, y = create_design_matrix(
            lineups, latent_profs, poss_seasons, rapm, poss_hm_off, y
        )
        logging.info('done with create_design_matrix for train')
        logging.info('len(X) == {}'.format(len(X)))
        for reg_name, reg_est in reg_ests.items():
            logging.info('starting regression: {}'.format(reg_name))
            reg_params_grid = model_selection.ParameterGrid(
                reg_param_grids[reg_name]
            )
            for reg_params in reg_params_grid:
                reg_est.set_params(**reg_params)
                logger.info('starting training for one param grid point...')
                cv_score = np.mean(model_selection.cross_val_score(
                    reg_est, X, y, cv=3, groups=poss_seasons,
                    scoring='neg_mean_squared_error', n_jobs=n_jobs
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
logging.info(res_df.sort_values('score').tail(5))
res_df.to_csv('data/models/selection_results.csv', index_label=False)
