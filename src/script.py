import multiprocessing as mp
import sys
import time

import numpy as np
import pandas as pd
from sklearn import cluster, preprocessing

from sportsref import nba


def process_player(p_id):
    print 'starting', p_id
    start = time.time()
    p = nba.Player(p_id)
    per100 = p.stats_per100()
    per100 = per100.ix[per100.has_class_full_table].iloc[-1]
    advanced = p.stats_advanced()
    advanced = advanced.ix[advanced.has_class_full_table].iloc[-1]
    shooting = p.stats_shooting()
    shooting = shooting.ix[shooting.has_class_full_table].iloc[-1]

    features1 = advanced[
        ['trb_pct', 'ast_pct', 'stl_pct', 'blk_pct', 'tov_pct', 'usg_pct']
    ].fillna(0).values
    features2 = shooting[
        ['fg2a_pct_fga', 'fg3a_pct_fga', 'fg2_pct', 'fg3_pct', 'fg2_pct_ast',
         'fg3_pct_ast']
    ].fillna(0).values
    features = np.concatenate((features1, features2))

    duration = time.time() - start
    print 'done {} in {:.3f} sec'.format(p_id, duration)
    return features


pool = mp.Pool(mp.cpu_count() - 1)

season = nba.Season(2016)
per100 = season.player_stats_per100()
per100 = per100.ix[per100['has_class_full_table']]
per100 = per100.ix[per100['mp'] >= 82 * 15]
players = per100.index

print 'Found {} players'.format(len(players))

cols = ['trb_pct', 'ast_pct', 'stl_pct', 'blk_pct', 'tov_pct', 'usg_pct',
        'fg2a_pct_fga', 'fg3a_pct_fga', 'fg2_pct', 'fg3_pct', 'fg2_pct_ast',
        'fg3_pct_ast']

results = pool.map_async(process_player, players).get(sys.maxint)
standardized_results = preprocessing.scale(results)
player_names = pd.Index([nba.Player(p_id).name() for p_id in players])
stats = pd.DataFrame(data=standardized_results, index=player_names,
                     columns=cols)

km = cluster.KMeans(n_clusters=8, verbose=1, tol=1e-6, n_jobs=3)
clusters = km.fit_predict(standardized_results)
groups = player_names.groupby(clusters)
centers = pd.DataFrame(data=km.cluster_centers_, columns=cols)


def group_distances(group_idx):
    return pd.Series(
        {player: np.linalg.norm(stats.ix[player].values -
                                centers.iloc[group_idx].values)
         for player in groups[group_idx]}).sort_values()
