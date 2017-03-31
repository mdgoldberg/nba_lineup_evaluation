import os

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cluster, metrics

from sportsref import nba

from src import helpers
from src.visualization import visualize

env_path = dotenv.find_dotenv()
dotenv.load_dotenv(env_path)
PROJ_DIR = os.environ['PROJ_DIR']

if __name__ == '__main__':

    profile_df = pd.concat(
        [helpers.get_profiles_data(yr) for yr in range(2007, 2017)]
    )
    profile_df.drop('RP', axis=0, level=0, inplace=True)

    profiles_scaled = profile_df.groupby(level=1).transform(
        lambda z: (z - z.mean()) / z.std()
    )
    profs_kmeans = profiles_scaled.copy()

    def_cols = ['blk_pct', 'stl_pct', 'pf_pct', 'opp_to_pct', 'drapm']
    reb_cols = ['orb_pct', 'drb_pct']

    profs_kmeans.loc[:, def_cols] *= 2
    profs_kmeans.loc[:, reb_cols] *= 3

    sil_scores = []
    nc_vals = range(5, 21)
    for nc in nc_vals:
        print nc
        km = cluster.KMeans(n_clusters=nc, n_init=20, max_iter=500)
        clusters = km.fit_predict(profs_kmeans)
        sil_scores.append(metrics.silhouette_score(profs_kmeans, clusters))

    plt.plot(nc_vals, sil_scores)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Silhouette Score')
    plt.savefig(os.path.join(
        PROJ_DIR, 'reports', 'thesis', 'figures', 'sil_scores.png'
    ))

    km = cluster.KMeans(n_clusters=8, n_init=20, max_iter=500)
    clusters = km.fit_predict(profs_kmeans)
    cluster_df = profiles_scaled.assign(cluster=clusters)
    # visualize.write_output(cluster_df['cluster'], 'clusters.csv')

    means = pd.DataFrame(
        profiles_scaled.groupby(clusters).mean(),
        columns=profiles_scaled.columns
    )
    means.index += 1
    means.index.name = 'Cluster'
    means.columns.name = 'Feature'
    means = means.T.applymap(lambda x: round(x, 2))

    # visualize.write_table(means, 'cluster_means.txt', column_format='c'*8)
