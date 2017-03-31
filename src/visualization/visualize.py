from collections import Counter
import logging
import os

import dotenv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib
from matplotlib import pyplot as plt

from sportsref import nba

from src import helpers
from src.models import predict_model

env_path = dotenv.find_dotenv()
dotenv.load_dotenv(env_path)
PROJ_DIR = os.environ['PROJ_DIR']


def get_logger():
    logging.config.fileConfig(
        os.path.join(PROJ_DIR, 'logging_config.ini')
    )
    logger = logging.getLogger()
    return logger


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def write_output(obj, filename):
    obj.to_csv(os.path.join(
        PROJ_DIR, 'data', 'results', filename
    ))


def write_table(obj, filename, **kwargs):
    with open(
        os.path.join(PROJ_DIR, 'reports', 'thesis', 'tables', filename), 'w'
    ) as f:
        obj.to_latex(buf=f, **kwargs)


def plot_matrix(grid, out_filename=None, midval=0):
    height = grid.shape[0]
    width = grid.shape[1]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'cmap', ['red', 'white', 'green'], 256
    )

    maxval = grid.values.max()
    minval = grid.values.min()
    mid = (midval - minval) / (maxval - minval)
    cmap = shiftedColorMap(cmap, start=0, midpoint=mid, stop=1)

    fig, ax = plt.subplots()
    img = ax.imshow(grid, cmap=cmap)
    plt.colorbar(img, cmap=cmap)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticks(np.arange(width))
    ax.set_xticklabels(grid.columns, rotation=90)
    ax.set_yticks(np.arange(height))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel(grid.columns.name)
    ax.set_ylabel(grid.index.name)
    fig.tight_layout()
    if out_filename:
        fig.savefig(os.path.join(
            PROJ_DIR, 'reports', 'thesis', 'figures', out_filename
        ))
    else:
        plt.show()


def produce_results_for_year(year):
    logger = get_logger()
    logger.info('loading PBP data')
    pbp = helpers.get_pbp_data(year).groupby('poss_id').tail(1)
    profiles = helpers.get_profiles_data(year).xs(year, level=1)
    tot_rapm = profiles[['orapm', 'drapm']].sum(axis=1)
    season = nba.Season(year)
    clusters = pd.read_csv(os.path.join(
        PROJ_DIR, 'data', 'results', 'clusters.csv'
    ), index_col=[0,1], header=None)[2].xs(year, level=1)

    logger.info('evaluating players')
    players_df = season.player_stats_totals().query(
        'has_class_full_table and mp >= 820'
    )
    players = players_df.player_id.values
    player_ratings = [
        predict_model.evaluate_player(player, year) for player in players
    ]
    orapms = [
        profiles.orapm.get(player, profiles.loc['RP', 'orapm']) * 100
        for player in players
    ]
    drapms = [
        profiles.drapm.get(player, profiles.loc['RP', 'drapm']) * 100
        for player in players
    ]
    rapms = [
        tot_rapm.get(player, tot_rapm.loc['RP']) * 100 for player in players
    ]
    player_clusts = [
        clusters.get(player, 'RP') for player in players
    ]
    player_evals = pd.DataFrame({
        'Player': players,
        'Cluster': player_clusts,
        'Rating': player_ratings,
        'ORAPM': orapms, 'DRAPM': drapms, 'RAPM': rapms
    })[['Player', 'Cluster', 'Rating', 'ORAPM', 'DRAPM', 'RAPM']]
    player_evals.sort_values('Rating', ascending=False, inplace=True)
    write_output(player_evals, 'player_evals.csv')

    logger.info('evaluating lineups')
    hm_lineups = Counter(
        map(tuple, map(sorted, pbp[nba.pbp.HM_LINEUP_COLS].values))
    )
    aw_lineups = Counter(
        map(tuple, map(sorted, pbp[nba.pbp.AW_LINEUP_COLS].values))
    )
    all_lineups = dict((hm_lineups + aw_lineups).most_common(100)).keys()
    lineup_ratings = [
        predict_model.evaluate_lineup(lineup, year) for lineup in all_lineups
    ]
    lineups = [list(lineup) for lineup in all_lineups]
    lineup_clusts = [
        tuple(clusters.get(p, 'RP') for p in lineup)
        for lineup in lineups
    ]
    rapm_sums = [
        sum(tot_rapm.get(p, tot_rapm.loc['RP']) for p in lineup)
        for lineup in lineups
    ]
    lineup_evals = pd.DataFrame(
        lineups, columns=['Player {}'.format(i) for i in range(1, 6)]
    )
    lineup_evals['Cluster Types'] = lineup_clusts
    lineup_evals['Total RAPM'] = np.array(rapm_sums) * 100.
    lineup_evals['Rating'] = lineup_ratings
    lineup_evals.sort_values('Rating', ascending=False, inplace=True)
    write_output(lineup_evals, 'lineup_evals.csv')

    logger.info('evaluating starting offense vs defense')
    off_def_matrix = (predict_model.year_off_def_matrix(year)
                      .sort_index().sort_index(axis=1))
    plot_matrix(off_def_matrix, 'off_def_matrix.png', midval=104.5)
    write_output(off_def_matrix, 'off_def_matrix.csv')

    logger.info('evaluating starting lineup point differential')
    point_diff_matrix = (predict_model.year_diff_matrix(year)
                         .sort_index().sort_index(axis=1))
    plot_matrix(point_diff_matrix, 'point_diff_matrix.png')
    write_output(point_diff_matrix, 'point_diff_matrix.csv')


if __name__ == '__main__':
    produce_results_for_year(2016)
