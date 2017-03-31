import logging
import os

import dotenv
import numpy as np
import pandas as pd
import matplotlib as mpl
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


def write_output(obj, filename):
    obj.to_csv(os.path.join(
        PROJ_DIR, 'data', 'results', filename
    ))


def write_table(obj, filename, **kwargs):
    with open(
        os.path.join(PROJ_DIR, 'reports', 'thesis', 'tables', filename), 'w'
    ) as f:
        obj.to_latex(buf=f, **kwargs)


def plot_matrix(grid, out_filename=None, figsize=None):
    height = grid.shape[0]
    width = grid.shape[1]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'cmap', ['red', 'white', 'green'], 256
    )
    fig, ax = plt.subplots(figsize=figsize)
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
    pbp = helpers.get_pbp_data(year)
    profiles = helpers.get_profiles_data(year).xs(year, level=1)
    tot_rapm = profiles[['orapm', 'drapm']].sum(axis=1)
    season = nba.Season(year)
    clusters = pd.read_csv(os.path.join(
        PROJ_DIR, 'data', 'results', 'clusters.csv'
    ), index_col=[0,1])['0'].xs(year, level=1)

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
    hm_lineups = set(map(tuple,
                         map(sorted, pbp[nba.pbp.HM_LINEUP_COLS].values)))
    aw_lineups = set(map(tuple,
                         map(sorted, pbp[nba.pbp.AW_LINEUP_COLS].values)))
    all_lineups = hm_lineups | aw_lineups
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
    lineup_evals['Total RAPM'] = rapm_sums
    lineup_evals['Rating'] = lineup_ratings
    lineup_evals.sort_values('Rating', ascending=False, inplace=True)
    write_output(lineup_evals, 'lineup_evals.csv')

    logger.info('evaluating starting offense vs defense')
    off_def_matrix = predict_model.year_off_def_matrix(year)
    plot_matrix(off_def_matrix, 'off_def_matrix.png')
    write_output(off_def_matrix, 'off_def_matrix.csv')

    logger.info('evaluating starting lineup point differential')
    point_diff_matrix = predict_model.year_diff_matrix(year)
    plot_matrix(point_diff_matrix, 'point_diff_matrix.png')
    write_output(point_diff_matrix, 'point_diff_matrix.csv')

    # TODO: make this faster?
    logger.info('evaluating starting decisions')
    team_ids = season.get_team_ids()
    sched_eval_df = pd.concat([
        predict_model.evaluate_team_schedule(team_id, year)
        for team_id in team_ids
    ])
    write_output(sched_eval_df, 'sched_evals.csv')

    # TODO: make this faster?
    logger.info('evaluating all trades')
    trade_evals = predict_model.evaluate_all_trades(year)
    write_output(trade_evals, 'trade_evals.csv')


if __name__ == '__main__':
    produce_results_for_year(2016)
