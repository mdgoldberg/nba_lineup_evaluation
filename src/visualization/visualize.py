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
    ), index_label=False)


def plot_matrix(grid, out_filename=None):
    n = grid.shape[0]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'cmap', ['red', 'white', 'green'], 256
    )
    fig, ax = plt.subplots()
    img = ax.imshow(grid, cmap=cmap)
    plt.colorbar(img, cmap=cmap)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(grid.columns, rotation=90)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(grid.index)
    ax.set_xlabel(grid.columns.name)
    ax.set_ylabel(grid.index.name)
    if out_filename:
        fig.savefig(os.path.join(PROJ_DIR, 'data', 'figures', out_filename))
    else:
        plt.show()


def produce_results_for_year(year):
    logger = get_logger()
    logger.info('loading PBP data')
    pbp = helpers.get_pbp_data(year)
    season = nba.Season(year)

    logger.info('evaluating players')
    players_df = season.player_stats_totals().query('mp >= 820')
    players = players_df.player_id.values[:250]
    player_evals = pd.Series({
        player: predict_model.evaluate_player(player, year)
        for player in players
    })
    write_output(player_evals, 'player_evals.csv')

    logger.info('evaluating lineups')
    hm_lineups = set(map(tuple,
                         map(sorted, pbp[nba.pbp.HM_LINEUP_COLS].values)))
    aw_lineups = set(map(tuple,
                         map(sorted, pbp[nba.pbp.AW_LINEUP_COLS].values)))
    all_lineups = hm_lineups | aw_lineups
    lineup_evals = pd.Series({
        lineup: predict_model.evaluate_lineup(lineup, year)
        for lineup in all_lineups
    })
    write_output(lineup_evals, 'lineup_evals.csv')

    logger.info('evaluating starting decisions')
    team_ids = season.get_team_ids()
    sched_eval_df = pd.concat([
        predict_model.evaluate_team_schedule(team_id, year)
        for team_id in team_ids
    ])
    write_output(sched_eval_df, 'sched_evals.csv')

    logger.info('evaluating starting offense vs defense')
    off_def_matrix = predict_model.year_off_def_matrix(year)
    plot_matrix(off_def_matrix, 'off_def_matrix.png')
    write_output(off_def_matrix, 'off_def_matrix.csv')

    logger.info('evaluating starting lineup point differential')
    point_diff_matrix = predict_model.year_diff_matrix(year)
    plot_matrix(point_diff_matrix, 'point_diff_matrix.png')
    write_output(point_diff_matrix, 'point_diff_matrix.csv')

    logger.info('evaluating all trades')
    trade_evals = predict_model.evaluate_all_trades(year)
    write_output(trade_evals, 'trade_evals.csv')
