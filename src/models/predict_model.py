from collections import Counter
import itertools

import dask
from dask import bag as db, diagnostics
import numpy as np
import pandas as pd
from sklearn.externals import joblib

from sportsref import nba, decorators

from src import helpers
from src.models import train_model

dr_model = joblib.load('models/dr_model.pkl')
reg_model = joblib.load('models/regression_model.pkl')

# 1. load regular profiles and compute RAPM for each player
print 'Please wait while loading profiles...'
profile_dfs = [
    helpers.get_profiles_data(season) for season in range(2007, 2017)
]
profile_df = pd.concat(profile_dfs)
rapm = profile_df.loc[:, ['orapm', 'drapm']].sum(axis=1)

# 2. load latent profiles for use in create_design_matrix
profiles_scaled = (
    profile_df.groupby(level=1).transform(lambda x: (x - x.mean()) / x.std())
)
latent_profiles = pd.DataFrame(
    dr_model.transform(profiles_scaled), index=profiles_scaled.index
)
print 'Done loading profiles!'


def expected_pd(hm_players, aw_players, hm_off, season):
    players = np.concatenate((hm_players, aw_players)).reshape(1, -1)
    lineups = pd.DataFrame(
        players, columns=nba.pbp.HM_LINEUP_COLS + nba.pbp.AW_LINEUP_COLS
    )
    hm_off = pd.Series([hm_off])
    seasons = pd.Series([season])
    X_off = train_model.create_design_matrix(
        lineups, latent_profiles, seasons, rapm, hm_off
    )
    X_def = train_model.create_design_matrix(
        lineups, latent_profiles, seasons, rapm, ~hm_off
    )
    return 100. * (reg_model.predict(X_off) - reg_model.predict(X_def))


def evaluate_player(player, season):
    hm_lineup = [player, 'RP', 'RP', 'RP', 'RP']
    aw_lineup = ['RP' for _ in range(5)]
    return expected_pd(hm_lineup, aw_lineup, True, season)


def evaluate_lineup(lineup, season):
    aw_lineup = ['RP' for _ in range(5)]
    return expected_pd(lineup, aw_lineup, True, season)


@decorators.memoize
def get_pbp_data_wrapper(year):
    return helpers.get_pbp_data(year)


def evaluate_game(bsid):
    bs = nba.BoxScore(bsid)
    season = bs.season()
    stats = bs.basic_stats()
    hm_players = stats.ix[stats.is_home, 'player_id'].values
    aw_players = stats.ix[~stats.is_home, 'player_id'].values
    hm_starters = hm_players[stats.is_starter]
    aw_starters = aw_players[stats.is_starter]
    home, away = bs.home(), bs.away()
    true_pd = bs.home_score() - bs.away_score()
    exp_pd = expected_pd(hm_starters, aw_starters, True, season)

    best_hm_pd = max(
        expected_pd(hm_lineup, aw_starters, True, season)
        for hm_lineup in itertools.combinations(hm_players, 5)
    )
    best_aw_pd = max(
        expected_pd(hm_starters, aw_lineup, True, season)
        for aw_lineup in itertools.combinations(aw_players, 5)
    )

    return {
        '{}_true_pd'.format(home): true_pd,
        '{}_true_pd'.format(away): -true_pd,
        '{}_exp_pd'.format(home): exp_pd,
        '{}_exp_pd'.format(away): -exp_pd,
        '{}_best_exp_pd'.format(home): best_hm_pd,
        '{}_best_exp_pd'.format(away): best_aw_pd,
    }


def evaluate_team_schedule(team_id, year):
    team = nba.Team(team_id)
    schedule = team.schedule(year)
    bsids = db.from_sequence(schedule.boxscore_id.values)
    with diagnostics.ProgressBar():
        results = (
            bsids.map(evaluate_game).compute(get=dask.multiprocessing.get)
        )
    res_df = pd.DataFrame(results)
    cols = ['{}_{}'.format(team_id, col)
            for col in ['true_pd', 'exp_pd', 'best_exp_pd']]
    return res_df.ix[:, cols]


@decorators.memoized
def get_starters(team_id, year):
    start_counter = Counter()
    for bs_id in nba.Team(team_id).schedule(year).boxscore_id.values:
        bs = nba.BoxScore(bs_id)
        stats = bs.basic_stats()
        starters = tuple(stats.ix[
            (stats.team_id == team_id) & stats.is_starter, 'player_id'
        ].values)
        start_counter[starters] += 1
    return list(max(start_counter, key=start_counter.get))


def process_trade(team1, team2, starters1, starters2, trade):
    if not trade:
        return starters1, starters2

    if set(trade[:2]) == set((team1, team2)):
        player1 = trade[2]
        player2 = trade[3]
        if player1 in starters1:
            # player 1 goes to team 2
            starters1.remove(player1)
            starters2.append(player1)
            # player 2 goes to team 1
            starters2.remove(player2)
            starters1.append(player2)
        else:
            assert player1 in starters2
            # player 1 goes to team 1
            starters2.remove(player1)
            starters1.append(player1)
            # player 2 goes to team 2
            starters1.remove(player2)
            starters2.append(player2)

    return starters1, starters2


def year_off_def_matrix(year, trade=None):
    season = nba.Season(year)
    team_ids = season.get_team_ids()
    n = len(team_ids)
    team_idxs = dict(zip(team_ids, range(n)))
    grid = np.zeros((n, n))
    for team1, team2 in itertools.combinations_with_replacement(team_ids, 2):
        starters1 = get_starters(team1, year)
        starters2 = get_starters(team2, year)
        starters1, starters2 = process_trade(team1, team2, starters1,
                                             starters2, trade)
        pd_12 = expected_pd(starters1, starters2, True, year)
        pd_21 = expected_pd(starters2, starters1, True, year)
        idx1 = team_idxs[team1]
        idx2 = team_idxs[team2]
        grid[idx1, idx2] = pd_12
        grid[idx2, idx1] = pd_21

    df = pd.DataFrame(grid, index=team_ids, columns=team_ids)
    df.index.name = 'offense'
    df.columns.name = 'defense'
    return df


def year_diff_matrix(year, trade=None):
    grid_df = year_off_def_matrix(year, trade=trade)
    season = nba.Season(year)
    team_ids = season.get_team_ids()
    n = len(team_ids)
    team_idxs = dict(zip(team_ids, range(n)))

    new_grid = np.zeros((n, n))
    for team1, team2 in itertools.combinations(team_ids, 2):
        idx1 = team_idxs[team1]
        idx2 = team_idxs[team2]
        new_grid[idx1, idx2] = (
            grid_df.loc[team1, team2] - grid_df.loc[team2, team1]
        )
        new_grid[idx2, idx1] = (
            grid_df.loc[team2, team1] - grid_df.loc[team1, team2]
        )

    for i in range(n):
        new_grid[i, i] = 0.

    df = pd.DataFrame(new_grid, index=team_ids, columns=team_ids)
    df.index.name = 'Team'
    df.columns.name = 'Opponent'
    return df


def evaluate_trade(team1, team2, player1, player2, year):
    # generate matrices for with and without trade
    trade = (team1, team2, player1, player2)
    reg_mat = year_diff_matrix(year)
    trade_mat = year_diff_matrix(year, trade)

    # compare matrices
    season = nba.Season(year)
    team_ids = season.get_team_ids()
    n = len(team_ids)

    ppp_advs = []
    rank_advs = []
    for team in (team1, team2):
        team_idx = team_ids.index(team)

        reg_ppp = np.sum(reg_mat[team_idx])
        trade_ppp = np.sum(trade_mat[team_idx])
        ppp_adv = trade_ppp - reg_ppp
        ppp_advs.append(ppp_adv)

        reg_rank = n - np.sum(reg_mat[team_idx] > 0)
        trade_rank = n - np.sum(trade_mat[team_idx] > 0)
        rank_adv = -(trade_rank - reg_rank)
        rank_advs.append(rank_adv)

    ppp_advs = np.array(ppp_advs)
    rank_advs = np.array(rank_advs)

    return ppp_advs, rank_advs


def evaluate_all_trades(year):
    season = nba.Season(year)
    team_ids = season.get_team_ids()
    trade_results = []
    for teams in itertools.combinations(team_ids, 2):
        starters = [get_starters(team, year) for team in teams]
        for players in itertools.product(*starters):
            ppps, ranks = evaluate_trade(teams[0], teams[1], players[0],
                                         players[1], year)
            result = {
                'team1': teams[0], 'team2': teams[1],
                'player1': players[0], 'player2': players[1],
                'team1_ppp_adv': ppps[0], 'team2_ppp_adv': ppps[1],
                'team1_rank_adv': ranks[0], 'team2_rank_adv': ranks[1]
            }
            trade_results.append(result)
    return pd.DataFrame(trade_results)
