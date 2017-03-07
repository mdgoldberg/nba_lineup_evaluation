import numpy as np
import pandas as pd
import dask

from sportsref import nba, utils, decorators
from dask import dataframe as dd, bag as db, diagnostics, delayed


def year_boxscore_ids(yr):
    season = nba.Season(yr)
    return season.get_schedule().boxscore_id.values


def diff_pm(bsid):
    bs = nba.BoxScore(bsid)
    df = bs.pbp(sparse_lineups=True)
    stats = bs.basic_stats().query('mp > 0')
    true_pm = stats.set_index('player_id').plus_minus
    calc_pm = {}
    for i, row in stats.iterrows():
        p = row['player_id']
        col = '{}_in'.format(p)
        hm_mult = 1 if row['is_home'] else -1
        try:
            sub_df = df[df[col]]
        except KeyError:
            print '{} error in {}'.format(col, bsid)
            continue
        calc_pm[p] = (sub_df.hm_pts - sub_df.aw_pts).sum() * hm_mult
    calc_pm = pd.Series(calc_pm)
    diff = (true_pm - calc_pm).dropna()
    diff = diff[diff != 0]
    diff_df = pd.DataFrame({
        'team': diff.index.map(stats.set_index('player_id').team.get),
        'pm_diff': diff
    })
    return true_pm, calc_pm, diff_df


def process_single(bsid):
    try:
        true, calc, diff = diff_pm(bsid)
    except Exception as e:
        return (bsid, 'error', None)

    is_warning = not diff.empty
    return (bsid, 'warning' if is_warning else 'success', diff)


def process_games(bsids):
    bsid_bag = db.from_sequence(bsids)
    result_bag = bsid_bag.map(process_single)
    return result_bag


def summary_from_tuples(tups):
    bsids, results, diffs = zip(*tups)
    res_df = pd.DataFrame({
        'boxscore_id': bsids,
        'result': results,
        'n_diff': [len(d) if d is not None else np.nan for d in diffs]
    })
    return res_df


if __name__ == '__main__':
    game_tuples = []

    for yr in range(2002, 2017):
        bsids = year_boxscore_ids(yr)
        result_bag = process_games(bsids)
        with diagnostics.ProgressBar():
            game_tuples.extend(result_bag.compute())

    res_df = summary_from_tuples(game_tuples)
