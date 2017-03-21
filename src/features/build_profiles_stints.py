import logging
import os
import time

import dotenv
import luigi
import numpy as np
import pandas as pd
from sklearn import linear_model, grid_search

from sportsref import nba, decorators
from src.data import pbp_fetch_data

env_path = dotenv.find_dotenv()
dotenv.load_dotenv(env_path)
PROJ_DIR = os.environ['PROJ_DIR']


def get_logger():
    logging.config.fileConfig(
        os.path.join(PROJ_DIR, 'logging_config.ini')
    )
    logger = logging.getLogger()
    return logger


class YearProfiles(luigi.Task):

    year = luigi.IntParameter()

    def requires(self):
        return [
            pbp_fetch_data.PBPYearFetcher(self.year-1),
            pbp_fetch_data.PBPYearFetcher(self.year),
        ]

    def output(self):
        path = os.path.join(
            PROJ_DIR, 'data', 'processed', 'profiles_{}.csv'.format(self.year)
        )
        return luigi.LocalTarget(path)

    def run(self):
        profiles = year_profiles(self.year)
        profiles.to_csv(self.output().path, index_label=False)


class RangeProfiles(luigi.Task):

    start_year = luigi.IntParameter()
    end_year = luigi.IntParameter()

    def requires(self):
        return [
            YearProfiles(year)
            for year in range(self.start_year, self.end_year+1)
        ]


def year_profiles(year):
    """Get player profiles to be used for possessions from a given year.

    :year: int representing the season
    :returns: DataFrame with 361 rows and n_features columns.
    """
    logger = get_logger()

    this_year = get_data(year)
    last_year = get_data(year-1)
    first_half, _ = split_data(this_year)

    all_pcols = (nba.pbp.sparse_lineup_cols(first_half) +
                   nba.pbp.sparse_lineup_cols(last_year))
    all_players = [col[:-3] for col in all_pcols]

    # compute each players # of off/def plays and determine replacement level
    all_off_plays = combine_stat(get_off_plays, last_year, first_half,
                                 all_players, [])
    all_def_plays = combine_stat(get_def_plays, last_year, first_half,
                                 all_players, [])
    all_plays = all_off_plays + all_def_plays
    all_plays.sort_values(ascending=False, inplace=True)
    players = all_plays.iloc[:360].index.values
    reps = all_plays.iloc[360:].index.values
    off_plays = all_off_plays[players]
    off_plays['RP'] = all_off_plays[reps].sum()
    def_plays = all_def_plays[players]
    def_plays['RP'] = all_def_plays[reps].sum()

    # offensive stats
    logger.info('starting offense')
    fga = combine_stat(get_tot_fga, last_year, first_half, players, reps)
    fga_by_region = combine_stat(get_fga_by_region, last_year, first_half,
                                 players, reps)
    asts = combine_stat(get_ast, last_year, first_half, players, reps)
    tm_fgm = combine_stat(get_teammate_fgm, last_year, first_half,
                          players, reps)
    fta = combine_stat(get_fta, last_year, first_half, players, reps)
    ftm = combine_stat(get_ftm, last_year, first_half, players, reps)
    bad_passes = combine_stat(get_bad_passes, last_year, first_half,
                              players, reps)
    lost_balls = combine_stat(get_lost_balls, last_year, first_half,
                              players, reps)
    travels = combine_stat(get_travels, last_year, first_half, players, reps)
    off_fouls = combine_stat(get_off_fouls, last_year, first_half,
                             players, reps)
    fouls_drawn = combine_stat(get_fouls_drawn, last_year, first_half,
                               players, reps)
    fg2m = combine_stat(get_fg2m, last_year, first_half, players, reps)
    fg3m = combine_stat(get_fg3m, last_year, first_half, players, reps)
    fg2m_astd = combine_stat(get_fg2m_assisted, last_year, first_half,
                             players, reps)
    fg3m_astd = combine_stat(get_fg3m_assisted, last_year, first_half,
                             players, reps)

    # offensive features
    logger.info('offensive features')
    off_profiles = pd.concat((
        fga/off_plays, asts/tm_fgm, ftm/fta, fta/fga, fouls_drawn/off_plays,
        lost_balls/off_plays, bad_passes/off_plays, travels/off_plays,
        off_fouls/off_plays, fg2m_astd/fg2m, fg3m_astd/fg3m
    ), axis=1)
    off_profiles.columns = [
        'fga_per_play', 'ast_pct', 'ft_pct', 'fta_per_fga', 'pf_drawn_pct',
        'lost_ball_pct', 'bad_pass_pct', 'travel_pct', 'off_foul_pct',
        'fg2m_ast_pct', 'fg3m_ast_pct'
    ]
    off_profiles = pd.concat((off_profiles, fga_by_region.divide(fga, axis=0)),
                             axis=1).fillna(0)

    # defensive stats
    logger.info('starting defense')
    blk = combine_stat(get_blk, last_year, first_half, players, reps)
    stl = combine_stat(get_stl, last_year, first_half, players, reps)
    opp_fg2a = combine_stat(get_opp_fg2a, last_year, first_half,
                             players, reps)
    opp_fga = combine_stat(get_opp_fga, last_year, first_half, players, reps)
    pf = combine_stat(get_pf, last_year, first_half, players, reps)
    opp_fta = combine_stat(get_opp_fta, last_year, first_half, players, reps)
    opp_to = combine_stat(get_opp_to, last_year, first_half, players, reps)

    # defensive features
    logger.info('defensive features')
    def_profiles = pd.concat((
        blk/opp_fg2a, stl/def_plays, pf/def_plays, opp_to/def_plays
    ), axis=1).fillna(0)
    def_profiles.columns = [
        'blk_pct', 'stl_pct', 'pf_pct', 'opp_to_pct'
    ]

    # rebounding stats
    logger.info('starting rebounding')
    orb = combine_stat(get_orb, last_year, first_half, players, reps)
    drb = combine_stat(get_drb, last_year, first_half, players, reps)
    orb_opp = combine_stat(get_orb_opps, last_year, first_half, players, reps)
    drb_opp = combine_stat(get_drb_opps, last_year, first_half, players, reps)

    # rebounding features
    logger.info('defensive features')
    reb_profiles = pd.concat((
        orb/orb_opp, drb/drb_opp
    ), axis=1).fillna(0)
    reb_profiles.columns = [
        'orb_pct', 'drb_pct'
    ]

    # RAPM
    logger.info('starting RAPM')
    combined_df = nba.pbp.clean_multigame_features(
        pd.concat((last_year, first_half))
    )
    rapm_profiles = combined_rapm(combined_df, players, reps)

    # TODO: IOT?

    logger.info('combining all profiles')
    profiles = pd.concat(
        (off_profiles, def_profiles, reb_profiles, rapm_profiles),
        axis=1
    )
    profiles.index = [(p, year) for p in profiles.index]

    return profiles


@decorators.memoize
def get_data(yr):
    return pd.read_csv(
        os.path.join(PROJ_DIR, 'data', 'raw', 'pbp_{}.csv'.format(yr))
    )


def split_data(df):
    df = df.sort_values(['year', 'month', 'day'])
    game_num = np.cumsum(df.boxscore_id != df.boxscore_id.shift(1))
    mid = game_num.max() / 2
    first_half = df.ix[game_num <= mid]
    second_half = df.ix[game_num > mid]
    return first_half, second_half


def combine_stat(stat_func, last_year, first_half, players, reps, weight=6):

    last_stat = stat_func(last_year, players, reps).fillna(0)
    half_stat = stat_func(first_half, players, reps).fillna(0)

    return last_stat + weight*half_stat


def get_off_plays(df, players, reps):

    def num_on_off(df, p):
        in_on_off = (df['{}_in'.format(p)] == (2*df.hm_off - 1))
        return df.ix[in_on_off, 'play_id'].nunique()

    on_floor = df.ix[df.is_fga | df.is_to | df.is_pf]
    player_ops = pd.Series({
        p: num_on_off(on_floor, p) if '{}_in'.format(p) in df else 0.
        for p in players
    })
    rep_ops = np.sum(
        num_on_off(on_floor, p) if '{}_in'.format(p) in df else 0.
        for p in reps
    )
    player_ops['RP'] = rep_ops
    return player_ops



def get_def_plays(df, players, reps):

    def num_on_def(df, p):
        in_on_def = (df['{}_in'.format(p)] == (-2*df.hm_off + 1))
        return df.ix[in_on_def, 'play_id'].nunique()

    on_floor = df.ix[df.is_fga | df.is_to | df.is_pf]
    player_dps = pd.Series({
        p: num_on_def(on_floor, p) if '{}_in'.format(p) in df else 0.
        for p in players
    })
    rep_dps = np.sum(
        num_on_def(on_floor, p) if '{}_in'.format(p) in df else 0.
        for p in reps
    )
    player_dps['RP'] = rep_dps
    return player_dps


def get_fga_by_region(df, players, reps):
    shots = df.ix[df.is_fga]
    conditions = [
        '~is_three & (shot_dist <= 4)',  # restricted area
        '~is_three & (4 < shot_dist <= 8)',  # rest of paint
        '~is_three & (8 < shot_dist <= 16)',  # mid-range
        '~is_three & (16 < shot_dist)',  # long twos
        'is_three & (shot_dist <= 23)',  # corner 3's
        'is_three & (23 < shot_dist <= 26)',  # regular 3's
        'is_three & (26 < shot_dist <= 30)',  # deep 3's
    ]
    counts = [
        df.query(cond).shooter.value_counts()
        for cond in conditions
    ]
    all_df = pd.concat(counts, axis=1)
    all_df.columns = [
        'rest_area', 'paint', 'midrange', 'long_two', 'corner_three',
        'reg_three', 'deep_three'
    ]
    players_df = all_df.ix[players].fillna(0)
    players_df.ix['RP'] = all_df.ix[reps].fillna(0).sum(axis=0)
    return players_df


def _simple_counts(df, players, reps, col, cond=None):

    # compute counts, depending on the value of cond
    if cond:
        if isinstance(cond, basestring):
            counts = df.query(cond).ix[:, col].value_counts()
        else:
            counts = df.ix[cond, col].value_counts()
    else:
        counts = df.ix[:, col].value_counts()

    player_counts = counts.ix[players]
    player_counts.ix['RP'] = counts.ix[reps].sum()
    return player_counts


def get_tot_fga(df, players, reps):
    return _simple_counts(df, players, reps, 'shooter')


def get_ast(df, players, reps):
    return _simple_counts(df, players, reps, 'assister')


def get_fta(df, players, reps):
    return _simple_counts(df, players, reps, 'ft_shooter',
                          cond='~is_tech_fta')


def get_ftm(df, players, reps):
    return _simple_counts(df, players, reps, 'ft_shooter',
                          cond='~is_tech_fta & is_ftm')


def get_fouls_drawn(df, players, reps):
    return _simple_counts(df, players, reps, 'drew_foul',
                          cond='is_pf & ~is_off_foul')


def get_bad_passes(df, players, reps):
    return _simple_counts(df, players, reps, 'to_by',
                          cond='to_type == "bad pass"')


def get_lost_balls(df, players, reps):
    return _simple_counts(df, players, reps, 'to_by',
                          cond='to_type == "lost ball"')


def get_travels(df, players, reps):
    return _simple_counts(df, players, reps, 'to_by',
                          cond='to_type == "traveling"')


def get_off_fouls(df, players, reps):
    return _simple_counts(df, players, reps, 'fouler',
                          cond='is_off_foul')


def get_fg2m(df, players, reps):
    return _simple_counts(df, players, reps, 'shooter',
                          cond='~is_three & is_fgm')


def get_fg2m_assisted(df, players, reps):
    return _simple_counts(df, players, reps, 'shooter',
                          cond='~is_three & is_fgm & is_assist')


def get_fg3m(df, players, reps):
    return _simple_counts(df, players, reps, 'shooter',
                          cond='is_three & is_fgm')


def get_fg3m_assisted(df, players, reps):
    return _simple_counts(df, players, reps, 'shooter',
                          cond='is_three & is_fgm & is_assist')


def get_teammate_fgm(df, players, reps):

    def num_off_teammate(df, p):
        col = '{}_in'.format(p)
        on_off = df[col] == (2*df.hm_off - 1)
        return (on_off & (df.shooter != p)).sum()

    fgm = df.ix[df.is_fgm]
    player_teammate_fgm = pd.Series({
        p: num_off_teammate(fgm, p) if '{}_in'.format(p) in df.columns else 0.
        for p in players
    })
    rep_teammate_fgm = np.sum(
        num_off_teammate(fgm, p) if '{}_in'.format(p) in df.columns else 0.
        for p in reps
    )
    player_teammate_fgm['RP'] = rep_teammate_fgm
    return player_teammate_fgm


def get_blk(df, players, reps):
    return _simple_counts(df, players, reps, 'blocker')


def get_stl(df, players, reps):
    return _simple_counts(df, players, reps, 'stealer')


def get_pf(df, players, reps):
    return _simple_counts(df, players, reps, 'fouler')


def get_opp_fg2a(df, players, reps):

    def num_on_def(df, p):
        return (df['{}_in'.format(p)] == (-2*df.hm_off + 1)).sum()

    fg2a = df.ix[df.is_fga & ~df.is_three]
    player_opp_fg2a = pd.Series({
        p: num_on_def(fg2a, p) if '{}_in'.format(p) in df.columns else 0.
        for p in players
    })
    rep_opp_fg2a = np.sum(
        num_on_def(fg2a, p) if '{}_in'.format(p) in df.columns else 0.
        for p in reps
    )
    player_opp_fg2a['RP'] = rep_opp_fg2a
    return player_opp_fg2a


def get_opp_fga(df, players, reps):

    def num_on_def(df, p):
        return (df['{}_in'.format(p)] == (-2*df.hm_off + 1)).sum()

    fga = df.ix[df.is_fga]
    player_opp_fga = pd.Series({
        p: num_on_def(fga, p) if '{}_in'.format(p) in df.columns else 0.
        for p in players
    })
    rep_opp_fga = np.sum(
        num_on_def(fga, p) if '{}_in'.format(p) in df.columns else 0.
        for p in reps
    )
    player_opp_fga['RP'] = rep_opp_fga
    return player_opp_fga


def get_opp_fta(df, players, reps):

    def num_on_def(df, p):
        return (df['{}_in'.format(p)] == (-2*df.hm_off + 1)).sum()

    fta = df.ix[df.is_fta & ~df.is_tech_fta]
    player_opp_fta = pd.Series({
        p: num_on_def(fta, p) if '{}_in'.format(p) in df.columns else 0.
        for p in players
    })
    rep_opp_fta = np.sum(
        num_on_def(fta, p) if '{}_in'.format(p) in df.columns else 0.
        for p in reps
    )
    player_opp_fta['RP'] = rep_opp_fta
    return player_opp_fta



def get_opp_to(df, players, reps):

    def num_on_def(df, p):
        return (df['{}_in'.format(p)] == (-2*df.hm_off + 1)).sum()

    to = df.ix[df.is_to]
    player_opp_to = pd.Series({
        p: num_on_def(to, p) if '{}_in'.format(p) in df.columns else 0.
        for p in players
    })
    rep_opp_to = np.sum(
        num_on_def(to, p) if '{}_in'.format(p) in df.columns else 0.
        for p in reps
    )
    player_opp_to['RP'] = rep_opp_to
    return player_opp_to


def get_orb(df, players, reps):
    return _simple_counts(df, players, reps, 'rebounder', cond='is_oreb')


def get_drb(df, players, reps):
    return _simple_counts(df, players, reps, 'rebounder', cond='is_dreb')


def get_orb_opps(df, players, reps):

    def num_on_off(df, p):
        return (df['{}_in'.format(p)] == (2*df.hm_off - 1)).sum()

    rebs = df.ix[df.is_reb]
    player_orb_opps = pd.Series({
        p: num_on_off(rebs, p) if '{}_in'.format(p) in df.columns else 0.
        for p in players
    })
    rep_orb_opps = pd.Series({
        p: num_on_off(rebs, p) if '{}_in'.format(p) in df.columns else 0.
        for p in players
    }).sum()
    player_orb_opps['RP'] = rep_orb_opps
    return player_orb_opps


def get_drb_opps(df, players, reps):

    def num_on_def(df, p):
        return (df['{}_in'.format(p)] == (-2*df.hm_off + 1)).sum()

    rebs = df.ix[df.is_reb]
    player_drb_opps = pd.Series({
        p: num_on_def(rebs, p) if '{}_in'.format(p) in df.columns else 0.
        for p in players
    })
    rep_drb_opps = pd.Series({
        p: num_on_def(rebs, p) if '{}_in'.format(p) in df.columns else 0.
        for p in players
    }).sum()
    player_drb_opps['RP'] = rep_drb_opps
    return player_drb_opps


def combined_rapm(combined_df, players, reps, weight=6):

    def on_off(df, p):
        col = '{}_in'.format(p)
        if col in df.columns:
            return (df[col] == (2*df.hm_off - 1)).astype(int)
        else:
            return np.zeros(df.shape[0])

    def on_def(df, p):
        col = '{}_in'.format(p)
        if col in df.columns:
            return (df[col] == (-2*df.hm_off + 1)).astype(int)
        else:
            return np.zeros(df.shape[0])

    logger = get_logger()

    diffs = combined_df.groupby('boxscore_id').is_sub.cumsum().diff().fillna(0)
    combined_df['stint_id'] = np.cumsum(np.where(diffs < 0, 1, diffs))
    stint_df = combined_df.loc[
        ~(combined_df.is_sub | combined_df.is_jump_ball |
          combined_df.is_dreb | combined_df.is_timeout |
          (combined_df.is_pf & ~combined_df.is_off_foul))
    ]
    # logger.info('Filtering possessions')
    # stint_df = stint_df.groupby('stint_id').filter(
    #     lambda x: x.groupby('hm_off').poss_id.nunique().shape[0] == 2
    # )
    stint_grouped = stint_df.groupby(['stint_id'])#, 'hm_off'])

    stint_end = stint_grouped.tail(1)
    stint_end = stint_end.loc[
        :, nba.pbp.sparse_lineup_cols(stint_end) +
        ['season', 'hm_off', 'stint_id']
    ]

    logger.info('computing off_df')
    off_df = pd.DataFrame({
        '{}_orapm'.format(p): on_off(stint_end, p) for p in players
    })
    off_df['RP_orapm'] = pd.DataFrame({
        '{}_orapm'.format(p): on_off(stint_end, p) for p in reps
    }).sum(axis=1).values

    logger.info('computing def_df')
    def_df = pd.DataFrame({
        '{}_drapm'.format(p): on_def(stint_end, p) for p in players
    })
    def_df['RP_drapm'] = pd.DataFrame({
        '{}_drapm'.format(p): on_def(stint_end, p) for p in reps
    }).sum(axis=1).values

    logger.info('computing X')
    X = pd.DataFrame(pd.concat((off_df, -def_df), axis=1))
    X['hm_off'] = stint_end.hm_off.values

    logger.info('computing y')
    pts = stint_grouped.hm_pts.sum().values - stint_grouped.aw_pts.sum().values
    num_poss = stint_grouped.poss_id.nunique().values
    y = 100. * pts / num_poss
    season_min = stint_end.season.min()
    weights = np.where(stint_end.season == season_min, num_poss,
                       weight*num_poss)

    # import ipdb; ipdb.set_trace()
    # for i, (lab, grp) in enumerate(stint_grouped):
    #     pass

    logger.info('fitting model with CV')
    ridge = linear_model.RidgeCV(alphas=np.logspace(-4, 5, num=10), cv=5)
    ridge.fit(X, y, sample_weight=weights)
    coefs = pd.Series(ridge.coef_, index=X.columns)

    logger.info('alpha: {}'.format(ridge.alpha_))
    logger.info('R^2: {}'.format(ridge.score(X, y, sample_weight=weights)))
    logger.info('home_offense coef: {}'.format(coefs['hm_off']))

    coefs.drop('hm_off', axis=0, inplace=True)
    coefs.index = pd.MultiIndex.from_tuples([
        idx.split('_') for idx in coefs.index
    ])
    coefs = coefs.unstack(level=1)

    import ipdb; ipdb.set_trace()

    return X, y, weights, ridge, coefs
