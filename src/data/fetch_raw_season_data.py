import functools

import click
import dask
from dask import bag as db
import pandas as pd

from sportsref import nba


@click.command()
@click.argument('output_filepath', type=click.Path(dir_okay=False))
@click.argument('start_year', type=int)
@click.argument('end_year', type=int)
def fetch_raw_season_data(output_filepath, start_year, end_year):
    """Saves a DataFrame where each row represents a player-season, and each
    column is a different statistic that might be used for clustering.

    :output_filepath: The path of the output CSV file.
    :start_year: The first season to include.
    :end_year: The last season to include.
    """
    years = db.from_sequence(xrange(start_year, end_year + 1), npartitions=16)
    all_player_ids = years.map(player_ids_from_year).concat().distinct()
    df = (all_player_ids
          .map(functools.partial(process_player_id,
                                 start_year=start_year, end_year=end_year))
          .fold(lambda x, y: pd.concat((x, y))))
    df = df.compute()
    df.to_csv(output_filepath, index=None)


def player_ids_from_year(year):
    """Returns a list of player IDs that qualify from the year given.

    :year: A year representing a season.
    :returns: A numpy array of player IDs.
    """
    season = nba.Season(year)
    per_game = season.player_stats_per_game()
    filtered = per_game.ix[per_game['has_class_full_table'] &
                           (per_game['mp_per_g'] >= 10) &
                           (per_game['g'] >= 10)]
    return filtered.player_id.values


def process_player_id(player_id, start_year=None, end_year=None):
    """Returns a table of all of the player's yearly stats."""
    player = nba.Player(player_id)
    print player
    per100 = player.stats_per100()
    years = per100.ix[per100['has_class_full_table'] &
                      (per100['mp'] / per100['g'] >= 10) &
                      (per100['g'] >= 10), 'season'].values
    years = [y for y in years if start_year <= y <= end_year]
    clean = functools.partial(clean_dataframe, years=years)
    try:
        per100 = clean(per100)
        adv = clean(player.stats_advanced())
        shooting = clean(player.stats_shooting())
        pbp = clean(player.stats_pbp())
        dfs = [per100, adv, shooting, pbp]
        df = reduce(merge_dataframes, dfs)
        df['player_id'] = player_id
        df['height'] = player.height()
        df['weight'] = player.weight()
        return df
    except Exception:
        print 'error!', player_id
        return pd.DataFrame()


def clean_dataframe(df, years):
    """Cleans DataFrames to only include relevant, full seasons."""
    df = df.ix[df['has_class_full_table'] & df.season.isin(years)]
    to_drop = [c for c in df.columns if c.startswith('has_class')]
    df = df.drop(to_drop, axis=1)
    return df


def merge_dataframes(df1, df2):
    overlap = list(set(df1.columns) & set(df2.columns))
    return pd.merge(df1, df2, on=overlap)


if __name__ == '__main__':
    dask.set_options(get=dask.multiprocessing.get)

    fetch_raw_season_data()
