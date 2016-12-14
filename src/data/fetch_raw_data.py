import click

from sportsref import nba


@click.command()
@click.argument('output_filepath', type=click.Path(dir_okay=False))
@click.argument('start_year', type=int)
@click.argument('end_year', type=int)
def fetch_raw_data(output_filepath, start_year, end_year):
    """Fetches raw data from bkref and saves it in the file pointed to by
    output_filepath."""
    for year in range(start_year, end_year + 1):
        season = nba.Season(year)
        players = season.player_stats_per_game()
        filtered = players.ix[players['has_class_full_table'] &
                              players['mp_per_g'] >= 10.]
        print(filtered.columns)
        player_ids = filtered['player_id'].values
        print(player_ids)


if __name__ == '__main__':
    fetch_raw_data()
