import itertools

import click
import pandas as pd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_filepath', type=click.Path(dir_okay=False))
def process_season_data(input_filepath, output_filepath):
    """TODO

    :input_filepath: TODO
    :output_filepath: TODO
    """
    df = pd.read_csv(input_filepath)

    COLS = [
        ['player_id', 'season', 'height', 'weight', 'age'],
        ['{}_{}'.format(*x) for x in
         itertools.product(['pct_fga', 'fg_pct'],
                           ['00_03', '03_10', '10_16', '16_xx'])],
        ['fg3a_pct_fga', 'fg3_pct', 'avg_dist'],
        ['usg_pct', 'fta_per_fga_pct', 'fg2_pct_ast', 'fg3_pct_ast'],
        ['orb_pct', 'drb_pct'],
        ['blk_pct', 'stl_pct', 'pf_per_poss', 'def_rtg'],
        ['ast_pct', 'tov_pct', 'tov_bad_pass', 'tov_lost_ball']
    ]
    COLS = [item for sublist in COLS for item in sublist]

    sub_df = df[COLS].fillna(0)
    sub_df['tov_bad_pass'] /= df['mp']
    sub_df['tov_lost_ball'] /= df['mp']
    sub_df.to_csv(output_filepath, index=None)


if __name__ == '__main__':
    process_season_data()
