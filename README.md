evaluating_nba_teams
==============================

Extending the concepts of adjusted plus/minus and complementary play styles to evaluate full NBA lineups and rosters.

Project Organization
------------

Below is a brief summary of the main files and directories in the repo; each top-level entry is a file or directory in the root of the repo.

- data: contains all data that is written to disk, typically in CSV form
    - pbp: contains cleaned, scraped play-by-play data
    - testing: contains results from testing integrity of data
    - profiles: contains player profile data computed from pbp data
    - models: contains output from model fitting and performance evaluation
    - results: contains output from applications of the model
- src: contains Python source code
    - data: code for scraping and cleaning PBP data
    - testing: code for testing integrity fo PBP data
    - features: code for generating player profiles
    - models: code for training, selecting, comparing, and applying models
    - visualizations: code for generating visualizations and results
- slurm: contains slurm scripts for running code on Odyssey research compute cluster
- reports: home for all report-related source and output
    - fall_submission: report for fall CS91r
    - thesis: the actual thesis LaTeX and output
- models: contains pickled trained models that can be read from disk


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
