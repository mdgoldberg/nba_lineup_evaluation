import os

import dotenv
import invoke

dotenv_path = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_path)

@invoke.task
def clean(ctx, cluster_data=False, pbp_data=False):
    patterns = []
    if cluster_data:
        patterns.append(
            os.path.join(os.environ['PROJ_DIR'], 'data', 'raw', 'cluster')
        )
    if pbp_data:
        patterns.append(
            os.path.join(os.environ['PROJ_DIR'], 'data', 'raw', 'pbp')
        )
    for pattern in patterns:
        ctx.run('rm -rf {}'.format(pattern))
