import os

import dotenv
import invoke

dotenv_path = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_path)

thesis_dir = os.path.join(os.environ['PROJ_DIR'], 'reports', 'thesis')

@invoke.task
def compile_pweave(ctx):
    for (path, _, filenames) in os.walk(thesis_dir, followlinks=True):
        for filename in filenames:
            src_path = os.path.join(path, filename)
            dst_path = os.path.join(path, filename[:-1])
            ctx.run('pweave -f texminted -m -g png -o {} {}'
                    .format(dst_path, src_path))

@invoke.task(compile_pweave)
def compile(ctx):
    start_dir = os.getcwd()
    os.chdir(thesis_dir)
    ctx.run('xelatex thesis')
    ctx.run('bibtex thesis')
    ctx.run('xelatex thesis')
    ctx.run('xelatex thesis')
    ctx.run('mv thesis.log .logged')
    ctx.run('latexmk -c')
    os.chdir(start_dir)

@invoke.task
def view(ctx):
    ctx.run('open {}/thesis.pdf'.format(thesis_dir))
