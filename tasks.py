import os

import dotenv
import invoke

dotenv_path = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_path)

thesis_dir = os.path.join(os.environ['PROJ_DIR'], 'reports', 'thesis')

@invoke.task
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
