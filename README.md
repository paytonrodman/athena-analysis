[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/fa3ffef347fd481bb44b5a1f6c1042e9)](https://www.codacy.com/gh/paytonrodman/athena-analysis/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=paytonrodman/athena-analysis&amp;utm_campaign=Badge_Grade)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# athena-analysis

A set of analysis and plotting scripts for use with Athena++ data files. The scripts (and the data they analyse) are based on [Athena++ v21.0](https://github.com/PrincetonUniversity/athena-public-version/releases/tag/v21.0) (archived) but may also work on other versions. Athena++ is a GRMHD and Adaptive Mesh Refinement (AMR) code. The latest release is available at https://github.com/PrincetonUniversity/athena.

## Dependencies

Most core scripts rely on a slightly modified version of Athena++'s `athena_read.py`, contained within the `dependencies/` folder. In addition, you will need:

  - AAT (Athena Analysis Tools), also found in `dependencies/`
  - [numpy](https://numpy.org/)
  - [matplotlib](https://matplotlib.org/) (for plotting scripts)
  - [pandas](https://pandas.pydata.org/) (for plotting scripts)
  - [scipy](https://www.scipy.org/)
  - [mpi4py](https://pypi.org/project/mpi4py/) (for parallel scripts)

## Running the scripts

**Note: Location of the `dependencies/` folder is hard-coded for all files. You must change these to match your system!**

For parallel `calc/` scripts, run using `mpirun`

`mpirun -n <procs> <script_name> <args> <*kwargs>`

For plotting scripts, run

`python <script_name> <args> <*kwargs>`

For any given script, running

`python <script_name> -h`

will show you the arguments (required and optional) that can be passed to that script.
