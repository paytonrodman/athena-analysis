[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/fa3ffef347fd481bb44b5a1f6c1042e9)](https://www.codacy.com/gh/paytonrodman/athena-analysis/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=paytonrodman/athena-analysis&amp;utm_campaign=Badge_Grade)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
# athena-analysis

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9b655f5f20e64d20ab5a41938e47c944)](https://app.codacy.com/gh/paytonrodman/athena-analysis?utm_source=github.com&utm_medium=referral&utm_content=paytonrodman/athena-analysis&utm_campaign=Badge_Grade_Settings)

A set of analysis and plotting scripts for use with Athena++ data files. The scripts (and the data they analyse) are based on [Athena++ v19.0](https://github.com/PrincetonUniversity/athena-public-version/releases/tag/v19.0) (archived) but may also work on later versions. Athena++ is a GRMHD and Adaptive Mesh Refinement (AMR) code. The latest release is available at https://github.com/PrincetonUniversity/athena.

## Dependencies

Most core scripts rely on a slightly modified version of Athena++'s `athena_read.py`, contained within the `Dependencies` folder. In addition, you will need:

  - AAT (Athena Analysis Tools), also found in 'Dependencies'
  - [numpy](https://numpy.org/)
  - [matplotlib](https://matplotlib.org/) (for plotting scripts)
  - [scipy](https://www.scipy.org/)

## File structure

Each script assumes a file structure of

```
root_dir
│
│
└───prob_id
    │
    │
    └───data
        │   prob_id.00000.athdf.xdmf
        │   prob_id.00000.athdf
        │   prob_id.00001.athdf.xdmf
        │   prob_id.00001.athdf
        │   ...
```

where `prob_id` is the same as that defined in your `athinput` file and which is passed to Athena++ during configuration, e.g.

`python configure.py --prob=disk_base --coord=spherical_polar --eos=adiabatic -b -mpi`

You will need to redefine `root_dir` and `data_dir` (`root_dir/prob_id/data`) to match your own file structure. Output is saved one level above `data`, i.e. in `root_dir/prob_id/`.

## Running the scripts

Run scripts through

`python <script_name> <args> <*kwargs>`

**Example 1.** If we wanted to calculate the average plasma beta value for our `disk_base`

`python calc_beta.py disk_base`

**Example 2.** If we have run our simulation `disk_base` for longer and now want to analyse the latest files

`python calc_beta.py disk_base -u`

For any given script, running

`python <script_name> -h`

will show you the arguments (required and optional) that can be passed to that script
