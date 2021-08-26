# athena-analysis

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9b655f5f20e64d20ab5a41938e47c944)](https://app.codacy.com/gh/paytonrodman/athena-analysis?utm_source=github.com&utm_medium=referral&utm_content=paytonrodman/athena-analysis&utm_campaign=Badge_Grade_Settings)

A set of analysis and plotting scripts for use with Athena++ data files. The scripts (and the data they analyse) are based on [Athena++ v19.0](https://github.com/PrincetonUniversity/athena-public-version/releases/tag/v19.0) (archived) but may also work on later versions. Athena++ is a GRMHD and Adaptive Mesh Refinement (AMR) code. The latest release is available at https://github.com/PrincetonUniversity/athena.

## Dependencies

Most core scripts rely on a slightly modified version of Athena++'s `athena_read.py`, contained within the `Dependencies` folder. In addition, you will need:

  - [numpy](https://numpy.org/)
  - [matplotlib](https://matplotlib.org/) (for plotting scripts)
  - [scipy](https://www.scipy.org/)


