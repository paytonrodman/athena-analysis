[![Codacy Badge](https://app.codacy.com/project/badge/Grade/fa3ffef347fd481bb44b5a1f6c1042e9)](https://www.codacy.com/gh/paytonrodman/athena-analysis/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=paytonrodman/athena-analysis&amp;utm_campaign=Badge_Grade)
# athena-analysis
A set of analysis and plotting scripts for use with Athena++ data files. The scripts (and the data they analyse) are based on [Athena++ v19.0](https://github.com/PrincetonUniversity/athena-public-version/releases/tag/v19.0) (archived) but may also work on later versions. Athena++ is a GRMHD and Adaptive Mesh Refinement (AMR) code. The latest release is available at https://github.com/PrincetonUniversity/athena.

## Dependencies

Most core scripts rely on a slightly modified version of Athena++'s `athena_read.py`, contained within the `Dependencies` folder. In addition, you will need:

  - [numpy](https://numpy.org/)
  - [matplotlib](https://matplotlib.org/) (for plotting scripts)
  - [scipy](https://www.scipy.org/)


