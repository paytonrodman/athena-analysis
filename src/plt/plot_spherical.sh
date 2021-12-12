#!/bin/bash
#ROOTDIR=/Users/paytonrodman/athena-sim
ROOTDIR=/home/per29/rds/rds-accretion-zyNhkonJSR8
PROBLEM=high_res

cd ${ROOTDIR}/${PROBLEM}/data/
for FILE in ./${PROBLEM}.cons.*.athdf; do
  NUMBER=$(echo $FILE | sed 's/[^0-9]*//g')
  NOZERO=$(echo $NUMBER | sed 's/^0*//')

  if ((${#NOZERO}==0)); then
    NOZERO=0
  fi

  REMAINDER=$(( $NOZERO % 10 ))
  if (($REMAINDER==0)); then
    python $ROOTDIR/athena-analysis/dependencies/plot_spherical.py \
    $FILE \
    dens \
    ../img/${FILE//[^0-9]/}.png \
    --midplane \
    --colormap viridis \
    --vmin 0.01 \
    --vmax 0.1 \
    --logc \
    --stream Bcc \
    --dpi 1200
   fi
done
