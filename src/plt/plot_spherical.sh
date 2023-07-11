#!/bin/bash
#ROOTDIR=/Users/paytonrodman/athena-sim
ROOTDIR=/home/per29/rds/rds-accretion-zyNhkonJSR8
PROBLEM=$1

if [ $# -lt 3 ]
  then
    echo "Must specify (1) problem ID, (2) start time, (3) end time"
    exit 1
fi

cd ${ROOTDIR}/${PROBLEM}/data/
for FILE in ./${PROBLEM}.cons.*.athdf; do
  NUMBER=$(echo "$FILE" | sed 's/[^0-9]*//g')
  NOZERO=$(echo "$NUMBER" | sed 's/^0*//')

  if ((${#NOZERO}==0)); then
    NOZERO=0
  fi

  if [ "$NOZERO" -gt "$2" ] && [ "$NOZERO" -lt "$3" ]; then # if in range
    REMAINDER=$(( $NOZERO % 10 ))
    if (($REMAINDER==0)); then # do every 10th file
      #if [[ ! -f ${ROOTDIR}/${PROBLEM}/img/${FILE//[^0-9]/}.png ]]; then # if image doesn't already exist
      python $ROOTDIR/athena-analysis/dependencies/plot_spherical.py \
      $FILE \
      dens \
      ../img/${FILE//[^0-9]/}.png \
      --colormap viridis \
      --vmin 0.001 \
      --vmax 0.05 \
      --logc \
      --time \
      --dpi 1200 \
      #--stream Bcc \
      #--output_file show \
      #--midplane
      #fi
    fi
  fi
done
