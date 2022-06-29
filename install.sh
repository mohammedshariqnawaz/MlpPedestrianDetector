#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then

  # put your install commands here:
  apt update
  apt clean
  conda install cython
  python -m pip install -v -e .
  python -m pip install timm
  python -m pip install tensorboardX
  python -m pip install mmcv==0.2.10
  python -m pip install wandb
  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi
