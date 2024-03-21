#!/bin/sh

cd $(dirname $(dirname "$0")) || exit
PYTHON=python

TEST_CODE=test.py

DATASET=scannet
CONFIG="None"
EXP_NAME=debug
WEIGHT="None"
RESUME=false
GPU=None
NUMMACHINE=1
MRANK=0
DIST="auto"
NUMWORKERS=16

while getopts "p:d:c:n:w:r:g:m:k:t:o" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=$OPTARG
      ;;
    g)
      GPU=$OPTARG
      ;;
    m)
      NUMMACHINE=$OPTARG
      ;;
    k)
      MRANK=$OPTARG
      ;;
    t)
      DIST=&OPTARG
      ;;
    o)
      NUMWORKERS=&OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $GPU"
echo "Machine Num: $NUMMACHINE"
echo "Machine Rank: $MRANK"
echo "Dist url: $DIST"

EXP_DIR=exp/${DATASET}/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py

if [ "${CONFIG}" = "None" ]
then
    CONFIG_DIR=${EXP_DIR}/config.py
else
    CONFIG_DIR=configs/${DATASET}/${CONFIG}.py
fi

echo "Loading config in:" $CONFIG_DIR
#export PYTHONPATH=./$CODE_DIR
export PYTHONPATH=./
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="

#$PYTHON -u "$CODE_DIR"/tools/$TEST_CODE \
$PYTHON -u tools/$TEST_CODE \
  --config-file "$CONFIG_DIR" \
  --num-gpus "$GPU" \
  --options save_path="$EXP_DIR" weight="${MODEL_DIR}"/"${WEIGHT}".pth
