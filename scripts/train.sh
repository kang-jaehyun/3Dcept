#!/bin/sh

cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

TRAIN_CODE=train.py

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


echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
if ${RESUME}
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_last.pth
else
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  cp -r scripts tools pointcept "$CODE_DIR"
fi

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=./$CODE_DIR
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="

if [ "${WEIGHT}" = "None" ]
then
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --num-machines "$NUMMACHINE" \
    --machine-rank "$MRANK" \
    --dist-url "$DIST" \
    --options save_path="$EXP_DIR" num_workers="$NUMWORKERS"
else
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --num-machines "$NUMMACHINE" \
    --machine-rank "$MRANK" \
    --dist-url "$DIST" \
    --options save_path="$EXP_DIR" num_workers="$NUMWORKERS" resume="$RESUME" weight="$WEIGHT"
fi