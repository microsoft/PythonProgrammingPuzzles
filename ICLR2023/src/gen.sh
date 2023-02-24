#!/bin/bash
# Grab the configuration variables
. babysit.sh

if [[ -z "${AMLT_DATA_DIR}" ]]; then
  # running locally on torch2020 we don't have AMLT enviroment variables defined, set them up
  AMLT_DATA_DIR="../data"
fi

echo RANK is:
echo $RANK

if [[ -z "${RANK}" ]]; then
  # running locally on torch2020 we don't have AMLT enviroment variables defined, set them up
  RANK=0
fi

GPU=$RANK
PUZZLE_CNT=32
SOLUTION_CNT=32
FULLNAME="125M_RL_TEST---0"

echo $# $0 $1 $2 $3 $4
if (( $# \>= 1 ))
then
    GPU=$1
fi

echo $RANK
echo $GPU

if (( $# \>= 2 ))
then
    PUZZLE_CNT=$2
fi

if (( $# \>= 3 ))
then
    SOLUTION_CNT=$3
fi

if (( $# \>= 4 ))
then
    FULLNAME=$4

fi

RANDOM_SEED=-1

if (( $# \>= 5 ))
then
    RANDOM_SEED=$5
    echo "Random seed is $RANDOM_SEED"
fi

SPLIT=(${FULLNAME//---/ })
EXPERIMENT=${SPLIT[0]}
ITERATION=${SPLIT[1]}
OUTPATH=$AMLT_DATA_DIR/$EXPERIMENT/iter_$ITERATION

echo GPU $GPU
echo EXPERIMENT $EXPERIMENT
echo ITERAION $ITERATION
echo OUTPATH $OUTPATH

BASE_MODEL_PATH=$AMLT_DATA_DIR/../models/gpt-neo-$MODEL
# 125M is copied locally to start
MODEL_PATH=$BASE_MODEL_PATH
MODEL_PATH=EleutherAI/gpt-neo-125M # !!! Just for paper release
# 13B is off in the cloud to start
if [[ "$MODEL" == "13B" ]]; then
    MODEL_PATH=EleutherAI/gpt-neo-1.3B
fi
# 27B is off in the cloud tto start
if [[ "$MODEL" == "27B" ]]; then
    MODEL_PATH=EleutherAI/gpt-neo-2.7B
fi

echo MODEL MODEL_PATH $MODEL $MODEL_PATH

PREV_ITERATION=$((ITERATION-1))
echo $PREV_ITERATION
TEST=$AMLT_DATA_DIR/../models/gpt-neo-$MODEL/$EXPERIMENT/iter_$PREV_ITERATION/epoch_$EPOCHS_MAX/finetuned

if [ -a $TEST ]  # exists
then
    MODEL_PATH=$TEST
    echo fine_tune.sh using previous iteration model
fi

echo "gen.sh using NEO model at: ${MODEL_PATH}"

python gen.py -out="$OUTPATH" -n=$PUZZLE_CNT -seed=$RANDOM_SEED -gpu=$GPU -train=$AMLT_DATA_DIR/155_train.json -prefix=$AMLT_DATA_DIR/train_prefix.txt -model_path=$MODEL_PATH -attempts=$SOLUTION_CNT -only_good=True
