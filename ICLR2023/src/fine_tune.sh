#!/bin/bash
echo fine_tune.sh args: $# $0 $1 $2 $3 $4
# Grab the configuration variables
. babysit.sh

# On AMLT machines we don't specify which GPU to use
# GPU="-1"
if [[ -z "${AMLT_DATA_DIR}" ]]; then
  # running locally on torch2020 so we don't have AMLT enviroment variables defined, so need to set them up
  AMLT_DATA_DIR=../data
  # On torch2020 we do specify which GPU to use
  # GPU="0"
fi

# assert that there are at least 2 argument
if (( $# \< 2 ))
then
    echo "Usage: $0 <GPU> <FULL EXPERIMENT NAME>"
    exit
fi

GPU=$1
FULLNAME=$2

# split by ; fullname string into experiment name and iteration
# e.g. "125M_RL---0" --> "125M_RL;0"
SPLIT=(${FULLNAME//---/ })
EXPERIMENT=${SPLIT[0]}
ITERATION=${SPLIT[1]}
OUTPATH=$AMLT_DATA_DIR/$EXPERIMENT/iter_$ITERATION

echo GPU $GPU
echo EXPERIMENT $EXPERIMENT
echo ITERAION $ITERATION
echo OUTPATH $OUTPATH

# GPU_SOLVE is the GPU we want solve to use.  Solve currently only uses 1 GPU - it would be great to make it use more when they are available.
# if GPU is negative - that tells fine_tune how many GPU to use on cluster - and we need to set GPU for solve to 0 on cluster
# if GPU is positive - we are running locally on torch2020 - and we need to leave the GPU set properly
GPU_SOLVE=$GPU
if (( $GPU \< 0 ))
then
   GPU_SOLVE=0
fi
echo GPU_SOLVE $GPU_SOLVE

python preprocess.py $OUTPATH

TRN_FILE=$OUTPATH/gen_ps_filtered.txt
echo TRN_FILE $TRN_FILE
TST_FILE=$AMLT_DATA_DIR/test_228.json

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

# Training incremental means use the previous iterations trained model, and just the additional iteration's new data to fine_tune on.
# Otherwise use the base model - and retrain from scratch on all the data from all previous iterations.
# They are sort of equivalent - except from scratch picks up any extra data that was generated - and mixes all the iterations data together - but slower.
if (( $TRAIN_INCREMENTAL \> 0 ))
then
   PREV_ITERATION=$((ITERATION-1))
   echo $PREV_ITERATION
   TEST=$AMLT_DATA_DIR/../models/gpt-neo-$MODEL/$EXPERIMENT/iter_$PREV_ITERATION/epoch_$EPOCHS_MAX/finetuned

   if [ -a $TEST ]  # exists
   then
      MODEL_PATH=$TEST
      echo fine_tune.sh using previous iteration model
   fi
fi

echo "fine_tune.sh starting from NEO model at: ${MODEL_PATH}"

# Pull all the results into 1 log file to look at more easily
LOG_FILE=$BASE_MODEL_PATH/$EXPERIMENT/iter_$ITERATION/"solve.txt"
echo solve LOG_FILE for fine_tune.sh is $LOG_FILE
rm $LOG_FILE

for (( epochs=$EPOCHS_START; epochs<=$EPOCHS_MAX; epochs++ ))
do
   EPOCH_NAME="epoch_"$epochs
   EPOCHS_STEP=$(($EPOCHS_PER_STEP * $epochs))
   python fine_tune.py -train_txt=$TRN_FILE -gpu=$GPU -output_dir=$BASE_MODEL_PATH/$EXPERIMENT/iter_$ITERATION -subdir=$EPOCH_NAME -model_path=$MODEL_PATH -epochs=$EPOCHS_STEP
   # measure the finetuned model's accuracy
   STEP_PATH=$BASE_MODEL_PATH/$EXPERIMENT/iter_$ITERATION/$EPOCH_NAME
   MODEL_PATH=$STEP_PATH/finetuned
   python solve.py -prefix=$AMLT_DATA_DIR/train_prefix.txt -attempts=$PASS_AT_K -model_path=$MODEL_PATH -gpu=$GPU_SOLVE -fixed_temp=0.8 -out=$STEP_PATH/"solve_"$PASS_AT_K"/" -puzzles=$TST_FILE
   head -$LINE_LOG_K $STEP_PATH/"solve_"$PASS_AT_K/results.json >> $LOG_FILE
done

cat $LOG_FILE