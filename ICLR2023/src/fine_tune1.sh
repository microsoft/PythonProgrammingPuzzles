#!/bin/bash
# This is for finetuning a model on 1 dataset only
echo fine_tune1.sh args: $# $0 $1 $2 $3 $4

# All Experiment Settings - constant through the experiment run
GPU=0 # which GPU to use
MODEL="125M" # MODEL is the size of the model: 125M, 13B, 27B
EXPERIMENT=$MODEL"_PAPER1" # Name of Experiment directory under data/* and models/base-model/* to store results
ITERATION=0 # Random seed for finetuning
EPOCHS_START=1 # inclusive index of epochs to start processing at - could continue prev run by starting at prev EPOCHS_MAX+1 - 0th epoch is the default model so epoch starts at 1
EPOCHS_MAX=10 # inclusive index of epochs to stop processing at
EPOCHS_PER_STEP=1 # How many EPOCHS through the data to do in each step
PASS_AT_K=100 # PASS_AT_K says do K trials to solve to compute Pass@K
LINE_LOG_K=11 # LINE_LOG_K is how many lines of results from solve have results for saving

# On AMLT machines we don't specify which GPU to use
if [[ -z "${AMLT_DATA_DIR}" ]]; then
  # running locally on torch2020 so we don't have AMLT enviroment variables defined, so need to set them up
  AMLT_DATA_DIR=../data
fi

if (( $# \>= 1 ))
then
    GPU=$1
fi

echo GPU $GPU
echo EXPERIMENT $EXPERIMENT
echo ITERAION $ITERATION

TRN_FILE=$AMLT_DATA_DIR/generated_sol_950k.txt
echo TRN_FILE $TRN_FILE
TST_FILE=$AMLT_DATA_DIR/test_228.json

# GPU_SOLVE is the GPU we want solve to use.  Solve currently only uses 1 GPU - it would be great to make it use more when they are available.
# if GPU is negative - that tells fine_tune how many GPU to use on cluster - and we need to set GPU for solve to 0 on cluster
# if GPU is positive - we are running locally on torch2020 - and we need to leave the GPU set properly
GPU_SOLVE=$GPU
if (( $GPU \< 0 ))
then
   GPU_SOLVE=0
fi
echo GPU_SOLVE $GPU_SOLVE

# measure the base model's accuracy - don't really need to do this very often - it doesn't change

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

# Training incremental means use the previous epochs model to start
# Otherwise use the base model to retrain from scratch
if (( $EPOCHS_START \> 1 ))
then
   PREV_EPOCH=$((EPOCHS_START-1))
   echo $PREV_EPOCH
   TEST=$AMLT_DATA_DIR/../models/gpt-neo-$MODEL/$EXPERIMENT/iter_$ITERATION/epoch_$PREV_EPOCH/finetuned

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
   python fine_tune.py -train_txt=$TRN_FILE -gpu=$GPU -output_dir=$BASE_MODEL_PATH/$EXPERIMENT/iter_$ITERATION -subdir=$EPOCH_NAME -model_path=$MODEL_PATH -epochs=$EPOCHS_STEP -seed=$ITERATION
   # measure the finetuned model's accuracy
   STEP_PATH=$BASE_MODEL_PATH/$EXPERIMENT/iter_$ITERATION/$EPOCH_NAME
   MODEL_PATH=$STEP_PATH/finetuned
   python solve.py -prefix=$AMLT_DATA_DIR/train_prefix.txt -attempts=$PASS_AT_K -model_path=$MODEL_PATH -gpu=$GPU_SOLVE -fixed_temp=0.8 -out=$STEP_PATH/"solve_"$PASS_AT_K"/" -puzzles=$TST_FILE -seed=$ITERATION -batch_size=256
   head -$LINE_LOG_K $STEP_PATH/"solve_"$PASS_AT_K/results.json >> $LOG_FILE
done

cat $LOG_FILE