#!/bin/bash
# All Experiment Settings - constant through the experiment run - passed to gen.sh and fine_tune.sh as needed
GPU=0 # which GPU to use
MODEL="125M" # MODEL is the size of the model: 125M, 13B, 27B
EXPERIMENT=$MODEL"_PAPER" # Name of Experiment directory under data/* and models/base-model/* to store results
TEST_LOCAL=1 # 0 means run gen/fine_tune on cluster remotely, 1 means run gen/fine_tune locally
TARGET_NUM_FILES=1 # How many files to generate in each iteration before starting fine_tuning.  Count of unique examples would have been better.
ITER_START=0 # inclusive index to start processing at - creates iter_# under data&models at each iteration.  Can continue prev runs by start at prev ITER_MAX
ITER_MAX=5 # exclusive index to stop processing iterations at
EPOCHS_START=1 # inclusive index of epochs to start processing at - could continue prev run by starting at prev EPOCHS_MAX+1 - 0th epoch is the default model so epoch starts at 1
EPOCHS_MAX=4 # inclusive index of epochs to stop processing at
EPOCHS_PER_STEP=1 # How many EPOCHS through the data to do in each step
TRAIN_INCREMENTAL=0 # Only train on data from the latest iteration, and start finetuning on the last finetuned model - otherwise start from scratch and use all the data generated
TRAIN_BOOST=0 # Initial generation of data from default model is slow - 1 means looks in 125M_RL_ALL to use previous generated initial data to bootstrap.
PASS_AT_K=100 # PASS_AT_K says do K trials to solve to compute Pass@K
LINE_LOG_K=11 # LINE_LOG_K is how many lines of results from solve have results for saving

echo babysit args: $# $0 $1 $2 $3 $4

if (( $# \!= 1 ))
then
    echo babysit.sh only takes 1 argument, unless called by another script to initialize configuration variables
    return
fi

if (( $# \>= 1 ))
then
    GPU=$1
fi

echo babysit GPU $GPU

for (( iteration=$ITER_START; iteration<$ITER_MAX; iteration++ ))
do 
    FULLNAME="${EXPERIMENT}---${iteration}" 
    echo FULLNAME $FULLNAME
    export FULLNAME  # Needed to pass variable off to yaml job
    DATAPATH=data/${EXPERIMENT}/iter_$iteration
    echo DATAPATH $DATAPATH

    if (( $TEST_LOCAL \> 0 )) 
    then
        count=`ls -lt ../${DATAPATH} | grep json | wc -l`
    else
        count=`amlt sto list ${DATAPATH} | grep json | wc -l`
    fi
    echo count $count

    # Instead of file count we might want to check if the amount of data from preprocess is sufficient
    # If not we call to generate more

    if (( $count \> 0 ))
    then
        echo "$FULLNAME has already been started"
        echo "You are resuming at iteration $iteration"
        echo "You already have $count files of data this iteration"
    else
        echo "$FULLNAME is starting generation for iteration $iteration"
    fi

    if (( $count \< $TARGET_NUM_FILES ))
    then
        if (( $TEST_LOCAL \> 0 ))
        then
            # ./gen.sh $GPU 2560 100 $FULLNAME -1
            # 2.7B 384 100 runs ~10 hours
            # 2.7B 160 100 runs ~4.5 hours
            ./gen.sh $GPU 256000 100 $FULLNAME -1
        else
            amlt run hyper_gen_octows.yaml $FULLNAME -d "$FULLNAME"
            exit
        fi
    fi

    # Running local you are done, but launching on the cloud, you have to wait
    for (( poll=0; poll<500; poll++ ))
    do
        if (( $TEST_LOCAL \> 0 ))
        then
            count=`ls -lt ../${DATAPATH} | grep json | wc -l`
        else
            count=`amlt sto list ${DATAPATH} | grep json | wc -l`
        fi

        echo "gen wait - Iteration: $iteration, Poll: $poll, Count: $count"

        if (( $count \>= $TARGET_NUM_FILES ))
        then
            echo "Finished generation iteration $iteration after $poll polls"
            break
        fi
        sleep 3m
    done

    # Start a finetune job
    if (( $TEST_LOCAL \> 0 ))
    then
        ./fine_tune.sh $GPU $FULLNAME
    else
        # Pass enviroment variable FULLNAME to amlt.yaml
        amlt run amlt_octo.yaml $FULLNAME -d "$FULLNAME"
        exit
    fi

    # On cluster we need to wait for finetune job to finish, run locally it's done
    # Check the log files for starting the running of solve have been created for the last epoch of training

    MODELPATH=models/gpt-neo-$MODEL/${EXPERIMENT}/iter_$iteration
    SOLVE_PATH=$MODELPATH/"epoch_"$EPOCHS_MAX/"solve_"$PASS_AT_K
    echo babysit.sh SOLVE_PATH $SOLVE_PATH

    for (( poll=0; poll<500; poll++ ))
    do
        if (( $TEST_LOCAL \> 0 ))
        then
            count=`ls -lt ../$SOLVE_PATH | grep json | wc -l`
        else
            count=`amlt sto list $SOLVE_PATH | grep json | wc -l`
        fi

        echo "fine_tune wait - Iteration: $iteration, Poll: $poll, Count: $count"

        if (( $count \>= 1 ))
        then
            echo "Finished fine_tune iteration $iteration after $poll polls"
            break
        fi
        sleep 3m
    done

done

# Pull all the results into 1 log file to look at more easily

if [[ -z "${AMLT_DATA_DIR}" ]]; 
then
  # running locally on torch2020 so we don't have AMLT enviroment variables defined, so need to set them up
  AMLT_DATA_DIR=../data
else
  # On remote we don't have access to the log files - maybe could do amlt sto download to do this summary below ?
  exit
fi

BASE_MODEL_PATH=$AMLT_DATA_DIR/../models/gpt-neo-$MODEL
LOG_FILE=$BASE_MODEL_PATH/$EXPERIMENT/"solve_"$PASS_AT_K".txt"
echo solve LOG_FILE for babysit.sh is $LOG_FILE
rm $LOG_FILE

for (( iteration=$ITER_START; iteration<$ITER_MAX; iteration++ ))
do
    for (( epochs=$EPOCHS_START; epochs<=$EPOCHS_MAX; epochs++ ))
    do
        EPOCH_NAME="epoch_"$epochs
        STEP_PATH=$BASE_MODEL_PATH/$EXPERIMENT/iter_$iteration/$EPOCH_NAME
        MODEL_PATH=$STEP_PATH/finetuned
        echo iteration $iteration epoch $epochs >> $LOG_FILE
        head -$LINE_LOG_K $STEP_PATH/"solve_"$PASS_AT_K/results.json >> $LOG_FILE
    done
done

cat $LOG_FILE
