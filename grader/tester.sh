#!/bin/bash

set -e

if [ "$1" = "" ]
  then
    printf "Warning: No arguments supplied\n"
    exit
fi

VIDEO='./challenge_workspace/test_video_long.mp4'
# VIDEO='./grader/valid_data.mp4'
#VIDEO='./grader/test_data.mp4'

# Set frames constant for test video
FRAMES=30.0

# Space for new command
SPACE=' '

# Make run command
RUN_CMD=$1$SPACE$VIDEO

# Set clock
start_time=$(date +'%s')

# Run suppiled function with image argument
# Outputs pixel labeled array
# Stored for comparison
$RUN_CMD > ./challenge_workspace/tester_data

# Output runtime
end_time=$(date +'%s')

printf "\nYour program has been ran, now grading...\n"

FPS=$(echo "scale=3; $FRAMES/($end_time - $start_time)" | bc)

printf "\nYour program runs at $FPS FPS\n\n"

CODE_CMD='python ./grader/score'
TMP_LOCATION='./challenge_workspace/tester_data'
TRUTH_LOCATION='./grader/test_truth.json'
#TRUTH_LOCATION='./grader/valid_truth.json'

PYTHON_CMD=$CODE_CMD$SPACE$TMP_LOCATION$SPACE$TRUTH_LOCATION


# # Run python program to calculate accuracy and store info for database
ACC=$($PYTHON_CMD 2>&1)

echo $ACC
echo ' '
