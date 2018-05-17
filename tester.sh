#!/bin/bash

set -e

if [ "$1" = "" ]
  then
    printf "Warning: No arguments supplied\n"
    exit
fi

VIDEO='./workspace/test_video_long.mp4'

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
$RUN_CMD > ./workspace/tester_data

# Output runtime
end_time=$(date +'%s')

printf "\nYour program has been ran, now grading...\n"

FPS=$(echo "scale=3; $FRAMES/($end_time - $start_time)" | bc)

# CODE_CMD='python /home/workspace/Example/score'
# TMP_LOCATION='/home/workspace/Example/tester_data'

# PYTHON_CMD=$CODE_CMD$SPACE$TMP_LOCATION$SPACE$FPS$SPACE$input_user

# # Run python program to calculate accuracy and store info for database
# ACC=$($PYTHON_CMD 2>&1)

printf "\nYour program runs at $FPS FPS\n\n"

# echo $ACC
echo ' '
