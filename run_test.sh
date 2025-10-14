#!/bin/bash
DATASETS=(
    "ml-1m"
    "ml-10m"
    "steam"
)
TEST_SCRIPTS=(
    "GRPO/test.sh"
    "GRPO/test_rl.sh"
    "GRPO/test_recaller.sh"
)

# Create a new tmux session
SESSION_NAME="test"
# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null || true
tmux new-session -d -s $SESSION_NAME

# 初始化计数器
window_id=0

# Run commands in tmux windows
for DATASET in "${DATASETS[@]}"; do
    for TEST_SCRIPT in "${TEST_SCRIPTS[@]}"; do
        DEVICE_ID=$((window_id % 8))
        
        if [ $window_id -eq 0 ]; then
            # 第一个命令在初始窗口执行
            tmux send-keys -t $SESSION_NAME:0 "CUDA_VISIBLE_DEVICES=$DEVICE_ID ${TEST_SCRIPT} $DATASET $DEVICE_ID" Enter
        else
            # 创建新窗口并执行命令
            tmux new-window -t $SESSION_NAME -n "window_${window_id}"
            tmux send-keys -t $SESSION_NAME:${window_id} "CUDA_VISIBLE_DEVICES=$DEVICE_ID ${TEST_SCRIPT} $DATASET $DEVICE_ID" Enter
        fi
        
        ((window_id++))
    done
done

# Attach to the tmux session
tmux attach-session -t $SESSION_NAME