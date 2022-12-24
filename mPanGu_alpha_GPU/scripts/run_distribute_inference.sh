#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distributed_train_gpu.sh RANK_SIZE HOSTFILE DATASET PER_BATCH_SIZE MODE"
echo "for example: bash scripts/run_distribute_inference.sh 8 /tmp/hostfile_8gpus 2.6B '8,9,10,11,12,13,14,15' "
echo "It is better to use absolute path."
echo "=============================================================================================================="

execute_path=$(pwd)
self_path=$(cd "$(dirname "$0")" || exit; pwd)
export RANK_SIZE=$1
export RANK_TABLE_FILE=$2
export MODE=$3
export CUDA_VISIBLE_DEVICES=$4
export LANGUAGE_IDX=$5
CKPT_PATH="/tmp/ckpt_filter/"
CKPT_NAME="mPanGu_Alpha-53_fp16.ckpt"
PARAM_TYPE="fp16"

mpirun --allow-run-as-root -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_DEBUG -x GLOG_v \
        -n $RANK_SIZE \
        --hostfile $RANK_TABLE_FILE \
        --output-filename log_output \
        --merge-stderr-to-stdout \
        python -s /tmp/predict.py \
        --mode $MODE \
        --run_type predict \
        --language_idx $LANGUAGE_IDX \
        --op_level_model_parallel_num $RANK_SIZE \
        --load_ckpt_path  $CKPT_PATH \
        --load_ckpt_name  $CKPT_NAME \
        --param_init_type $PARAM_TYPE #>inference_$LANGUAGE_IDX.log 2>&1 &



