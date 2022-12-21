#!/bin/bash

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

source scripts/env_npu.sh;
export GLOG_v=3
################基础配置参数，需要模型审视修改##################
# number of Ascend910 device
export RANK_SIZE=1


# ensure GPU_8P_log_336 dir exists
DIR=./outputs
if [[ ! -d "$DIR" ]]; then
    mkdir $DIR
fi

################# strat training #################
python train.py --coco_path=/data/coco2017 \
                --output_dir=outputs/ \
                --mindrecord_dir=data/ \
                --clip_max_norm=0.1 \
                --no_aux_loss \
                --dropout=0.1 \
                --pretrained=ms_resnet_50.ckpt \
                --epochs=300 \
                --device_target="Ascend" \
                --device_id=3 > ${DIR}/train.log 2>&1

cat ${DIR}/train.log | grep loss > ${DIR}/train_loss.log