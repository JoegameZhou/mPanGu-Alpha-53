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
"""
PanguAlpha train script
"""
import datetime
import glob
import os
import math
import time
from pathlib2 import Path

from mindspore import context
from mindspore.train.model import Model
import mindspore.communication.management as D
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.train.callback import TimeMonitor, Callback
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
import mindspore.common.dtype as mstype
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.nn.wrap.cell_wrapper import PipelineCell, _VirtualDatasetCell
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dataset import create_dataset
from src.pangu_alpha import PanguAlpha, PanguAlphaWithLoss, CrossEntropyLoss
from src.pangu_alpha_wrapcell import PanguAlphaTrainOneStepWithLossScaleCell, PanguAlphaTrainPipelineWithLossScaleCell
from src.pangu_alpha_config import PANGUALPHAConfig, set_parse
from src.utils_m53 import LearningRate, get_args, FP32StateAdamWeightDecay
from src.utils_m53 import download_data, ckpt_copy_tar_new, get_ckpt_file_list
from src.utils_m53 import StrategySaveCallback, CheckpointSaveCallback, LossSummaryCallback

from download_dataset import DatasetDownloader, BUCKET_DIR, LOCAL_PATH
# from obs import ObsUploader
import moxing as mox

from mindspore.common import Parameter
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
import numpy as np

class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    """

    def __init__(self, dataset_size=-1, local_rank=0, has_trained_epoch=0, has_trained_step=0, micro_size=1):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size
        self.local_rank = local_rank
        self.has_trained_epoch = has_trained_epoch
        self.has_trained_step = has_trained_step
        self.micro_size = micro_size
        print("load has trained epoch :{} and step: {}".format(has_trained_epoch, has_trained_step), flush=True)

    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0 and self.local_rank % 8 == 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num /
                                           self._dataset_size)
            if percent == 0:
                epoch_num -= 1
            date = time.asctime(time.localtime(time.time()))
            loss_value = cb_params.net_outputs[0].asnumpy() / self.micro_size
            D.init()
            rank = D.get_rank()
            if rank%8 == 0:
                print("time: {} local_rank: {}, epoch: {}, step: {}, loss is {}, overflow is {}, scale is {}, lr is {}".##lr is {}, 
                      format(date, 
                             int(self.local_rank), 
                             int(epoch_num) + int(self.has_trained_epoch),
                             cb_params.cur_step_num + int(self.has_trained_step), 
                             loss_value,
                             cb_params.net_outputs[1].asnumpy(), 
                             cb_params.net_outputs[2].asnumpy(),
                             cb_params.net_outputs[3].asnumpy()))


project_root = os.path.abspath(
    os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


def run_train(args_opt):
    r"""
    The main training process.
    """
    # Set hccl connect time
    os.environ['HCCL_CONNECT_TIMEOUT'] = "6000"
    
    # Set execution mode
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    context.set_context(variable_memory_max_size="30GB")
    print(args_opt)
    # Set parallel context
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        print("rank_id is {}, device_num is {}".format(rank, device_num))

        context.reset_auto_parallel_context()
        
        local_strategy_ckpt_path="/cache/ckpt_strategy.ckpt"
        if args_opt.pre_trained:
                os.system('ulimit -s 102400')
                mox.file.copy(src_url=args_opt.strategy_load_ckpt_path, dst_url=local_strategy_ckpt_path)

        if args_opt.incremental_training:
            local_strategy_ckpt_path="/cache/ckpt_strategy.ckpt"
            if rank % 8 == 0:
                print("Incremental training", flush=True)
                os.system('ulimit -s 102400')
                mox.file.copy(src_url=args_opt.strategy_load_ckpt_path, dst_url=local_strategy_ckpt_path)
                ckpt_copy_tar_new(args_opt.load_ckpt_obs_path, target_path=args_opt.load_ckpt_local_path)
                mox.file.copy(f'{args_opt.load_ckpt_obs_path}_word_embedding.npy', f'{args_opt.load_ckpt_local_path}/word_embedding.npy')
                mox.file.copy(f'{args_opt.load_ckpt_obs_path}_position_embedding.npy', f'{args_opt.load_ckpt_local_path}/position_embedding.npy')
                mox.file.copy(f'{args_opt.load_ckpt_obs_path}_top_query_embedding.npy', f'{args_opt.load_ckpt_local_path}/top_query_embedding.npy')
                print("setting env success.")
                # 下载模型文件结束后，写一个文件来表示下载成功
                f = open("/tmp/download_ckpt.txt", 'w')
                f.close()
        if args_opt.incremental_training:
            # 此处用于阻塞其他进程，直到刷包以及下载数据集完成为止
            while not os.path.exists("/tmp/download_ckpt.txt"):
                time.sleep(1)
            print("\n\n************Checkpoint download succeed!*************\n\n", flush=True)
            if rank % 8 == 0:
                print(os.listdir(args_opt.load_ckpt_local_path), flush=True)
        
        if args_opt.incremental_training or args_opt.pre_trained:
            if args_opt.device_num > 64:
                context.set_auto_parallel_context(
                    parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, 
                    gradients_mean=False,
                    full_batch=bool(args_opt.full_batch), 
                    strategy_ckpt_load_file=local_strategy_ckpt_path,
                    enable_parallel_optimizer=bool(args_opt.optimizer_shard), 
                    optimizer_weight_shard_size=64,
                    strategy_ckpt_save_file='/cache/strategy.ckpt')
#                     optimizer_weight_shard_aggregated_save=True,
                    
            else:
                context.set_auto_parallel_context(
                    parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, 
                    gradients_mean=False,
                    full_batch=bool(args_opt.full_batch), 
                    strategy_ckpt_load_file=local_strategy_ckpt_path,
                    enable_parallel_optimizer=bool(args_opt.optimizer_shard), 
                    optimizer_weight_shard_aggregated_save=True,
                    strategy_ckpt_save_file='/cache/strategy.ckpt')
        else:
            if args_opt.device_num > 64:
                context.set_auto_parallel_context(
                    parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, 
                    gradients_mean=False,
                    full_batch=bool(args_opt.full_batch),
                    enable_parallel_optimizer=bool(args_opt.optimizer_shard), 
                    optimizer_weight_shard_size=64,
                    strategy_ckpt_save_file='/cache/strategy.ckpt')
                    ## optimizer_weight_shard_aggregated_save=True,
            else:
                context.set_auto_parallel_context(
                    parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, 
                    gradients_mean=False,
                    full_batch=bool(args_opt.full_batch), 
                    enable_parallel_optimizer=bool(args_opt.optimizer_shard),
                    optimizer_weight_shard_aggregated_save=True,
                    strategy_ckpt_save_file='/cache/strategy.ckpt')
        
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()
    else:
        rank = 0
        device_num = 1
    context.set_context(save_graphs=False, save_graphs_path="./graphs_of_device_id_" + str(rank))
    ###################################################
    ## context.set_context(enable_graph_kernel=True)
    ###################################################
    # copy data from the cloud to the /cache/Data
    cache_url = "/cache/Data/" ##LOCAL_PATH ## '/cache/Data/'
    if args_opt.offline:
        cache_url = args_opt.data_url
    else:
        download_data(src_data_url=args_opt.data_url, tgt_data_path=cache_url, rank=rank)
    # Set model property
    model_parallel_num = args_opt.op_level_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    batch_size = args_opt.per_batch_size * data_parallel_num
    config = PANGUALPHAConfig(
        data_parallel_num=data_parallel_num, 
        model_parallel_num=model_parallel_num, 
        batch_size=batch_size,
        seq_length=args_opt.seq_length, 
        vocab_size=args_opt.vocab_size, 
        embedding_size=args_opt.embedding_size,
        num_layers=args_opt.num_layers, 
        num_heads=args_opt.num_heads, 
        expand_ratio=4, dropout_rate=0.1,
        compute_dtype=mstype.float16, 
        stage_num=args_opt.stage_num, 
        micro_size=args_opt.micro_size,
        eod_reset=bool(args_opt.eod_reset), 
        load_ckpt_path=None, ##args_opt.load_ckpt_local_path,## incremental_training ckpt load，None
        param_init_type=mstype.float32 if args_opt.param_init_type == 'fp32' else mstype.float16,
        word_emb_dp=bool(args_opt.word_emb_dp))
    print("===config is: ", config, flush=True)

    # Define network
    pangu_alpha = PanguAlpha(config)
    loss = CrossEntropyLoss(config)
    pangu_alpha_with_loss = PanguAlphaWithLoss(config, pangu_alpha, loss)
    pangu_alpha_with_loss = _VirtualDatasetCell(pangu_alpha_with_loss)

    print("=====args_opt is: ", args_opt, flush=True)

    # Warm-up and cosine decay learning rate
    lr = LearningRate(learning_rate=args_opt.start_lr, end_learning_rate=args_opt.end_lr,
                      warmup_steps=args_opt.warmup_step, decay_steps=args_opt.decay_steps)

    # Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    params = pangu_alpha.trainable_params()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-1
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': params
    }]
    if args_opt.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    else:
        optimizer = FP32StateAdamWeightDecay(group_params, learning_rate=lr, eps=1e-8, beta1=0.9, beta2=0.94)## 0.95
    # Initial scaling sens
    loss_scale_value = math.pow(2, 12)
    epoch_num = args_opt.epoch_size
    
    # Dataset loading mindrecord files
    ds = create_dataset(config.batch_size, 
                        data_path=cache_url,
                        data_start_index=args_opt.data_start_index, 
                        eod_reset=config.eod_reset, 
                        full_batch=bool(args_opt.full_batch),
                        eod_id=args_opt.eod_id, 
                        device_num=device_num, 
                        rank=rank,
                        column_name=args_opt.data_column_name, 
                        epoch=epoch_num)
    step_per_epoch = ds.get_dataset_size()
    callback_size = args_opt.sink_size
    actual_epoch_num = int(epoch_num * step_per_epoch / callback_size)
    print("\n=====dataset size: ", ds.get_dataset_size(), flush=True)
    print("=====batchsize: ", batch_size, flush=True)
    print("=====actual_epoch_num: ", actual_epoch_num, flush=True)
    print(f"=====mp: {model_parallel_num}, dp: {data_parallel_num}\n")
    #################################################################
    if args_opt.incremental_training or args_opt.pre_trained:
#         callback = [
#         TimeMonitor(callback_size), LossCallBack(callback_size, rank, 0, 0)]
        callback = [
            TimeMonitor(callback_size),
            LossCallBack(callback_size, rank, args_opt.has_trained_epoches, args_opt.has_trained_steps)
        ]
    else:
        callback = [
        TimeMonitor(callback_size), LossCallBack(callback_size, rank, 0, 0)]
    ##################################################################

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value, scale_factor=2, scale_window=1500)
    pangu_alpha_with_grads = PanguAlphaTrainOneStepWithLossScaleCell(
        pangu_alpha_with_loss, optimizer=optimizer, scale_update_cell=update_cell, enable_global_norm=True,
        config=config)
    model = Model(pangu_alpha_with_grads)

    if not mox.file.exists(args_opt.save_checkpoint_bucket_dir):
        mox.file.make_dirs(args_opt.save_checkpoint_bucket_dir) 
    add_checkpoint_callback_policy(args_opt, callback, rank)
    ## -----------------------------------------------------------------------------------------------
    if args_opt.pre_trained:
        restore_checkpoint_from_obs(args_opt, callback_size, ds, model, pangu_alpha, optimizer, actual_epoch_num)
        # restore_checkpoint_from_obs_64(args_opt, callback_size, ds, model, pangu_alpha, optimizer, actual_epoch_num)
    if args_opt.incremental_training:
        from mindspore.train.serialization import load_distributed_checkpoint
        ckpt_file_list = get_ckpt_file_list(args_opt.load_ckpt_local_path, device_num=512)
        print("Start to load distributed checkpoint", flush=True)
        print(f"Loading from path {ckpt_file_list[0]}", flush=True)
        load_distributed_checkpoint(model.train_network, ckpt_file_list)
    ## -----------------------------------------------------------------------------------------------
    
    print("Dataset size: {}, actual_epoch_num: {}".format(ds.get_dataset_size(), actual_epoch_num), flush=True)
    model.train(actual_epoch_num, ds, callbacks=callback, sink_size=callback_size, dataset_sink_mode=True)


def add_checkpoint_callback_policy(args_param, callback, rank_id):
    r"""
    Add checkpoint policy to callback.
    """
    if args_param.save_checkpoint:
        if not os.path.exists(args_param.save_checkpoint_path):
            os.makedirs(args_param.save_checkpoint_path, exist_ok=True)
        # checkpoint store epoch_num and step_num info
        ckpt_append_info = [{"epoch_num": args_param.has_trained_epoches, "step_num": args_param.has_trained_steps}]
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args_param.save_checkpoint_steps,
                                       keep_checkpoint_max=1,
                                       integrated_save=False,
                                       append_info=ckpt_append_info
                                       )
        save_dir_rank = os.path.join(args_param.save_checkpoint_path, f"rank_{rank_id}")
        save_ckptfile_name = args_param.ckpt_name_prefix + '_' + str(rank_id)
        if not os.path.exists(save_dir_rank):
            os.makedirs(save_dir_rank, exist_ok=True)
        ckpoint_cb = ModelCheckpoint(prefix=args_param.ckpt_name_prefix + '_' + str(rank_id),
                                     directory=save_dir_rank,
                                     config=ckpt_config)
    
        ckpt_save_obs_cb = CheckpointSaveCallback(local_ckpt_dir=save_dir_rank, 
                                              local_rank=rank_id, 
                                              has_trained_epoch=args_param.has_trained_epoches,
                                              has_trained_step=args_param.has_trained_steps, 
                                              bucket=args_param.save_checkpoint_bucket_dir,
                                              syn_obs_steps=args_param.save_checkpoint_steps)
        callback.append(ckpoint_cb)
        callback.append(ckpt_save_obs_cb)
        
    if rank_id == 0:
        sub_dir = args_param.save_checkpoint_bucket_dir.split('/')[-1]
        callback.append(LossSummaryCallback(summary_dir="summary", 
                                            local_rank=0, 
                                            has_trained_epoch=args_param.has_trained_epoches,
                                            has_trained_step=args_param.has_trained_steps, 
                                            bucket='obs://research-my/taoht-13b/summary/' + sub_dir,
                                            syn_times=40))
        callback.append(StrategySaveCallback(strategy_path='/cache/', 
                                            local_rank=0, 
                                            has_trained_epoch=args_param.has_trained_epoches,
                                            has_trained_step=args_param.has_trained_steps, 
                                            bucket=args_param.save_strategy_bucket_dir))
                                            ## bucket='obs://research-my/taoht-13b/strategy_ckpt/' + sub_dir))
        
def restore_checkpoint_from_obs(args_param, sink_size, dataset, model, network, optimizer, epoch):
    r"""
    Load checkpoint process.
    """
    import logging
    import moxing as mox
    from pathlib2 import Path
    if args_param.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank_id = D.get_rank()
        local_rank = rank_id

    ckpt_dir = os.path.join("/cache/ckpt/", f"rank_{str(rank_id)}")
    if not os.path.exists(ckpt_dir):
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    if args_param.pre_trained:
        ckpt_number = device_num
        os.system('ulimit -s 102400')
        logging.info(f"rank_{rank_id}: start restoring ckpt.")

        restore_ckptname=f"{args_param.train_ckpt_name_load}_{rank_id}-{args_param.load_ckpt_num}_2.ckpt"
        # restore_ckptname=f"pangu-multilangs_{rank_id}-{args_param.load_ckpt_num}_2.ckpt"

        restore_bueckt_dir = os.path.join(args_param.load_checkpoint_bucket_dir + f"/rank_{str(rank_id)}", restore_ckptname)
        local_ckpt_dir = os.path.join(ckpt_dir, str(rank_id)+'_ckpt.ckpt')
        print(f"copy ckpt from {restore_bueckt_dir} to {local_ckpt_dir} ...")
        time.sleep((rank_id%16)*10)
        mox.file.copy(restore_bueckt_dir, local_ckpt_dir)
        time.sleep((int(rank_id+15) % 16 )*10)
        # 下载模型文件结束后，写一个文件来表示下载成功
        f = open(f"/tmp/download_ckpt{rank_id}.txt", 'w')
        f.close()
    # 此处用于阻塞其他进程，直到刷包以及下载数据集完成为止
    while not os.path.exists(f"/tmp/download_ckpt{rank_id}.txt"):
        print(f"copy ckpt waitting 1s ...")
        time.sleep(1)
    print(f"Copy ckpt finished !")
    params_dict = load_checkpoint(local_ckpt_dir)
    
    has_trained_epoch = int(params_dict["epoch_num"].data.asnumpy())
    has_trained_step = int(params_dict["step_num"].data.asnumpy())
    global_step = int(params_dict["global_step"].data.asnumpy())
    print(f'\n\nhas_trained_epoch: {has_trained_epoch} \n has_trained_step: {has_trained_step}\n global_step: {global_step}\n\n')
    
    if args_param.has_trained_epoches == 0:
        # reset epoch_num, step_num
        params_dict["epoch_num"] = Parameter(Tensor(0, dtype=mstype.int64), name="epoch_num")
        params_dict["step_num"] = Parameter(Tensor(0, dtype=mstype.int64), name="step_num")
        params_dict["step"] = Parameter(Tensor(0, dtype=mstype.int32), name="step")
        params_dict["scale_sense"] = Parameter(Tensor(65536, dtype=mstype.float32), name="scale_sense")
        params_dict["current_iterator_step"] = Parameter(Tensor(0, dtype=mstype.int32), name="current_iterator_step")
        params_dict["last_overflow_iterator_step"] = Parameter(Tensor(0, dtype=mstype.int32), name="last_overflow_iterator_step")
        params_dict["global_step"] = Parameter(Tensor(np.zeros((1, ), dtype=np.int32), dtype=mstype.int32), name="global_step")
    
    if params_dict:
        params_dict["scale_sense"] = Parameter(Tensor(65536, dtype=mstype.float32), name="scale_sense")
        model._init(train_dataset=dataset, sink_size=sink_size)
        load_param_into_net(model.train_network, params_dict)

        
def restore_checkpoint_from_obs_64(args_param, sink_size, dataset, model, network, optimizer, epoch):
    r"""
    Load checkpoint process.
    """
    import logging
    import moxing as mox
    from pathlib2 import Path
    if args_param.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank_id = D.get_rank()
        local_rank = rank_id

    ckpt_dir = os.path.join("/cache/ckpt/", f"rank_{str(local_rank)}")
    # create dir for ckpt
    if not os.path.exists(ckpt_dir):
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    if args_param.pre_trained:
        ckpt_number = device_num
        os.system('ulimit -s 102400')
        logging.info(f"rank_{rank_id}: start restoring ckpt.")
        rank_id_64 = (rank_id + 64)%128
        restore_ckptname=f"{args_param.train_ckpt_name_load}_{rank_id}-{args_param.load_ckpt_num}_2.ckpt"
        restore_ckptname_64=f"{args_param.train_ckpt_name_load}_{rank_id_64}-{args_param.load_ckpt_num}_2.ckpt"
            
        restore_bueckt_dir = os.path.join(args_param.load_checkpoint_bucket_dir + f"/rank_{str(rank_id)}", restore_ckptname)
        restore_bueckt_dir_64 = os.path.join(args_param.load_checkpoint_bucket_dir + f"/rank_{str(rank_id_64)}", restore_ckptname_64)
        # local_ckpt_dir = os.path.join(ckpt_dir, restore_ckptname)
        # local_ckpt_dir_64 = os.path.join(ckpt_dir, restore_ckptname_64)
        local_ckpt_dir = os.path.join(ckpt_dir, str(rank_id)+'_ckpt.ckpt')
        local_ckpt_dir_64 = os.path.join(ckpt_dir, str(rank_id_64)+'_ckpt.ckpt')
        time.sleep((rank_id%16)*1)
        print(f"copy ckpt from {restore_bueckt_dir} to {local_ckpt_dir} ...")
        mox.file.copy(restore_bueckt_dir, local_ckpt_dir)
        time.sleep(1)
        print(f"copy ckpt from {restore_bueckt_dir_64} to {local_ckpt_dir_64} ...")
        mox.file.copy(restore_bueckt_dir_64, local_ckpt_dir_64)
        time.sleep((int(rank_id+15) % 16 ) * 1)
        # 下载模型文件结束后，写一个文件来表示下载成功
        f = open(f"/tmp/download_ckpt{rank_id}.txt", 'w')
        f.close()
    # 此处用于阻塞其他进程，直到刷包以及下载数据集完成为止
    while not os.path.exists(f"/tmp/download_ckpt{rank_id}.txt"):
        print(f"copy ckpt waitting 1s ...")
        time.sleep(1)
    print(f"Copy ckpt finished !")
    ##logging.info(f"rank_{rank_id}: start loading {local_ckpt_dir}")
    try:
        logging.info(f"rank_{rank_id}: start loading {local_ckpt_dir}")
        params_dict = load_checkpoint(local_ckpt_dir)
        logging.info(f"rank_{rank_id}: success restoring ckpt...")
        if params_dict:
            if args_param.has_trained_epoches == 0:
                ## reset epoch_num, step_num
                params_dict["epoch_num"] = Parameter(Tensor(0, dtype=mstype.int64), name="epoch_num")
                params_dict["step_num"] = Parameter(Tensor(0, dtype=mstype.int64), name="step_num")
                params_dict["step"] = Parameter(Tensor(0, dtype=mstype.int32), name="step")
                params_dict["scale_sense"] = Parameter(Tensor(65536, dtype=mstype.float32), name="scale_sense")
                params_dict["current_iterator_step"] = Parameter(Tensor(0, dtype=mstype.int32), name="current_iterator_step")
                params_dict["last_overflow_iterator_step"] = Parameter(Tensor(0, dtype=mstype.int32), name="last_overflow_iterator_step")
                params_dict["global_step"] = Parameter(Tensor(np.zeros((1, ), dtype=np.int32), dtype=mstype.int32), name="global_step")
            
            model._init(train_dataset=dataset, sink_size=sink_size)
            load_param_into_net(model.train_network, params_dict) 
    except:
        try:
            print('\n\nLoad ', str(rank_id)*40, ' Failed !!!\n\n')
            logging.info(f"111111, rank_{rank_id}: load {local_ckpt_dir} Failed !")
            logging.info(f"rank_{rank_id}: start loading {local_ckpt_dir_64} checkpoint ...\n")
            params_dict = load_checkpoint(local_ckpt_dir_64)
            logging.info(f"rank_{rank_id}: success restoring ckpt...\n")
            if params_dict:
                if args_param.has_trained_epoches == 0:
                    ## reset epoch_num, step_num
                    params_dict["epoch_num"] = Parameter(Tensor(0, dtype=mstype.int64), name="epoch_num")
                    params_dict["step_num"] = Parameter(Tensor(0, dtype=mstype.int64), name="step_num")
                    params_dict["step"] = Parameter(Tensor(0, dtype=mstype.int32), name="step")
                    params_dict["scale_sense"] = Parameter(Tensor(65536, dtype=mstype.float32), name="scale_sense")
                    params_dict["current_iterator_step"] = Parameter(Tensor(0, dtype=mstype.int32), name="current_iterator_step")
                    params_dict["last_overflow_iterator_step"] = Parameter(Tensor(0, dtype=mstype.int32), name="last_overflow_iterator_step")
                    params_dict["global_step"] = Parameter(Tensor(np.zeros((1, ), dtype=np.int32), dtype=mstype.int32), name="global_step")
                
                model._init(train_dataset=dataset, sink_size=sink_size)
                load_param_into_net(model.train_network, params_dict) 
        except:
            logging.info(f"\n\n22222222222222-------\n\n, rank_{rank_id}: load {rank_id_64} ckpt Failed !\n\n")
            print('\n\nLoad ', str(rank_id_64)*40, ' Failed !!!\n\n')

def restore_checkpoint(args_param, sink_size, dataset, model, network, epoch):
    r"""
    Load checkpoint process.
    """
    print("======start single checkpoint", flush=True)
    ckpt_name = args_param.ckpt_name_prefix
    ckpt_pattern = os.path.join(args_param.save_checkpoint_path, "rank_{}".format(D.get_rank()),
                                f"{ckpt_name}*.ckpt")
    ckpt_files = glob.glob(ckpt_pattern)
    if not ckpt_files:
        print(f"There is no ckpt file in {args_param.save_checkpoint_path}, "
              f"current ckpt_files found is {ckpt_files} "
              f"with pattern {ckpt_pattern}, so skip the loading.")
        return
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    time_stamp = datetime.datetime.now()
    print(f"time stamp {time_stamp.strftime('%Y.%m.%d-%H:%M:%S')} pre trained ckpt model {ckpt_files} loading",
          flush=True)
    # Load checkpoint files latest file
    print(f'Start to load from {ckpt_files[0]}')
    param_dict = load_checkpoint(ckpt_files[0])
    if param_dict.get("epoch_num") and param_dict.get("step_num"):
        args_param.has_trained_epoches = int(param_dict["epoch_num"].data.asnumpy())
        args_param.has_trained_steps = int(param_dict["step_num"].data.asnumpy())

    model.build(train_dataset=dataset, sink_size=sink_size, epoch=epoch)
    load_param_into_net(network, param_dict)


def run_train_pipeline(args_opt):
    r"""
    The main training process in pipeline.
    """
    # Set hccl connect time
    os.environ['HCCL_CONNECT_TIMEOUT'] = "6000"

    context.set_context(save_graphs=False, mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    context.set_context(variable_memory_max_size="31GB")
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank_id = D.get_rank()
        print("rank_id is {}, device_num is {}".format(rank_id, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            full_batch=bool(args_opt.full_batch),
            loss_repeated_mean=True,
            device_num=device_num,
            enable_parallel_optimizer=bool(args_opt.optimizer_shard),
            pipeline_stages=args_opt.stage_num)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()
    else:
        rank_id = int(os.getenv("RANK_ID"))
        device_num = 1
    # copy data from the cloud to the /cache/Data
    cache_url = '/cache/Data/'
    if args_opt.offline:
        cache_url = args_opt.data_url
    else:
        download_data(src_data_url=args_opt.data_url, tgt_data_path=cache_url, rank=rank_id)
    model_parallel_num = args_opt.op_level_model_parallel_num
    stage_device_num = int(device_num / args_opt.stage_num)
    data_parallel_num = int(stage_device_num / model_parallel_num)
    per_batch_size = args_opt.per_batch_size
    batch_size = per_batch_size * data_parallel_num * args_opt.micro_size
    config = PANGUALPHAConfig(
        data_parallel_num=data_parallel_num,
        model_parallel_num=model_parallel_num,
        batch_size=batch_size,
        seq_length=args_opt.seq_length,
        vocab_size=args_opt.vocab_size,
        embedding_size=args_opt.embedding_size,
        num_layers=args_opt.num_layers,
        num_heads=args_opt.num_heads,
        expand_ratio=4,
        post_layernorm_residual=False,
        dropout_rate=0.1,
        compute_dtype=mstype.float16,
        use_past=False,
        stage_num=args_opt.stage_num,
        micro_size=args_opt.micro_size,
        word_emb_dp=bool(args_opt.word_emb_dp))
    print("===config is: ", config, flush=True)
    pangu_alpha = PanguAlpha(config)
    loss = CrossEntropyLoss(config)
    pangu_alpha_with_loss = PipelineCell(PanguAlphaWithLoss(config, pangu_alpha, loss), config.micro_size)
    pangu_alpha_with_loss = _VirtualDatasetCell(pangu_alpha_with_loss)
    print("=====args_opt is: ", args_opt, flush=True)
    lr = LearningRate(learning_rate=args_opt.start_lr, end_learning_rate=args_opt.end_lr,
                      warmup_steps=args_opt.warmup_step, decay_steps=args_opt.decay_steps)
    params = pangu_alpha.infer_param_pipeline_stage()
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-1
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': params
    }]
    if args_opt.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    else:
        optimizer = nn.AdamWeightDecay(group_params, learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)

    ds = create_dataset(config.batch_size, data_path=cache_url, device_num=stage_device_num,
                        rank=rank_id % stage_device_num, eod_reset=True, data_start_index=0,
                        full_batch=context.get_auto_parallel_context("full_batch"),
                        column_name=args_opt.data_column_name)
    epoch_num = args_opt.epoch_size
    step_per_epoch = ds.get_dataset_size()
    callback_size = args_opt.sink_size
    actual_epoch_num = int(epoch_num * step_per_epoch / callback_size)
    callback = [TimeMonitor(callback_size),
                LossCallBack(callback_size, local_rank=rank_id, micro_size=config.micro_size)]
    loss_scale_value = math.pow(2, 32)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value, scale_factor=2, scale_window=1000)
    pangu_alpha_with_grads = PanguAlphaTrainPipelineWithLossScaleCell(
        pangu_alpha_with_loss, optimizer=optimizer, config=config, scale_update_cell=update_cell)
    model = Model(pangu_alpha_with_grads)
    model.train(actual_epoch_num, ds, callbacks=callback,
                sink_size=callback_size, dataset_sink_mode=True)


if __name__ == "__main__":
    opt = get_args()
    set_parse(opt)
    if opt.per_batch_size == 0:
        raise ValueError("The per_batch_size has not been configured.")
    if opt.stage_num > 1:
        run_train_pipeline(opt)
    else:
        run_train(opt)
