#  
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
network config setting, gradient clip function and dynamic learning rate function
"""
import argparse
import ast
import os
import time
import hashlib
import numpy as np
import mindspore.nn as nn
from mindspore import context
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_rank, get_group_size, create_group
from mindspore.common import Parameter, ParameterTuple
from mindspore.common.initializer import initializer
from mindspore.nn.optim.optimizer import Optimizer

from mindspore.nn.optim.adam import _adam_opt, _check_param_value


class AdamWeightDecay(Optimizer):
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(AdamWeightDecay, self).__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self.parameters.clone(prefix="adam_m", init='zeros')
        self.moments2 = self.parameters.clone(prefix="adam_v", init='zeros')
        self.hyper_map = C.HyperMap()

    def construct(self, gradients):
        lr = self.get_lr()
        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(F.partial(_adam_opt, self.beta1, self.beta2, self.eps),
                                              lr, self.weight_decay, self.parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.hyper_map(F.partial(_adam_opt, self.beta1, self.beta2, self.eps, lr),
                                              self.weight_decay, self.parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
        else:
            optim_result = self.hyper_map(F.partial(_adam_opt, self.beta1, self.beta2, self.eps, lr, self.weight_decay),
                                          self.parameters, self.moments1, self.moments2,
                                          gradients, self.decay_flags, self.optim_filter)
        if self.use_parallel:
            self.broadcast_params(optim_result)
        return optim_result, lr

class FP32StateAdamWeightDecay(AdamWeightDecay):
    r"""
        This class is almost same with the mindspore's AdamWeightDecay implements, the
        only difference is the optimizer's state will be always initialized with float32,
        where the original AdamWeightDecay will initialize the optimizer's state with float16,
        if the parameters are initialized with fp16.
        This setting will avoid overflow in training PanGu-Alpha model using fp16.
    """

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(FP32StateAdamWeightDecay, self).__init__(params, learning_rate=learning_rate,
                                                       beta1=beta1,
                                                       beta2=beta2,
                                                       eps=eps,
                                                       weight_decay=weight_decay)

        self.moments1 = self.clone_state(self.parameters, prefix='adam_m', init='zeros')
        self.moments2 = self.clone_state(self.parameters, prefix='adam_v', init='zeros')

    def clone_state(self, parameter_tuple, prefix, init):
        r"""
            parameter_tuple: ParameterTuple. The parameters of the network
            prefix: str. The prefix name of the parameters
            init: str. The initialization method
        """
        new = []
        for old_param in parameter_tuple:
            new_state = Parameter(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.param_info = old_param.param_info.clone()
            new_state.is_init = False
            new_state.set_data(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.name = prefix + '.' + new_state.name
            new.append(new_state)
        return ParameterTuple(new)


get_square_sum = C.MultitypeFuncGraph("get_square_sum")


@get_square_sum.register("Tensor", "Number")
def _get_square_sum(grad, value):
    norm = P.ReduceSum(False)(F.square(grad), ()) / value
    norm = F.expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")


@apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm(clip_norm, global_norm, grad):
    grad = grad * clip_norm / global_norm
    return grad

def _get_model_parallel_group(mp):
    """

    Calculate the communication group of model parallel dim in one pipeline stage

    """
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    stage_id = rank // per_stage_device_nums
    local_stage_rank_id = rank % per_stage_device_nums
    index = local_stage_rank_id // mp
    group = range(0, mp)
    rank_str_list = [str(x + index * mp + stage_id * per_stage_device_nums) for x in group]
    rank_list_str = "-".join(rank_str_list)
    rank_list = [x + index * mp + stage_id * per_stage_device_nums for x in group]
    return rank_list, rank_list_str


def _get_pipeline_group():
    """

    Calculate the communication group between all pipeline stages

    """
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    local_stage_rank_id = rank % per_stage_device_nums
    group = range(0, stage_nums)
    rank_list = [local_stage_rank_id + x * per_stage_device_nums for x in group]
    rank_str_list = [str(local_stage_rank_id + x * per_stage_device_nums) for x in group]
    rank_list_str = "-".join(rank_str_list)
    return rank_list, rank_list_str


class GlobalNorm(nn.Cell):
    """
    Calculate the global norm value of given tensors
    """

    def __init__(self, params, config):
        super(GlobalNorm, self).__init__()
        self.norm = nn.Norm()
        self.hyper_map = C.HyperMap()
        self.is_pipeline = (config.stage_num > 1)
        if self.is_pipeline:
            if context.get_auto_parallel_context("enable_parallel_optimizer"):
                group_size = get_group_size() // config.stage_num
            else:
                group_size = config.mp
            group_list, group_name = _get_model_parallel_group(group_size)
            # In avoid of the group name too long
            hashed = hashlib.md5(group_name.encode()).hexdigest()[:48]
            print(f"Creating hash value for the group_name hash({group_name})={hashed}")
            group_name = str(hashed)
            create_group(group_name, group_list)
            self.allreduce = P.AllReduce(group=group_name)
            pipeline_group_list, pipeline_group_name = _get_pipeline_group()
            hashed = hashlib.md5(pipeline_group_name.encode()).hexdigest()[:48]
            print(f"Creating hash value for the group_name hash({pipeline_group_name})={hashed}")
            pipeline_group_name = str(hashed)
            create_group(pipeline_group_name, pipeline_group_list)
            self.allreduce2 = P.AllReduce(group=pipeline_group_name)
        else:
            group_size = get_group_size()
        self.allreduce_group_size = ()
        for x in params:
            if "projection.bias" not in x.name and "layernorm" not in x.name and "embedding_table" not in x.name:
                self.allreduce_group_size = self.allreduce_group_size + (1.0,)
            elif "embedding_table" not in x.name:
                self.allreduce_group_size = self.allreduce_group_size + (group_size * 1.0,)
            else:
                if not config.word_emb_dp and "position_embedding.embedding_table" not in x.name \
                        and "top_query_embedding_table" not in x.name:
                    self.allreduce_group_size = self.allreduce_group_size + (config.dp * 1.0,)
                else:
                    self.allreduce_group_size = self.allreduce_group_size + (group_size * 1.0,)

    def construct(self, grads):
        """Calculate global norm construct"""
        square_sum = self.hyper_map(get_square_sum, grads, self.allreduce_group_size)
        square_reduce_sum = F.addn(square_sum)
        if self.is_pipeline:
            stage_square_reduce_sum = self.allreduce(square_reduce_sum)
            global_square_reduce_sum = self.allreduce2(stage_square_reduce_sum)
            global_norms = F.sqrt(global_square_reduce_sum)
        else:
            global_norms = F.sqrt(P.AllReduce()(square_reduce_sum))
        return global_norms


class ClipByGlobalNorm(nn.Cell):
    """

    Clip grads by global norm

    """

    def __init__(self, params, config, clip_norm=1.0):
        super(ClipByGlobalNorm, self).__init__()
        self.global_norm = GlobalNorm(params, config)
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()

    def construct(self, grads):
        """Clip grads by global norm construct"""
        global_norm_value = self.global_norm(grads)
        cond = P.GreaterEqual()(global_norm_value, self.clip_norm)
        global_norm = F.select(cond, global_norm_value, self.clip_norm)
        grads = self.hyper_map(F.partial(apply_global_norm, self.clip_norm, global_norm), grads)
        return grads, global_norm_value


class LearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for PanguAlpha network.
    """

    def __init__(self,
                 learning_rate,
                 end_learning_rate,
                 warmup_steps,
                 decay_steps,
                 power=1.0,
                 use_cosine=True):
        super(LearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate,
                                          decay_steps, power)
        self.cosine_decay_lr = CosineDecayLR(end_learning_rate, learning_rate,
                                             decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine

    def construct(self, global_step):
        """dynamic learning rate"""
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step),
                                  mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


def add_inference_params(opt):
    """Add inference params"""
    opt.add_argument("--frequency_penalty",
                     type=float,
                     default=1.5,
                     help="coefficient for frequency_penalty")
    opt.add_argument("--presence_penalty",
                     type=float,
                     default=0.3,
                     help="coefficient for presence_penalty")
    opt.add_argument("--max_generate_length",
                     type=int,
                     default=512,
                     help="the maximum number of generated token")
    opt.add_argument("--top_k_num",
                     type=int,
                     default=1,
                     help="the number for top_k sampling")
    opt.add_argument("--top_p",
                     type=float,
                     default=1.0,
                     help="top_p sampling threshold, enabled if less than 1.0, if top_p is set to 1.0, use top_k sampling")
    opt.add_argument("--end_token",
                     type=int,
                     default=128298,
                     help="the token id for <end of document>")
    opt.add_argument("--use_pynative_op",
                     type=int,
                     default=0,
                     help="Whether use pynative op for postproecess")
    opt.add_argument("--use_past",
                     type=str,
                     default="true",
                     choices=["true", "false"],
                     help="Whether enable state reuse")

def add_training_params(opt):
    """Add training params"""
    opt.add_argument("--seq_length",
                     type=int,
                     default=1024,
                     help="sequence length, default is 1024.")
    opt.add_argument("--vocab_size",
                     type=int,
                     default=128320,
                     help="vocabulary size, default is 40000.")
    opt.add_argument("--embedding_size",
                     type=int,
                     default=16384,
                     help="embedding table size, default is 16384.")
    opt.add_argument("--num_layers",
                     type=int,
                     default=64,
                     help="total layers, default is 64.")
    opt.add_argument("--num_heads",
                     type=int,
                     default=128,
                     help="head size, default is 128.")
    opt.add_argument("--stage_num",
                     type=int,
                     default=1,
                     help="Pipeline stage num, default is 4.")
    opt.add_argument("--micro_size",
                     type=int,
                     default=1,
                     help="Pipeline micro_size, default is 1.")
    opt.add_argument("--eod_reset",
                     type=int,
                     default=1,
                     help="Enable eod mask, default is 1.")
    opt.add_argument("--warmup_step",
                     type=int,
                     default=2000,
                     help="Warmup step, default is 2000.")
    opt.add_argument("--decay_steps",
                     type=int,
                     default=150000,
                     help="Decay step, default is 200000.")
    opt.add_argument("--optimizer",
                     type=str,
                     default="adam",
                     choices=["adam", "lamb"],
                     help="select which optimizer to be used, default adam")
    opt.add_argument("--eod_id",
                     type=int,
                     default=128298,
                     help="The id of end of document,6")
    opt.add_argument("--epoch_size",
                     type=int,
                     default=1,
                     help="The training epoch")
    opt.add_argument("--sink_size",
                     type=int,
                     default=2,
                     help="The sink size of the training. default is 2")
    opt.add_argument("--full_batch",
                     default=0,
                     type=int,
                     help="Import the full size of a batch for each card, default is 1")
    opt.add_argument("--optimizer_shard",
                     type=int,
                     default=1,
                     help="Enable optimizer parallel, default is 1")
    opt.add_argument("--per_batch_size",
                     type=int,
                     default=4,
                     help="The batch size for each data parallel way. default 6")
    opt.add_argument("--start_lr",
                     type=float,
                     default=5e-6,
                     help="The start learning rate. default 5e-5")
    opt.add_argument("--end_lr",
                     type=float,
                     default=5e-7,
                     help="The end learning rate. default 1e-6")
    opt.add_argument("--op_level_model_parallel_num",
                     type=int,
                     default=1,
                     help="The model parallel way. default 8")
    opt.add_argument("--word_emb_dp",
                     type=int,
                     default=1,
                     choices=[0, 1],
                     help="Whether do data parallel in word embedding. default 1")
    opt.add_argument("--data_column_name",
                     type=str,
                     default="input_ids",
                     help="Column name of datasets")
# 中断恢复配置， pre_trained
def add_retrain_params(opt):
    """
    Add parameters about retrain.
    """
    opt.add_argument("--pre_trained",
                     type=ast.literal_eval,
                     default=True,
                     help="Pretrained checkpoint path.")
    opt.add_argument("--save_checkpoint_path",
                     type=str,
                     default="",
                     help="Save checkpoint path.")
    opt.add_argument("--save_checkpoint_bucket_dir",
                     type=str,
                     default="",
                     help="Save checkpoint path.")
    opt.add_argument("--save_strategy_bucket_dir",
                     type=str,
                     default="",
                     help="Save checkpoint path.")
    opt.add_argument("--strategy_load_ckpt_path",
                        type=str,
                        default="",
                        help="The training prallel strategy for the model.")
    opt.add_argument("--keep_checkpoint_max",
                     type=int,
                     default=1,
                     help="Max checkpoint save number.")
    opt.add_argument("--save_checkpoint_steps",
                     type=int,
                     default=3000,
                     help="Save checkpoint step number.")
    opt.add_argument("--save_checkpoint",
                     type=ast.literal_eval,
                     default=True,
                     help="Whether save checkpoint in local disk.")
    opt.add_argument("--train_ckpt_name_load",
                     type=str,
                     default="mPanGu53-26b-stage1-2",
                     help="Load checkpoint name prefix.")
    opt.add_argument("--ckpt_name_prefix",
                     type=str,
                     default="mPanGu53-26b-stage2",
                     help="Saving checkpoint name prefix.")
    opt.add_argument("--load_checkpoint_bucket_dir",
                     type=str,
                     default="s3://research-my2/taoht-13b/ckpt/mPanGu_53-26b-128k-m1d128-22m2d24-exp1",
                     help="Load checkpoint path, continue trainning.")
    opt.add_argument("--load_ckpt_num",
                     type=int,
                     default=204000,
                     help="Load ckpt num steps. 0")
    opt.add_argument("--has_trained_epoches",
                     type=int,
                     default=0,
                     help="Epoches has been trained before. 0")
    opt.add_argument("--has_trained_steps",
                     type=int,
                     default=0,
                     help="Steps has been trained before. 0")
    opt.add_argument("--data_start_index",
                     type=int,
                     default=0,
                     help="datasets has been trained before. 0")
    
    
def get_args(inference=False):
    """train function for PanguAlpha"""
    parser = argparse.ArgumentParser(description="PanguAlpha training")
    parser.add_argument('--device_id',
                        type=int,
                        default=0,
                        help="Device id, default is 0.")
    parser.add_argument("--device_num",
                        type=int,
                        default=128,
                        help="Use device nums, default is 128.")
    parser.add_argument("--distribute",
                        type=str,
                        default="true",
                        choices=["true", "false"],
                        help="Run distribute, default is true.")
    parser.add_argument('--data_url',
                        required=False,
                        default='obs://datasets/V3-Multi-langs-53-1T-bpe-128k-jieba/Exp_3-langs53-140G-hybrid/',
                        help='Location of data.')
    parser.add_argument('--train_url',
                        required=False,
                        default=None,
                        help='Location of training outputs.')
    parser.add_argument("--run_type",
                        type=str,
                        default="pre_trained",
                        choices=["train", "predict", "incremental_train", "pre_trained"],
                        help="The run type")
    parser.add_argument("--mode",
                        type=str,
                        default="2.6B",
                        choices=["200B", "13B", "2.6B", "self_define"],
                        help="The scale of the model parameters")
    parser.add_argument("--device_target",
                        type=str,
                        default="Ascend",
                        choices=["Ascend", "GPU"],
                        help="The running device")
    parser.add_argument("--tokenizer_path",
                        type=str,
                        default="./tokenizer_path",
                        help="The path where stores vocab and vocab model file")
    parser.add_argument("--param_init_type",
                        type=str,
                        default="fp32",
                        help="The initialization type for parameters. Default fp32.")
    parser.add_argument("--offline",
                        type=int,
                        default=0,
                        help="Running on cloud of not. Default 1.")
    parser.add_argument("--export",
                        type=int,
                        default=0,
                        help="Whether export mindir for serving.")
    parser.add_argument("--load_ckpt_name",
                        type=str,
                        default='PanGu_Alpha.ckpt',
                        help="checkpint file name.")
    parser.add_argument("--load_ckpt_obs_path",
                        type=str,##s3://mindspore-file/huangxinjing/filtered_ckpt/Newexp69_GPT3_5-684_2/Newexp69_GPT3_5-684_2
                        default='s3://research-my2/taoht-13b/filtered_ckpt/mPanGu_53-26b-128k-m1d128-22m1d24/mPanGu_53-26b-128k-m1d128-22m3d26-exp4/integrated_save/',
                        help="predict file path.")## 
    parser.add_argument("--load_obs_ckptname",
                        type=str,
                        default='exp4-54000_mPanGu53-26b-stage4_',
                        help="ckpt file name in obs.")## 
    parser.add_argument("--load_ckpt_local_path",
                        type=str,
                        default='/cache/ckpt_file',
                        help="incremental training or predict ckpt file path. Default None")
    parser.add_argument("--incremental_training",
                        type=int,
                        default=0,
                        help="Enable incremental training. Default 0.")
    parser.add_argument("--language",
                        type=str,
                        default="en",
                        choices=["en", "ar", "bg", "bn", "de", "es", "fa", "fr", "he", "hi", "ko", "pt", "ru", "sr", "ta", "uk", "ur", "vi"],
                        help="The language to translation")
    parser.add_argument("--language_idx",
                        type=int,
                        default=0,
                        help="The language idx to translation， 0-17")
    parser.add_argument("--language_idx_wmt",
                    type=int,
                    default=0,
                    help="The language idx to translation， 0:en, 1:zh")
    add_training_params(parser)
    add_retrain_params(parser)
    if inference:
        add_inference_params(parser)
    args_opt = parser.parse_args()

    return args_opt
