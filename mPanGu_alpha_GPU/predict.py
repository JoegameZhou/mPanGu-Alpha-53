# Copyright 2022 PCL
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
PanGu predict run
"""
import os
import numpy as np
import mindspore
import mindspore.common.dtype as mstype
import mindspore.communication.management as D
from mindspore import context, Tensor
from mindspore import export
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
# from mindspore.nn.transformer.transformer import TransformerOpParallelConfig
from src.transformer import TransformerOpParallelConfig
from mindspore.train.model import Model
from src.serialization import load_distributed_checkpoint
from src.pangu_alpha import EvalNet, PanguAlphaModel
from src.pangu_alpha_config import set_parse, PanguAlphaConfig
from src.utils import get_args

from mindspore.common import Parameter
from mindspore.train.serialization import load_distributed_checkpoint, load_checkpoint, load_param_into_net

import time
from mindspore.train.serialization import load_checkpoint

def load_model(args_opt):
    r"""
     The main function for load model
    """
    # Set execution mode
    context.set_context(save_graphs=False,
                        mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target)
    context.set_context(variable_memory_max_size="30GB")
    # Set parallel context
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        print("rank_id is {}, device_num is {}".format(rank, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            full_batch=True,
            loss_repeated_mean=True,
            enable_parallel_optimizer=False,
            strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path,
            pipeline_stages=args_opt.stage_num)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()

    else:
        rank = 0
        device_num = 1
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path)

    use_past = (args_opt.use_past == "true")
    print('local_rank:{}, start to run...'.format(rank), flush=True)
    if args_opt.export:
        use_past = True
    # Set model property
    model_parallel_num = device_num # args_opt.op_level_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    print("*("*20)
    print(model_parallel_num, data_parallel_num)

    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num,
                                                  model_parallel=model_parallel_num,
                                                  pipeline_stage=args_opt.stage_num,
                                                  micro_batch_num=args_opt.micro_size,
                                                  recompute=True)

    per_batch_size = args_opt.per_batch_size
    batch_size = per_batch_size * data_parallel_num
    # Now only support single batch_size for predict
    if args_opt.run_type == "predict":
        batch_size = 1
    config = PanguAlphaConfig(
        batch_size=batch_size,
        seq_length=args_opt.seq_length,
        vocab_size=args_opt.vocab_size,
        hidden_size=args_opt.embedding_size,
        num_layers=args_opt.num_layers,
        num_heads=args_opt.num_heads,
        post_layernorm_residual=False,
        dropout_rate=0.0,
        ffn_hidden_size=args_opt.embedding_size * 4,
        use_past=use_past,
        eod_reset=False,
        parallel_config=parallel_config,
        load_ckpt_path=None,
        run_type=args_opt.run_type,
        param_init_type=mstype.float32 if args_opt.param_init_type == 'fp32' else mstype.float16)
    print("===config is: ", config, flush=True)
    print("=====args_opt is: ", args_opt, flush=True)

    ckpt_name = args_opt.load_ckpt_name
    # Define network
    pangu_alpha = PanguAlphaModel(config)
    eval_net = EvalNet(pangu_alpha)
    eval_net.set_train(False)
    model_predict = Model(eval_net)
    # Compile network and obtain tensor layout for loading ckpt
    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    current_index = Tensor(np.array([0]), mstype.int32)

    if args_opt.distribute == "false":
        predict_layout = None
    elif config.use_past:
        batch_valid_length = Tensor(np.array([0]), mstype.int32)
        init_true = Tensor([True], mstype.bool_)
        inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
        predict_layout = model_predict.infer_predict_layout(inputs_np, current_index, init_true, batch_valid_length)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
        _ = model_predict.infer_predict_layout(inputs_np_1, current_index, init_true, batch_valid_length)
    else:
        predict_layout = model_predict.infer_predict_layout(inputs_np, current_index)

    print("======start load checkpoint", flush=True)
    local_ckpt_path = args_opt.load_ckpt_path + args_opt.load_ckpt_name
    # NPU训练的模型两边代码版本不同有几个parameters名字不一致，需要对应下
    parameters = load_checkpoint(local_ckpt_path)
    parameters["backbone.embedding.word_embedding.embedding_table"] = parameters["embedding_table"]
    parameters.pop("embedding_table")
    parameters["backbone.embedding.position_embedding.embedding_table"] = parameters["backbone.embedding.embedding.position_embedding.embedding_table"]
    parameters.pop("backbone.embedding.embedding.position_embedding.embedding_table")
    parameters["backbone.top_query_embedding.embedding_table"] = parameters["backbone.top_query_embedding_table"]
    parameters.pop("backbone.top_query_embedding_table")

    load_param_into_net(eval_net, parameters)
    # load_checkpoint(local_ckpt_path, eval_net)
    print("================load param ok=================", flush=True)

    return model_predict, config


def export_mindir(model_predict, config):
    """Export mindir model"""
    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    current_index = Tensor(np.array([0]), mstype.int32)

    batch_valid_length = Tensor(np.array([0]), mstype.int32)
    init_true = Tensor([True], mstype.bool_)
    inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)

    model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
    export(model_predict.predict_network, inputs_np, current_index,
           init_true, batch_valid_length, file_name='pangu_alpha_1024', file_format='MINDIR')
    model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
    export(model_predict.predict_network, inputs_np_1, current_index,
           init_true, batch_valid_length, file_name='pangu_alpha_1', file_format='MINDIR')
    print("Export finished and now exit.")


def run_predict(model_predict, config, args_opt):
    """run predict"""
    from src.tokenization_jieba import JIEBATokenizer
    from src.generate import generate, generate_increment
    # Define tokenizer
    tokenizer = JIEBATokenizer(os.path.join(args_opt.tokenizer_path, 'vocab10.vocab'),
                               os.path.join(args_opt.tokenizer_path, 'vocab10.model'))

    # Tokenize input sentence to ids
    sample = "今天是一个好天气"
    tokenized_token = tokenizer.tokenize(sample)
    start_sentence = tokenizer.convert_tokens_to_ids(tokenized_token)
    input_ids = np.array(start_sentence).reshape(1, -1)
    # Call inference
    generate_func = generate_increment if config.use_past else generate
    output_ids = generate_func(model_predict, input_ids, args_opt)
    # Decode output ids to sentence
    output_samples = tokenizer.convert_ids_to_tokens(output_ids.tolist())
    print('Output is:', output_samples, flush=True)

def run_predict_langs(model_predict, config, args_opt):
    """run predict"""
    from tokenizer.tokenizer_spm import SpmTokenizer
    from src.generate import generate, generate_increment
    from tokenizer.tokenizer_spm import langs_ID, translate_ID
    import jieba
    D.init()
    rank = D.get_rank()
    
    work_dir = '/tmp'
    # Define tokenizer
    vocab_file = work_dir + '/tokenizer/spm.128k.model.1'
    tokenizer = SpmTokenizer(vocab_file)
    EOT = tokenizer.eot_id
    # inference mode
    generate_func = generate_increment if config.use_past else generate
    
    #------------------------------------------------------------------
    # Tokenize input sentence to ids, example
    sample = "你 今天 中午 吃的 什么 ？"
    tokenized_token = tokenizer.tokenize(sample)
    start_sentence = tokenizer.convert_tokens_to_ids(tokenized_token)
    input_ids = np.array(start_sentence).reshape(1, -1)
    # Call inference
    print('000000000000'*20)
    print(input_ids)
    output_ids = generate_func(model_predict, input_ids, args_opt, dynamic_generate_length=20)
    # Decode output ids to sentence
    output_samples = tokenizer.convert_ids_to_tokens(output_ids.tolist())
    print('\nExample output is:', output_samples, flush=True)
    #------------------------------------------------------------------
    result_save_dir = f"{work_dir}/inference/flores_zh-langs/"

    if not os.path.exists(result_save_dir):
        print("Creating translate output bueckt dir {}".format(result_save_dir))
        os.mkdir(result_save_dir)

    langs_zh = [['vi', 'ko', 'en', 'nl'], 
                ['de', 'ms', 'id', 'tl'], 
                ['mn', 'my', 'th', 'lo'], 
                ['km', 'lt', 'et', 'lv'], 
                ['hu', 'pl', 'cs', 'sk'], 
                ['sl', 'hr', 'bs', 'sr'],
                ['bg', 'mk', 'ru', 'uk'], 
                ['be', 'el', 'ka', 'hy'], 
                ['ro', 'fr', 'es', 'pt'],
                ['fa', 'he', 'ar', 'ps'], 
                ['tr', 'kk', 'uz'], 
                ['az', 'hi', 'ta'], 
                ['ur', 'bn', 'ne']]

    langs_5 = langs_zh[args_opt.language_idx]

    for langs in langs_5:
        try:
            result = []
            times_stat = []
            translate_file_path = work_dir + f'/tokenizer/langs_53/flores101_50_zh/zh-{langs}_flores-test.txt'
            result_upload_path = os.path.join(result_save_dir, f"output_zh-{langs}-flores.txt")

            src_langs = 'zh'
            tag_langs = langs

            with open(translate_file_path, 'r', encoding='utf-8') as f:
                for idx, data in enumerate(f.read().split("\n\n")):
                    if data:
                        src_txt, tag_txt = data.split('\t')

                        if src_langs == 'zh':
                            tokenized_src_langs = tokenizer.tokenize('' + src_txt)
                        else:
                            tokenized_src_langs = tokenizer.tokenize(''.join(jieba.cut('' + src_txt)))
                        src_id = tokenizer.convert_tokens_to_ids(tokenized_src_langs)

                        tokenized_tag_langs = tokenizer.tokenize('' + tag_txt)
                        tag_id = tokenizer.convert_tokens_to_ids(tokenized_tag_langs)
                        # Tokenize input sentence to ids

                        src_trans2_tag_input = [langs_ID[src_langs], langs_ID[src_langs], langs_ID[src_langs]] + \
                                              src_id + \
                                              [translate_ID, translate_ID, translate_ID] + \
                                              [langs_ID[tag_langs], langs_ID[tag_langs], langs_ID[tag_langs]]

                        tag_trans2_src_input = [langs_ID[tag_langs], langs_ID[tag_langs], langs_ID[tag_langs]] + \
                                             tag_id + \
                                             [translate_ID, translate_ID, translate_ID] + \
                                             [langs_ID[src_langs], langs_ID[src_langs], langs_ID[src_langs]]

                        # Call inference
                        time_start = time.time()
                        tag_out_max_len = min(len(src_id)*2 + 50, 512)
                        src2tag_output_ids = generate_func(model_predict,
                                                          np.array([src_trans2_tag_input]),
                                                          args_opt,
                                                          dynamic_generate_length=tag_out_max_len).tolist()
                        time1 = time.time()
                        src_out_max_len = min(len(tag_id)*2 + 50, 512)
                        en2zh_output_ids = generate_func(model_predict, np.array([tag_trans2_src_input]), args_opt,
                                                         dynamic_generate_length=src_out_max_len).tolist()

                        times_stat.append((time1 - time_start) / len(src2tag_output_ids[len(src_trans2_tag_input):]))
                        times_stat.append((time.time() - time1) / len(en2zh_output_ids[len(tag_trans2_src_input):]))

                        # Decode output ids to sentence
                        tag_output = tokenizer.convert_ids_to_tokens(src2tag_output_ids[len(src_trans2_tag_input):])
                        src_output = tokenizer.convert_ids_to_tokens(en2zh_output_ids[len(tag_trans2_src_input):])
                        if src_langs == 'zh':
                            src_output.replace(' ', '')
                        result.append(tag_output + '\t' + src_output)
                        if rank == 0:
                            print("----------------------------------------------------------")
                            print(idx, " INPUT is : ", data, "\n")
                            print(idx, " OUTPUT is : " + src_output + '\t' + tag_output + "\n")

                        if rank == 0 and idx % 10 == 0:
                            with open(result_upload_path, 'w')as f_output:
                                for i, i_txt in enumerate(result):
                                    f_output.writelines(str(i) + '\t' + i_txt + "\n")
                            print(tag_langs, "translate time: ", np.average(times_stat), " s/tokens" + '\n\n')
                time.sleep(1)
                if rank == 0:
                    with open(result_upload_path, 'w')as f_output:
                        for i, i_txt in enumerate(result):
                            f_output.writelines(str(i) + '\t' + i_txt + "\n")
                    print("Save the output file to the :{}".format(result_upload_path))
        except Exception as error:
            print(langs, " error, continue...", error)
                
    
def main():
    """Main process for predict or export model"""
    opt = get_args(True)
    set_parse(opt)
    model_predict, config = load_model(opt)
    if opt.export:
        export_mindir(model_predict, config)
    else:
        run_predict_langs(model_predict, config, opt)


if __name__ == "__main__":
    main()