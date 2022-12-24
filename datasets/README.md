
<!-- TOC -->
- [数据使用&处理](#数据使用)
     - [1、数据准备](#原始数据准备)
     - [2、数据抽样](#数据抽样)
     - [3、数据转化](#数据转化、打乱)
- [数据处理案例](#数据处理案例)
<!-- /TOC -->

## 数据使用
[一带一路多语言1T数据集详情](.。/docs/B&R-M-1T_dataset.md) 
### 原始数据准备
* 靶场开源数据下载
```bash
#导入数据集 一带一路多语言数据集
#测试数据集将下载到当前目录下的"./SAMPLE_DATA"文件夹下，如需改变目录名称请自行修改
import wfio
_INPUT = '{"type":25,"uri":"sample_data/2114/"}'
wfio.load_files(_INPUT, dir_name='./SAMPLE_DATA')
```

### 数据抽样
这里以“一带一路多语言数据集”为例（单双语数据格式及容量，详见[一带一路多语言数据集](./doc/B&R-M-1T_dataset.md)）：
```bash
# 设定抽样文件本地路径和抽样保存路径
data_dir = '/cache/data/'
save_path = '/cache/data_sample/'
# 设定抽样策略， 配置中、英单语文件抽取容量， MB
mono_sample_strategy = {'zh': 1024, 'en': 1024}
# 设定抽样策略， 配置中、英双语文件抽取容量， MB
corpus_sample_strategy = {'zh-en': 1024}

# 单语抽取
!python ./dataset/dataset_sample.py \
        --data_path $data_dir \ # 原始数据路径
        --output_path $save_path \ # 抽取数据保存路径
        --sample_strategy "{mono_sample_strategy}" \ # 数据抽取的策略
        --mode 'mono' # 单语抽取模式
# 双语抽取
!python ./dataset/dataset_sample.py \
        --data_path $data_dir \ # 原始数据路径
        --output_path $save_path \ # 抽取数据保存路径
        --sample_strategy "{corpus_sample_strategy}" \ # 数据抽取的策略
        --mode 'corpus' # 双语抽取模式
```

<div align=center>
<img src="../doc/dataset_sample.png" width="1500"/><br/>
</div>

### 数据转化、打乱
这里以转化为mindRecord数据文件为例：
```bash
data_dir = '/cache/data_sample/*.txt'
save_path = '/cache/MindRecord/mPanGu_zh-en_mindrecord'
# 多进程，一个进程绑定一个mindrecord文件写
num_mindrecord = 50 #不要设置的太小，不然进程少，转换较慢

!python ./dataset/pre_process_bc.py \
        --data_path "{data_dir}" \
        --output_file $save_path \
        --num_process $num_mindrecord \
        --tokenizer 'spm_13w'

# 文件内打乱
!python ./dataset/mindrecord_shuffle.py \
        --input-dir "/cache/MindRecord/" \
        --output-dir "/cache/MindRecord_shuffle/" 
```

### 数据处理案例
client 1, client 2数据处理流程一致
```bash
#导入数据集 一带一路多语言数据集
#测试数据集将下载到当前目录下的"./SAMPLE_DATA"文件夹下，如需改变目录名称请自行修改
import wfio
_INPUT = '{"type":25,"uri":"sample_data/2114/"}'
wfio.load_files(_INPUT, dir_name='./SAMPLE_DATA')

# 设定抽样文件本地路径和抽样保存路径
data_dir = '/cache/data/'
save_path = '/cache/data_sample/'
# 设定抽样策略， 配置中、英单语文件抽取容量， MB
mono_sample_strategy = {'zh': 1024, 'en': 1024}
# 设定抽样策略， 配置中、英双语文件抽取容量， MB
corpus_sample_strategy = {'zh-en': 1024}

# 单语抽取
!python ./dataset/dataset_sample.py --data_path $data_dir --output_path $save_path --sample_strategy "{mono_sample_strategy}" --mode 'mono'
# 双语抽取
!python ./dataset/dataset_sample.py --data_path $data_dir --output_path $save_path --sample_strategy "{corpus_sample_strategy}" --mode 'corpus'

data_dir = '/cache/data_sample/*.txt'
save_path = '/cache/MindRecord/mPanGu_zh-en_mindrecord'
# 多进程，一个进程绑定一个mindrecord文件写
num_mindrecord = 50 #不要设置的太小，不然进程少，转换较慢

!python ./dataset/pre_process_bc.py --data_path "{data_dir}" --output_file $save_path --num_process $num_mindrecord --tokenizer 'spm_13w'

# 文件内打乱
!python ./dataset/mindrecord_shuffle.py --input-dir "/cache/MindRecord/" --output-dir "/cache/MindRecord_shuffle/" 
```

### 交流通道
- 提问：https://git.openi.org.cn/PCL-Platform.Intelligence/mPanGu-Alpha-53/issues
- 邮箱：taoht@pcl.ac.cn

### 项目信息
鹏城实验室-智能部-高效能云计算所-分布式计算研究室

### 许可证

[Apache License 2.0]

