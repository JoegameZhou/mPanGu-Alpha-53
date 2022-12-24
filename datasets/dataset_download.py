

#导入数据集 一带一路多语言数据集
#测试数据集将下载到当前目录下的"./SAMPLE_DATA"文件夹下，如需改变目录名称请自行修改
import wfio
_INPUT = '{"type":25,"uri":"sample_data/103/"}'
wfio.load_files(_INPUT, dir_name='/cache/')

