import os
import multiprocessing
import mindspore.dataset as ds
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, default='/cache/mpangu2/')
parser.add_argument('--output-dir', type=str, default='/cache/shuffled_mpangu2/')
args = parser.parse_args()

INPUT_MINDRECORD_DIR=args.input_dir
OUTPUT_MINDRECORD_DIR=args.output_dir
os.system(f"rm -rf {args.output_dir}*")

print(f'shuffle input dir is : {INPUT_MINDRECORD_DIR}')
print(f'shuffle output dir is : {OUTPUT_MINDRECORD_DIR}')

if not os.path.exists(OUTPUT_MINDRECORD_DIR):
    os.makedirs(OUTPUT_MINDRECORD_DIR)

def shuffle_mindrecord(file):
    # os.remove(OUTPUT_MINDRECORD_DIR + file)
    data_set = ds.MindDataset(dataset_files=[INPUT_MINDRECORD_DIR + file], shuffle=True)
    data_set.save(OUTPUT_MINDRECORD_DIR + file)
    print("MindRecord file: {} shuffled success.".format(file))

record = []
mindrecord_files = sorted(os.listdir(INPUT_MINDRECORD_DIR))

files = []
for file in mindrecord_files:
    if str.endswith(file, "db"):
        continue
    files.append(file)

for file in files:
    print("Begin shuffle file: {}".format(file))
    shuffle_mindrecord(file)
#     process = multiprocessing.Process(target=shuffle_mindrecord, args=(str(file),))
#     process.start()
#     record.append(process)

# for p in record:
#     p.join()







