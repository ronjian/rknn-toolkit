from rknn.api import RKNN
from multiprocessing import Process
import os


input_size_list = [[3, 260, 260]]

jit_path = "/workspace/once-for-all/jiangrong/assets/note10_lat_64ms_top1_80.2_finetune_75.jit"
rknn_path = "/workspace/once-for-all/jiangrong/assets/note10_lat_64ms_top1_80.2_finetune_75.rknn"

# Create RKNN object
rknn = RKNN()
# pre-process config
print('--> config model')
rknn.config(channel_mean_value='0. 0. 0. 255.'
            , reorder_channel='0 1 2'
            , target_platform='rv1126'
            , batch_size = 1 # for quantize
            )
print('done')
# Load pytorch model
print('--> Loading model')
ret = rknn.load_pytorch(model=jit_path, input_size_list=input_size_list)
if ret != 0:
    print('Load pytorch model failed!')
    print("================= {} failed ================\n".format(idx))
    exit()
print('done')

# Build model
print('--> Building model')
ret = rknn.build(do_quantization=True
            , dataset='../yolov5s/dataset.txt'
            , pre_compile=True
            , rknn_batch_size=1)
if ret != 0:
    print('Build pytorch failed!')
    print("================= {} failed ================\n".format(idx))
    exit()
print('done')

# Export rknn model
print('--> Export RKNN model')
ret = rknn.export_rknn(rknn_path)
if ret != 0:
    print('Export failed!')
    print("================= {} failed ================\n".format(idx))
    exit()
print('done')
