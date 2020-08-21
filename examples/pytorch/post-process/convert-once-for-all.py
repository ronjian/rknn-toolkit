from rknn.api import RKNN
from multiprocessing import Process
import os

def convert(line):
    idx, resolution, op_str, input_size = line.strip().split(',')
    # if idx != "71": return 
    input_size_list = [[int(each) for each in input_size.split('x')[1:]]]
    for posix in ["test", "fake"]:
        jit_path = "/workspace/once-for-all/jiangrong/assets/jits-new/{}_{}.jit".format(idx, posix)
        rknn_path = "/workspace/once-for-all/jiangrong/assets/rknns/{}_{}.rknn".format(idx, posix)
        if os.path.exists(rknn_path):
            print("{} already exists...".format(rknn_path))
            continue
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
            return
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
            return
        print('done')

        # Export rknn model
        print('--> Export RKNN model')
        ret = rknn.export_rknn(rknn_path)
        if ret != 0:
            print('Export failed!')
            print("================= {} failed ================\n".format(idx))
            return
        print('done')
    print("================= {} passed ================\n".format(idx))

if __name__ == "__main__":
    with open("/workspace/once-for-all/jiangrong/assets/jit-latency-lookuptable.meta", 'r') as rf:
        for line in rf:
            P = Process(target=convert,args=(line,))
            P.start()
            P.join()


    # P = Process(target=convert,args=("fail,160,expanded_conv-input:40x40x32-output:40x40x32-expand:96-kernel:5-stride:1-idskip:1-se:0-hs:0,1x3x40x40",))
    # P.start()
    # P.join()
