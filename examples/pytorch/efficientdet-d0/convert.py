
from rknn.api import RKNN
import torch

if __name__ == '__main__':
        # Create RKNN object
        rknn = RKNN()
        input_size_list = [[3,512,512]]

        # pre-process config
        print('--> config model')
        rknn.config(channel_mean_value='123.675 116.28 103.53 58.395', reorder_channel='0 1 2', target_platform='rv1126', batch_size = 10)
        print('done')

        # Load pytorch model
        print('--> Loading model')
        # ret = rknn.load_pytorch(model='/workspace/Yet-Another-EfficientDet-Pytorch/weights/efficientdet-d0_jit.pt', input_size_list=input_size_list)
        # ret = rknn.load_pytorch(model='/workspace/Yet-Another-EfficientDet-Pytorch/weights/efficientdet.pt', input_size_list=input_size_list)
        ret = rknn.load_pytorch(model='/workspace/Yet-Another-EfficientDet-Pytorch/weights/pad_jit.pt', input_size_list=input_size_list)
        if ret != 0:
            print('Load pytorch model failed!')
            exit(ret)
        print('done')

        # Build model
        print('--> Building model')
        # ret = rknn.build(do_quantization=False, pre_compile=True, rknn_batch_size=1)
        ret = rknn.build(do_quantization=True
                        , dataset='./dataset.txt'
                        , pre_compile=True
                        , rknn_batch_size=1)
        if ret != 0:
            print('Build pytorch failed!')
            exit(ret)
        print('done')

        # Export rknn model
        print('--> Export RKNN model')
        ret = rknn.export_rknn('./efficientdet-d0.rknn')
        if ret != 0:
            print('Export failed!')
            exit(ret)
        print('done')

