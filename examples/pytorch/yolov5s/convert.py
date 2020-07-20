
from rknn.api import RKNN
import torch
FLOAT = True
if __name__ == '__main__':
        # Create RKNN object
        rknn = RKNN()
        input_size_list = [[3,640,640]]

        # pre-process config
        print('--> config model')
        if FLOAT:
            rknn.config(channel_mean_value='0. 0. 0. 255.', reorder_channel='0 1 2')
        else:
            rknn.config(channel_mean_value='0. 0. 0. 255.', reorder_channel='0 1 2', target_platform='rv1126', batch_size = 10)
        print('done')

        # Load pytorch model
        print('--> Loading model')
        ret = rknn.load_pytorch(model='/workspace/yolov5/weights/best_yolov5s_robo_inconv_jit.pt', input_size_list=input_size_list)
        if ret != 0:
            print('Load pytorch model failed!')
            exit(ret)
        print('done')

        # Build model
        print('--> Building model')
        if FLOAT:
            ret = rknn.build(do_quantization=False)
        else:
            ret = rknn.build(do_quantization=True
                        , dataset='/workspace/rockchip/RV1126_RV1109/rv1126_rv1109/external/rknn-toolkit/examples/pytorch/centernet/baiguang-val-dataset.txt'
                        , pre_compile=True
                        , rknn_batch_size=1)
        if ret != 0:
            print('Build pytorch failed!')
            exit(ret)
        print('done')

        # Export rknn model
        print('--> Export RKNN model')
        if FLOAT:
            ret = rknn.export_rknn('./best_yolov5s_robo_inconv.rknn-float')
        else:
            ret = rknn.export_rknn('./best_yolov5s_robo_inconv.rknn')
        if ret != 0:
            print('Export failed!')
            exit(ret)
        print('done')

        if not FLOAT:
            print('--> Analysis')
            rknn.accuracy_analysis('/workspace/rockchip/RV1126_RV1109/rv1126_rv1109/external/rknn-toolkit/examples/pytorch/centernet/baiguang_anaysis.txt'
                                    , output_dir='./snapshot', calc_qnt_error=True)
            print('done')
