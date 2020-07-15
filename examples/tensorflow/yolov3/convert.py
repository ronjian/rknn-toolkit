from rknn.api import RKNN

rknn = RKNN()

rknn.config(channel_mean_value='127.5 127.5 127.5 127.5', reorder_channel='0 1 2', target_platform='rv1126')
# rknn.config(channel_mean_value='127.5 127.5 127.5 127.5', reorder_channel='0 1 2')

rknn.load_tensorflow(tf_pb='./yolov3_darknet_608.pb',
                         inputs=['conv2d/Conv2D'],
                         outputs=['predict_conv_1/Conv2D', 'predict_conv_2/Conv2D', 'predict_conv_3/Conv2D'],
                         input_size_list=[[608, 608, 3]])

rknn.build(do_quantization=True, dataset='./dataset.txt', pre_compile = True)

rknn.export_rknn('./yolov3_darknet_608.rknn')
