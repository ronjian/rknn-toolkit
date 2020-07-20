from rknn.api import RKNN

rknn = RKNN()

rknn.config(channel_mean_value='127.5 127.5 127.5 127.5', reorder_channel='0 1 2', target_platform='rv1126')
# rknn.config(channel_mean_value='127.5 127.5 127.5 127.5', reorder_channel='0 1 2')

rknn.load_tensorflow(tf_pb='./efficientdet-d0_frozen_1.15.pb',
                         inputs=['image_arrays'],
                         outputs=['class_net/class-predict/BiasAdd',
                                'class_net/class-predict_1/BiasAdd',
                                'class_net/class-predict_2/BiasAdd',
                                'class_net/class-predict_3/BiasAdd',
                                'class_net/class-predict_4/BiasAdd',
                                'box_net/box-predict/BiasAdd',
                                'box_net/box-predict_1/BiasAdd',
                                'box_net/box-predict_2/BiasAdd',
                                'box_net/box-predict_3/BiasAdd',
                                'box_net/box-predict_4/BiasAdd'],
                         input_size_list=[[512, 512, 3]])

rknn.build(do_quantization=True, dataset='./dataset.txt', pre_compile = True)

rknn.export_rknn('./efficientdet-d0.rknn')
