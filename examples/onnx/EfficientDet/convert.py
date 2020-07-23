from rknn.api import RKNN

rknn = RKNN()

rknn.config(channel_mean_value='123.675 116.28 103.53 58.82', reorder_channel='0 1 2')

ret = rknn.load_onnx(model='./efficientdet-d0_opset9.onnx')

