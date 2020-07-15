from rknn.api import RKNN
import cv2

rknn = RKNN()

rknn.config(channel_mean_value='123.675 116.28 103.53 58.395', reorder_channel='0 1 2')

ret = rknn.load_rknn('./centernet_mbv2.rknn')

img = cv2.imread('./space_shuttle_224.jpg')

img = cv2.resize(img, (384, 288))

ret = rknn.init_runtime()

outputs = rknn.inference(inputs=[img])
print(type(outputs), len(outputs), outputs[0].shape, outputs[1].shape, outputs[2].shape, outputs[3].shape)
