from rknn.api import RKNN
import cv2
import numpy as np
import os

# downSampleScale = 4.0
image_width = 640
image_height = 640
# pad = 0
origin_width = 1280
origin_height = 960
val_dir = '/workspace/centernet/data/baiguang/images/val/'
txt_dir = './assets/baiguang_text/'

rknn = RKNN()
rknn.config(channel_mean_value='0. 0. 0. 255.', reorder_channel='0 1 2')
ret = rknn.load_rknn('./best_yolov5s_robo_inconv.rknn-float')
assert ret == 0

for imgname in os.listdir(val_dir):
    if not imgname.endswith('.jpeg'): continue
    if imgname != 'StereoVision_L_990964_10_0_0_5026_D_FakePoop_719_-149.jpeg': continue
    print(imgname)
    origin_img = cv2.imread(val_dir + imgname)
    print(origin_img.shape)
    input_img = cv2.resize(origin_img, (image_width, image_height))
    ret = rknn.init_runtime()
    assert ret == 0
    outputs = rknn.inference(inputs=[input_img])
    print(len(outputs), outputs[0].shape, outputs[1].shape, outputs[2].shape)
    np.save('assets/output0.npy', outputs[0])
    np.save('assets/output1.npy', outputs[1])
    np.save('assets/output2.npy', outputs[2])

rknn.release()