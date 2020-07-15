from rknn.api import RKNN
import cv2
import numpy as np
import os

downSampleScale = 4.0
image_width = 384
image_height = 288
pad = 0
origin_width = 1280
origin_height = 1920
val_dir = '/workspace/centernet/data/baiguang/images/val/'
txt_dir = './assets/baiguang_text/'

rknn = RKNN()
rknn.config(channel_mean_value='123.675 116.28 103.53 58.395', reorder_channel='0 1 2')
ret = rknn.load_rknn('./centernet_mbv2.rknn')
assert ret == 0

for imgname in os.listdir(val_dir):
    if not imgname.endswith('.jpeg'): continue
    # if imgname != 'StereoVision_L_990964_10_0_0_5026_D_FakePoop_719_-149.jpeg': continue
    print(imgname)
    origin_img = cv2.imread(val_dir + imgname)
    input_img = cv2.resize(origin_img, (image_width, image_height))
    ret = rknn.init_runtime()
    assert ret == 0
    outputs = rknn.inference(inputs=[input_img])
    # print(type(outputs), len(outputs), outputs[0].shape, outputs[1].shape, outputs[2].shape, outputs[3].shape)
    # np.save('assets/hm.npy', outputs[0])
    # np.save('assets/pool.npy', outputs[1])
    # np.save('assets/wh.npy', outputs[2])
    # np.save('assets/reg.npy', outputs[3])
    # hm = np.load('assets/hm.npy')
    # pool = np.load('assets/pool.npy')
    # wh = np.load('assets/wh.npy')
    # reg = np.load('assets/reg.npy')
    hm = outputs[0]
    pool = outputs[1]
    wh = outputs[2]
    reg = outputs[3]
    _,C,H,W = hm.shape
    with open(txt_dir + imgname.replace('.jpeg', '.txt'), 'w') as wf:
        wf.write(val_dir + imgname)
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    if hm[0,c,h,w] > 0.01 and hm[0,c,h,w] == pool[0,c,h,w]:
                        score = hm[0,c,h,w]
                        label = c+1
                        xreg = reg[0,0,h,w]
                        yreg = reg[0,1,h,w]
                        width = wh[0,0,h,w]
                        height = wh[0,1,h,w]
                        x1 = int(max(((w + xreg) - width / 2.0) * downSampleScale / image_width, 0.0) * origin_width)
                        y1 = int(max((((h + yreg) - height / 2.0) * downSampleScale - pad) / image_height, 0.0) * origin_height)
                        x2 = int(min(((w + xreg) + width / 2.0) * downSampleScale / image_width, 1.0) * origin_width)
                        y2 = int(min((((h + yreg) + height / 2.0) * downSampleScale - pad) / image_height, 1.0) * origin_height)
                        # print(label, score, x1, y1, x2, y2)
                        wf.write(" " + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + "," + str(label) + "," + str(score))
