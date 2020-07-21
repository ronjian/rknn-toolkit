import numpy as np
import cv2

an_c = 102
class_cnt = 29
an_vec = 29 + 1 + 4

img = cv2.imread('/workspace/centernet/data/baiguang/images/val/StereoVision_L_990964_10_0_0_5026_D_FakePoop_719_-149.jpeg')
# img = cv2.imread('./assets/input.jpg')
H, W, _ = img.shape
image_width = 640
image_height = int(640/1280*960)
pad_top = (640 - image_height) // 2
pad_bottom = 640 - image_height - pad_top

def clip(val):
    return min(max(val, 0.0), 1.0)

def parse_det(an, an_h, an_w, an_s, an_wh):
    res = []
    for h in range(an_h):
        for w in range(an_w):
            for c in range(an_c):
                val = an[int(c * an_h * an_w + h * an_w + w)]
                if c % an_vec == 0:
                    c_x, c_y, b_w, b_h, obj_conf, score = val, None, None, None, None, None
                    c_x = (c_x * 2. - 0.5 + w) * an_s
                elif c % an_vec  == 1:
                    c_y = val
                    c_y = (c_y * 2. - 0.5 + h) * an_s
                elif c % an_vec == 2:
                    b_w = val
                    b_w = (b_w * 2.) ** 2. * an_wh[c // an_vec][0]
                elif c % an_vec == 3:
                    b_h = val
                    b_h = (b_h * 2.) ** 2. * an_wh[c // an_vec][1]
                elif c % an_vec == 4:
                    obj_conf = val
                else:
                    # if obj_conf and obj_conf > 0.3 and val > 0.5:
                    if obj_conf > 0.1:
                        score = obj_conf * val
                        print(an_h, an_w, h, w, c % an_vec, c // an_vec, c_x, c_y, b_w, b_h, obj_conf, val, score)
                        x0 = int(clip((c_x - b_w / 2.) / image_width) * W)
                        y0 = int(clip((c_y - b_h / 2. - pad_top) / image_height) * H)
                        x1 = int(clip((c_x + b_w / 2.) / image_width) * W)
                        y1 = int(clip((c_y + b_h / 2. - pad_top) / image_height) * H)
                        res.append((x0, y0, x1, y1))
    return res            

R = []
R += parse_det(np.load('./assets/output2.npy').flatten().tolist()
        , 20
        , 20
        , 32
        , [[214,99], [287,176], [376,365]])

R += parse_det(np.load('./assets/output1.npy').flatten().tolist()
        , 40
        , 40
        , 16
        , [[94,219], [120,86], [173,337]])

R += parse_det(np.load('./assets/output0.npy').flatten().tolist()
        , 80
        , 80
        , 8
        , [[28,31], [53,73], [91,39]] )

for box in R:
    x0, y0, x1, y1 = box
    img = cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0,0), 2)
cv2.imwrite('./assets/StereoVision_L_990964_10_0_0_5026_D_FakePoop_719_-149.jpeg', img)