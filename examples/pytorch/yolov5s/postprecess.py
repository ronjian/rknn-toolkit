import numpy as np

an_c = 102
class_cnt = 29
an_vec = 29 + 1 + 4

def parse_det(an, an_h, an_w, an_s, an_wh):
    for h in range(an_h):
        for w in range(an_w):
            for c in range(an_c):
                val = an[int(c * an_h * an_w + h * an_w + w)]
                if c % an_vec == 0:
                    c_x, c_y, b_w, b_h, obj_conf = val, None, None, None, None
                    c_x = (c_x * 2. - 0.5 + w) * an_s
                elif c % an_vec  == 1:
                    c_y = val
                    c_y = (c_y * 2. - 0.5 + h) * an_s
                elif c % an_vec == 2:
                    b_w = val
                    b_w = (b_w * 2) ** 2 * an_wh[c // an_vec][0]
                elif c % an_vec == 3:
                    b_h = val
                    b_h = (b_h * 2) ** 2 * an_wh[c // an_vec][1]
                elif c % an_vec == 4:
                    obj_conf = val
                else:
                    # if obj_conf and obj_conf > 0.5 and val > 0.5:
                    if obj_conf and obj_conf > 0.1:
                        print(an_h, an_w, h, w, c % an_vec, c_x, c_y, b_w, b_h, obj_conf, val)

parse_det(np.load('./assets/output2.npy').flatten().tolist()
        , 20
        , 20
        , 32
        , [[116,90], [156,198], [373,326]])

parse_det(np.load('./assets/output1.npy').flatten().tolist()
        , 40
        , 40
        , 16
        , [[30,61], [62,45], [59,119]])

parse_det(np.load('./assets/output0.npy').flatten().tolist()
        , 80
        , 80
        , 8
        , [[10,13], [16,30], [33,23]] )

