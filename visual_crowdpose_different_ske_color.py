import cv2

import matplotlib.pyplot as plt

import json
#/media/zxl/E/zxl/MIPNet/hg_ce_output/crowdpose/hg_ca_se/w48_256x192_adam_lr1e-3/results/keypoints_valid_results_epoch-1.json
#/media/zxl/E/zxl/datasets/crowdpose/json/crowdpose_val.json
json_data = open('/media/zxl/E/zxl/MIPNet/hg_ce_output/crowdpose/hg_ca_se/w48_256x192_adam_lr1e-3/results/keypoints_valid_results_epoch-1.json', 'r')
# json_data = open('/media/zxl/E/zxl/datasets/crowdpose/json/crowdpose_val.json', 'r')
print('json_data',json_data)
json_data = json.load(json_data)
print('json_data',len(json_data))

vis_id = 105335


def find_filename(img_id, json_data):
    for i in range(len(json_data)):
        # print('block',block)
        if json_data[i]['image_id'] == img_id:
            return str(json_data[i]['image_id'])+'.jpg'
        continue
    return None

def get_annokpts(img_id, json_data):
    kpts = []

    bboxes = []
    for i in range(len(json_data)):
        if json_data[i]['image_id'] == img_id:
            kpts.append(json_data[i]['keypoints'])

            bboxes.append(json_data[i]['bbox'])

        continue

    return kpts, bboxes

def vis_keypoints(img, kptss):
    import numpy as np
    color = [(0,0,255),
             (0, 255, 255),(0, 255, 255),(0, 255, 255),
             (49,163,84) ,(49,163,84),(49,163,84),
            (240,2,127), (240,2,127), (240,2,127),
            (255,255,0), (255,255,0),(255,255,0)]
    thr = 0.5
    for j in range(len(kptss)):
        kpts = np.array(kptss[j]).reshape(-1, 3)
        # skelenton = [[0, 2], [1, 3], [2, 4], [3, 5], [6, 8], [8, 10], [7, 9], [9, 11], [12, 13], [0, 13], [1, 13],
        #              [6, 13], [7, 13]]
        skelenton = [[12, 13], [0, 13],[0, 2],[2, 4], [1, 13], [1, 3], [3, 5], [6, 13],[6, 8], [8, 10], [7, 13], [7, 9], [9, 11],
                     ]
        points_num = [num for num in range(14)]
        for i,sk in enumerate(skelenton):
            pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
            pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
            if pos1[0] > 0 and pos1[1] > 0 and pos2[0] > 0 and pos2[1] > 0 and kpts[sk[0], 2] > thr and kpts[
                sk[1], 2] > thr:
                cv2.line(img, pos1, pos2, color[i], 2, 8)
        for points in points_num:
            pos = (int(kpts[points, 0]), int(kpts[points, 1]))
            if pos[0] > 0 and pos[1] > 0 and kpts[points, 2] > thr:
                cv2.circle(img, pos, 4, (0,0,255), -1)  # 为肢体点画红色实心圆
    return img

file_name= find_filename(vis_id, json_data)

kpts, bboxes = get_annokpts(vis_id, json_data)
# print('kpts',len(kpts))#3
# print('kpts0',kpts[0])#3
# print('kpts1',kpts[1])#3
# print('kpts2',kpts[2])#3
img = cv2.imread('/media/zxl/E/zxl/datasets/crowdpose/images/' + file_name)

plt.figure(figsize=(12, 10))

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.show()

# img = vis_box(img, bboxes)

plt.figure(figsize=(12, 10))

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# plt.show()

img = vis_keypoints(img, kpts)

plt.figure(figsize=(12, 10))

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# plt.show()
plt.savefig('/media/zxl/E/zxl/STIP/imgcrowd/105335.png')
