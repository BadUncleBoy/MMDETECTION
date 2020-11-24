from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import cv2
import json
import random 
colors = []
for _ in range(10):
    colors.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))
config_file = './configs/vg/faster_rcnn_r50_fpn_sample1e-3_mstrain_1x_vg1000.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = './exps/vg/r50/latest.pth'
# # build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

def test_single(img_path):
    result= inference_detector(model, img_path)
    img = cv2.imread(img_path)
    for each in result:
        if len(each):
            for rect in each:
               if rect[4]<0.2:
                   continue 
               img = cv2.rectangle(img, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])),colors[random.randint(0,100)%(len(colors))], 2)
    cv2.imwrite("infer.jpg",img)


def test_multi(test_path):
    test_jpgs = [each.strip().split(" ")[1] for each in open(test_path).readlines()]
    for each in test_jpgs:
        img_path = "/data/zy/dataset/voc/"+ each
        result = inference_detector(model, img_path)[0]
        # img = cv2.imread(img_path)
        # for i in range(result.shape[0]):
        #     rect = result[i]
        #     img = cv2.rectangle(img,(int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])),colors[i%(len(colors))], 2)
        # cv2.imwrite("infer.jpg",img)
        print(each)



img_path = "demo/demo.jpg"
test_path = "/data/zy/dataset/voc/val.txt"
test_single(img_path)
#test_multi(test_path)
