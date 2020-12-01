import json 
import os
import cv2
ds_path = "/data/zy/efficientdet/names/vg_names.txt"
with open(ds_path, "r") as f:
    ols = [int(each.strip()) for each in f.readlines()]

vg_json = json.load(open("/data/zy/dataset/vg/annotations/objects_val.json", "r"))
coco_json = json.load(open("/data/zy/dataset/cocods/annotations/instances_val2017.json", "r"))
vg_ann_json = []
vg_img_json  = []
vg_cate_json = []
vg_lic_json  = coco_json["licenses"]
vg_inf_json  = coco_json["info"]
index_list = []

for index, each_img_json in enumerate(vg_json):
    img_id = each_img_json["image_id"]
    width  = each_img_json["width"]
    height = each_img_json["height"]
    img = cv2.imread("/data/zy/dataset/vg/img/"+str(img_id)+".jpg")
    h, w = img.shape[:-1]
    if h!=height or width != w:
        print("h,w not fixed",img_id)
        continue
    num_bboxes = 0
    for each_bbox_json in each_img_json["objects"]:
        box = [each_bbox_json["x"], each_bbox_json["y"], each_bbox_json["w"], each_bbox_json["h"]]
        x,y =box[:2]
        if float(box[2] * box[3]) < 1:
            continue
        if x>width or y>height or x<0 or y<0:
            print("h, w out", img_id)
            continue
        id = ols.index(each_bbox_json["category_id"])
        if id >=36:
            continue
        num_bboxes += 1
        _json = {
            'image_id': img_id,
            'id':each_bbox_json["object_id"],
            'category_id': id,
            'bbox': box,
            'area':float(box[2] * box[3]),
            "iscrowd": 0
        }
        vg_ann_json.append(_json)
    if num_bboxes > 0 :
        index_list.append(index)
        vg_img_json.append(
                        {"id":img_id, 
                        "width":width, 
                        "height":height, 
                        "file_name":str(img_id)+".jpg"}
                    )
    if num_bboxes == 0:
        print(img_id,"no bboxes")
for i in range(len(ols)):
    vg_cate_json.append({"supercategory":"none",
                          'name':str(ols[i]),
                          "id":i})
coco_json_data = {
    'annotations':vg_ann_json, 
    'info':vg_inf_json, 
    'images':vg_img_json, 
    'licenses':vg_lic_json, 
    'categories':vg_cate_json
}
with open("cocoformat_vgval_rare.json","w+") as f:
    json.dump(coco_json_data, f)
print(index_list)

