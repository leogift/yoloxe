import cv2
import json
import numpy as np

__all__ = [
    "COCO2Anchors",
]

def COCO2Anchors(coco_json_path, input_shape, K):
    with open(coco_json_path, "r") as f:
        j = json.load(f)
        f.close()
    if 'images' not in list(j.keys()) \
        or 'annotations' not in list(j.keys()):
            return None
        
    images = j['images']
    annotations = j['annotations']

    anchor_list = []
    whratio_list = []
    for anno in annotations:
        image_id = anno['image_id']
        flag_found_img = False
        img_height, img_width = 0, 0
        img_file_name = ""
        for img in images:
            if img['id'] == image_id:
                flag_found_img = True
                img_width= img['width']
                img_height = img['height']
                img_file_name = img['file_name']
                break
        if img_height<=0 or img_width<=0 or len(img_file_name)<=4:
            continue
        if flag_found_img:
            anchor_w = anno['bbox'][2] * input_shape[1] / img_width
            anchor_h = anno['bbox'][3] * input_shape[0] / img_height
            if(anchor_w<=0 or anchor_h<=0):
                print("bug:",anno,img)
                continue

            anchor_list.extend([anchor_w, anchor_h])
            whratio_list.append(anchor_w/anchor_h)
            
    whratio_array = np.array(whratio_list, np.float32)
    print("W/H Ratio:", np.mean(whratio_array))

    scale_data = np.array(anchor_list, np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS +cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    compactness, labels, centers = cv2.kmeans(scale_data, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    anchors = centers.squeeze().tolist()
    anchors.sort()
    return anchors
