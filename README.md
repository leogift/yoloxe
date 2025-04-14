# YOLOXE
## Introduction
This repository is a modification version of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), dealing with both detection and keypoints estimation and supporting rknn models.

Kudos to the YOLOX team.

## Complaint
After my three-year struggling with YOLOXE based on YOLOX, I decide to give it up and start using the YOLO framework of Ultralytics.
Reality has proven that no matter how hard one tries, it can't match the effectiveness of a well-functioning team. 

The output for keypoints still employs traditional regression methods, suitable for rigid bodies with fixed relative positions among keypoints but not ideal for dynamic points like pointers or joints - an inherent issue of regression. I recommend to use a two-stage model instead in these dynamic situtaions.

I have also implemented multi-layer IPM on the basis of this YOLOXE, and will release it later.

## Modification
- use anchors/scales, which performs better without DFL
- use regnet as backbone
- use ReLU as activation
- including keypoints dataloader/eval and estimation head
- including aux head
- including fusion IoU loss
- including uncertainty balance
- including rknn exporting tools
- including grad accumulation training

* For more operations, please refer to the files under the 'exps' directory.

## Quick Start

<details>
<summary>Install</summary>

```shell
pip3 install -r requirements.txt
```
</details>

<details>
<summary>Train</summary>

Step1. Prepare dataset
```shell
cd <YOLOXE_HOME>
ln -s /path/to/your/COCO ./datasets/COCO
```

Step2. Train on COCO:

```shell
python3 -m yoloxe.tools.train -f exps/active/yoloxe_s_coco.py -b32
```

* -f: exp config file
* -b: total batch size

</details>


<details>
<summary>Export</summary>

Use the following command:
```shell
python3 -m yoloxe.tools.export_onnx --output-name onnx_outputs/test.onnx -f exps/active/yoloxe_s_coco.py -c yoloxe_s.pth
```

* -output-name: output onnx file path.
* -f: exp config file
* -c: ckpt file

</details>


<details>
<summary>Export RKNN</summary>

Use the following command:
```shell
python3 -m yoloxe.tools.onnx2rknn -m onnx_outputs/test.onnx --bgr
```

* -m: onnx model file
* --bgr: bgr mode

</details>
