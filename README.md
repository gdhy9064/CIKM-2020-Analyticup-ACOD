# The Solution of Alibaba-Tsinghua Adversarial Challengeon Object Detection
## Introduction
This repository contains the solution of [Alibaba-Tsinghua Adversarial Challenge on Object Detection in CIKM 2020 Analyticup](http://www.cikm2020.org/adversarial-challenge-on-object-detection/).

## File List
- eval_code: Modified source of YOLO-V4 and official evaluation code. Because of the file size limitation of github, you may download [offical source](https://tianchi-competition.oss-accelerate.aliyuncs.com/531806/data.zip), manually uncompress the `eval_code.zip` and copy the directory `eval_code/models` to the same place in our source, and then uncompress the `images.zip` to `eval_code/select1000_new/` with directory `images` retained.
- mmdetection: Modified source of mmdetection, including the specified Faster-RCNN. 
- eval_code.diff: A file shows the differences between original source and modified source in `eval_code`. 
- mmdetection.diff: A file shows the differences between original source and modified source in `mmdetection`. 
- train.py: Adversarial training with selected parameters.  
- metrics.py: Evaluate the attack score of Faster-RCNN and YOLO-V4, generate the files required by `fusion.py`.  
- fusion.py: Make a fusion of results with selected parameters for higher score.








