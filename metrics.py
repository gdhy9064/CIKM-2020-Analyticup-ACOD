
import sys
sys.path.insert(0, './eval_code')
sys.path.insert(0, './mmdetection')
from eval import *

t = 't3_0.80'
MAX_TOTAL_AREA_RATE = 0.02  # 5000/(500*500) = 0.02
selected_path = f'{t}/'
max_patch_number = 10
output_dir = '.'

# compute_connected_domin_score
cd_json_name = f'connected_domin_score_{t}.json'
count_connected_domin_score(MAX_TOTAL_AREA_RATE, selected_path, max_patch_number, cd_json_name, output_dir)

# compute_boundingbox_score
bb_json_name = f'whitebox_rcnn_boundingbox_score_{t}.json'
whitebox_yolo_result = f'whitebox_rcnn_overall_score_{t}.json'
count_detection_score_fasterrcnn(selected_path, bb_json_name, output_dir)
compute_overall_score(cd_json_name, bb_json_name, output_dir, whitebox_yolo_result)


bb_json_name = f'whitebox_yolo_boundingbox_score_{t}.json'
whitebox_rcnn_result = f'whitebox_yolo_overall_score_{t}.json'
count_detection_score_yolov4(selected_path, bb_json_name, output_dir)
compute_overall_score(cd_json_name, bb_json_name, output_dir, whitebox_rcnn_result)