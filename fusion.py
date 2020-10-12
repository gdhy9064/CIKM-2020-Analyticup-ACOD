import shutil
t_lst=['t3_0.80', 't4_0.80', 't5_0.80', 't5_0.70', 't6_0.70']
yolo_lst = []
rcnn_lst = []

if not os.path.exists('images'):
    os.makedirs('images')

for t in t_lst:
    with open(f'whitebox_rcnn_overall_score_{t}.json') as f:
        rcnn_lst.append(eval(f.read()))
    with open(f'whitebox_yolo_overall_score_{t}.json') as f:
        yolo_lst.append(eval(f.read()))

lst = yolo_lst[1].keys()
for i in lst:
    max_score = 0
    max_t = 0
    for j in range(len(t_lst)):
        if rcnn_lst[j].get(i,0) + yolo_lst[j].get(i,0) > max_score:
            max_score = rcnn_lst[j].get(i, 0) + yolo_lst[j].get(i,0)
            max_t=t_lst[j]

    shutil.copy(f'{max_t}/{i}', 'images/')