from ensemble_boxes import *
import csv 
import argparse 

parser = argparse.ArgumentParser(description='ensemble') 
parser.add_argument('--csv_files', '--names-list', nargs='+', default=[])   # list of strings(file names)
parser.add_argument('--weights', nargs='+', type=int)                       # list of integers 
parser.add_argument('--num', type=int, default=0)
parser.add_argument('--skip_box_thresh', type=float, default=0.3)
parser.add_argument('--IoU_thresh', type=float, default=0.55)

args = parser.parse_args() 


# open inference files 
inference = [] 
for file in args.csv_files: 
    f = open(file, 'r')
    data = list(csv.reader(f, delimiter=","))
    inference.append(data)
    f.close()

# get weights 
weights = list(args.weights)


# divide inference into label / score / box lists
num_files = len(inference)
num_img = len(inference[0])

iou_thresh = 0.55
skip_box_thresh = 0.3

wbf_label, wbf_score, wbf_box = [], [], []
for i in range(1, num_img): # for number of images
    label, score, box = [[]]*num_files, [[]]*num_files, [[]]*num_files

    for j in range(num_files): # for number of inference files 
        one_img = inference[j][i][0].split(' ') # 0th = numbers, 1st = img name 

        for k in range(0, len(one_img)-1, 6): 
            label[j].append(int(one_img[k]))
            score[j].append(float(one_img[k+1]))
            box[j].append(([float(one_img[w])/1024. for w in range(k+2, k+6)]))  # normalize for WBF library 

    # print(label, score, box)
    b, s, l = weighted_boxes_fusion(box, score, label, weights=weights, iou_thr=iou_thresh, skip_box_thr=skip_box_thresh)

    wbf_label.append(l) 
    wbf_score.append(s) 
    wbf_box.append(b)

# make a submission file in a csv format 
f = open('ensemble_file_{}.csv'.format(args.num), 'w') 
writer = csv.writer(f) 
writer.writerow(['PredictionString', 'image_id'])
for i in range(num_img-1): 
    s = ''
    for j in range(len(wbf_label[i])): 
        s += str(int(wbf_label[i][j])) + ' '
        s += str(wbf_score[i][j]) + ' '
        for k in range(4): 
            s += str(wbf_box[i][j][k] * 1024) + ' '

    writer.writerow([s, 'test/{0:04d}.jpg'.format(i)])




