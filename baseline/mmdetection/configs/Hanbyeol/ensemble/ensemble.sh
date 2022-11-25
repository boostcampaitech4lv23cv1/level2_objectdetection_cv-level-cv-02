#!/bin/bash 

python ensemble.py \
    --csv_files \
        data_paths/output_1_gyubo.csv \
        data_paths/output_2_gyubo.csv \
    --weights 2 1 \
    --num 5 \
    --skip_box_thresh 0.05 \
    --IoU_thresh 0.55