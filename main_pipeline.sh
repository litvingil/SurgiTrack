#!/bin/bash
pwd
eval "$(conda shell.bash hook)"
conda activate botsort_env

# source /home/user_219/miniconda3/condabin/conda
# conda activate botsort_env

video_paths=$(ls ./videos)
cd BoT-SORT
pwd
#for video_path in $video_paths; do
    # Run the Python script with the video file path as an argument
#    python tools/demo.py video --path "../videos/$video_path" -f yolox/exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --with-reid --fuse-score --fp16 --fuse --save_result
    # echo "./videos/$video_path"
#done

conda activate dinov2
python tools/filter_and_get_best_bodypart.py

cd ../Self-Correction-Human-Parsing
cd Self-Correction-Human-Parsing
python get_best_foot_parts.py 
python human_parse_and_embed.py

cd ..
python BoT-SORT/tools/assign_ids.py

# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 14, 15]), array([12, 14,  8,  1,  0,  7,  3,  9, 10, 11,  5, 13,  2,  4,  6]))
# (array([ 1, 10, 11, 14, 16, 17, 18, 20, 21,  4,  5,  6,  7,  8,  9]), array([ 6,  8, 23, 12,  1, 21, 14, 24, 25,  5,  2,  7, 13, 16, 20]))