#!/bin/bash
pwd
eval "$(conda shell.bash hook)"

cd BoT-SORT

conda activate dinov2

# python /home/user_219/omer_gil/BoT-SORT/tools/clustering.py

# # Define the folder paths
# folder1="/home/user_219/omer_gil/BoT-SORT/images_final_run"
# folder2="/home/user_219/omer_gil/BoT-SORT/images_clustered"

# # Define temporary folder path
# temp_folder="/home/user_219/omer_gil/BoT-SORT/temp_swap_folder"

# # Check if the temporary folder already exists
# if [ -d "$temp_folder" ]; then
#     echo "Temporary folder already exists. Please remove or rename it and try again."
#     exit 1
# fi

# # Rename the folders using a temporary folder
# mv "$folder1" "$temp_folder"
# mv "$folder2" "$folder1"
# mv "$temp_folder" "$folder2"


rm -r "/home/user_219/omer_gil/BoT-SORT/best_images_parts"
rm -r "/home/user_219/omer_gil/BoT-SORT/best_images_parts_embeddings"


# delete /home/user_219/omer_gil/BoT-SORT/best_images_parts
python tools/filter_and_get_best_bodypart.py /home/user_219/omer_gil/BoT-SORT/best_images_parts_embeddings

cd ../Self-Correction-Human-Parsing
cd Self-Correction-Human-Parsing
python get_best_foot_parts.py 
python human_parse_and_embed.py

cd ..
python BoT-SORT/tools/assign_ids.py

# (array([0, 1, 2, 3]), array([0, 1, 2, 5]))
# (array([0, 1, 2, 3]), array([0, 1, 2, 5]))