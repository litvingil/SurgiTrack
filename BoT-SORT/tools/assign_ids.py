import os
from scipy.optimize import linear_sum_assignment
import numpy as np
import copy
from scipy.spatial.distance import cosine
from tqdm import tqdm
from sys import maxsize

# def create_dist_matrix():
#     max_int = 100
#     n = 5
#     cost_matrix = np.random.randint(max_int, size=(n,n))
#     print(cost_matrix)
#     return cost_matrix

def compute_cosine_dist(mat_a, mat_b):
    # Step 1: Compute the dot product of A and B
    dot_product = np.dot(mat_a, mat_b.T)

    # Step 2: Compute the norms of the rows of A and B
    norms_A = np.linalg.norm(mat_a, axis=1)
    norms_B = np.linalg.norm(mat_b, axis=1)

    # Reshape norms_A and norms_B to allow broadcasting
    norms_A = norms_A[:, np.newaxis]
    norms_B = norms_B[np.newaxis, :]

    # Step 3: Calculate the cosine similarity
    cosine_similarity = dot_product / (norms_A * norms_B)

    # Step 4: Convert cosine similarity to cosine distance
    cosine_distance = 1 - cosine_similarity
    return cosine_distance
    
def create_dist_matrix():
    features_root_path = '/home/user_219/omer_gil/BoT-SORT/best_images_parts_embeddings/'
    # I can make it smarter, but for now, let's assume there are at most 100 ID's:
    num_ids = 39
    cameras = [1, 2]
    embeddings_1 = [] #np.zeros((0, 4, 1600, 768))
    embeddings_2 = [] #np.zeros((0, 4, 1600, 768))
    ids_1 = []
    ids_2 = []
    for camera in cameras: # sorted(os.listdir(features_root_path)):
        for id in tqdm(sorted(os.listdir(os.path.join(features_root_path, str(camera))))):
            current_id_folder = os.path.join(features_root_path, str(camera), str(id))
            embed_dict = np.load(os.path.join(current_id_folder, "features.npy"))
            # shape is (4, 1600, 768)
            if camera == 1:
                # import pdb; pdb.set_trace()
                embeddings_1.append(embed_dict)
                ids_1.append(id)
            if camera == 2:
                embeddings_2.append(embed_dict)
                ids_2.append(id)
                
    # Now, we have an array of all embedding dicts. 
    # Compute cosine distances sum, between all embedding dicts, and insert into matrix:
    embeddings_1 = np.stack(embeddings_1, axis = 0)
    embeddings_2 = np.stack(embeddings_2, axis = 0)
    dist_matrix = np.zeros((len(ids_1), len(ids_2)))
    # import pdb; pdb.set_trace()
    for i, dict1 in tqdm(enumerate(embeddings_1)):
        for j, dict2 in enumerate(embeddings_2):
            total_distance = 0
            for part_idx in range(dict1.shape[0]):
                if dict1[part_idx].all() == 0 or dict2[part_idx].all() == 0:
                    # Add 0.5 to account for missing parts, but don't give unfair advantage:
                    total_distance += 0.5
                    continue
                total_distance += compute_cosine_dist(dict1[part_idx], dict2[part_idx])
            # print("i = ", i, ", j = ", j)
            dist_matrix[i,j] = np.sum(total_distance)
                
    # print(dist_matrix)
    return dist_matrix, ids_1, ids_2

def main():
    dist_mat, ids_1, ids_2 = create_dist_matrix()
    assignment = linear_sum_assignment(dist_mat)
    print(assignment)
    assignment_ammended = assignment
    # import pdb; pdb.set_trace()
    for i, assign in enumerate(assignment[0]):
        assignment_ammended[0][i] = int(ids_1[assign])
    for i, assign in enumerate(assignment[1]):
        assignment_ammended[1][i] = int(ids_2[assign])
    print(assignment_ammended)
    return 0

if __name__ == "__main__":
    main()
    
# 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38
# 0, 24, 17, 1, 5, 21, 6, 13, 32, 3, 26, 4, 16, 7, 22, 9, 10, 11, 2, 18, 15, 8, 29, 37, 12, 25, 19, 20, 23, 27, 28, 30, 31, 33, 34, 35, 36, 14, 38
# ([ 1, 10, 12, 13, 14, 18, 21, 22, 23, 24, 25,  3, 31, 37, 38,  4,  5]), array([ 1, 12, 13, 14, 16,  2, 21, 22, 24, 25, 26, 30, 32,  5,  6,  7,  8]))
# ([ 1, 10, 12, 13, 14, 18, 21, 22, 23, 24, 25,  3, 31, 37, 38,  4,  5]), array([ 1, 12, 13, 14, 16,  2, 21, 22, 24, 25, 26, 30, 32,  5,  6,  7,  8]))

# [ 1, 10, 11, 14, 16, 17, 18, 20, 21,  3,  4,  5,  6,  7,  8,  9]
# [53, 57, 47, 56,  1, 21, 12, 61, 24, 54, 58,  2, 59, 13, 36, 55]
# [ 1,  1,0.5,  1,  0,  1,  0,  0,  1,  1, ... ]