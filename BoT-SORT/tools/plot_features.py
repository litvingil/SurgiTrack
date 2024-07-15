from __future__ import print_function
import time

import numpy as np

from sklearn.manifold import TSNE
import umap

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import os

from tqdm import tqdm

# vid_ids = [1, 2, 4]
vid_ids = [1]
def main_gil():
    data3d = {}
    features_folder = '/home/user_219/omer_gil/BoT-SORT/images_feature_extracted'
    for vid_id in vid_ids:
        current_vid_folder =os.path.join(features_folder, str(vid_id))
        ids_dir = [f for f in os.listdir(current_vid_folder) if os.path.isdir(os.path.join(current_vid_folder, f))]
        for id in tqdm(ids_dir):
            data = []
            if int(id) < 23:
                print("ID: ", id)
                # import pdb; pdb.set_trace()
                for f in os.listdir(os.path.join(current_vid_folder, str(id))):
                    current_data = np.load(os.path.join(current_vid_folder, str(id), f))
                    nsamples, nx, ny = current_data.shape
                    current_data = current_data.reshape(nsamples, -1)
                    data.append(current_data)
                data = np.array(data)
                print("Data pre concat: ", data.shape)
                data = np.concatenate(data, axis=0)
                print("Data post concat: ", data.shape)
                tsne = TSNE(n_components = 3, random_state = 0)
                transformed_vectors = tsne.fit_transform(data)
                print("Transformed vectors: ", transformed_vectors.shape)
                data3d[id] = transformed_vectors
    # Plot the features:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for id in data3d:
        data = data3d[id]
        color = np.random.rand(3,)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], color=color, label=id)
    plt.title('Features')
    plt.savefig("Features_gil23.png")

def main_gil_TSNE():
    data3d = {}
    features_folder = '/home/user_219/omer_gil/BoT-SORT/images_feature_extracted'
    for vid_id in vid_ids:
        target = []
        current_vid_folder =os.path.join(features_folder, str(vid_id))
        ids_dir = [f for f in os.listdir(current_vid_folder) if os.path.isdir(os.path.join(current_vid_folder, f))]
        data = []
        for id in tqdm(ids_dir):
            count = 0
            if int(id) < 23:
                print("ID: ", id)
                import pdb; pdb.set_trace()
                for f in os.listdir(os.path.join(current_vid_folder, str(id))):
                    current_data = np.load(os.path.join(current_vid_folder, str(id), f))
                    nsamples, nx, ny = current_data.shape
                    current_data = current_data.reshape(nsamples, -1)
                    data.append(current_data)
                    count += nsamples
            target +=[int(id)]*count
        import pdb; pdb.set_trace()
        data = np.array(data)
        print("Data pre concat: ", data.shape)
        data = np.concatenate(data, axis=0)
        print("Data post concat: ", data.shape)
        tsne = TSNE(n_components = 3, random_state = 0)
        transformed_vectors = tsne.fit_transform(data)
        print("Transformed vectors: ", transformed_vectors.shape)
        data3d = transformed_vectors
    import pdb; pdb.set_trace()
    # Plot the features:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data3d[:, 0], data3d[:, 1], data3d[:, 2],c=target ,cmap='Spectral', label=id)
    plt.title('Features')
    plt.savefig("Features_gil12.png")



def main_gil_UMAP():
    data3d = {}
    features_folder = '/home/user_219/omer_gil/BoT-SORT/images_feature_extracted'
    for vid_id in vid_ids:
        target = []
        current_vid_folder =os.path.join(features_folder, str(vid_id))
        ids_dir = [f for f in os.listdir(current_vid_folder) if os.path.isdir(os.path.join(current_vid_folder, f))]
        data = []
        for id in tqdm(ids_dir):
            count = 0
            if int(id) < 23:
                print("ID: ", id)
                # import pdb; pdb.set_trace()
                for f in os.listdir(os.path.join(current_vid_folder, str(id))):
                    current_data = np.load(os.path.join(current_vid_folder, str(id), f))
                    nsamples, nx, ny = current_data.shape
                    current_data = current_data.reshape(nsamples, -1)
                    data.append(current_data)
                    count += nsamples
            target +=[int(id)]*count
        import pdb; pdb.set_trace()
        data = np.array(data)
        print("Data pre concat: ", data.shape)
        data = np.concatenate(data, axis=0)
        print("Data post concat: ", data.shape)
        reducer = umap.UMAP(a=None, angular_rp_forest=False, b=None,
        force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
        local_connectivity=1.0, low_memory=False, metric='euclidean',
        metric_kwds=None, min_dist=0.1, n_components=3, n_epochs=None,
        n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
        output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
        set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
        target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
        transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)
        transformed_vectors = reducer.fit_transform(data)
        print("Transformed vectors: ", transformed_vectors.shape)
        data3d = transformed_vectors
    import pdb; pdb.set_trace()
    # Plot the features:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data3d[:, 0], data3d[:, 1], data3d[:, 2],c=target ,cmap='Spectral', label=id)
    plt.title('Features')
    plt.savefig("Features_UMAP.png")

def main():
    data = []
    features_folder = '/home/user_219/omer_gil/BoT-SORT/images_feature_extracted'
    for vid_id in range(1, 4):
        current_vid_folder =os.path.join(features_folder, str(vid_id))
        ids_dir = [f for f in os.listdir(current_vid_folder) if os.path.isdir(os.path.join(current_vid_folder, f))]
        for id in ids_dir:
            for f in os.listdir(os.path.join(current_vid_folder, str(id))):
                current_data = np.load(f)
                nsamples, nx, ny = current_data.shape
                current_data = current_data.reshape((nsamples * nx * ny))
                data = data.append(current_data)
    # data1_2 = np.load("/home/user_219/omer_gil/BoT-SORT/2/1/features.npy")
    # nsamples, nx, ny = data1_2.shape
    # data1_2 = data1_2.reshape((nsamples * nx * ny))
    # data2_2 = np.load("/home/user_219/omer_gil/BoT-SORT/2/2/features.npy")
    # nsamples, nx, ny = data2_2.shape
    # data2_2 = data2_2.reshape((nsamples * nx * ny))
    # data4_2 = np.load("/home/user_219/omer_gil/BoT-SORT/2/4/features.npy")
    # nsamples, nx, ny = data4_2.shape
    # data4_2 = data4_2.reshape((nsamples * nx * ny))
    # data1_3 = np.load("/home/user_219/omer_gil/BoT-SORT/3/1/features.npy")
    # nsamples, nx, ny = data1_3.shape
    # data1_3 = data1_3.reshape((nsamples * nx * ny))
    # data2_3 = np.load("/home/user_219/omer_gil/BoT-SORT/3/2/features.npy")
    # nsamples, nx, ny = data2_3.shape
    # data2_3 = data2_3.reshape((nsamples * nx * ny))
    # data4_3 = np.load("/home/user_219/omer_gil/BoT-SORT/3/4/features.npy")
    # nsamples, nx, ny = data4_3.shape
    # data4_3 = data4_3.reshape((nsamples * nx * ny))
    # data1_4 = np.load("/home/user_219/omer_gil/BoT-SORT/4/1/features.npy")
    # nsamples, nx, ny = data1_4.shape
    # data1_4 = data1_4.reshape((nsamples * nx * ny))
    # data2_4 = np.load("/home/user_219/omer_gil/BoT-SORT/4/2/features.npy")
    # nsamples, nx, ny = data2_4.shape
    # data2_4 = data2_4.reshape((nsamples * nx * ny))
    # data4_4 = np.load("/home/user_219/omer_gil/BoT-SORT/4/4/features.npy")
    # nsamples, nx, ny = data4_4.shape
    # data4_4 = data4_4.reshape((nsamples * nx * ny))
    data = np.vstack((data))
    tsne = TSNE(n_components = 3, random_state = 0)
    transformed_vectors = tsne.fit_transform(data)
    # colors = ["r", "g", "b", "y", "m", "r","r", "g", "r"]
    # Plot the features:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(transformed_vectors)):
        ax.scatter(transformed_vectors[:, 0], transformed_vectors[:, 1], transformed_vectors[:, 2])
    plt.title('Features')
    plt.savefig("Features.png")

if __name__ == "__main__":
    main_gil_UMAP()