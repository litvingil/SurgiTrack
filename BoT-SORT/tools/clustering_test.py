# import os
# import torch
# import torchvision.transforms as T
# import hubconf
# from PIL import Image
import numpy as np
from sklearn.cluster import AgglomerativeClustering


batch_size = 50

def list_files_recursive(directory):
    all_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    return all_files


def extract_features():
    input_folder = '/home/user_219/omer_gil/BoT-SORT/best_images_segmented/1'
    feature_folder = '/home/user_219/omer_gil/BoT-SORT/features_for_clustering'
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)
    # Load the largest dino model
    dino = hubconf.dinov2_vitb14()
    dino = dino.cuda()
    # Preprocess & convert to tensor
    transform = T.Compose([
              T.ToTensor(),
              T.Resize(560, interpolation=T.InterpolationMode.BICUBIC),
              T.CenterCrop(560),
              T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
    
    files_list = list_files_recursive(input_folder)
    all_embeddings = []
    files_num = len(files_list)
    batches = [list(range(i, min(i + batch_size, files_num))) for i in range(0, files_num, batch_size)]
    print("Number of batches: ", len(batches))
    batch_num = 0
    avg_features = np.zeros(0)
    for batch in batches:
        print("Batch number: ", batch_num)
        images = []
        for index in batch:
            img = Image.open(files_list[index]).convert("RGB") 
            img_array = np.array(img)
            images.append(img_array)

        # Use image tensor to load in a batch of images
        imgs_tensor = torch.zeros(batch_size, 3, 560, 560)

        for i, img in enumerate(images):
            imgs_tensor[i] = transform(img)[:3]

        # Inference
        with torch.no_grad():
            features_dict = dino.forward_features(imgs_tensor.cuda())
            features = features_dict['x_norm_patchtokens']

        print(features.shape)
        batch_num += 1
        all_embeddings.append(features.cpu().numpy())
    file_path = os.path.join(feature_folder, f'features.npy')
    # Save features vector:
    with open(file_path, 'wb') as f:
        np.save(f, all_embeddings[:files_num])

# run clustering

if __name__ == "__main__":
    all_embs = np.load('/home/user_219/omer_gil/BoT-SORT/features_for_clustering/features.npy')
    # import pdb; pdb.set_trace()
    clustering = AgglomerativeClustering(distance_threshold=80,n_clusters=None).fit(all_embs.reshape(76, 1600 * 768))
    import pdb; pdb.set_trace()
    print(clustering.labels_)
    # clustering = AgglomerativeClustering(n_clusters=k).fit(all_embs)
