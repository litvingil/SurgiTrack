import os

pre_segment_paths = "/home/user_219/omer_gil/BoT-SORT/images"
post_segment_paths = "/home/user_219/omer_gil/BoT-SORT/images_segmented"
# batches_paths = "/home/user_219/omer_gil/BoT-SORT/images_feature_extracteds"
batches_paths = "/home/user_219/omer_gil/BoT-SORT/images_feature_extracted"

def count_images(path):
    count = 0 
    videos = os.listdir(path)
    for vid in videos:
        # Get all the folders in the video directory
        ids = os.listdir(os.path.join(path, vid))
        for id in ids:
            # Get all the images in the id directory
            images = os.listdir(os.path.join(path, vid, id))
            count += len(images)
    return count

# print("Counting images before segmentation...")
# count_pre_segment = count_images(pre_segment_paths)
# print("Counting images after segmentation...")
# count_post_segment = count_images(post_segment_paths)
# print("Counting batches...")
# count_batches = count_images(batches_paths)
# print("Number of images before segmentation: ", count_pre_segment)
# print("Number of images after segmentation: ", count_post_segment)
# print("Number of batches: ", count_batches)

pre_clustering = '/home/user_219/omer_gil/BoT-SORT/images'
post_clustering = '/home/user_219/omer_gil/BoT-SORT/images_clustered'

print("Counting images before clustering...")
print(count_images(pre_clustering))
print("Counting images after clustering...")
print(count_images(post_clustering))