import os, glob
import numpy as np
import pandas as pd
import faiss

feature_layer = "res5c_branch2c"
model_name = "resnet50"

d = 512                           # dimension
xb = np.load("output/resnet50_feature_{}_index.npy".format(feature_layer))
xq = np.load("output/resnet50_feature_{}_query.npy".format(feature_layer))
# print(xb.shape)
# print(xq.shape)

# build a flat (CPU) index
index_flat = faiss.IndexFlatL2(d)

index_flat.add(xb)         # add vectors to the index
# print(index_flat.ntotal)

k = 100                          # we want to see 4 nearest neighbors
D, I = index_flat.search(xq, k)  # actual search
# print(I[:5])                   # neighbors of the 5 first queries
# print(I[-5:])                  # neighbors of the 5 last queries

np.save("output/I_{}.npy".format(feature_layer), I)
np.save("output/D_{}.npy".format(feature_layer), D)

### make submission
index_path = "input/index/"
index_list = sorted(glob.glob(index_path + "*")) # 1091756
index_list = pd.DataFrame(index_list, columns=['id'])
index_list['id'] = index_list['id'].apply(lambda x: os.path.basename(x)[:-4])
index_list = np.array(index_list['id'])
query_path = "input/query/"
query_list = sorted(glob.glob(query_path + "*")) # 114943

sub = pd.DataFrame(query_list, columns=['id'])
sub['id'] = sub['id'].apply(lambda x: os.path.basename(x)[:-4])

images_list =  index_list[I]
images_list = images_list + " "
images_list = np.sum(images_list, axis=1)

sub['images'] = images_list
sub2 = pd.read_csv("input/sample_submission.csv")
sub2['images'] = ""
sub = pd.concat([sub, sub2])
sub = sub.drop_duplicates(['id'])
sub.to_csv("output/sub_resnet50_pred.csv".format(model_name, feature_layer), index=None)