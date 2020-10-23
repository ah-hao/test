import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min



os.chdir('./data/data_kmeans/')

name_list = os.listdir()

img_list = []

for n in name_list:
    img = cv2.imread(n, cv2.IMREAD_UNCHANGED)
    img_list.append(img)

img_list_np = np.array(img_list, dtype='float32')



img_list_np = img_list_np.reshape(len(img_list_np), -1)

print(img_list_np.shape)

img_list_np = img_list_np / 255.0

total_clusters = 3
# Initialize the K-Means model
kmeans = MiniBatchKMeans(n_clusters=total_clusters)
# Fitting the model to training set
kmeans.fit(img_list_np)


a = kmeans.labels_
unique, counts = np.unique(a, return_counts=True)
c = dict(zip(unique, counts))

print(c)

closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, img_list_np)


# cen = kmeans.cluster_centers_
# cen = cen.reshape((total_clusters, 32, 32))
# cen = cen * 255
# cen = np.array(cen, dtype='uint8')
# i = cen[1]

cv2.imshow('123', img_list[closest[2]])
cv2.waitKey(0)
cv2.destroyAllWindows()