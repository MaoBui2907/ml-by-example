from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
# ! Chuẩn bị dữ liệu
# * load dữ liệu gồm 3 class, 150 samples, mỗi sample là vector 4 chiều
dataset = datasets.load_iris()
# * Láy ra dữ liệu chỉ gồm 2 thuộc tính cuối
X = dataset.data[:,2:4]
# * Lấy ra nhãn của từng sample (0, 1 hoặc 2)
Y = dataset.target
# print(Y)
# ! Trực quan hóa dữ liệu
# * gom các dữ liệu cùng nhãn
y_0 = np.where(Y==0)
plt.scatter(X[y_0,0], X[y_0,1])

y_1 = np.where(Y==1)
plt.scatter(X[y_1,0], X[y_1,1])

y_2 = np.where(Y==2)
plt.scatter(X[y_2,0], X[y_2,1])

# ! 1. Chọn K
k = 3

# ! 2. Chọn centroid mặc đinh
random_index = np.random.choice(range(len(X)), k)
centroids = X[random_index]
plt.scatter(centroids[:,0], centroids[:,1])

# ! Gắn nhãn dựa vào centroid
cluster = []
for i in X:
    cluster.append(np.argmin([np.linalg.norm(i-j) for j in centroids]))
print(cluster)
for i in range(k):
    print(i)
    clusters_i = np.where(cluster == i)
    print(X[clusters_i])
    centroids[i] = np.mean(X[clusters_i],axis=0)
plt.show()