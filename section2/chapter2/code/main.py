from sklearn.datasets import fetch_20newsgroups
import seaborn as sns
import matplotlib.pyplot as plt

# Dữ liệu chỉ tải một lần rồi lưu lại để có thể sử dụng lần sau
dataset = fetch_20newsgroups()

# Các thuộc tính của dữ liệu
#? print(dataset.keys())
#* dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR'])

# Danh sách các nhóm tin
#? print(dataset['target_names'])
#* ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

# Hiện nhóm tin dưới dạng số mà tin của tất cả các tin
# Các nhóm tin được đánh số từ 0 đến 19
#? print(dataset['target'])
#* [7 4 4 ... 3 1 8]

# Vẽ đồ thị với dữ liệu dataset
# Số tin tức tương ứng với mỗi nhóm tin
sns.distplot(dataset.target)
plt.show()


