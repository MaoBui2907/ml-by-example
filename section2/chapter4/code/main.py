import glob
import os
from collections import defaultdict

import numpy as np

from nltk.corpus import names
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer

def is_letter_only(word):
    return word.isalpha()


mails = []
labels = []

# ? Đọc dữ liệu, spam có nhãn 1
file_path = "../tmp/enron1/spam/"
for mail in glob.glob(os.path.join(file_path, '*.txt')):
    with open(mail, 'r', encoding="ISO-8859-1") as file:
        mails.append(file.read())
        labels.append(1)
        pass

file_path = "../tmp/enron1/ham/"
for mail in glob.glob(os.path.join(file_path, '*.txt')):
    with open(mail, 'r', encoding="ISO-8859-1") as file:
        mails.append(file.read())
        labels.append(0)
        pass

# print(len(mails))
# print(len(labels))

# ? Làm sạch dữ liệu
names = set(names.words())
lemmatizer = WordNetLemmatizer()
cleaned_mails = []
for mail in mails:
    tokens = mail.split()
    features = [lemmatizer.lemmatize(token) for token in tokens if is_letter_only(token) and token not in names]
    cleaned_mail = " ".join(features)
    cleaned_mails.append(cleaned_mail)
# print(cleaned_mails)

feature_size = 1000

# ? Tạo bag of words
cv = CountVectorizer(stop_words="english", max_features=feature_size, max_df=0.5, min_df=2)
mails_cv = cv.fit_transform(cleaned_mails)
# print(mails_cv[0].todense())

words_bag = cv.get_feature_names()
# print(words_bag)

# ? Tính prior
label_index = defaultdict(list)
for index, label in enumerate(labels):
    label_index[label].append(index)
prior = {}
for label in label_index:
    prior[label] = len(label_index[label])/float(len(labels))

# print(prior)

# ? Tính likelihood

label_group = defaultdict(list)
for label in label_index:
    for ind in label_index[label]:
        label_group[label].append(mails_cv[ind].todense())
    label_group[label] = np.array(label_group[label]).resize(len(label_index[label]),feature_size)
print(label_group[0])

exit()

label_word = {}
for label in label_index:
    word_count = []
    for ind in label_index[label]:
        word_count.append(mails_cv[ind].sum(axis=1))
    label_word[label] = int(np.array(word_count).sum(axis=0))

# print(label_word)
laplace = 1

