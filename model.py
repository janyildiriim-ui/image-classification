import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DATASET_PATH = "dataset"

folders = ["coin", "pottery", "object"]
images = []
labels = []
label_map = {"coin": 0, "pottery": 1, "object": 2}

for folder in folders:
    folder_path = DATASET_PATH + "/" + folder
    for file in os.listdir(folder_path):
        img_path = folder_path + "/" + file
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))
            img = img.flatten()
            images.append(img)
            labels.append(label_map[folder])
        except:
            pass

images = np.array(images)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=3
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print("Accuracy:", acc)