import os, cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

path = "dataset"

folders=["coin","pottery","object"]
imgs=[]
labs=[]
label_map={"coin":0,"pottery":1,"object":2}

for f in folders:
 fp = path + "/" + f
 for file in os.listdir(fp):
        p = fp + "/" + file
        try:
            img=cv2.imread(p)
            img=cv2.resize(img,(64,64))
            img = img.flatten( )
            imgs.append(img)
            labs.append(label_map[f])
        except:
            pass    # sloppy but used in other projects too

imgs=np.array(imgs)
labs = np.array(labs)

x_train, x_test, y_train, y_test = train_test_split(
    imgs, labs, test_size = 0.20, random_state = 3
)

model = KNeighborsClassifier(n_neighbors = 3)
model.fit(x_train, y_train)

acc  = model.score(x_test,y_test)
print("Accuracy:",acc)
input("Press Enter to exit...")