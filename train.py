import numpy as np
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.impute import SimpleImputer;
from sklearn.compose import ColumnTransformer;
from sklearn.pipeline import Pipeline;
from sklearn.preprocessing import LabelEncoder;
from sklearn.preprocessing import StandardScaler;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression ;
from sklearn.linear_model import LogisticRegression;
from sklearn.linear_model import Ridge, Lasso;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import r2_score;
from sklearn.preprocessing import PolynomialFeatures;
from sklearn.svm import SVR;
from sklearn.svm import SVC;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.tree import DecisionTreeRegressor;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.neighbors import KNeighborsRegressor;
from sklearn.naive_bayes import GaussianNB;
import xgboost as xgb;
from xgboost import XGBClassifier;
from xgboost import XGBRegressor;

import tensorflow as tf
import keras;
from keras_preprocessing import image;
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam;
from keras.callbacks import ModelCheckpoint;
from keras.models import Sequential;
from tensorflow.keras.applications import VGG16;
from tensorflow.keras.applications import InceptionResNetV2;
from keras.applications.vgg16 import preprocess_input;
from tensorflow.keras.applications.vgg16 import decode_predictions;
from tensorflow.keras.callbacks import EarlyStopping;

import os;
from os import listdir;
from PIL import Image as PImage;
import cv2

from tqdm import tqdm

We have imported the libraries.

Its Time to export the data
Data

train = pd.read_csv('../input/identify-snake-breed-hackerearth/dataset/train.csv');
test = pd.read_csv('../input/identify-snake-breed-hackerearth/dataset/test.csv')

How many types of breeds are there?

train['breed'].unique()

array(['nerodia-erythrogaster', 'pantherophis-vulpinus',
       'thamnophis-sirtalis', 'pantherophis-obsoletus',
       'agkistrodon-contortrix', 'crotalus-atrox',
       'lampropeltis-triangulum', 'crotalus-horridus', 'crotalus-ruber',
       'heterodon-platirhinos', 'nerodia-sipedon', 'thamnophis-elegans',
       'thamnophis-marcianus', 'crotalus-viridis', 'nerodia-fasciata',
       'haldea-striatula', 'storeria-dekayi', 'agkistrodon-piscivorus',
       'nerodia-rhombifer', 'storeria-occipitomaculata',
       'thamnophis-radix', 'coluber-constrictor', 'natrix-natrix',
       'diadophis-punctatus', 'masticophis-flagellum',
       'pantherophis-spiloides', 'rhinocheilus-lecontei',
       'lampropeltis-californiae', 'pituophis-catenifer',
       'opheodrys-aestivus', 'pantherophis-guttatus',
       'pantherophis-alleghaniensis', 'thamnophis-proximus',
       'pantherophis-emoryi', 'crotalus-scutulatus'], dtype=object)

train.head()

	image_id 	breed
0 	a8b3ad1dde 	nerodia-erythrogaster
1 	8b492b973d 	pantherophis-vulpinus
2 	929b99ea92 	thamnophis-sirtalis
3 	bbac7385e2 	pantherophis-obsoletus
4 	ef776b1488 	agkistrodon-contortrix
X

We have images in train folder. And, we have name of those breeds in the csv format. We have to merge between these two to make the machine understand which image belongs to which breed.

Lets run a simple for loop

img_width = 150;
img_height = 150;
x=[];

for i in tqdm (range(train.shape[0])):
    path = '../input/identify-snake-breed-hackerearth/dataset/train/' + train['image_id'][i] + '.jpg'
    img = image.load_img(path, target_size = (img_width, img_height, 3))
    img = image.img_to_array(img)
    img = img/255.0
    x.append(img)
    
#x = np.array(x);
#rm path

100%|██████████| 5508/5508 [00:16<00:00, 338.98it/s]

Nice, the images are extracted. We have got our x. Now we need y

 y = train.drop(columns = ['image_id'], axis = 1)

y = pd.get_dummies(y)

y = y.to_numpy()

y

Nice. So, we have converted our y file into numpy array. But how about x? Shouldn't it be a numpy array also?

x = np.array(x)

Simple Train Test Split

mtrain, mtest, ntrain, ntest = train_test_split(x, y, test_size = 0.2, random_state = 0)

mtrain.shape, ntrain.shape

((4406, 150, 150, 3), (4406, 35))

Deep Model

mod = keras.models.Sequential([
                        keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = mtrain[0].shape),
                        keras.layers.BatchNormalization(),
                        keras.layers.MaxPooling2D(pool_size=(2,2)),
                        keras.layers.Dropout(0.30),
    
                        keras.layers.Conv2D(16, (3,3), activation = 'relu'),
                        keras.layers.BatchNormalization(),
                        keras.layers.MaxPooling2D(pool_size=(2,2)),
                        keras.layers.Dropout(0.30),
    
                        keras.layers.Conv2D(16, (3,3), activation = 'relu'),
                        keras.layers.BatchNormalization(),
                        keras.layers.MaxPooling2D(pool_size=(2,2)),
                        keras.layers.Dropout(0.40),
    
                        keras.layers.Conv2D(16, (3,3), activation = 'relu'),
                        keras.layers.BatchNormalization(),
                        keras.layers.MaxPooling2D(pool_size=(2,2)),
                        keras.layers.Dropout(0.50),
    
                        keras.layers.Flatten(),
     
                        keras.layers.Dense(units = 128, activation = 'relu'),
                        keras.layers.BatchNormalization(),
                        keras.layers.Dropout(0.50),
    
                        keras.layers.Dense(units = 128, activation = 'relu'),
                        keras.layers.BatchNormalization(),
                        keras.layers.Dropout(0.50),
    
    
                        keras.layers.Dense(units = 35, activation = 'softmax')
])
                        

mod.compile(optimizer= 'Adam' , loss = 'categorical_crossentropy' , metrics= ['accuracy'])

Training

mod.fit(mtrain, ntrain, epochs= 500, batch_size = 64, validation_data = (mtest, ntest))

Epoch 1/500
69/69 [==============================] - 3s 38ms/step - loss: 4.5876 - accuracy: 0.0350 - val_loss: 3.5919 - val_accuracy: 0.0617
Epoch 2/500
69/69 [==============================] - 2s 28ms/step - loss: 4.2645 - accuracy: 0.0433 - val_loss: 3.7274 - val_accuracy: 0.0617
Epoch 3/500
69/69 [==============================] - 2s 31ms/step - loss: 3.9987 - accuracy: 0.0520 - val_loss: 3.8151 - val_accuracy: 0.0617
Epoch 4/500
69/69 [==============================] - 2s 34ms/step - loss: 3.7955 - accuracy: 0.0710 - val_loss: 3.7804 - val_accuracy: 0.0581
Epoch 5/500
69/69 [==============================] - 2s 34ms/step - loss: 3.6884 - accuracy: 0.0706 - val_loss: 3.7813 - val_accuracy: 0.0617
Epoch 6/500
69/69 [==============================] - 2s 32ms/step - loss: 3.5896 - accuracy: 0.0887 - val_loss: 3.8459 - val_accuracy: 0.0608
Epoch 7/500
69/69 [==============================] - 2s 29ms/step - loss: 3.5446 - accuracy: 0.0933 - val_loss: 4.0219 - val_accuracy: 0.0608
Epoch 8/500
69/69 [==============================] - 2s 28ms/step - loss: 3.4920 - accuracy: 0.0926 - val_loss: 3.6789 - val_accuracy: 0.0672
Epoch 9/500
69/69 [==============================] - 2s 28ms/step - loss: 3.4547 - accuracy: 0.0919 - val_loss: 3.7609 - val_accuracy: 0.0690
Epoch 10/500
69/69 [==============================] - 2s 31ms/step - loss: 3.4342 - accuracy: 0.1035 - val_loss: 3.7173 - val_accuracy: 0.0681
Epoch 11/500
69/69 [==============================] - 2s 28ms/step - loss: 3.3852 - accuracy: 0.1060 - val_loss: 3.6566 - val_accuracy: 0.0717
Epoch 12/500
69/69 [==============================] - 2s 28ms/step - loss: 3.3720 - accuracy: 0.1044 - val_loss: 3.7139 - val_accuracy: 0.0726
Epoch 13/500
69/69 [==============================] - 2s 28ms/step - loss: 3.3528 - accuracy: 0.1069 - val_loss: 3.6882 - val_accuracy: 0.0690
Epoch 14/500
69/69 [==============================] - 2s 28ms/step - loss: 3.3227 - accuracy: 0.1110 - val_loss: 3.5574 - val_accuracy: 0.0762
Epoch 15/500
69/69 [==============================] - 2s 28ms/step - loss: 3.3006 - accuracy: 0.1126 - val_loss: 3.5081 - val_accuracy: 0.0771
Epoch 16/500
69/69 [==============================] - 2s 30ms/step - loss: 3.3081 - accuracy: 0.1128 - val_loss: 3.5852 - val_accuracy: 0.0726
Epoch 17/500
69/69 [==============================] - 2s 27ms/step - loss: 3.2909 - accuracy: 0.1144 - val_loss: 3.4995 - val_accuracy: 0.0762
Epoch 18/500
69/69 [==============================] - 2s 28ms/step - loss: 3.2739 - accuracy: 0.1241 - val_loss: 3.5178 - val_accuracy: 0.0826
Epoch 19/500
69/69 [==============================] - 2s 28ms/step - loss: 3.2765 - accuracy: 0.1296 - val_loss: 3.6314 - val_accuracy: 0.0726
Epoch 20/500
69/69 [==============================] - 2s 30ms/step - loss: 3.2557 - accuracy: 0.1244 - val_loss: 3.5323 - val_accuracy: 0.0853
Epoch 21/500
69/69 [==============================] - 2s 34ms/step - loss: 3.2585 - accuracy: 0.1244 - val_loss: 3.5433 - val_accuracy: 0.0799
Epoch 22/500
69/69 [==============================] - 2s 28ms/step - loss: 3.2478 - accuracy: 0.1251 - val_loss: 3.5473 - val_accuracy: 0.0835
Epoch 23/500
69/69 [==============================] - 2s 27ms/step - loss: 3.2271 - accuracy: 0.1328 - val_loss: 3.4111 - val_accuracy: 0.0926
Epoch 24/500
69/69 [==============================] - 2s 27ms/step - loss: 3.2402 - accuracy: 0.1273 - val_loss: 3.4392 - val_accuracy: 0.0880
Epoch 25/500
69/69 [==============================] - 2s 28ms/step - loss: 3.2219 - accuracy: 0.1346 - val_loss: 3.4523 - val_accuracy: 0.0862
Epoch 26/500
69/69 [==============================] - 2s 28ms/step - loss: 3.2158 - accuracy: 0.1300 - val_loss: 3.4259 - val_accuracy: 0.0998
Epoch 27/500
69/69 [==============================] - 2s 30ms/step - loss: 3.2150 - accuracy: 0.1339 - val_loss: 3.4032 - val_accuracy: 0.0953
Epoch 28/500
69/69 [==============================] - 2s 27ms/step - loss: 3.2107 - accuracy: 0.1344 - val_loss: 3.4597 - val_accuracy: 0.0898
Epoch 29/500
69/69 [==============================] - 2s 27ms/step - loss: 3.2039 - accuracy: 0.1360 - val_loss: 3.5191 - val_accuracy: 0.0771
Epoch 30/500
69/69 [==============================] - 2s 28ms/step - loss: 3.2003 - accuracy: 0.1350 - val_loss: 3.4815 - val_accuracy: 0.0817
Epoch 31/500
69/69 [==============================] - 2s 28ms/step - loss: 3.1891 - accuracy: 0.1305 - val_loss: 3.6275 - val_accuracy: 0.0844
Epoch 32/500
69/69 [==============================] - 2s 30ms/step - loss: 3.1889 - accuracy: 0.1366 - val_loss: 3.4112 - val_accuracy: 0.0980
Epoch 33/500
69/69 [==============================] - 2s 33ms/step - loss: 3.1869 - accuracy: 0.1335 - val_loss: 3.4024 - val_accuracy: 0.1053
Epoch 34/500
69/69 [==============================] - 2s 28ms/step - loss: 3.1769 - accuracy: 0.1362 - val_loss: 3.5862 - val_accuracy: 0.0862
Epoch 35/500
69/69 [==============================] - 2s 30ms/step - loss: 3.1688 - accuracy: 0.1362 - val_loss: 3.4457 - val_accuracy: 0.0935
Epoch 36/500
69/69 [==============================] - 2s 30ms/step - loss: 3.1718 - accuracy: 0.1337 - val_loss: 3.4245 - val_accuracy: 0.1044
Epoch 37/500
69/69 [==============================] - 2s 34ms/step - loss: 3.1606 - accuracy: 0.1412 - val_loss: 3.5610 - val_accuracy: 0.0799
Epoch 38/500
69/69 [==============================] - 2s 29ms/step - loss: 3.1563 - accuracy: 0.1434 - val_loss: 3.5237 - val_accuracy: 0.0799
Epoch 39/500
69/69 [==============================] - 2s 27ms/step - loss: 3.1454 - accuracy: 0.1441 - val_loss: 3.4600 - val_accuracy: 0.0871
Epoch 40/500
69/69 [==============================] - 2s 27ms/step - loss: 3.1457 - accuracy: 0.1457 - val_loss: 3.6850 - val_accuracy: 0.0708
Epoch 41/500
69/69 [==============================] - 2s 27ms/step - loss: 3.1505 - accuracy: 0.1493 - val_loss: 3.4697 - val_accuracy: 0.0907
Epoch 42/500
69/69 [==============================] - 2s 28ms/step - loss: 3.1338 - accuracy: 0.1487 - val_loss: 3.4652 - val_accuracy: 0.0880
Epoch 43/500
69/69 [==============================] - 2s 30ms/step - loss: 3.1181 - accuracy: 0.1539 - val_loss: 3.3949 - val_accuracy: 0.1143
Epoch 44/500
69/69 [==============================] - 2s 27ms/step - loss: 3.1353 - accuracy: 0.1498 - val_loss: 3.3584 - val_accuracy: 0.1080
Epoch 45/500
69/69 [==============================] - 2s 28ms/step - loss: 3.1257 - accuracy: 0.1453 - val_loss: 3.2757 - val_accuracy: 0.1261
Epoch 46/500
69/69 [==============================] - 2s 27ms/step - loss: 3.1244 - accuracy: 0.1491 - val_loss: 3.4266 - val_accuracy: 0.0871
Epoch 47/500
69/69 [==============================] - 2s 28ms/step - loss: 3.1064 - accuracy: 0.1491 - val_loss: 3.5236 - val_accuracy: 0.0835
Epoch 48/500
69/69 [==============================] - 2s 28ms/step - loss: 3.1113 - accuracy: 0.1475 - val_loss: 3.5363 - val_accuracy: 0.0789
Epoch 49/500
69/69 [==============================] - 2s 29ms/step - loss: 3.0935 - accuracy: 0.1566 - val_loss: 3.4919 - val_accuracy: 0.0835
Epoch 50/500
69/69 [==============================] - 2s 27ms/step - loss: 3.1015 - accuracy: 0.1546 - val_loss: 3.4216 - val_accuracy: 0.0944
Epoch 51/500
69/69 [==============================] - 2s 27ms/step - loss: 3.0993 - accuracy: 0.1584 - val_loss: 3.4114 - val_accuracy: 0.0917
Epoch 52/500
69/69 [==============================] - 2s 28ms/step - loss: 3.0972 - accuracy: 0.1464 - val_loss: 3.3557 - val_accuracy: 0.1025
Epoch 53/500
69/69 [==============================] - 2s 30ms/step - loss: 3.0850 - accuracy: 0.1571 - val_loss: 3.5198 - val_accuracy: 0.0853
Epoch 54/500
69/69 [==============================] - 2s 32ms/step - loss: 3.0836 - accuracy: 0.1557 - val_loss: 3.3502 - val_accuracy: 0.1125
Epoch 55/500
69/69 [==============================] - 2s 28ms/step - loss: 3.0768 - accuracy: 0.1655 - val_loss: 3.4526 - val_accuracy: 0.0935
Epoch 56/500
69/69 [==============================] - 2s 27ms/step - loss: 3.0675 - accuracy: 0.1548 - val_loss: 3.5740 - val_accuracy: 0.0799
Epoch 57/500
69/69 [==============================] - 2s 28ms/step - loss: 3.0802 - accuracy: 0.1643 - val_loss: 3.4950 - val_accuracy: 0.0862
Epoch 58/500
69/69 [==============================] - 2s 27ms/step - loss: 3.0534 - accuracy: 0.1725 - val_loss: 3.4143 - val_accuracy: 0.1053
Epoch 59/500
69/69 [==============================] - 2s 27ms/step - loss: 3.0664 - accuracy: 0.1609 - val_loss: 3.5023 - val_accuracy: 0.0880
Epoch 60/500
69/69 [==============================] - 2s 30ms/step - loss: 3.0494 - accuracy: 0.1668 - val_loss: 3.5152 - val_accuracy: 0.0844
Epoch 61/500
69/69 [==============================] - 2s 28ms/step - loss: 3.0598 - accuracy: 0.1659 - val_loss: 3.3591 - val_accuracy: 0.1034
Epoch 62/500
69/69 [==============================] - 2s 27ms/step - loss: 3.0327 - accuracy: 0.1664 - val_loss: 3.3165 - val_accuracy: 0.1152
Epoch 63/500
69/69 [==============================] - 2s 31ms/step - loss: 3.0362 - accuracy: 0.1650 - val_loss: 3.5140 - val_accuracy: 0.0844
Epoch 64/500
69/69 [==============================] - 2s 28ms/step - loss: 3.0411 - accuracy: 0.1586 - val_loss: 3.3610 - val_accuracy: 0.1116
Epoch 65/500
69/69 [==============================] - 2s 32ms/step - loss: 3.0295 - accuracy: 0.1655 - val_loss: 3.4521 - val_accuracy: 0.0971
Epoch 66/500
69/69 [==============================] - 2s 32ms/step - loss: 3.0379 - accuracy: 0.1686 - val_loss: 3.3555 - val_accuracy: 0.1171
Epoch 67/500
69/69 [==============================] - 2s 28ms/step - loss: 3.0306 - accuracy: 0.1652 - val_loss: 3.2952 - val_accuracy: 0.1116
Epoch 68/500
69/69 [==============================] - 2s 29ms/step - loss: 3.0227 - accuracy: 0.1786 - val_loss: 3.3571 - val_accuracy: 0.1025
Epoch 69/500
69/69 [==============================] - 2s 28ms/step - loss: 3.0266 - accuracy: 0.1666 - val_loss: 3.2955 - val_accuracy: 0.1180
Epoch 70/500
69/69 [==============================] - 2s 33ms/step - loss: 3.0144 - accuracy: 0.1686 - val_loss: 3.3436 - val_accuracy: 0.1071
Epoch 71/500
69/69 [==============================] - 2s 29ms/step - loss: 3.0101 - accuracy: 0.1695 - val_loss: 3.4037 - val_accuracy: 0.0980
Epoch 72/500
69/69 [==============================] - 2s 27ms/step - loss: 2.9956 - accuracy: 0.1734 - val_loss: 3.4605 - val_accuracy: 0.0989
Epoch 73/500
69/69 [==============================] - 2s 28ms/step - loss: 3.0063 - accuracy: 0.1775 - val_loss: 3.5372 - val_accuracy: 0.0889
Epoch 74/500
69/69 [==============================] - 2s 28ms/step - loss: 2.9886 - accuracy: 0.1795 - val_loss: 3.3768 - val_accuracy: 0.1098
Epoch 75/500
69/69 [==============================] - 2s 28ms/step - loss: 2.9850 - accuracy: 0.1750 - val_loss: 3.3180 - val_accuracy: 0.1171
Epoch 76/500
69/69 [==============================] - 2s 30ms/step - loss: 2.9727 - accuracy: 0.1745 - val_loss: 3.4366 - val_accuracy: 0.1034
Epoch 77/500
69/69 [==============================] - 2s 27ms/step - loss: 2.9763 - accuracy: 0.1736 - val_loss: 3.3608 - val_accuracy: 0.1152
Epoch 78/500
69/69 [==============================] - 2s 28ms/step - loss: 2.9592 - accuracy: 0.1804 - val_loss: 3.2814 - val_accuracy: 0.1316
Epoch 79/500
69/69 [==============================] - 2s 28ms/step - loss: 2.9668 - accuracy: 0.1859 - val_loss: 3.4433 - val_accuracy: 0.1034
Epoch 80/500
69/69 [==============================] - 2s 28ms/step - loss: 2.9666 - accuracy: 0.1779 - val_loss: 3.4794 - val_accuracy: 0.0980
Epoch 81/500
69/69 [==============================] - 2s 28ms/step - loss: 2.9672 - accuracy: 0.1825 - val_loss: 3.3632 - val_accuracy: 0.1189
Epoch 82/500
69/69 [==============================] - 2s 30ms/step - loss: 2.9444 - accuracy: 0.1929 - val_loss: 3.3555 - val_accuracy: 0.1216
Epoch 83/500
69/69 [==============================] - 2s 29ms/step - loss: 2.9700 - accuracy: 0.1820 - val_loss: 3.4310 - val_accuracy: 0.1053
Epoch 84/500
69/69 [==============================] - 2s 28ms/step - loss: 2.9625 - accuracy: 0.1882 - val_loss: 3.4742 - val_accuracy: 0.1016
Epoch 85/500
69/69 [==============================] - 2s 28ms/step - loss: 2.9306 - accuracy: 0.1906 - val_loss: 3.4003 - val_accuracy: 0.0998
Epoch 86/500
69/69 [==============================] - 2s 31ms/step - loss: 2.9444 - accuracy: 0.1813 - val_loss: 3.3957 - val_accuracy: 0.1098
Epoch 87/500
69/69 [==============================] - 2s 32ms/step - loss: 2.9507 - accuracy: 0.1854 - val_loss: 3.5119 - val_accuracy: 0.0935
Epoch 88/500
69/69 [==============================] - 2s 28ms/step - loss: 2.9173 - accuracy: 0.1882 - val_loss: 3.3719 - val_accuracy: 0.1071
Epoch 89/500
69/69 [==============================] - 2s 28ms/step - loss: 2.9238 - accuracy: 0.1943 - val_loss: 3.3978 - val_accuracy: 0.1152
Epoch 90/500
69/69 [==============================] - 2s 27ms/step - loss: 2.9252 - accuracy: 0.1927 - val_loss: 3.2608 - val_accuracy: 0.1325
Epoch 91/500
69/69 [==============================] - 2s 28ms/step - loss: 2.9181 - accuracy: 0.1970 - val_loss: 3.3199 - val_accuracy: 0.1216
Epoch 92/500
69/69 [==============================] - 2s 28ms/step - loss: 2.9109 - accuracy: 0.1929 - val_loss: 3.4055 - val_accuracy: 0.1152
Epoch 93/500
69/69 [==============================] - 2s 34ms/step - loss: 2.9338 - accuracy: 0.1741 - val_loss: 3.4177 - val_accuracy: 0.1180
Epoch 94/500
69/69 [==============================] - 2s 27ms/step - loss: 2.8990 - accuracy: 0.1941 - val_loss: 3.3819 - val_accuracy: 0.1143
Epoch 95/500
69/69 [==============================] - 2s 29ms/step - loss: 2.9135 - accuracy: 0.2004 - val_loss: 3.4053 - val_accuracy: 0.1171
Epoch 96/500
69/69 [==============================] - 2s 29ms/step - loss: 2.8986 - accuracy: 0.1950 - val_loss: 3.4933 - val_accuracy: 0.1207
Epoch 97/500
69/69 [==============================] - 2s 27ms/step - loss: 2.8777 - accuracy: 0.1968 - val_loss: 3.4220 - val_accuracy: 0.1107
Epoch 98/500
69/69 [==============================] - 2s 30ms/step - loss: 2.8978 - accuracy: 0.1857 - val_loss: 3.3476 - val_accuracy: 0.1198
Epoch 99/500
69/69 [==============================] - 2s 27ms/step - loss: 2.8904 - accuracy: 0.2047 - val_loss: 3.3861 - val_accuracy: 0.1207
Epoch 100/500
69/69 [==============================] - 2s 27ms/step - loss: 2.8933 - accuracy: 0.1947 - val_loss: 3.2915 - val_accuracy: 0.1316
Epoch 101/500
69/69 [==============================] - 2s 27ms/step - loss: 2.8723 - accuracy: 0.1929 - val_loss: 3.5066 - val_accuracy: 0.1071
Epoch 102/500
69/69 [==============================] - 2s 28ms/step - loss: 2.8577 - accuracy: 0.2059 - val_loss: 3.3640 - val_accuracy: 0.1162
Epoch 103/500
69/69 [==============================] - 2s 32ms/step - loss: 2.8827 - accuracy: 0.1984 - val_loss: 3.3927 - val_accuracy: 0.1225
Epoch 104/500
69/69 [==============================] - 2s 29ms/step - loss: 2.8853 - accuracy: 0.2074 - val_loss: 3.3148 - val_accuracy: 0.1216
Epoch 105/500
69/69 [==============================] - 2s 27ms/step - loss: 2.8857 - accuracy: 0.2070 - val_loss: 3.4197 - val_accuracy: 0.1180
Epoch 106/500
69/69 [==============================] - 2s 28ms/step - loss: 2.8601 - accuracy: 0.2054 - val_loss: 3.4036 - val_accuracy: 0.1171
Epoch 107/500
69/69 [==============================] - 2s 27ms/step - loss: 2.8501 - accuracy: 0.2079 - val_loss: 3.4066 - val_accuracy: 0.1198
Epoch 108/500
69/69 [==============================] - 2s 28ms/step - loss: 2.8553 - accuracy: 0.1993 - val_loss: 3.4917 - val_accuracy: 0.1116
Epoch 109/500
69/69 [==============================] - 2s 30ms/step - loss: 2.8461 - accuracy: 0.2020 - val_loss: 3.3633 - val_accuracy: 0.1107
Epoch 110/500
69/69 [==============================] - 2s 29ms/step - loss: 2.8556 - accuracy: 0.1986 - val_loss: 3.3941 - val_accuracy: 0.1143
Epoch 111/500
69/69 [==============================] - 2s 27ms/step - loss: 2.8332 - accuracy: 0.2093 - val_loss: 3.3046 - val_accuracy: 0.1143
Epoch 112/500
69/69 [==============================] - 2s 28ms/step - loss: 2.8498 - accuracy: 0.2002 - val_loss: 3.2969 - val_accuracy: 0.1316
Epoch 113/500
69/69 [==============================] - 2s 28ms/step - loss: 2.8636 - accuracy: 0.1988 - val_loss: 3.3787 - val_accuracy: 0.1198
Epoch 114/500
69/69 [==============================] - 2s 28ms/step - loss: 2.8317 - accuracy: 0.2081 - val_loss: 3.3431 - val_accuracy: 0.1289
Epoch 115/500
69/69 [==============================] - 2s 29ms/step - loss: 2.8265 - accuracy: 0.2088 - val_loss: 3.4573 - val_accuracy: 0.1089
Epoch 116/500
69/69 [==============================] - 2s 27ms/step - loss: 2.8231 - accuracy: 0.2129 - val_loss: 3.4337 - val_accuracy: 0.1152
Epoch 117/500
69/69 [==============================] - 2s 28ms/step - loss: 2.8386 - accuracy: 0.2074 - val_loss: 3.3783 - val_accuracy: 0.1125
Epoch 118/500
69/69 [==============================] - 2s 27ms/step - loss: 2.8225 - accuracy: 0.2129 - val_loss: 3.3784 - val_accuracy: 0.1143
Epoch 119/500
69/69 [==============================] - 2s 31ms/step - loss: 2.8273 - accuracy: 0.2077 - val_loss: 3.2843 - val_accuracy: 0.1298
Epoch 120/500
69/69 [==============================] - 2s 30ms/step - loss: 2.8243 - accuracy: 0.2077 - val_loss: 3.3446 - val_accuracy: 0.1261
Epoch 121/500
69/69 [==============================] - 2s 28ms/step - loss: 2.8330 - accuracy: 0.2106 - val_loss: 3.4383 - val_accuracy: 0.1171
Epoch 122/500
69/69 [==============================] - 2s 28ms/step - loss: 2.8071 - accuracy: 0.2070 - val_loss: 3.3605 - val_accuracy: 0.1261
Epoch 123/500
69/69 [==============================] - 2s 32ms/step - loss: 2.8137 - accuracy: 0.2158 - val_loss: 3.3037 - val_accuracy: 0.1279
Epoch 124/500
69/69 [==============================] - 2s 32ms/step - loss: 2.8169 - accuracy: 0.2090 - val_loss: 3.3953 - val_accuracy: 0.1261
Epoch 125/500
69/69 [==============================] - 2s 29ms/step - loss: 2.8056 - accuracy: 0.2072 - val_loss: 3.4673 - val_accuracy: 0.1143
Epoch 126/500
69/69 [==============================] - 2s 31ms/step - loss: 2.7906 - accuracy: 0.2163 - val_loss: 3.3545 - val_accuracy: 0.1198
Epoch 127/500
69/69 [==============================] - 2s 28ms/step - loss: 2.8152 - accuracy: 0.2111 - val_loss: 3.4108 - val_accuracy: 0.1180
Epoch 128/500
69/69 [==============================] - 2s 27ms/step - loss: 2.8185 - accuracy: 0.2120 - val_loss: 3.4203 - val_accuracy: 0.1098
Epoch 129/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7963 - accuracy: 0.2020 - val_loss: 3.4445 - val_accuracy: 0.1044
Epoch 130/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7859 - accuracy: 0.2224 - val_loss: 3.4132 - val_accuracy: 0.1162
Epoch 131/500
69/69 [==============================] - 2s 30ms/step - loss: 2.7893 - accuracy: 0.2224 - val_loss: 3.3516 - val_accuracy: 0.1143
Epoch 132/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7984 - accuracy: 0.2174 - val_loss: 3.3714 - val_accuracy: 0.1171
Epoch 133/500
69/69 [==============================] - 2s 27ms/step - loss: 2.7990 - accuracy: 0.2233 - val_loss: 3.3623 - val_accuracy: 0.1143
Epoch 134/500
69/69 [==============================] - 2s 27ms/step - loss: 2.7844 - accuracy: 0.2199 - val_loss: 3.4431 - val_accuracy: 0.1171
Epoch 135/500
69/69 [==============================] - 2s 32ms/step - loss: 2.7973 - accuracy: 0.2177 - val_loss: 3.4439 - val_accuracy: 0.1098
Epoch 136/500
69/69 [==============================] - 2s 30ms/step - loss: 2.7943 - accuracy: 0.2186 - val_loss: 3.4632 - val_accuracy: 0.1134
Epoch 137/500
69/69 [==============================] - 2s 29ms/step - loss: 2.7860 - accuracy: 0.2272 - val_loss: 3.4108 - val_accuracy: 0.1189
Epoch 138/500
69/69 [==============================] - 2s 27ms/step - loss: 2.7734 - accuracy: 0.2245 - val_loss: 3.4631 - val_accuracy: 0.1207
Epoch 139/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7749 - accuracy: 0.2227 - val_loss: 3.2846 - val_accuracy: 0.1243
Epoch 140/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7587 - accuracy: 0.2290 - val_loss: 3.3862 - val_accuracy: 0.1270
Epoch 141/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7753 - accuracy: 0.2099 - val_loss: 3.3892 - val_accuracy: 0.1252
Epoch 142/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7677 - accuracy: 0.2299 - val_loss: 3.3790 - val_accuracy: 0.1171
Epoch 143/500
69/69 [==============================] - 2s 27ms/step - loss: 2.7672 - accuracy: 0.2245 - val_loss: 3.3880 - val_accuracy: 0.1180
Epoch 144/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7552 - accuracy: 0.2217 - val_loss: 3.3737 - val_accuracy: 0.1261
Epoch 145/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7675 - accuracy: 0.2197 - val_loss: 3.3766 - val_accuracy: 0.1289
Epoch 146/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7458 - accuracy: 0.2347 - val_loss: 3.4027 - val_accuracy: 0.1207
Epoch 147/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7387 - accuracy: 0.2347 - val_loss: 3.3660 - val_accuracy: 0.1270
Epoch 148/500
69/69 [==============================] - 2s 30ms/step - loss: 2.7633 - accuracy: 0.2267 - val_loss: 3.3366 - val_accuracy: 0.1325
Epoch 149/500
69/69 [==============================] - 2s 27ms/step - loss: 2.7529 - accuracy: 0.2222 - val_loss: 3.4207 - val_accuracy: 0.1171
Epoch 150/500
69/69 [==============================] - 2s 27ms/step - loss: 2.7574 - accuracy: 0.2326 - val_loss: 3.3733 - val_accuracy: 0.1225
Epoch 151/500
69/69 [==============================] - 2s 33ms/step - loss: 2.7562 - accuracy: 0.2222 - val_loss: 3.3440 - val_accuracy: 0.1270
Epoch 152/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7530 - accuracy: 0.2186 - val_loss: 3.3866 - val_accuracy: 0.1225
Epoch 153/500
69/69 [==============================] - 2s 36ms/step - loss: 2.7496 - accuracy: 0.2238 - val_loss: 3.4125 - val_accuracy: 0.1152
Epoch 154/500
69/69 [==============================] - 2s 32ms/step - loss: 2.7508 - accuracy: 0.2286 - val_loss: 3.4056 - val_accuracy: 0.1162
Epoch 155/500
69/69 [==============================] - 2s 29ms/step - loss: 2.7313 - accuracy: 0.2286 - val_loss: 3.3626 - val_accuracy: 0.1261
Epoch 156/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7643 - accuracy: 0.2297 - val_loss: 3.3820 - val_accuracy: 0.1152
Epoch 157/500
69/69 [==============================] - 2s 29ms/step - loss: 2.7406 - accuracy: 0.2274 - val_loss: 3.4659 - val_accuracy: 0.1098
Epoch 158/500
69/69 [==============================] - 2s 30ms/step - loss: 2.7463 - accuracy: 0.2315 - val_loss: 3.5256 - val_accuracy: 0.1098
Epoch 159/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7092 - accuracy: 0.2306 - val_loss: 3.4406 - val_accuracy: 0.1080
Epoch 160/500
69/69 [==============================] - 2s 27ms/step - loss: 2.7319 - accuracy: 0.2342 - val_loss: 3.3865 - val_accuracy: 0.1152
Epoch 161/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7188 - accuracy: 0.2376 - val_loss: 3.4243 - val_accuracy: 0.1180
Epoch 162/500
69/69 [==============================] - 2s 27ms/step - loss: 2.7083 - accuracy: 0.2310 - val_loss: 3.4559 - val_accuracy: 0.1243
Epoch 163/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7157 - accuracy: 0.2315 - val_loss: 3.4210 - val_accuracy: 0.1116
Epoch 164/500
69/69 [==============================] - 2s 31ms/step - loss: 2.7068 - accuracy: 0.2374 - val_loss: 3.4834 - val_accuracy: 0.1143
Epoch 165/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7314 - accuracy: 0.2304 - val_loss: 3.4259 - val_accuracy: 0.1270
Epoch 166/500
69/69 [==============================] - 2s 31ms/step - loss: 2.6959 - accuracy: 0.2394 - val_loss: 3.3699 - val_accuracy: 0.1171
Epoch 167/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7157 - accuracy: 0.2338 - val_loss: 3.5658 - val_accuracy: 0.1080
Epoch 168/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6913 - accuracy: 0.2345 - val_loss: 3.4156 - val_accuracy: 0.1143
Epoch 169/500
69/69 [==============================] - 2s 30ms/step - loss: 2.6987 - accuracy: 0.2383 - val_loss: 3.3829 - val_accuracy: 0.1198
Epoch 170/500
69/69 [==============================] - 2s 29ms/step - loss: 2.7175 - accuracy: 0.2288 - val_loss: 3.4786 - val_accuracy: 0.1216
Epoch 171/500
69/69 [==============================] - 2s 27ms/step - loss: 2.7214 - accuracy: 0.2320 - val_loss: 3.3285 - val_accuracy: 0.1171
Epoch 172/500
69/69 [==============================] - 2s 27ms/step - loss: 2.7096 - accuracy: 0.2379 - val_loss: 3.4592 - val_accuracy: 0.1143
Epoch 173/500
69/69 [==============================] - 2s 27ms/step - loss: 2.7057 - accuracy: 0.2231 - val_loss: 3.4742 - val_accuracy: 0.1198
Epoch 174/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6866 - accuracy: 0.2429 - val_loss: 3.4371 - val_accuracy: 0.1171
Epoch 175/500
69/69 [==============================] - 2s 31ms/step - loss: 2.7070 - accuracy: 0.2306 - val_loss: 3.4303 - val_accuracy: 0.1171
Epoch 176/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7053 - accuracy: 0.2433 - val_loss: 3.4384 - val_accuracy: 0.1252
Epoch 177/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6947 - accuracy: 0.2347 - val_loss: 3.4714 - val_accuracy: 0.1216
Epoch 178/500
69/69 [==============================] - 2s 28ms/step - loss: 2.7076 - accuracy: 0.2419 - val_loss: 3.4173 - val_accuracy: 0.1143
Epoch 179/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6935 - accuracy: 0.2374 - val_loss: 3.3943 - val_accuracy: 0.1388
Epoch 180/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6853 - accuracy: 0.2358 - val_loss: 3.4361 - val_accuracy: 0.1189
Epoch 181/500
69/69 [==============================] - 2s 31ms/step - loss: 2.7062 - accuracy: 0.2313 - val_loss: 3.4988 - val_accuracy: 0.1125
Epoch 182/500
69/69 [==============================] - 2s 31ms/step - loss: 2.6771 - accuracy: 0.2515 - val_loss: 3.4930 - val_accuracy: 0.1152
Epoch 183/500
69/69 [==============================] - 2s 35ms/step - loss: 2.6776 - accuracy: 0.2397 - val_loss: 3.4289 - val_accuracy: 0.1171
Epoch 184/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6758 - accuracy: 0.2453 - val_loss: 3.5089 - val_accuracy: 0.1098
Epoch 185/500
69/69 [==============================] - 2s 29ms/step - loss: 2.6829 - accuracy: 0.2410 - val_loss: 3.4018 - val_accuracy: 0.1207
Epoch 186/500
69/69 [==============================] - 2s 35ms/step - loss: 2.6899 - accuracy: 0.2372 - val_loss: 3.4983 - val_accuracy: 0.1189
Epoch 187/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6729 - accuracy: 0.2406 - val_loss: 3.4746 - val_accuracy: 0.1189
Epoch 188/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6748 - accuracy: 0.2417 - val_loss: 3.3483 - val_accuracy: 0.1198
Epoch 189/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6601 - accuracy: 0.2524 - val_loss: 3.4337 - val_accuracy: 0.1225
Epoch 190/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6647 - accuracy: 0.2519 - val_loss: 3.3588 - val_accuracy: 0.1198
Epoch 191/500
69/69 [==============================] - 2s 30ms/step - loss: 2.6625 - accuracy: 0.2499 - val_loss: 3.4030 - val_accuracy: 0.1298
Epoch 192/500
69/69 [==============================] - 2s 29ms/step - loss: 2.6468 - accuracy: 0.2547 - val_loss: 3.4652 - val_accuracy: 0.1289
Epoch 193/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6838 - accuracy: 0.2490 - val_loss: 3.4534 - val_accuracy: 0.1243
Epoch 194/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6585 - accuracy: 0.2547 - val_loss: 3.4512 - val_accuracy: 0.1143
Epoch 195/500
69/69 [==============================] - 2s 26ms/step - loss: 2.6606 - accuracy: 0.2392 - val_loss: 3.5479 - val_accuracy: 0.1080
Epoch 196/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6790 - accuracy: 0.2478 - val_loss: 3.4123 - val_accuracy: 0.1270
Epoch 197/500
69/69 [==============================] - 2s 31ms/step - loss: 2.6712 - accuracy: 0.2469 - val_loss: 3.4418 - val_accuracy: 0.1171
Epoch 198/500
69/69 [==============================] - 2s 32ms/step - loss: 2.6574 - accuracy: 0.2485 - val_loss: 3.4006 - val_accuracy: 0.1107
Epoch 199/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6536 - accuracy: 0.2501 - val_loss: 3.5164 - val_accuracy: 0.1152
Epoch 200/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6880 - accuracy: 0.2417 - val_loss: 3.4335 - val_accuracy: 0.1116
Epoch 201/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6760 - accuracy: 0.2369 - val_loss: 3.3991 - val_accuracy: 0.1180
Epoch 202/500
69/69 [==============================] - 2s 29ms/step - loss: 2.6870 - accuracy: 0.2390 - val_loss: 3.4041 - val_accuracy: 0.1189
Epoch 203/500
69/69 [==============================] - 2s 29ms/step - loss: 2.6537 - accuracy: 0.2503 - val_loss: 3.3974 - val_accuracy: 0.1289
Epoch 204/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6844 - accuracy: 0.2476 - val_loss: 3.4472 - val_accuracy: 0.1171
Epoch 205/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6506 - accuracy: 0.2483 - val_loss: 3.3863 - val_accuracy: 0.1171
Epoch 206/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6427 - accuracy: 0.2456 - val_loss: 3.3952 - val_accuracy: 0.1270
Epoch 207/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6598 - accuracy: 0.2497 - val_loss: 3.4356 - val_accuracy: 0.1216
Epoch 208/500
69/69 [==============================] - 2s 30ms/step - loss: 2.6606 - accuracy: 0.2526 - val_loss: 3.3630 - val_accuracy: 0.1343
Epoch 209/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6403 - accuracy: 0.2617 - val_loss: 3.3417 - val_accuracy: 0.1416
Epoch 210/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6550 - accuracy: 0.2492 - val_loss: 3.4258 - val_accuracy: 0.1171
Epoch 211/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6381 - accuracy: 0.2549 - val_loss: 3.3469 - val_accuracy: 0.1307
Epoch 212/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6513 - accuracy: 0.2490 - val_loss: 3.3819 - val_accuracy: 0.1289
Epoch 213/500
69/69 [==============================] - 2s 35ms/step - loss: 2.6488 - accuracy: 0.2476 - val_loss: 3.3687 - val_accuracy: 0.1307
Epoch 214/500
69/69 [==============================] - 2s 34ms/step - loss: 2.6576 - accuracy: 0.2549 - val_loss: 3.4610 - val_accuracy: 0.1252
Epoch 215/500
69/69 [==============================] - 2s 32ms/step - loss: 2.6381 - accuracy: 0.2508 - val_loss: 3.3854 - val_accuracy: 0.1216
Epoch 216/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6321 - accuracy: 0.2549 - val_loss: 3.4080 - val_accuracy: 0.1261
Epoch 217/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6207 - accuracy: 0.2481 - val_loss: 3.4546 - val_accuracy: 0.1298
Epoch 218/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6405 - accuracy: 0.2531 - val_loss: 3.4331 - val_accuracy: 0.1234
Epoch 219/500
69/69 [==============================] - 2s 31ms/step - loss: 2.6408 - accuracy: 0.2542 - val_loss: 3.3944 - val_accuracy: 0.1234
Epoch 220/500
69/69 [==============================] - 2s 29ms/step - loss: 2.6543 - accuracy: 0.2499 - val_loss: 3.3519 - val_accuracy: 0.1361
Epoch 221/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6327 - accuracy: 0.2483 - val_loss: 3.3926 - val_accuracy: 0.1189
Epoch 222/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6331 - accuracy: 0.2583 - val_loss: 3.4105 - val_accuracy: 0.1279
Epoch 223/500
69/69 [==============================] - 2s 29ms/step - loss: 2.6346 - accuracy: 0.2601 - val_loss: 3.3947 - val_accuracy: 0.1361
Epoch 224/500
69/69 [==============================] - 2s 30ms/step - loss: 2.6459 - accuracy: 0.2508 - val_loss: 3.4708 - val_accuracy: 0.1216
Epoch 225/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6271 - accuracy: 0.2549 - val_loss: 3.4722 - val_accuracy: 0.1270
Epoch 226/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6398 - accuracy: 0.2501 - val_loss: 3.3807 - val_accuracy: 0.1434
Epoch 227/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6240 - accuracy: 0.2587 - val_loss: 3.3982 - val_accuracy: 0.1289
Epoch 228/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6134 - accuracy: 0.2599 - val_loss: 3.3740 - val_accuracy: 0.1325
Epoch 229/500
69/69 [==============================] - 2s 30ms/step - loss: 2.6182 - accuracy: 0.2551 - val_loss: 3.4272 - val_accuracy: 0.1270
Epoch 230/500
69/69 [==============================] - 2s 33ms/step - loss: 2.6101 - accuracy: 0.2581 - val_loss: 3.3594 - val_accuracy: 0.1443
Epoch 231/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6297 - accuracy: 0.2458 - val_loss: 3.4615 - val_accuracy: 0.1234
Epoch 232/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6143 - accuracy: 0.2617 - val_loss: 3.3681 - val_accuracy: 0.1243
Epoch 233/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6287 - accuracy: 0.2551 - val_loss: 3.4513 - val_accuracy: 0.1298
Epoch 234/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6241 - accuracy: 0.2535 - val_loss: 3.4220 - val_accuracy: 0.1198
Epoch 235/500
69/69 [==============================] - 2s 30ms/step - loss: 2.6297 - accuracy: 0.2615 - val_loss: 3.4328 - val_accuracy: 0.1243
Epoch 236/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6264 - accuracy: 0.2469 - val_loss: 3.4273 - val_accuracy: 0.1143
Epoch 237/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6010 - accuracy: 0.2549 - val_loss: 3.4026 - val_accuracy: 0.1343
Epoch 238/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6332 - accuracy: 0.2567 - val_loss: 3.3583 - val_accuracy: 0.1279
Epoch 239/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5883 - accuracy: 0.2626 - val_loss: 3.4082 - val_accuracy: 0.1243
Epoch 240/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6134 - accuracy: 0.2549 - val_loss: 3.4652 - val_accuracy: 0.1171
Epoch 241/500
69/69 [==============================] - 2s 30ms/step - loss: 2.6134 - accuracy: 0.2571 - val_loss: 3.4479 - val_accuracy: 0.1234
Epoch 242/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6103 - accuracy: 0.2631 - val_loss: 3.4499 - val_accuracy: 0.1270
Epoch 243/500
69/69 [==============================] - 2s 32ms/step - loss: 2.6105 - accuracy: 0.2583 - val_loss: 3.3773 - val_accuracy: 0.1325
Epoch 244/500
69/69 [==============================] - 2s 29ms/step - loss: 2.6221 - accuracy: 0.2619 - val_loss: 3.4032 - val_accuracy: 0.1307
Epoch 245/500
69/69 [==============================] - 2s 31ms/step - loss: 2.5996 - accuracy: 0.2601 - val_loss: 3.4916 - val_accuracy: 0.1243
Epoch 246/500
69/69 [==============================] - 2s 35ms/step - loss: 2.5907 - accuracy: 0.2617 - val_loss: 3.4874 - val_accuracy: 0.1261
Epoch 247/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6044 - accuracy: 0.2615 - val_loss: 3.4465 - val_accuracy: 0.1243
Epoch 248/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5940 - accuracy: 0.2660 - val_loss: 3.4644 - val_accuracy: 0.1261
Epoch 249/500
69/69 [==============================] - 2s 27ms/step - loss: 2.6074 - accuracy: 0.2542 - val_loss: 3.4510 - val_accuracy: 0.1207
Epoch 250/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5875 - accuracy: 0.2603 - val_loss: 3.4289 - val_accuracy: 0.1234
Epoch 251/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6126 - accuracy: 0.2601 - val_loss: 3.4231 - val_accuracy: 0.1234
Epoch 252/500
69/69 [==============================] - 2s 30ms/step - loss: 2.5841 - accuracy: 0.2662 - val_loss: 3.4826 - val_accuracy: 0.1216
Epoch 253/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5742 - accuracy: 0.2655 - val_loss: 3.4217 - val_accuracy: 0.1352
Epoch 254/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5909 - accuracy: 0.2660 - val_loss: 3.4374 - val_accuracy: 0.1307
Epoch 255/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5924 - accuracy: 0.2653 - val_loss: 3.4031 - val_accuracy: 0.1289
Epoch 256/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5932 - accuracy: 0.2578 - val_loss: 3.4106 - val_accuracy: 0.1343
Epoch 257/500
69/69 [==============================] - 2s 30ms/step - loss: 2.5970 - accuracy: 0.2669 - val_loss: 3.3939 - val_accuracy: 0.1352
Epoch 258/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5686 - accuracy: 0.2680 - val_loss: 3.5657 - val_accuracy: 0.1216
Epoch 259/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5767 - accuracy: 0.2724 - val_loss: 3.4826 - val_accuracy: 0.1298
Epoch 260/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5814 - accuracy: 0.2683 - val_loss: 3.4396 - val_accuracy: 0.1279
Epoch 261/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5954 - accuracy: 0.2549 - val_loss: 3.4114 - val_accuracy: 0.1289
Epoch 262/500
69/69 [==============================] - 2s 30ms/step - loss: 2.5866 - accuracy: 0.2665 - val_loss: 3.4739 - val_accuracy: 0.1152
Epoch 263/500
69/69 [==============================] - 2s 30ms/step - loss: 2.5784 - accuracy: 0.2626 - val_loss: 3.5282 - val_accuracy: 0.1216
Epoch 264/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5922 - accuracy: 0.2642 - val_loss: 3.3360 - val_accuracy: 0.1407
Epoch 265/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6210 - accuracy: 0.2678 - val_loss: 3.4021 - val_accuracy: 0.1252
Epoch 266/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5764 - accuracy: 0.2671 - val_loss: 3.3840 - val_accuracy: 0.1298
Epoch 267/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5788 - accuracy: 0.2651 - val_loss: 3.4559 - val_accuracy: 0.1207
Epoch 268/500
69/69 [==============================] - 2s 30ms/step - loss: 2.6075 - accuracy: 0.2635 - val_loss: 3.4762 - val_accuracy: 0.1234
Epoch 269/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5680 - accuracy: 0.2719 - val_loss: 3.3741 - val_accuracy: 0.1289
Epoch 270/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5782 - accuracy: 0.2617 - val_loss: 3.4262 - val_accuracy: 0.1216
Epoch 271/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5946 - accuracy: 0.2610 - val_loss: 3.4989 - val_accuracy: 0.1252
Epoch 272/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5987 - accuracy: 0.2578 - val_loss: 3.4069 - val_accuracy: 0.1316
Epoch 273/500
69/69 [==============================] - 2s 31ms/step - loss: 2.5884 - accuracy: 0.2685 - val_loss: 3.3782 - val_accuracy: 0.1361
Epoch 274/500
69/69 [==============================] - 2s 31ms/step - loss: 2.5731 - accuracy: 0.2724 - val_loss: 3.4723 - val_accuracy: 0.1279
Epoch 275/500
69/69 [==============================] - 2s 30ms/step - loss: 2.5845 - accuracy: 0.2621 - val_loss: 3.3996 - val_accuracy: 0.1325
Epoch 276/500
69/69 [==============================] - 2s 28ms/step - loss: 2.6078 - accuracy: 0.2651 - val_loss: 3.4272 - val_accuracy: 0.1252
Epoch 277/500
69/69 [==============================] - 2s 32ms/step - loss: 2.5585 - accuracy: 0.2705 - val_loss: 3.3792 - val_accuracy: 0.1379
Epoch 278/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5520 - accuracy: 0.2724 - val_loss: 3.4161 - val_accuracy: 0.1279
Epoch 279/500
69/69 [==============================] - 2s 31ms/step - loss: 2.5806 - accuracy: 0.2617 - val_loss: 3.5110 - val_accuracy: 0.1279
Epoch 280/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5522 - accuracy: 0.2814 - val_loss: 3.4765 - val_accuracy: 0.1162
Epoch 281/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5362 - accuracy: 0.2776 - val_loss: 3.4654 - val_accuracy: 0.1162
Epoch 282/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5654 - accuracy: 0.2655 - val_loss: 3.4623 - val_accuracy: 0.1261
Epoch 283/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5698 - accuracy: 0.2637 - val_loss: 3.5313 - val_accuracy: 0.1207
Epoch 284/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5696 - accuracy: 0.2644 - val_loss: 3.4283 - val_accuracy: 0.1316
Epoch 285/500
69/69 [==============================] - 2s 30ms/step - loss: 2.5686 - accuracy: 0.2692 - val_loss: 3.4370 - val_accuracy: 0.1279
Epoch 286/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5501 - accuracy: 0.2710 - val_loss: 3.4615 - val_accuracy: 0.1307
Epoch 287/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5687 - accuracy: 0.2735 - val_loss: 3.4273 - val_accuracy: 0.1316
Epoch 288/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5491 - accuracy: 0.2667 - val_loss: 3.3583 - val_accuracy: 0.1461
Epoch 289/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5586 - accuracy: 0.2749 - val_loss: 3.4572 - val_accuracy: 0.1252
Epoch 290/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5551 - accuracy: 0.2671 - val_loss: 3.4449 - val_accuracy: 0.1234
Epoch 291/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5667 - accuracy: 0.2653 - val_loss: 3.4105 - val_accuracy: 0.1307
Epoch 292/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5676 - accuracy: 0.2746 - val_loss: 3.3822 - val_accuracy: 0.1397
Epoch 293/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5558 - accuracy: 0.2839 - val_loss: 3.4060 - val_accuracy: 0.1361
Epoch 294/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5567 - accuracy: 0.2646 - val_loss: 3.4997 - val_accuracy: 0.1225
Epoch 295/500
69/69 [==============================] - 2s 26ms/step - loss: 2.5749 - accuracy: 0.2658 - val_loss: 3.4093 - val_accuracy: 0.1307
Epoch 296/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5416 - accuracy: 0.2730 - val_loss: 3.4239 - val_accuracy: 0.1252
Epoch 297/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5519 - accuracy: 0.2631 - val_loss: 3.5863 - val_accuracy: 0.1171
Epoch 298/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5471 - accuracy: 0.2644 - val_loss: 3.3941 - val_accuracy: 0.1261
Epoch 299/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5782 - accuracy: 0.2662 - val_loss: 3.3820 - val_accuracy: 0.1352
Epoch 300/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5664 - accuracy: 0.2737 - val_loss: 3.4114 - val_accuracy: 0.1307
Epoch 301/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5435 - accuracy: 0.2839 - val_loss: 3.3903 - val_accuracy: 0.1307
Epoch 302/500
69/69 [==============================] - 2s 30ms/step - loss: 2.5617 - accuracy: 0.2749 - val_loss: 3.4363 - val_accuracy: 0.1234
Epoch 303/500
69/69 [==============================] - 2s 30ms/step - loss: 2.5464 - accuracy: 0.2773 - val_loss: 3.4932 - val_accuracy: 0.1270
Epoch 304/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5454 - accuracy: 0.2776 - val_loss: 3.5011 - val_accuracy: 0.1252
Epoch 305/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5361 - accuracy: 0.2742 - val_loss: 3.3920 - val_accuracy: 0.1307
Epoch 306/500
69/69 [==============================] - 2s 31ms/step - loss: 2.5400 - accuracy: 0.2735 - val_loss: 3.4333 - val_accuracy: 0.1243
Epoch 307/500
69/69 [==============================] - 2s 32ms/step - loss: 2.5538 - accuracy: 0.2710 - val_loss: 3.4311 - val_accuracy: 0.1243
Epoch 308/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5424 - accuracy: 0.2601 - val_loss: 3.5358 - val_accuracy: 0.1225
Epoch 309/500
69/69 [==============================] - 2s 30ms/step - loss: 2.5534 - accuracy: 0.2733 - val_loss: 3.3972 - val_accuracy: 0.1325
Epoch 310/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5610 - accuracy: 0.2703 - val_loss: 3.4785 - val_accuracy: 0.1198
Epoch 311/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5496 - accuracy: 0.2717 - val_loss: 3.4098 - val_accuracy: 0.1370
Epoch 312/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5262 - accuracy: 0.2803 - val_loss: 3.3927 - val_accuracy: 0.1352
Epoch 313/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5274 - accuracy: 0.2733 - val_loss: 3.4436 - val_accuracy: 0.1307
Epoch 314/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5306 - accuracy: 0.2735 - val_loss: 3.4844 - val_accuracy: 0.1252
Epoch 315/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5504 - accuracy: 0.2753 - val_loss: 3.5059 - val_accuracy: 0.1207
Epoch 316/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5464 - accuracy: 0.2724 - val_loss: 3.3792 - val_accuracy: 0.1397
Epoch 317/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5447 - accuracy: 0.2835 - val_loss: 3.4190 - val_accuracy: 0.1352
Epoch 318/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5299 - accuracy: 0.2773 - val_loss: 3.3902 - val_accuracy: 0.1298
Epoch 319/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5374 - accuracy: 0.2798 - val_loss: 3.4294 - val_accuracy: 0.1316
Epoch 320/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5458 - accuracy: 0.2669 - val_loss: 3.3866 - val_accuracy: 0.1325
Epoch 321/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5458 - accuracy: 0.2767 - val_loss: 3.3489 - val_accuracy: 0.1379
Epoch 322/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5299 - accuracy: 0.2798 - val_loss: 3.3543 - val_accuracy: 0.1425
Epoch 323/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5347 - accuracy: 0.2767 - val_loss: 3.3457 - val_accuracy: 0.1416
Epoch 324/500
69/69 [==============================] - 2s 30ms/step - loss: 2.5364 - accuracy: 0.2742 - val_loss: 3.4285 - val_accuracy: 0.1261
Epoch 325/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5388 - accuracy: 0.2762 - val_loss: 3.4140 - val_accuracy: 0.1316
Epoch 326/500
69/69 [==============================] - 2s 30ms/step - loss: 2.5339 - accuracy: 0.2842 - val_loss: 3.4580 - val_accuracy: 0.1243
Epoch 327/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5578 - accuracy: 0.2719 - val_loss: 3.4664 - val_accuracy: 0.1216
Epoch 328/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5310 - accuracy: 0.2746 - val_loss: 3.3971 - val_accuracy: 0.1289
Epoch 329/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5295 - accuracy: 0.2814 - val_loss: 3.4769 - val_accuracy: 0.1216
Epoch 330/500
69/69 [==============================] - 2s 30ms/step - loss: 2.5348 - accuracy: 0.2701 - val_loss: 3.3978 - val_accuracy: 0.1352
Epoch 331/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5224 - accuracy: 0.2846 - val_loss: 3.4373 - val_accuracy: 0.1261
Epoch 332/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5195 - accuracy: 0.2821 - val_loss: 3.4140 - val_accuracy: 0.1270
Epoch 333/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5404 - accuracy: 0.2787 - val_loss: 3.5015 - val_accuracy: 0.1261
Epoch 334/500
69/69 [==============================] - 2s 33ms/step - loss: 2.5437 - accuracy: 0.2678 - val_loss: 3.4127 - val_accuracy: 0.1307
Epoch 335/500
69/69 [==============================] - 2s 31ms/step - loss: 2.5531 - accuracy: 0.2801 - val_loss: 3.3977 - val_accuracy: 0.1343
Epoch 336/500
69/69 [==============================] - 2s 31ms/step - loss: 2.5454 - accuracy: 0.2817 - val_loss: 3.4365 - val_accuracy: 0.1234
Epoch 337/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5091 - accuracy: 0.2796 - val_loss: 3.4363 - val_accuracy: 0.1216
Epoch 338/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5454 - accuracy: 0.2742 - val_loss: 3.4004 - val_accuracy: 0.1279
Epoch 339/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5267 - accuracy: 0.2805 - val_loss: 3.3947 - val_accuracy: 0.1361
Epoch 340/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5180 - accuracy: 0.2860 - val_loss: 3.4153 - val_accuracy: 0.1289
Epoch 341/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5047 - accuracy: 0.2894 - val_loss: 3.4022 - val_accuracy: 0.1334
Epoch 342/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5051 - accuracy: 0.2842 - val_loss: 3.4494 - val_accuracy: 0.1243
Epoch 343/500
69/69 [==============================] - 2s 32ms/step - loss: 2.5136 - accuracy: 0.2860 - val_loss: 3.4423 - val_accuracy: 0.1316
Epoch 344/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5083 - accuracy: 0.2789 - val_loss: 3.4542 - val_accuracy: 0.1316
Epoch 345/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5198 - accuracy: 0.2760 - val_loss: 3.4333 - val_accuracy: 0.1261
Epoch 346/500
69/69 [==============================] - 2s 31ms/step - loss: 2.5055 - accuracy: 0.2846 - val_loss: 3.5536 - val_accuracy: 0.1189
Epoch 347/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5106 - accuracy: 0.2753 - val_loss: 3.4809 - val_accuracy: 0.1289
Epoch 348/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5304 - accuracy: 0.2751 - val_loss: 3.3956 - val_accuracy: 0.1443
Epoch 349/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5361 - accuracy: 0.2724 - val_loss: 3.4335 - val_accuracy: 0.1298
Epoch 350/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5093 - accuracy: 0.2819 - val_loss: 3.4592 - val_accuracy: 0.1198
Epoch 351/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5205 - accuracy: 0.2737 - val_loss: 3.4092 - val_accuracy: 0.1397
Epoch 352/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5179 - accuracy: 0.2860 - val_loss: 3.4617 - val_accuracy: 0.1252
Epoch 353/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4812 - accuracy: 0.3030 - val_loss: 3.5485 - val_accuracy: 0.1207
Epoch 354/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5034 - accuracy: 0.2932 - val_loss: 3.4058 - val_accuracy: 0.1316
Epoch 355/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4796 - accuracy: 0.2935 - val_loss: 3.3841 - val_accuracy: 0.1361
Epoch 356/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5298 - accuracy: 0.2783 - val_loss: 3.4194 - val_accuracy: 0.1252
Epoch 357/500
69/69 [==============================] - 2s 29ms/step - loss: 2.5204 - accuracy: 0.2880 - val_loss: 3.5623 - val_accuracy: 0.1116
Epoch 358/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5254 - accuracy: 0.2842 - val_loss: 3.5233 - val_accuracy: 0.1207
Epoch 359/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5171 - accuracy: 0.2848 - val_loss: 3.5015 - val_accuracy: 0.1243
Epoch 360/500
69/69 [==============================] - 2s 31ms/step - loss: 2.5092 - accuracy: 0.2962 - val_loss: 3.4255 - val_accuracy: 0.1270
Epoch 361/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5004 - accuracy: 0.2835 - val_loss: 3.4597 - val_accuracy: 0.1298
Epoch 362/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5050 - accuracy: 0.2844 - val_loss: 3.3917 - val_accuracy: 0.1270
Epoch 363/500
69/69 [==============================] - 2s 31ms/step - loss: 2.4913 - accuracy: 0.2808 - val_loss: 3.4444 - val_accuracy: 0.1180
Epoch 364/500
69/69 [==============================] - 2s 33ms/step - loss: 2.5136 - accuracy: 0.2826 - val_loss: 3.4546 - val_accuracy: 0.1279
Epoch 365/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5108 - accuracy: 0.2828 - val_loss: 3.3849 - val_accuracy: 0.1316
Epoch 366/500
69/69 [==============================] - 2s 31ms/step - loss: 2.5297 - accuracy: 0.2842 - val_loss: 3.4891 - val_accuracy: 0.1252
Epoch 367/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4948 - accuracy: 0.2812 - val_loss: 3.4977 - val_accuracy: 0.1207
Epoch 368/500
69/69 [==============================] - 2s 29ms/step - loss: 2.4888 - accuracy: 0.2842 - val_loss: 3.3999 - val_accuracy: 0.1416
Epoch 369/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5401 - accuracy: 0.2773 - val_loss: 3.4354 - val_accuracy: 0.1325
Epoch 370/500
69/69 [==============================] - 2s 26ms/step - loss: 2.4863 - accuracy: 0.2937 - val_loss: 3.4722 - val_accuracy: 0.1289
Epoch 371/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5048 - accuracy: 0.2871 - val_loss: 3.4416 - val_accuracy: 0.1207
Epoch 372/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4828 - accuracy: 0.2944 - val_loss: 3.4188 - val_accuracy: 0.1243
Epoch 373/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5175 - accuracy: 0.2780 - val_loss: 3.3954 - val_accuracy: 0.1397
Epoch 374/500
69/69 [==============================] - 2s 31ms/step - loss: 2.5065 - accuracy: 0.2814 - val_loss: 3.3784 - val_accuracy: 0.1388
Epoch 375/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5064 - accuracy: 0.2914 - val_loss: 3.4592 - val_accuracy: 0.1279
Epoch 376/500
69/69 [==============================] - 2s 33ms/step - loss: 2.4957 - accuracy: 0.2855 - val_loss: 3.4464 - val_accuracy: 0.1180
Epoch 377/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5208 - accuracy: 0.2819 - val_loss: 3.4482 - val_accuracy: 0.1252
Epoch 378/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5105 - accuracy: 0.2830 - val_loss: 3.4419 - val_accuracy: 0.1298
Epoch 379/500
69/69 [==============================] - 2s 31ms/step - loss: 2.5011 - accuracy: 0.2833 - val_loss: 3.4083 - val_accuracy: 0.1307
Epoch 380/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4995 - accuracy: 0.2828 - val_loss: 3.5219 - val_accuracy: 0.1234
Epoch 381/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5041 - accuracy: 0.2812 - val_loss: 3.4610 - val_accuracy: 0.1125
Epoch 382/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4872 - accuracy: 0.2869 - val_loss: 3.4310 - val_accuracy: 0.1270
Epoch 383/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5105 - accuracy: 0.2939 - val_loss: 3.4596 - val_accuracy: 0.1325
Epoch 384/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4673 - accuracy: 0.2916 - val_loss: 3.4314 - val_accuracy: 0.1279
Epoch 385/500
69/69 [==============================] - 2s 30ms/step - loss: 2.4930 - accuracy: 0.2869 - val_loss: 3.4430 - val_accuracy: 0.1270
Epoch 386/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5127 - accuracy: 0.2762 - val_loss: 3.5440 - val_accuracy: 0.1198
Epoch 387/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4963 - accuracy: 0.2826 - val_loss: 3.5281 - val_accuracy: 0.1234
Epoch 388/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5069 - accuracy: 0.2871 - val_loss: 3.3916 - val_accuracy: 0.1416
Epoch 389/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5001 - accuracy: 0.2898 - val_loss: 3.4464 - val_accuracy: 0.1289
Epoch 390/500
69/69 [==============================] - 2s 29ms/step - loss: 2.4752 - accuracy: 0.2946 - val_loss: 3.3830 - val_accuracy: 0.1361
Epoch 391/500
69/69 [==============================] - 2s 29ms/step - loss: 2.4972 - accuracy: 0.2805 - val_loss: 3.4609 - val_accuracy: 0.1162
Epoch 392/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4865 - accuracy: 0.2839 - val_loss: 3.3873 - val_accuracy: 0.1334
Epoch 393/500
69/69 [==============================] - 2s 31ms/step - loss: 2.4982 - accuracy: 0.2894 - val_loss: 3.4632 - val_accuracy: 0.1307
Epoch 394/500
69/69 [==============================] - 2s 31ms/step - loss: 2.4989 - accuracy: 0.2867 - val_loss: 3.4814 - val_accuracy: 0.1225
Epoch 395/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4790 - accuracy: 0.2916 - val_loss: 3.4888 - val_accuracy: 0.1279
Epoch 396/500
69/69 [==============================] - 2s 34ms/step - loss: 2.4842 - accuracy: 0.2935 - val_loss: 3.4831 - val_accuracy: 0.1216
Epoch 397/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5013 - accuracy: 0.2860 - val_loss: 3.4110 - val_accuracy: 0.1334
Epoch 398/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4733 - accuracy: 0.2894 - val_loss: 3.5098 - val_accuracy: 0.1252
Epoch 399/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4999 - accuracy: 0.2873 - val_loss: 3.4435 - val_accuracy: 0.1279
Epoch 400/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4760 - accuracy: 0.2916 - val_loss: 3.4811 - val_accuracy: 0.1270
Epoch 401/500
69/69 [==============================] - 2s 30ms/step - loss: 2.4808 - accuracy: 0.2848 - val_loss: 3.4849 - val_accuracy: 0.1279
Epoch 402/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4729 - accuracy: 0.2919 - val_loss: 3.5137 - val_accuracy: 0.1316
Epoch 403/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5007 - accuracy: 0.2862 - val_loss: 3.4005 - val_accuracy: 0.1298
Epoch 404/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4856 - accuracy: 0.2941 - val_loss: 3.3973 - val_accuracy: 0.1270
Epoch 405/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5036 - accuracy: 0.2828 - val_loss: 3.4130 - val_accuracy: 0.1252
Epoch 406/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4632 - accuracy: 0.2907 - val_loss: 3.4645 - val_accuracy: 0.1216
Epoch 407/500
69/69 [==============================] - 2s 30ms/step - loss: 2.5009 - accuracy: 0.2901 - val_loss: 3.4827 - val_accuracy: 0.1243
Epoch 408/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4951 - accuracy: 0.2926 - val_loss: 3.4517 - val_accuracy: 0.1325
Epoch 409/500
69/69 [==============================] - 2s 31ms/step - loss: 2.4884 - accuracy: 0.2926 - val_loss: 3.4594 - val_accuracy: 0.1343
Epoch 410/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4616 - accuracy: 0.3021 - val_loss: 3.4099 - val_accuracy: 0.1388
Epoch 411/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5100 - accuracy: 0.2851 - val_loss: 3.4714 - val_accuracy: 0.1261
Epoch 412/500
69/69 [==============================] - 2s 30ms/step - loss: 2.4767 - accuracy: 0.2953 - val_loss: 3.5497 - val_accuracy: 0.1225
Epoch 413/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4860 - accuracy: 0.2894 - val_loss: 3.4633 - val_accuracy: 0.1379
Epoch 414/500
69/69 [==============================] - 2s 27ms/step - loss: 2.5183 - accuracy: 0.2792 - val_loss: 3.4264 - val_accuracy: 0.1397
Epoch 415/500
69/69 [==============================] - 2s 26ms/step - loss: 2.4854 - accuracy: 0.2901 - val_loss: 3.4483 - val_accuracy: 0.1370
Epoch 416/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4772 - accuracy: 0.2862 - val_loss: 3.4312 - val_accuracy: 0.1225
Epoch 417/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4917 - accuracy: 0.2728 - val_loss: 3.4921 - val_accuracy: 0.1207
Epoch 418/500
69/69 [==============================] - 2s 30ms/step - loss: 2.4493 - accuracy: 0.3005 - val_loss: 3.3976 - val_accuracy: 0.1407
Epoch 419/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4585 - accuracy: 0.3023 - val_loss: 3.3992 - val_accuracy: 0.1289
Epoch 420/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4862 - accuracy: 0.2928 - val_loss: 3.3611 - val_accuracy: 0.1388
Epoch 421/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4436 - accuracy: 0.2960 - val_loss: 3.4726 - val_accuracy: 0.1225
Epoch 422/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4751 - accuracy: 0.2964 - val_loss: 3.5200 - val_accuracy: 0.1307
Epoch 423/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4777 - accuracy: 0.2994 - val_loss: 3.4584 - val_accuracy: 0.1234
Epoch 424/500
69/69 [==============================] - 2s 35ms/step - loss: 2.4736 - accuracy: 0.2957 - val_loss: 3.3938 - val_accuracy: 0.1470
Epoch 425/500
69/69 [==============================] - 2s 32ms/step - loss: 2.4625 - accuracy: 0.2937 - val_loss: 3.3843 - val_accuracy: 0.1361
Epoch 426/500
69/69 [==============================] - 2s 29ms/step - loss: 2.4549 - accuracy: 0.2953 - val_loss: 3.4419 - val_accuracy: 0.1307
Epoch 427/500
69/69 [==============================] - 2s 29ms/step - loss: 2.4727 - accuracy: 0.2969 - val_loss: 3.4660 - val_accuracy: 0.1225
Epoch 428/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4469 - accuracy: 0.3050 - val_loss: 3.4684 - val_accuracy: 0.1307
Epoch 429/500
69/69 [==============================] - 2s 31ms/step - loss: 2.4986 - accuracy: 0.2928 - val_loss: 3.5088 - val_accuracy: 0.1207
Epoch 430/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4679 - accuracy: 0.2944 - val_loss: 3.5066 - val_accuracy: 0.1279
Epoch 431/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4792 - accuracy: 0.2951 - val_loss: 3.4398 - val_accuracy: 0.1325
Epoch 432/500
69/69 [==============================] - 2s 28ms/step - loss: 2.5048 - accuracy: 0.2844 - val_loss: 3.4413 - val_accuracy: 0.1370
Epoch 433/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4715 - accuracy: 0.2969 - val_loss: 3.4420 - val_accuracy: 0.1307
Epoch 434/500
69/69 [==============================] - 2s 30ms/step - loss: 2.4558 - accuracy: 0.2955 - val_loss: 3.4642 - val_accuracy: 0.1289
Epoch 435/500
69/69 [==============================] - 2s 29ms/step - loss: 2.4677 - accuracy: 0.2878 - val_loss: 3.4255 - val_accuracy: 0.1352
Epoch 436/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4784 - accuracy: 0.2964 - val_loss: 3.4513 - val_accuracy: 0.1343
Epoch 437/500
69/69 [==============================] - 2s 29ms/step - loss: 2.4912 - accuracy: 0.2830 - val_loss: 3.5562 - val_accuracy: 0.1207
Epoch 438/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4732 - accuracy: 0.2901 - val_loss: 3.4604 - val_accuracy: 0.1307
Epoch 439/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4739 - accuracy: 0.2951 - val_loss: 3.4083 - val_accuracy: 0.1416
Epoch 440/500
69/69 [==============================] - 2s 30ms/step - loss: 2.4820 - accuracy: 0.2826 - val_loss: 3.5001 - val_accuracy: 0.1270
Epoch 441/500
69/69 [==============================] - 2s 32ms/step - loss: 2.4580 - accuracy: 0.3005 - val_loss: 3.4973 - val_accuracy: 0.1279
Epoch 442/500
69/69 [==============================] - 2s 26ms/step - loss: 2.4728 - accuracy: 0.2994 - val_loss: 3.5524 - val_accuracy: 0.1225
Epoch 443/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4970 - accuracy: 0.2821 - val_loss: 3.4843 - val_accuracy: 0.1289
Epoch 444/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4528 - accuracy: 0.2971 - val_loss: 3.3911 - val_accuracy: 0.1443
Epoch 445/500
69/69 [==============================] - 2s 29ms/step - loss: 2.4701 - accuracy: 0.2844 - val_loss: 3.4166 - val_accuracy: 0.1370
Epoch 446/500
69/69 [==============================] - 2s 29ms/step - loss: 2.4443 - accuracy: 0.2982 - val_loss: 3.4501 - val_accuracy: 0.1307
Epoch 447/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4904 - accuracy: 0.2873 - val_loss: 3.4956 - val_accuracy: 0.1207
Epoch 448/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4309 - accuracy: 0.2991 - val_loss: 3.4357 - val_accuracy: 0.1388
Epoch 449/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4553 - accuracy: 0.2980 - val_loss: 3.4320 - val_accuracy: 0.1343
Epoch 450/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4804 - accuracy: 0.2905 - val_loss: 3.4807 - val_accuracy: 0.1279
Epoch 451/500
69/69 [==============================] - 2s 31ms/step - loss: 2.4457 - accuracy: 0.2969 - val_loss: 3.5655 - val_accuracy: 0.1243
Epoch 452/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4568 - accuracy: 0.2923 - val_loss: 3.4964 - val_accuracy: 0.1316
Epoch 453/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4483 - accuracy: 0.2941 - val_loss: 3.4746 - val_accuracy: 0.1334
Epoch 454/500
69/69 [==============================] - 2s 32ms/step - loss: 2.4870 - accuracy: 0.2801 - val_loss: 3.4189 - val_accuracy: 0.1434
Epoch 455/500
69/69 [==============================] - 2s 29ms/step - loss: 2.4679 - accuracy: 0.2885 - val_loss: 3.4355 - val_accuracy: 0.1388
Epoch 456/500
69/69 [==============================] - 2s 31ms/step - loss: 2.4629 - accuracy: 0.2969 - val_loss: 3.5262 - val_accuracy: 0.1243
Epoch 457/500
69/69 [==============================] - 3s 40ms/step - loss: 2.4481 - accuracy: 0.2978 - val_loss: 3.4564 - val_accuracy: 0.1352
Epoch 458/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4675 - accuracy: 0.2998 - val_loss: 3.4390 - val_accuracy: 0.1334
Epoch 459/500
69/69 [==============================] - 2s 29ms/step - loss: 2.4621 - accuracy: 0.2951 - val_loss: 3.4167 - val_accuracy: 0.1379
Epoch 460/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4653 - accuracy: 0.2930 - val_loss: 3.4777 - val_accuracy: 0.1261
Epoch 461/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4638 - accuracy: 0.3046 - val_loss: 3.4100 - val_accuracy: 0.1434
Epoch 462/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4553 - accuracy: 0.3064 - val_loss: 3.4731 - val_accuracy: 0.1261
Epoch 463/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4402 - accuracy: 0.2946 - val_loss: 3.4364 - val_accuracy: 0.1334
Epoch 464/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4740 - accuracy: 0.2916 - val_loss: 3.4895 - val_accuracy: 0.1261
Epoch 465/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4244 - accuracy: 0.3030 - val_loss: 3.4258 - val_accuracy: 0.1379
Epoch 466/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4339 - accuracy: 0.3062 - val_loss: 3.4925 - val_accuracy: 0.1225
Epoch 467/500
69/69 [==============================] - 2s 30ms/step - loss: 2.4608 - accuracy: 0.2946 - val_loss: 3.5302 - val_accuracy: 0.1261
Epoch 468/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4377 - accuracy: 0.3030 - val_loss: 3.4716 - val_accuracy: 0.1325
Epoch 469/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4643 - accuracy: 0.2903 - val_loss: 3.4624 - val_accuracy: 0.1325
Epoch 470/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4591 - accuracy: 0.2982 - val_loss: 3.4324 - val_accuracy: 0.1352
Epoch 471/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4784 - accuracy: 0.2903 - val_loss: 3.4702 - val_accuracy: 0.1289
Epoch 472/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4801 - accuracy: 0.2987 - val_loss: 3.5735 - val_accuracy: 0.1234
Epoch 473/500
69/69 [==============================] - 2s 35ms/step - loss: 2.4314 - accuracy: 0.2998 - val_loss: 3.4561 - val_accuracy: 0.1388
Epoch 474/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4277 - accuracy: 0.3071 - val_loss: 3.4075 - val_accuracy: 0.1443
Epoch 475/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4452 - accuracy: 0.2994 - val_loss: 3.4982 - val_accuracy: 0.1279
Epoch 476/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4311 - accuracy: 0.3078 - val_loss: 3.5476 - val_accuracy: 0.1261
Epoch 477/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4652 - accuracy: 0.2932 - val_loss: 3.4908 - val_accuracy: 0.1261
Epoch 478/500
69/69 [==============================] - 2s 29ms/step - loss: 2.4609 - accuracy: 0.3071 - val_loss: 3.5479 - val_accuracy: 0.1180
Epoch 479/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4591 - accuracy: 0.3007 - val_loss: 3.4555 - val_accuracy: 0.1298
Epoch 480/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4402 - accuracy: 0.3012 - val_loss: 3.4127 - val_accuracy: 0.1388
Epoch 481/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4344 - accuracy: 0.3066 - val_loss: 3.5016 - val_accuracy: 0.1279
Epoch 482/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4253 - accuracy: 0.3082 - val_loss: 3.5062 - val_accuracy: 0.1243
Epoch 483/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4592 - accuracy: 0.2928 - val_loss: 3.4411 - val_accuracy: 0.1388
Epoch 484/500
69/69 [==============================] - 2s 35ms/step - loss: 2.4272 - accuracy: 0.3071 - val_loss: 3.4564 - val_accuracy: 0.1352
Epoch 485/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4417 - accuracy: 0.3000 - val_loss: 3.5394 - val_accuracy: 0.1180
Epoch 486/500
69/69 [==============================] - 2s 29ms/step - loss: 2.4326 - accuracy: 0.3048 - val_loss: 3.4881 - val_accuracy: 0.1289
Epoch 487/500
69/69 [==============================] - 2s 32ms/step - loss: 2.4759 - accuracy: 0.2935 - val_loss: 3.4885 - val_accuracy: 0.1316
Epoch 488/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4336 - accuracy: 0.2982 - val_loss: 3.5205 - val_accuracy: 0.1252
Epoch 489/500
69/69 [==============================] - 3s 38ms/step - loss: 2.4658 - accuracy: 0.3000 - val_loss: 3.4032 - val_accuracy: 0.1388
Epoch 490/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4586 - accuracy: 0.2894 - val_loss: 3.4974 - val_accuracy: 0.1261
Epoch 491/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4052 - accuracy: 0.3069 - val_loss: 3.5461 - val_accuracy: 0.1270
Epoch 492/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4216 - accuracy: 0.3187 - val_loss: 3.4453 - val_accuracy: 0.1352
Epoch 493/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4483 - accuracy: 0.3041 - val_loss: 3.5244 - val_accuracy: 0.1261
Epoch 494/500
69/69 [==============================] - 2s 29ms/step - loss: 2.4591 - accuracy: 0.2960 - val_loss: 3.4936 - val_accuracy: 0.1252
Epoch 495/500
69/69 [==============================] - 2s 29ms/step - loss: 2.4518 - accuracy: 0.2869 - val_loss: 3.3675 - val_accuracy: 0.1434
Epoch 496/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4492 - accuracy: 0.2926 - val_loss: 3.4696 - val_accuracy: 0.1307
Epoch 497/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4395 - accuracy: 0.3066 - val_loss: 3.4746 - val_accuracy: 0.1307
Epoch 498/500
69/69 [==============================] - 2s 27ms/step - loss: 2.4485 - accuracy: 0.2960 - val_loss: 3.4124 - val_accuracy: 0.1370
Epoch 499/500
69/69 [==============================] - 2s 28ms/step - loss: 2.4370 - accuracy: 0.2969 - val_loss: 3.4688 - val_accuracy: 0.1352
Epoch 500/500
69/69 [==============================] - 2s 30ms/step - loss: 2.4437 - accuracy: 0.3121 - val_loss: 3.4359 - val_accuracy: 0.1397

<tensorflow.python.keras.callbacks.History at 0x7f705d87b3d0>

Test Data

img_width = 150;
img_height = 150;
z=[];

for i in tqdm (range(test.shape[0])):
    pathe = '../input/identify-snake-breed-hackerearth/dataset/test/' + test['image_id'][i] + '.jpg'
    img1 = image.load_img(pathe, target_size = (img_width, img_height, 3))
    img1 = image.img_to_array(img1)
    img1 = img1/255.0
    #img1 = img1.reshape(1,img_width, img_height, 3 )
    z.append(img1)



z

z = np.array(z)


res = mod.predict(z)

res = np.array(res)

Getting the Submission File Ready

cl = train['breed'].unique()

c = res.argmax(axis = 1);
c.shape



ped = [cl [index] for index in c];
hehe = pd.DataFrame({'breed': ped})

hehe.set_index('breed').to_csv('submissionnew.csv')

