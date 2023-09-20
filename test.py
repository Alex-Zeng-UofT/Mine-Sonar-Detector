import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv('data.csv', header=None)

data = sonar_data.drop(columns=60, axis=1)
labels = sonar_data[60]

d_train, d_test, l_train, l_test = train_test_split(data, labels, test_size = 0.1, stratify=labels, random_state=1)

model = LogisticRegression()
model.fit(data, labels)

input_data = (0.0139,0.0222,0.0089,0.0108,0.0215,0.0136,0.0659,0.0954,0.0786,0.1015,0.1261,0.0828,0.0493,0.0848,0.1514,0.1396,0.1066,0.1923,0.2991,0.3247,0.3797,0.5658,0.7483,0.8757,0.9048,0.7511,0.6858,0.7043,0.5864,0.3773,0.2206,0.2628,0.2672,0.2907,0.1982,0.2288,0.3186,0.2871,0.2921,0.2806,0.2682,0.2112,0.1513,0.1789,0.1850,0.1717,0.0898,0.0656,0.0445,0.0110,0.0024,0.0062,0.0072,0.0113,0.0012,0.0022,0.0025,0.0059,0.0039,0.0048)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a mine')
