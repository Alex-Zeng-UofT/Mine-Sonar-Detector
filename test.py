import pandas as pd 
import numpy as np

sonar_data = pd.read_csv('data.csv', header=None)
print(sonar_data[0])
