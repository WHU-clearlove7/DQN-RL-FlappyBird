import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import csv

plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False

csv=pd.read_csv('analysis.csv')
csv = csv.T
list = csv.values.tolist()
# print(list)
x = list[0]
y = list[1]

plt.scatter(x,y,s = 0.7)
plt.xlabel('ITERATION') 
plt.ylabel('SCORE')

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.grid() 
plt.savefig('analysis_pic', dpi=300, bbox_inches = 'tight')
plt.show()
