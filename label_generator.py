import pandas as pd
import numpy as np
img_num = 100
names=''
for i in range(100):
    names = names+str(i+1)+'.jpg,'
names=names.split(',')[0:img_num]
#print(names)
labels=np.zeros(100)
img_info = pd.Series(labels,names)
img_info.to_csv('long-march1/labels.csv')