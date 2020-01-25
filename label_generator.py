import pandas as pd
import numpy as np
import imghdr
import os
img_num = 100
name=''
names=''
directory = 'long-march5/'
j=0
for i in range(100):
    name = directory+str(i+1)
    try:
        imgType = imghdr.what(name)
    except:
        imgType = 'empty'
    if imgType!='empty':
        print(imgType)
        newName = (i-j+1 + '.' + imgType)
        os.rename(name,newName)
    else:
        print('the file is not a pic or is empty')
        imgType = 'empty'
        #os.remove(name)
    names = names+newName+','
names=names.split(',')[0:img_num]
#print(names)
labels=np.zeros(100)
img_info = pd.Series(labels,names)
img_info.to_csv('long-march5/labels.csv')