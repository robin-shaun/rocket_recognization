import pandas as pd
import numpy as np
import imghdr
import os
img_num = 100
name=''
names=''
directory = 'data/src/9/'
j=0
num=0
for i in range(img_num):
    name = directory+str(i+1)
    try:
        imgType = imghdr.what(name)
    except:
        imgType = None
    if imgType=='jpeg' or imgType == 'png':
        print(imgType)
        num=i-j+1
        newName = (directory+str(num) + '.' + imgType)
        os.rename(name,newName)
        names = names+newName+','
    else:
        print('the file is invalid')
        #imgType = 'empty'
        j=j+1
        try:
            os.remove(name)
        except:
            pass
names=names.split(',')[0:num]
#print(names)
labels=np.ones(num)
labels = labels.astype(int)
img_info = pd.Series(labels,names)
#img_info.to_csv('long-march5/labels.csv')