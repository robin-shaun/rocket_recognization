import imghdr,os
#filename = 'img.py'
filename = 'long-march5/85.jpg'
imgType = imghdr.what(filename)
if imgType:
   print(imgType)
   newName = (filename.split('.')[0] + '.' + imgType)
   os.rename(filename,newName)
else:
   print('the file is not a pic,rm it now')
   os.remove(filename)