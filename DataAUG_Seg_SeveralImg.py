import numpy as np
import cv2
import glob
import os
from ColorSegmentor import Segmentor

#read normals files
normals=[]
main_path='./Converted Dataset/Normal/'
main_folders=next(os.walk(main_path))[1]
for i in main_folders:
    path=main_path+i+'/'
    folders=next(os.walk(path))[1]
    for x in folders:
        new_path=path+x+'/'
        data=glob.glob(new_path+'*.jpg')
        if len(data)<1:
            indent_folders=next(os.walk(new_path))[1]
            for y in indent_folders:
                new_path=new_path+y+'/'
                data=glob.glob(new_path+'*.jpg')
        normals.extend(data)



#read sicks files
sicks=[]
main_path='./Converted Dataset/Sick/'
main_folders=next(os.walk(main_path))[1]
for i in main_folders:
    path=main_path+i+'/'
    folders=next(os.walk(path))[1]
    for x in folders:
        new_path=path+x+'/'
        data=glob.glob(new_path+'*.jpg')
        if len(data)<1:
            indent_folders=next(os.walk(new_path))[1]
            for y in indent_folders:
                new_path=new_path+y+'/'
                data=glob.glob(new_path+'*.jpg')
        sicks.extend(data)
    


counter=1
#load normal files
data_n=[]
for id in normals:    
    img=cv2.imread(id)
    img=Segmentor(img)
    cv2.imwrite(f'./Converted Dataset/Normal_AUG_BGRHSV/Normal_{counter}.jpg',img)
    counter+=1


counter=1
#load sick files
data_s=[]
for id in sicks:    
    img=cv2.imread(id)
    img=Segmentor(img)
    cv2.imwrite(f'./Converted Dataset/Sick_AUG_BGRHSV/Sick_{counter}.jpg',img)
    counter+=1

