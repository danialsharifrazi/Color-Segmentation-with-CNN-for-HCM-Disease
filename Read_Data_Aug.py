import numpy as np
import cv2
import glob
import os


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


#load normal files
labels_n=[]
data_n=[]
for id in normals:    
    img=cv2.imread(id)
    if np.max(img) !=0:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(50,50))
        if np.max(img) !=0:
            img=img/np.max(img)
            data_n.append(img)
            labels_n.append(0)

   
#load sick files
labels_s=[]
data_s=[]
for id in sicks:    
    img=cv2.imread(id)
    if np.max(img) !=0:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(50,50))
        if np.max(img) !=0:
            img=img/np.max(img)
            data_s.append(img)
            labels_s.append(1)



# read augmented data
normals_aug=[]
main_path='D:/A Researcher/Articles/New Cardiac/Converted Dataset/Normal_AUG_Gray/'
normals_aug=glob.glob(main_path+'*.jpg')+glob.glob(main_path+'*.png')

sicks_aug=[]
main_path='D:/A Researcher/Articles/New Cardiac/Converted Dataset/Sick_AUG_Gray/'
sicks_aug=glob.glob(main_path+'*.jpg')+glob.glob(main_path+'*.png')


# load normal_aug files
labels_n_aug=[]
data_n_aug=[]
for id in normals_aug:    
    img=cv2.imread(id)
    if np.max(img) !=0:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(50,50))
        if np.max(img) !=0:
            img=img/np.max(img)
            data_n_aug.append(img)
            labels_n_aug.append(0)




# load sick_aug files
labels_s_aug=[]
data_s_aug=[]
for id in sicks_aug:    
    img=cv2.imread(id)
    if np.max(img) !=0:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(50,50))
        if np.max(img) !=0:
            img=img/np.max(img)
            data_s_aug.append(img)
            labels_s_aug.append(1)



# Extend Data
data_n.extend(data_n_aug)
data_s.extend(data_s_aug)
data_n.extend(data_s)

labels_n.extend(labels_n_aug)
labels_s.extend(labels_s_aug)
labels_n.extend(labels_s)


# Retype to numpy all data
x_data=np.array(data_n)
y_data=np.array(labels_n)
print('data with aug: ',x_data.shape,y_data.shape)


import CNN
CNN.DeepCNN(x_data,y_data)

