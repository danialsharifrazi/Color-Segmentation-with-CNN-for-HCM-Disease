
def Segmentor(img):
    threshold=70
    import numpy as np
    import cv2

    b,g,r=128,128,128
    bgr=np.uint8([[[128,128,128]]])
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    h,s,v=int(h[0,0]),int(s[0,0]),int(v[0,0])

    light_green=(b-threshold,g-threshold,r-threshold)
    dark_green=(b+threshold,g+threshold,r+threshold)

    mask_green=cv2.inRange(img,light_green,dark_green)
    result_green=cv2.bitwise_and(img,img,mask=mask_green)

    result_green=cv2.cvtColor(result_green,cv2.COLOR_HSV2BGR)
    return result_green



