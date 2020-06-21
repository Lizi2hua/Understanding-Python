import cv2
import os

image_path=r'C:\Users\Administrator\Desktop\dataset\mchar_train'
files=os.listdir(image_path)

for i in files[0:50]:
    img_src=cv2.imread(image_path+'/'+i)
     # guass
    img=cv2.GaussianBlur(img_src,(3,3),0)
    # median
    img=cv2.medianBlur(img,3)
    img_resize=cv2.resize(img,(80,60),interpolation=cv2.INTER_CUBIC )
    # print(img_resize.shape)
    # img_canny=cv2.Canny(img_resize,50,200)

    cv2.imshow('raw_pic{}'.format(i),img_src)
    cv2.imshow('pic{}'.format(i),img_resize)

    cv2.waitKey(1000)
    cv2.destroyAllWindows()



