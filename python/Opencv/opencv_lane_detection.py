import cv2
import numpy as np

def calcTheta(lines):
    x0,y0,x1,y1=lines[0]
    return (y1-y0)/(x1-x0)

def reject_abnormal_lines(lines,threshold):
    slope=[calcTheta(line) for line in lines]

    while len(lines)>0:
        mean=np.mean(slope)
        diff=[abs(s-mean) for s in slope]
        idx=np.argmax(diff)
        if diff[idx]>threshold:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break
    return lines

def least_square_fit(lines):
    # 得到所有的坐标
    x_coords=np.ravel([[line[0][0],line[0][2]]for line in lines])
    y_coords=np.ravel([[line[0][1],line[0][3]]for line in lines])
    # 进行直线拟合
    poly=np.polyfit(x_coords,y_coords,deg=1)
    # 根据多项式系数，计算两个直线上的点，用于唯一确定这条直线,计算可得最小x,最小y
    point_min=(np.min(x_coords),np.polyval(poly,np.min(x_coords)))
    point_max=(np.max(x_coords),np.polyval(poly,np.max(x_coords)))
    return np.array([point_min,point_max],dtype=np.int)


def showlane(img_src,img):
    #
    # dst=cv2.Canny(img,20,70)
    # ROI
    # mask
    x=10
    y=29
    mask=np.zeros_like(img)
    mask=cv2.fillPoly(mask,np.array([[[423,879-y],[834,648-y],[901,643-y],[1192,879-y]]]),color=255)
    masked=cv2.bitwise_and(mask,img)
    print(masked)
    # canny
    canny=cv2.Canny(masked,10,65)
    print(canny)
    # hough得到直线
    lines=cv2.HoughLinesP(canny,1,np.pi/180,10,minLineLength=40,maxLineGap=20)
    print(lines)

    # 判断斜率，区分左右车道线
    left_lines=[line for line in lines if calcTheta(line)>0]
    right_lines=[line for line in lines if calcTheta(line)<0]

    # print(left_lines)
    # print(lines[0])
    # 离群值过滤

    reject_abnormal_lines(left_lines,threshold=0.2)
    # print(len(left_lines))
    reject_abnormal_lines(right_lines,threshold=0.2)
    # print(len(right_lines))

    # 最小二乘拟合
    left_coord=least_square_fit(left_lines)
    right_coord=least_square_fit(right_lines)
    print(left_coord)
    print(right_coord)

    # 直线绘制

    cv2.line(img_src,tuple(left_coord[0]),tuple(left_coord[1]),color=(0,255,255),thickness=5)
    cv2.line(img_src,tuple(right_coord[0]),tuple(right_coord[1]),color=(0,255,255),thickness=5)

    return  img_src


cap=cv2.VideoCapture(r'C:\Users\Administrator\Desktop\lane1.mp4')
# img_src=cv2.imread(r'C:\Users\Administrator\Desktop\lane1.jpg')
# print(img_src)
while True:
    ret,img_src=cap.read()
    # img_src=np.swapaxes(img_src,1,0)
    img=cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
    # print(img_src)
    lane=showlane(img_src,img)
    cv2.imshow('lane',img_src)
    cv2.waitKey(20)