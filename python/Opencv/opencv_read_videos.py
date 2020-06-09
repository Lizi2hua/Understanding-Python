import cv2


# 创建视频对象
# cap=cv2.VideoCapture('http://ivi.bupt.edu.cn/hls/cctv1.m3u8')
cap=cv2.VideoCapture(r'C:\Users\Administrator\Desktop\test.mp4')
# cv2.VideoCapture(0)
# 0代表摄像头
while True:
    ret,frame=cap.read()
#     ret bool型，表示是否读取成功，frame每帧画面
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # print(frame)
    resize_factor=1/25
    w,h=frame.shape
    frame=cv2.resize(frame,(int(w*resize_factor),int(h*resize_factor)))
    w,h=frame.shape
    # frame=cv2.resize(frame,(w//2,h//2),interpolation=cv2.INTER_NEAREST)

    cv2.imshow('Bad Apple!',frame)
    if cv2.waitKey(31) & 0xFF==ord('q'):
        break
cap.release()
# 释放视频资源
cv2.destroyWindow('Bad Apple!')
# 关闭窗口qq