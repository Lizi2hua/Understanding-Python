import cv2

input_dir=r'C:\Users\Administrator\Desktop\test.mp4'
file_name="test"
output_dir=r'C:\Users\Administrator\Desktop\testdata_store'
# 第一步：读取视频
cap=cv2.VideoCapture(input_dir)
# 第二步：读取帧并保存
frame_count=0
while True:
    frame_count+=1
    ret,frame=cap.read()
    file_path=output_dir+'/'+file_name+'{}.jpg'.format(frame_count)
    cv2.imwrite(file_path,frame)
cap.release()