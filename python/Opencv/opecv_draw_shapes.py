import cv2
img=cv2.imread(r"images\1.jpg")
# 矩形
cv2.rectangle(img,(100,30),(210,180),color=(0,0,255),thickness=2 )

# 文字
font=cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img,'girl',(210,30),font,0.5,(0,0,0),1)

cv2.imshow("rectangle",img)
cv2.waitKey(0)
cv2.destroyAllWindows()