'''
Basic cv2 operation
'''
import os
import cv2


# cap = cv2.VideoCapture(0) #開啟第0號camera

# frame_in_w = 640
# frame_in_h = 480
# videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w) #設定擷取寬度
# videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h) #設定擷取高度

# print("capture device is open: " + str(videoIn.isOpened()))

# while cap.isOpened() #判別camera是否開啟成功
#     ret, outframe = cap.read() #捕獲影像
#     if cap.waitKey(0) & 0xff == ord('p') #若按下p則終止camera
#         break
        




# # while(True):
# #   # 從攝影機擷取一張影像
# #   ret, frame = cap.read()

# #   # 顯示圖片
# #   cv2.imshow('frame', frame)

# #   # 若按下 q 鍵則離開迴圈
# #   if cv2.waitKey(1) & 0xFF == ord('q'):
# #     break

# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print("Image Size: %d x %d" % (width, height))

# # 釋放攝影機
# cap.release()

# # 關閉所有 OpenCV 視窗
# # cv2.destroyAllWindows()


for m in range(2):
    print(m)
