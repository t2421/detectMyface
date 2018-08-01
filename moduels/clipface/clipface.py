# @see https://qiita.com/ShirataHikaru/items/a1dab6c6b5ba088123e0

import cv2

cascade_path = "./haarcascade_frontalface_alt.xml"
origin_image_path = "lenna.jpg"
dir_path = "./"

image = cv2.imread(origin_image_path,0)
if image is None:
    quit()

cascade = cv2.CascadeClassifier(cascade_path)
facerect = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(10, 10))

if len(facerect) > 0:
    for rect in facerect:
        # 顔だけ切り出して保存
        x = rect[0]
        y = rect[1]
        width = rect[2]
        height = rect[3]
        dst = image[y:y + height, x:x + width]
        save_path = dir_path + '/' + 'clip_image' + '.jpg'
        #認識結果の保存
        cv2.imwrite(save_path, dst)
        print("save!")