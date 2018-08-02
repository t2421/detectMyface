import cv2, os
import numpy as np
from PIL import Image

print(cv2.__version__)
train_path = './train'
test_path = './test'

#顔を認識する
cascadePath = "./haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer = cv2.face.EigenFaceRecognizer_create()
human_labels = {
    "Al_Gore":0,
    "John_Travolta":1,
    "shun_kiyo":2
}

# 指定されたpath内の画像を取得
def get_images_and_labels(path):
    # 画像を格納する配列
    images = []
    # ラベルを格納する配列
    labels = []
    # ファイル名を格納する配列
    files = []
    for f in os.listdir(path):
        if "DS_Store" in f:
            continue
        # print(os.path.join(path, f))
        # 画像のパス
        image_path = os.path.join(path, f)
        # グレースケールで画像を読み込む
        image_pil = Image.open(image_path).convert('L')
        # NumPyの配列に格納
        image = np.array(image_pil, 'uint8')
        if image is None:
            return
        # Haar-like特徴分類器で顔を検知
        faces = faceCascade.detectMultiScale(image)
        # 検出した顔画像の処理
        for (x, y, w, h) in faces:
            # 顔を 200x200 サイズにリサイズ
            roi = cv2.resize(image[y: y + h, x: x + w], (200, 200), interpolation=cv2.INTER_LINEAR)
        #     # 画像を配列に格納
            images.append(roi)
            # cv2.imwrite("./"+"clip_"+f, roi)
        #     # ファイル名からラベルを取得
            labels.append(human_labels[get_label(f)])
            
        #     # ファイル名を配列に格納
            files.append(f)
    
    return images, labels, files

def get_label(filename):
    filename_arr = filename.split('_')
    return filename_arr[0]+"_"+filename_arr[1]

# # トレーニング画像を取得
images, labels, files = get_images_and_labels(train_path)

# # トレーニング実施
recognizer.train(images, np.array(labels))

# # テスト画像を取得
test_images, test_labels, test_files = get_images_and_labels(test_path)

i = 0
while i < len(test_labels):
    # テスト画像に対して予測実施
    
    label, confidence = recognizer.predict(test_images[i])
    
    # 予測結果をコンソール出力
    print("Test Image: {}, Predicted Label: {}, Confidence: {}".format(test_files[i], label, confidence))
    # テスト画像を表示
    cv2.imshow("test image", test_images[i])
    
    cv2.waitKey(300)

    i += 1

# 終了処理
cv2.destroyAllWindows()