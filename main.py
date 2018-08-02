import cv2, os, json
import numpy as np
from PIL import Image

train_path = './train'
test_path = './test'
train_data = "train.yml"
is_train = os.path.isfile(train_data)

#Haar-like特徴分類器で顔を認識するための準備
cascadePath = "./haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer = cv2.face.EigenFaceRecognizer_create()

# labelをintで管理するため
human_labels = {}
result = {}

def init_labels(path):
    count = 0
    for f in os.listdir(path):
        if "DS_Store" in f:
            continue

        # 解析で渡せるラベルはintだけなので、それぞれの人をintに割り当てる
        use_label = get_label(f)
        if use_label not in human_labels:
            human_labels[use_label] = count
            count = count+1
        

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
        use_label = get_label(f)
        labels.append(human_labels[use_label])
        # 検出した顔画像の処理
        for (x, y, w, h) in faces:
            # 検出した顔の部分をクリップして 200x200 サイズにリサイズ
            roi = cv2.resize(image[y: y + h, x: x + w], (200, 200), interpolation=cv2.INTER_LINEAR)
            images.append(roi)
            # ファイル名を配列に格納
            files.append(f)
    return images, labels, files

def get_label(filename):
    filename_arr = filename.split('_')
    return filename_arr[0]+"_"+filename_arr[1]

init_labels(train_path)

if(is_train):
    recognizer.read(train_data)
else:
    images, labels, files = get_images_and_labels(train_path)
    recognizer.train(images, np.array(labels))
    recognizer.save(train_data)


# # テスト画像を取得
test_images, test_labels, test_files = get_images_and_labels(test_path)

i = 0


def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]


while i < len(test_labels):
    # テスト画像に対して予測実施
    label, confidence = recognizer.predict(test_images[i])
    
    # 予測結果をコンソール出力
    print("Test Image: {}, Predicted Label: {}, Confidence: {}".format(test_files[i], label, confidence))
    # テスト画像を表示
    cv2.imshow("test image", test_images[i]) 
    result["label"] = label
    result["confidence"] = confidence
    result["label_str"] = get_keys_from_value(human_labels,label)[0]
    print(json.dumps(result))
    i += 1


# 終了処理
cv2.destroyAllWindows()