import cv2, os, json
import numpy as np
from PIL import Image
import shutil
import json

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
    count = get_start_count()
    for f in os.listdir(path):
        if "DS_Store" in f:
            continue

        # 解析で渡せるラベルはintだけなので、それぞれの人をintに割り当てる
        use_label = get_label(f)
        if use_label not in human_labels:
            human_labels[use_label] = count
            count = count+1

# jsonが既に存在していたらそのjsonの続きから採番をする        
def get_start_count():
    tmp = 0
    for id in human_labels:
        if tmp < int(human_labels[id]):
            tmp = int(human_labels[id])
    print(tmp)
    return tmp

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
       
        main_face = get_main_face(faces)
        (x,y,w,h) = (main_face[0],main_face[1],main_face[2],main_face[3])
        
        # 検出した顔の部分をクリップして 200x200 サイズにリサイズ
        roi = cv2.resize(image[y: y + h, x: x + w], (200, 200), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(use_label+".jpg",roi)
        images.append(roi)
        # ファイル名を配列に格納
        files.append(f)
    return images, labels, files


# 画像から主として表示されている顔を抽出する
def get_main_face(faces):
    main_face = []
    tmp_width = 0
    # 検出した顔画像の処理
    # 2つ以上検出されたときは大きいほうをメインとする
    for (x1, y1, w1, h1) in faces:
        if tmp_width < w1:
            tmp_width = w1
            main_face = [x1,y1,w1,h1]
    return main_face


def get_label(filename):
    filename_arr = filename.split('_')
    return filename_arr[0]+"_"+filename_arr[1]

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def load_json(path):
    f = open("labels.json","r")
    json_data = json.load(f)
    return json_data

# jsonの読み書き
if os.path.exists('./labels.json'):
    tmp_labels = load_json('./labels.json')  
    human_labels = tmp_labels
    
init_labels(train_path)
fw = open('labels.json','w')
json.dump(human_labels,fw,indent=4)



mkdir("./trained")
if(is_train):
    images, labels, files = get_images_and_labels(train_path)
    recognizer.read(train_data)
    if images:
        recognizer.train(images, np.array(labels))
        recognizer.save(train_data)
        
else:
    images, labels, files = get_images_and_labels(train_path)
    if images:
        recognizer.train(images, np.array(labels))
        recognizer.save(train_data)
    else:
        print("train data is empty")

for filename in files:
    print(filename)
    shutil.move("./train/"+filename,"./trained")

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