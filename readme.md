## 使用している画像データ
http://vis-www.cs.umass.edu/lfw/index.html

## pip modules
+ opencv-contrib-python
+ opencv-python
+ numpy

## つまづいたこと
pipのモジュールが足りずに、エラーが出た。  
```
recognizer = cv2.face.EigenFaceRecognizer_create()
AttributeError: module 'cv2.cv2' has no attribute 'face'
```
@see https://stackoverflow.com/questions/45655699/attributeerror-module-cv2-face-has-no-attribute-createlbphfacerecognizer
を見て以下でopencvをアップデートしたら動いた  
`python -m pip install opencv-contrib-python`  

## 実行
`python main.py`  
1. trainフォルダに入っている画像から顔の部分を抽出
2. EigenFaceRecognizerで1の画像を学習
3. testフォルダに入っている画像から顔の部分を抽出
4. 3の画像をrecognizerでテスト、結果を出力する