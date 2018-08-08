import shutil
import os
for f in os.listdir("./trained/"):
    shutil.move("./trained/"+f,"./train")

os.remove("train.yml")
os.remove("labels.json")