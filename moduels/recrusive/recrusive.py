#指定したフォルダのファイル一覧を取得

import os

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)

for file in find_all_files('./test'):
    path = file.replace('\\','/')
    if '.jpg' in path:
        arr = path.split("/")
        print(arr[len(arr)-1])