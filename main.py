import os
import pandas as pd

if __name__ == '__main__':

    # 读取CSV文件
    df = pd.read_csv('data/train.csv')
    count =0 

    # 获取video_file列
    video_files = df['video_file']

    # 检查每个文件是否存在
    missing_files = []
    for file in video_files:
        if not os.path.exists("/home/zehong/Desktop/NLP/VG-SUM/data/"+file):
            missing_files.append(file)
            count = count+1

    # 输出不存在的文件
    if missing_files:
        print("以下文件不存在:")
        for missing_file in missing_files:
            print(missing_file)
        print(count)
    else:
        print("所有文件都存在.")
