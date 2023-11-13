import os
import pandas as pd
import subprocess
from tqdm import tqdm

def is_video_corrupted(file_path):
    try:
        # 使用ffprobe的命令行工具进行文件检测
        subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries','stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1', file_path],stderr=subprocess.STDOUT)
        return False  # 如果没有抛出异常，文件没有损坏
    except subprocess.CalledProcessError as e:
        return True  # 文件损坏


def batch_check_corruption(file_path,broken_files):
    # 遍历指定目录下的所有文件
    if is_video_corrupted(file_path):
        print(f'文件 {file_path} 损坏.')
        broken_files.append(file_path)
    # else:
    #     print(f'文件 {file_path} 正常.')

if __name__ == '__main__':

    # 读取CSV文件
    df = pd.read_csv('data/train.csv')
    count =0 

    # 获取video_file列
    video_files = df['video_file']


    missing_files =[]
    files=[]
    broken_files=[]
    # 检查每个文件是否存在
    for file in tqdm(video_files):
        if not os.path.exists("/home/zehong/Desktop/NLP/VG-SUM/data/"+file):
            missing_files.append(file)
            count = count+1
        else:
            files.append(file)

    # 输出不存在的文件
    if missing_files:
        print("以下文件不存在:")
        for missing_file in missing_files:
            print(missing_file)
        print(count)
    else:
        print("所有文件都存在.")

    for file in tqdm(files):
        batch_check_corruption("/home/zehong/Desktop/NLP/VG-SUM/data/"+file,broken_files)
    print(broken_files)

    

