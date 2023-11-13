import os
import subprocess

def is_video_corrupted(file_path):
    try:
        # 使用ffmpeg的命令行工具进行文件检测
        subprocess.check_output(['ffmpeg', '-v', 'error', '-i', file_path, '-f', 'null', '-'],
                                stderr=subprocess.STDOUT)
        return False  # 如果没有抛出异常，文件没有损坏
    except subprocess.CalledProcessError as e:
        error_output = e.output.decode('utf-8')
        # 在错误输出中查找特定的错误标志，例如 "Error while decoding stream"
        if 'Error while decoding stream' in error_output:
            return True  # 文件损坏
        return False  # 其他错误

def batch_check_corruption(directory):
    # 遍历指定目录下的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv')):
                file_path = os.path.join(root, file)
                if is_video_corrupted(file_path):
                    print(f'文件 {file_path} 损坏.')
                else:
                    print(f'文件 {file_path} 正常.')

if __name__ == "__main__":
    # 指定要检测的目录
    directory_to_check = '/path/to/your/directory'
    
    # 执行批量检测
    batch_check_corruption(directory_to_check)
