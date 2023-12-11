import csv
import os

# CSV文件路径
csv_file = './data/demo/demo.csv'  # 替换为你的CSV文件路径

# 打开CSV文件并读取内容
with open(csv_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    count = 0
    
    for row in reader:
        # 提取文件前缀和描述文本
        file_prefix = row['id']
        description_path = row['text']
        trans_txt = row['transport']

        
        # 生成.txt文件名
        txt_filename = 'demo_summary.txt'   # summary
        desc_filename = 'demo_desc.txt'  # 语音文本

        # 读取.description文本内容
        if os.path.exists(description_path):
            count = count+1
            with open(description_path, 'r', encoding='utf-8') as description_file:
                description_text = description_file.read()
                
                # 生成.txt文件并写入内容
                with open(txt_filename, 'a', encoding='utf-8') as txt_file:
                    txt_file.write(f"{file_prefix} {description_text.strip()}\n")
                    print(f"已写入{count}行内容。")

                # 生成.txt文件并写入内容
                with open(desc_filename, 'a', encoding='utf-8') as txt_file:
                    txt_file.write(f"{file_prefix} {trans_txt.strip()}\n")
                    print(f"已写入{count}行内容。")
        else:
            print(f"文件 '{description_path}' 不存在。")
            
print("生成完成")
