from pathlib import Path
import os

import re
import cairosvg
import random
import string
#递归遍历文件夹 输出png后缀的文件路径

#dir = "/Users/sam/material-design-icons/src"    
dir = "/Users/sam/dataset/material_icon"


def find_png_files(root_dir):
    root_path = Path(root_dir)
    return list(root_path.rglob('*.png'))




# 示例用法
png_paths = find_png_files(dir)
print(len(png_paths))
a = set(list(map(lambda x: x.name, png_paths)))
print(a)
for path in png_paths[:20]:
    print(path.resolve())  
    #print(path.name)

#都是svg图 两种路径模式提取标签 
#/Users/sam/material-design-icons/src/{type}/{label}/xxxx/xxxx/xxxx48px.svg

skipSet = set()
#只处理特定的icon_type
icon_type_set = set(["navigation","search","toggle", "home", "file"])
icon_data = {}
test_path = [path.resolve() for path in png_paths]
pattern = r"src/([^/]+)/([a-zA-Z0-9_]+)"
for path in test_path: 
    str_path = str(path) 
    parent = path.parent
    match = re.search(pattern, str_path)
    if match:
        icon_type = match.group(1)   # 提取 {type} 部分
        label = match.group(2)  # 提取 {label} 部分
        if parent in skipSet:
            continue
        skipSet.add(parent)
        if icon_type not in icon_type_set:
            continue
        print(icon_type, label)  # 输出: account_balance
        if label not in icon_data:
            icon_data[label] = []
        icon_data[label].append(str_path)
        
    else:
        print("未匹配s到 label")

print(len(icon_data))


#按label 转换成png写入文件 最终都会转换成灰度图
output_dir = "/Users/sam/dataset/material_design"
for label, paths in icon_data.items():
    #检查目录是否存在 不存在就新建
    label_dir = os.path.join(output_dir, label) 
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    for path in paths:
        prefix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        cairosvg.svg2png(url=path, 
                # SVG 转 JPG（默认输出 PNG，需通过 PIL 转 JPG）
                output_width=64,
                output_height=64,
                write_to=os.path.join(label_dir, prefix+"_"+os.path.basename(path).replace(".svg", ".png")))





