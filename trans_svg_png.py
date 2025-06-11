import os
import sys
import cairosvg

def convert_svgs_to_pngs(input_dir, output_dir, width, height):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    i = 0
    for root, _, files in os.walk(input_dir):    
        for filename in files:
            if filename.lower().endswith('.svg'):
                filename_no_ext = filename.split('.')[0]
                svg_path = os.path.join(root, filename)
                png_filename = os.path.splitext(filename_no_ext)[0] + '.png'
                i = i + 1
                png_path = os.path.join(output_dir, f'{i}_{png_filename}')
                cairosvg.svg2png(
                    url=svg_path,
                    write_to=png_path,
                    output_width=width,
                    output_height=height
                )
                print(f"Converted {svg_path} -> {png_path}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("用法: python main.py <svg目录> <输出目录> <宽度> <高度>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    width = int(sys.argv[3])
    height = int(sys.argv[4])
    convert_svgs_to_pngs(input_dir, output_dir, width, height) 