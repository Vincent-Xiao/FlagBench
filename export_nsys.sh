#!/bin/bash
set -e

# 定义要处理的目录，先cuda后gems
dirs=("nsys-cuda" "nsys-gems")

for d in "${dirs[@]}"; do
    if [ ! -d "$d" ]; then
        echo "Directory $d not found, skipping."
        continue
    fi

    echo "Processing directory: $d"

    # 遍历目录下的所有 .nsys-rep 文件
    # 使用 find 加上 sort 确保处理顺序相对稳定，或者直接 glob
    # 考虑到可能没有文件，开启 nullglob 更好，但在 bash 脚本中 find 比较通用
    
    find "$d" -maxdepth 1 -name "*.nsys-rep" | while read -r report_file; do
        # 获取不带扩展名的文件名 (例如 report-cuda-tts)
        filename=$(basename -- "$report_file")
        basename="${filename%.*}"
        
        # 输出文件路径 (例如 nsys-cuda/report-cuda-tts.sqlite)
        sqlite_out="$d/$basename.sqlite"
        txt_out="$d/$basename.txt"

        echo "  Converting $report_file..."

        echo "    -> SQLite: $sqlite_out"
        nsys export --type sqlite --force-overwrite true -o "$sqlite_out" "$report_file"

        # echo "    -> Stats: $txt_out"
        # nsys stats --force-overwrite true "$report_file" > "$txt_out"
    done
done

echo "Done."
