from pathlib import Path
from fsmnvad import FSMNVad
import os
import subprocess
import json


def convert_to_16000hz(input_path, output_path):
    """
    将输入视频或音频文件转换为 16000 Hz 的 WAV 文件
    :param input_path: 输入文件路径
    :param output_path: 输出 WAV 文件路径
    """
    # 如果输出文件已存在，先删除它
    if os.path.exists(output_path):
        os.remove(output_path)
    
    command = [
        'ffmpeg', '-i', input_path,  # 输入文件
        '-ar', '16000',  # 设置采样率为 16000 Hz
        '-ac', '1',  # 单声道
        output_path  # 输出文件
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def generate_srt_from_segments(segments, output_srt_path):
    """
    根据切分的语音片段生成SRT字幕文件
    :param segments: 语音片段列表，格式为 [(start_time, end_time), ...]
    :param output_srt_path: 输出的SRT文件路径
    """
    with open(output_srt_path, 'w', encoding='utf-8') as srt_file:
        for i, (start_time, end_time) in enumerate(segments, start=1):
            # 格式化时间戳为SRT格式 (HH:MM:SS,ms)
            start_time = start_time / 1000
            end_time = end_time / 1000

            start_time_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time % 1) * 1000):03}"
            end_time_str = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time % 1) * 1000):03}"
            
            # 写入序号、时间戳和内容
            srt_file.write(f"{i}\n")
            srt_file.write(f"{start_time_str} --> {end_time_str}\n")
            srt_file.write(f"语音片段 {i}\n\n")


def extract_audio_segments(srt_path, wav_path, output_folder, base_name,prompt):
    """
    根据SRT文件切分WAV文件并生成JSONL文件
    :param srt_path: SRT文件路径
    :param wav_path: WAV文件路径
    :param output_folder: 输出文件夹路径
    :param base_name: 输入文件的基本名称（不含扩展名）
    """
    # 读取SRT文件
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 解析SRT文件并提取音频片段
    jsonl_data = []
    for i in range(0, len(lines), 4):
        index = lines[i].strip()
        time_range = lines[i+1].strip()
        text = lines[i+2].strip()

        # 解析时间戳
        start_time, end_time = time_range.split(' --> ')
        start_time = start_time.replace(',', '.')
        end_time = end_time.replace(',', '.')

        # 生成WAV文件名
        wav_filename = f"{base_name}_{index}.wav"
        wav_path_segment = os.path.join(output_folder, wav_filename)


        # If output file exists, delete it first
        if os.path.exists(wav_path_segment):
            os.remove(wav_path_segment)

        # Use ffmpeg to extract audio segment
        command = [
            'ffmpeg', '-i', wav_path,
            '-ss', start_time, '-to', end_time,
            '-q:a', '0', '-map', 'a', wav_path_segment
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 生成JSONL数据
        prompt = prompt
        gt = f"{text}{prompt}{text}"
        source = f"video_{prompt[2:5]}_{prompt[9:12]}"
        jsonl_data.append({
            "audio": wav_filename,
            "prompt": prompt,
            "gt": gt,
            "source": source
        })

    # 保存JSONL文件
    jsonl_path = os.path.join(output_folder, f"{base_name}.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in jsonl_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main(output_path, input_video_path, prompt):
    """
    主函数：处理视频文件，生成SRT和JSONL
    :param input_video_path: 输入视频文件路径
    """
    # 获取输入文件的基本名称（不含扩展名）
    base_name = Path(input_video_path).stem

    # 创建输出文件夹，名称为输入文件的基本名称
    output_folder = Path(output_path) / base_name
    os.makedirs(output_folder, exist_ok=True)

    # 将视频文件转换为 16000 Hz 的 WAV 文件
    wav_path = os.path.join(output_folder, f"{base_name}.wav")
    convert_to_16000hz(input_video_path, wav_path)
    print("wav saved")

    # 使用 FSMNVad 切分音频
    vad = FSMNVad()
    segments = vad.segments_offline(Path(wav_path))
    print("切分结果：", segments)

    # 生成 SRT 文件
    srt_path = os.path.join(output_folder, f"{base_name}.srt")
    generate_srt_from_segments(segments, srt_path)

    # 根据 SRT 文件切分 WAV 文件并生成 JSONL
    extract_audio_segments(srt_path, wav_path, output_folder, base_name,prompt)

    return base_name

import json
from datetime import datetime

def srt_time_to_seconds(time_str):
    """将SRT时间格式转换为秒数"""
    try:
        dt = datetime.strptime(time_str, "%H:%M:%S,%f")
        return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    except ValueError:
        return 0.0

def parse_srt_file(srt_path):
    """解析SRT文件，返回包含时间信息和文本的列表"""
    subtitles = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    i = 0
    while i < len(lines):
        # 读取序号
        if not lines[i].isdigit():
            i += 1
            continue
        
        # 读取时间轴
        time_line = lines[i+1] if i+1 < len(lines) else ""
        if '-->' not in time_line:
            i += 1
            continue
        
        start_time, end_time = [t.strip() for t in time_line.split('-->')]
        
        # 读取文本内容（可能有多行）
        text_lines = []
        j = i + 2
        while j < len(lines) and not lines[j].isdigit():
            text_lines.append(lines[j])
            j += 1
        
        subtitles.append({
            'index': int(lines[i]),
            'startTime': start_time,
            'endTime': end_time,
            'startTimeInSeconds': srt_time_to_seconds(start_time),
            'endTimeInSeconds': srt_time_to_seconds(end_time),
            'text': ' '.join(text_lines)
        })
        i = j
    
    return subtitles

def convert_srt_to_jsonl(srt_path, jsonl_path, original_jsonl_path=None):
    """将SRT文件转换为JSONL，或与现有JSONL文件合并"""
    subtitles = parse_srt_file(srt_path)
    
    if original_jsonl_path:
        # 合并模式：读取原始JSONL并与SRT时间信息合并
        with open(original_jsonl_path, 'r', encoding='utf-8') as f_in, \
             open(jsonl_path, 'w', encoding='utf-8') as f_out:
            
            for i, line in enumerate(f_in):
                try:
                    data = json.loads(line.strip())
                    if i < len(subtitles):
                        duration = subtitles[i]['endTimeInSeconds']-subtitles[i]['startTimeInSeconds']
                        if duration < 0.8:
                            continue
                        data.update({
                            'index': subtitles[i]['index'],
                            'startTime': subtitles[i]['startTime'],
                            'endTime': subtitles[i]['endTime'],
                            'startTimeInSeconds': subtitles[i]['startTimeInSeconds'],
                            'endTimeInSeconds': subtitles[i]['endTimeInSeconds'],
                            'duration': duration,
                        })
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                except json.JSONDecodeError:
                    print(f"跳过无效的JSON行: {line}")
    else:
        # 纯转换模式：直接从SRT创建JSONL
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for sub in subtitles:
                f.write(json.dumps(sub, ensure_ascii=False) + '\n')

def split(prompt, input_video_path, output_path):
    name = main(output_path, input_video_path, prompt)

    srt_file = Path(output_path) / name / f"{name}.srt"  # 输入的SRT文件路径
    original_jsonl = Path(output_path) / name / f"{name}.jsonl"  # 原始JSONL文件路径（可选）
    output_jsonl = Path(output_path) / name / f"{name}_out.jsonl"  # 输出文件路径


    # 执行转换（如果不需要合并原始JSONL，省略第二个参数）
    convert_srt_to_jsonl(srt_file, output_jsonl, original_jsonl)

    print(f"转换完成，结果已保存到 {output_jsonl}")
    return (name, Path(output_path) / output_jsonl)

# 示例调用
if __name__ == "__main__":
    name = "gaowen"

    input_video_path = f"{name}.wav"  # 输入视频文件路径
    prompt = "<|zho|><|eng|>"

    main('.', input_video_path, prompt)

    srt_file = f"{name}/{name}.srt"  # 输入的SRT文件路径
    original_jsonl = f"{name}/{name}.jsonl"  # 原始JSONL文件路径（可选）
    output_jsonl = f"{name}/{name}_out.jsonl"  # 输出文件路径


    # 执行转换（如果不需要合并原始JSONL，省略第二个参数）
    convert_srt_to_jsonl(srt_file, output_jsonl, original_jsonl)

    print(f"转换完成，结果已保存到 {output_jsonl}")

    
