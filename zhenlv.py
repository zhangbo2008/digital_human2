import subprocess


# result = subprocess.run('ffmpeg -i video/5.mp4 -vf "fps=fps=25" video/5_25.mp4')



def get_video_fps(video_path):
    # 调用 FFmpeg 的 `ffprobe` 工具获取视频信息
    command = [
        'ffprobe',
        '-v', 'error',                 # 只输出错误信息
        '-select_streams', 'v:0',     # 选择第一个视频流
        '-show_entries', 'stream=avg_frame_rate',  # 只要帧率
        '-of', 'default=noprint_wrappers=1:nokey=1',  # 格式化输出
        video_path                     # 视频文件的路径
    ]
    
    # 运行命令并获取输出
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # 返回帧率的字符串
    return result.stdout.strip()

# 使用示例
video_file = 'video/5_25.mp4'  # 这里替换为你的视频文件
fps = get_video_fps(video_file)
print(f"视频的帧率是: {fps}")



