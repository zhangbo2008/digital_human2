# 群聊前后端架构:
1. 消息推送: https://blog.csdn.net/qq_40098459/article/details/136712515
2. 一个聊天室内A窗口发送aaa, 那么A,B都会在窗口刷出来aaa的文字.
   一个聊天室看做一个lists的列表.里面塞入每条消息.
   架构逻辑:
   点击按钮, 进入send函数, send函数先把消息写入lists里面,然后发送给后端服务器写入数据库.
















# 安装: 
# 安装: conda install -c conda-forge dlib
#  pip install mediapipe==0.10.9
# 这个代码做了colab 适配,  colab也能跑.
# 原始的训练代码在: https://github.com/Weizhi-Zhong/IP_LAP
# 阅读里面的issue不分启发非常大.

# 做了cuda, cpu的适配, 在本地可以直接跑. 用来学习整个结构.


#数字人中文素材: 

https://www.vipazoo.cn/CMLR.html










colab:https://colab.research.google.com/drive/1gnL_pOfawbPyFmIUuhWEP2EvtIaom1a4#scrollTo=evkOpLFnRGLk


素材下载:
https://weixinabcdefjqq.weixuke.com/thread-1422-1-1.html
并且都是1080p的效果好.
经过我的测试这份代码只有1080p时候效果才行. 我分析是480p这种误差太高.会让嘴定位不准.



train2.py核心代码:
712行:        
    predict_content=model(T_mels, T_pose, Nl_pose, Nl_content)  #(B*T,2,57)
    T_content=torch.cat([T_content[i] for i in range(T_content.size(0))],dim=0)

    landmarks_generator模型输入当前mel,当前pose,其他帧pose,其他帧content, 输出当前帧content
    T_content就是当前帧content的ground_true









# LIP_MASK：具有地标和外观先验的保持身份的说话人脸生成（CVPR 2023）

CVPR2023论文“**I**dentity-**P**reserving Talking Face Generation with **L**andmark and **A**ppearance **P**riors” 的PyTorch官方方案的升级。

[[论文]](https://arxiv.org/abs/2305.08293) [[演示视频]](https://youtu.be/wtb689iTJC8)

## 如何实现效果的提升？
- 通过dlib优化关键点检测方式
- 对齐音频视频帧
- 二次生成说话脸速度提升
- 通过锐化滤波加强清晰度

## 环境要求
- Python 3.9
- torch  2.0.0
- torchvision 0.15.1
- ffmpeg

我们在1个24G的RTX3090上使用CUDA 118进行实验。更多细节，请参考 `requirements.txt`。我们建议首先安装[pytorch](https://pytorch.org/)，然后运行以下命令：
```
pip install -r requirements.txt
```

## 测试
从[FoxCloud]([http://cloud.foxyear.cn/s/jMtW](https://foxyear.oss-cn-shenzhen.aliyuncs.com/web/FoxAvatar_Model_V2.1.zip)下载预训练模型，并将其放置在 `checkpoints` 文件夹中。然后运行以下命令：
```
python inference.py
```

## 运行
从[FoxCloud]([http://cloud.foxyear.cn/s/jMtW](https://foxyear.oss-cn-shenzhen.aliyuncs.com/web/FoxAvatar_Model_V2.1.zip)下载预训练模型，并将其放置在 `checkpoints` 文件夹中。然后运行以下命令：
```
python inference.py  --input ./video/videoxx.mp4 --audio ./audio/testxx.wav
```

## 致谢
该项目在公开可用的代码 [IP_LAP](https://github.com/Weizhi-Zhong/IP_LAP) , [DFRF](https://github.com/sstzal/DFRF) , [pix2pixHD](https://github.com/NVIDIA/pix2pixHD), [vico_challenge](https://github.com/dc3ea9f/vico_challenge_baseline/tree/a282472ea99a1983ca2ce194665a51c2634a1416/evaluations) 和 [Wav2Lip](https://github.com/Rudrabha/Wav2Lip/tree/master) 基础上构建而成。感谢这些作品和代码的作者将他们优秀的工作公开发布。


## 联系
深度开发合作交流，联系加微信：

<img src='./2.jpg' width=200>

交流群及资料教程：

<img src='./1.jpg' width=200>


## 引用和点赞
如果你在研究中使用了这个库，请引用以下论文并为该项目点赞。谢谢！
```
@InProceedings{Zhong_2023_CVPR,
    author    = {Zhong, Weizhi and Fang, Chaowei and Cai, Yinqi and Wei, Pengxu and Zhao, Gangming and Lin, Liang and Li, Guanbin},
    title     = {Identity-Preserving Talking Face Generation With Landmark and Appearance Priors},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {9729-9738}
}
```
# digital_human


# 原始训练readme
# IP_LAP: Identity-Preserving Talking Face Generation with Landmark and Appearance Priors （CVPR 2023）

Pytorch official implementation for our CVPR2023 paper "**I**dentity-**P**reserving Talking Face Generation with **L**andmark and **A**ppearance **P**riors".

<img src='./CVPR2023framework.png' width=900>

TODO:
- [x] Demo videos
- [x] pre-trained model
- [x] code for testing
- [x] code for training
- [x] code for preprocess dataset
- [x] guideline 
- [x] arxiv paper release

[[Paper]](https://arxiv.org/abs/2305.08293) [[Demo Video]](https://youtu.be/wtb689iTJC8)

## Requirements
- Python 3.7.13
- torch 1.10.0
- torchvision 0.11.0
- ffmpeg

We conduct the experiments with 4 24G RTX3090 on CUDA 11.1. For more details, please refer to the `requirements.txt`. We recommend to install [pytorch](https://pytorch.org/) firstly, and then run:
```
pip install -r requirements.txt
```
## Test
Download the pre-trained models from [OneDrive](https://1drv.ms/f/s!Amqu9u09qiUGi7UJIADzCCC9rThkpQ?e=P1jG5N) or [jianguoyun](https://www.jianguoyun.com/p/DeXpK34QgZ-EChjI9YcFIAA), and place them to the folder `test/checkpoints` . Then run the following command:
```
CUDA_VISIBLE_DEVICES=0 python inference_single.py
```
To inference on other videos, please specify the `--input` and `--audio` option and see more details in code.

The evaluation code is similar to [this repo](https://github.com/dc3ea9f/vico_challenge_baseline/tree/a282472ea99a1983ca2ce194665a51c2634a1416/evaluations).
## Train
### download LRS2 dataset
Our models are trained on LRS2. Please go to the [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) website to download the dataset. LRS2 dataset folder structure is following:
```
data_root (mvlrs_v1)
├── main, pretrain (we use only main folder in this work)
|	├── list of folders
|	│   ├── five-digit numbered video IDs ending with (.mp4)
```
`main folder` is the `lrs2_video` mentioned below.

### preprocess the audio
extract the raw audio and Mel-spectrum feature from video files by running: 
```
CUDA_VISIBLE_DEVICES=0 python preprocess_audio.py --data_root ....../lrs2_video/ --out_root ..../lrs2_audio
```
### preprocess the videos' face 

extract the cropped face, landmarks and sketches from video files by running: 

```
CUDA_VISIBLE_DEVICES=0 python preprocess_video.py --dataset_video_root ....../lrs2_video/ --output_sketch_root ..../lrs2_sketch --output_face_root ..../lrs2_face --output_landmark_root ..../lrs2_landmarks
```

### train Landmark generator

train the landmark generator network by running:

```
CUDA_VISIBLE_DEVICES=0 python train_landmarks_generator.py --pre_audio_root ..../lrs2_audio --landmarks_root ..../lrs2_landmarks
```
The models are trained until the eval_L1_loss no longer decreases (about 6e-3).
Under the default batchsize setting on a single RTX 3090, our model stopped at epoch 1837(610k iteration) with eval_L1_loss 5.866 e-3, using no more than one day.

### train Video Renderer
Training for the video renderer is similar (on four RTX 3090). Train it until the FID no longer decreases (about 20 or less).
train the video renderer network by running:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_video_renderer.py --sketch_root ..../lrs2_sketch --face_img_root ..../lrs2_face  --audio_root ..../lrs2_audio
```
Note that the translation module will only be trained  after 25 epochs, thus the fid and running_gen_loss will only decrease after epoch 25. 


## Acknowledgement
This project is built upon the publicly available code [DFRF](https://github.com/sstzal/DFRF) , [pix2pixHD](https://github.com/NVIDIA/pix2pixHD), [vico_challenge](https://github.com/dc3ea9f/vico_challenge_baseline/tree/a282472ea99a1983ca2ce194665a51c2634a1416/evaluations) and [Wav2Lip](https://github.com/Rudrabha/Wav2Lip/tree/master). Thank the authors of these works for making their excellent work and codes publicly available.


## Citation and Star
Please cite the following paper and star this project if you use this repository in your research. Thank you!
```
@InProceedings{Zhong_2023_CVPR,
    author    = {Zhong, Weizhi and Fang, Chaowei and Cai, Yinqi and Wei, Pengxu and Zhao, Gangming and Lin, Liang and Li, Guanbin},
    title     = {Identity-Preserving Talking Face Generation With Landmark and Appearance Priors},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {9729-9738}
}
```



# digital_human2
