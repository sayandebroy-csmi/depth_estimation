# 3D-depth-estimation on RGB video data

## Due to Limited computational power, I have used the MiDaS v2.1 - Small (lowest accuracy, highest inference speed) model for 3D depth estimation on RGB video data along with FastAPI. 

## Else the alternative could be DepthAnythingV2 [CVPR'24] [link](https://depth-anything.github.io/)


### Installation

Set-up environment
```
conda create -n "videodepth" python=3.8

conda activate videodepth
```

Install Pytorch on GPU Platform
```
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

Install other necessary Python packages
```
pip install -r requirements.txt
```

Clone this repository
```
git clone https://github.com/sayandebroy-csmi/depth_estimation.git

cd depth_estimation
```

Run the code
```
python app.py
```

Go to http://0.0.0.0:8000

![demo image](demo/rgb_video.gif)

![demo image](demo/gray_depth.gif)