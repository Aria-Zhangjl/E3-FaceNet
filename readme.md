# $E^3$-FaceNet
This is an official implementation of ICML 2024 Paper ["Fast Text-to-3D-Aware Face Generation and Manipulation via Direct Cross-modal Mapping and Geometric Regularization"](https://arxiv.org/abs/2403.06702). The proposed $E^3$-FaceNet is an **E**nd-to-End **E**fficient and **E**ffective network for fast and accurate T3D face generation and manipulation, which can not only achieve picture-like 3D face generation and manipulation, but also improve inference speed by orders of magnitudes.



## 🚀 Overview
![](figure/overview-readme.png)



## 🖥️ Setup
### Environment
The codebase is tested on 
* Python 3.7
* PyTorch 1.7.1

For additional python libraries, please install by:

```
pip install -r requirements.txt
```

Please refer to https://github.com/NVlabs/stylegan2-ada-pytorch for additional software/hardware requirements.

### Data Preparation
We train our $E^3$-FaceNet on [Multi-Modal-CelebA-HQ Dataset](https://github.com/IIGROUP/MM-CelebA-HQ-Dataset) (MM-CelebA) and evaluation on MMCelebA, [CelebAText-HQ](https://github.com/cripac-sjx/SEA-T2F) and [FFHQ-Text](https://github.com/Yutong-Zhou-cv/FFHQ-Text_Dataset). 

Before training, please dowload the [dataset2.json](https://drive.google.com/file/d/1MZda_8w96EAOWjwvGyTQBPzP1Dl9afdl/view?usp=sharing), and place the file in the MMceleba dataset directory.

### Pretrained Checkpoints
The model weight can be download at [here](https://drive.google.com/file/d/1msBAgRYo_o3yT9Nx1q86KMZRoboxkpxB/view?usp=sharing)

### Training $E^3$-FaceNet
use the shell script,

```
bash train_train_4_E3_Face.sh
```

Please check configuration files at ```conf/model``` and ```conf/spec```. You can always add your own model config. More details on how to use hydra configuration please follow https://hydra.cc/docs/intro/.

### Evaluate $E^3$-FaceNet

use the shell script,

```
bash run_eval_4_E3_Face.sh
```

### Text-Guided Generation and Manipulation

use the shell script,

```
bash sample.sh
```



### Visual Results

#### Compare with Text-to-3D Face Methods

<img src="figure/3D-readme.png" style="zoom:67%;" />

#### Compare with Text-to-2D Face Methods

<img src="figure/2D-readme.png" style="zoom: 67%;" />


## 🖊️ Citation
If $E^3$-FaceNet is helpful for your research or you wish to refer the baseline results published here, we'd really appreciate it if you could cite this paper:
```
@article{zhang2024fast,
  title={Fast Text-to-3D-Aware Face Generation and Manipulation via Direct Cross-modal Mapping and Geometric Regularization},
  author={Zhang, Jinlu and Zhou, Yiyi and Zheng, Qiancheng and Du, Xiaoxiong and Luo, Gen and Peng, Jun and Sun, Xiaoshuai and Ji, Rongrong},
  journal={arXiv preprint arXiv:2403.06702},
  year={2024}
}
```

## 🎫 Acknowledgment
This project largely references [StyleNeRF](https://github.com/facebookresearch/StyleNeRF). Thanks for their amazing work!
