
### Introduction

Code for our paper "Disentangled Sample Guidance Learning for Unsupervised Person Re-identification".

### Prerequisites

- Necessary packages
     torch==1.1.0
     scipy==1.5.4
     torchvision==0.3.0
     seaborn==0.11.0
     numpy==1.19.4
     faiss==1.6.4
     Pillow==7.1.2
     scikit_learn==0.21.2

- Training Data
  
  (Market-1501 and MSMT-17. You can download these datasets from [Zhong's repo](https://github.com/zhunzhong07/ECN))

   Unzip all datasets and ensure the file structure is as follow:
   
   ```
   MetaPRD/examples/data    
   │
   └───market1501 OR msmt17
        │   
        └───Market-1501-v15.09.15 OR MSMT17_V1
            │   
            └───bounding_box_train
            │   
            └───bounding_box_test
            | 
            └───query
   ```

### Optimized model

# on Market-1501
[Market-1501](https://drive.google.com/file/d/1VyXHMzBdCRNusBUc931xzsrz1wwe-vMz/view?usp=drive_link)

# on MSMT17
[MSMT17](https://drive.google.com/file/d/1QybLvlDzgmm8X-nkIQ7__gpkGrzVjjks/view?usp=drive_link)

### Usage 

# on Market-1501
python examples/train.py -b 64 -a resnet50 -d market1501 --iters 400 --momentum 0.1 --eps 0.45 --num-instances 16 --pooling-type gem --memorybank DLhybrid --epochs 60 --logs-dir examples/logs --step-size 25 

# on MSMT17
python examples/train.py -b 64 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.7 --num-instances 16 --pooling-type gem --memorybank DLhybrid --epochs 60 --logs-dir examples/logs --step-size 25 

### pre-trained model
When training with the backbone of [IBN-ResNet](https://arxiv.org/abs/1807.09441), you need to download the ImageNet-pretrained model from this [link](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S) 
The pre-trained model of CNN are saved in examples/pretrained
ImageNet-pretrained models for **ResNet-50** will be automatically downloaded in the python script.
use `-a resnet50` (default) for the backbone of ResNet-50, and `-a resnet_ibn50a` for the backbone of IBN-ResNet.

If this repo is helpful for your research, please consider citing the paper:

```BibTeX
@article{ji2024dsgl,
  title={Disentangled Sample Guidance Learning for Unsupervised Person Re-identification},
  author={Ji, Haoxuanye and Wang, Le and Zhou, Sanping and Tang, Wei and Hua, Gang},
  booktitle={submitted to T-IP},
  year={2024}
}
```

### Acknowledgments
This repo borrows partially from 
[SpCL](https://github.com/yxgeee/SpCL),
[cluster-contrast-reid](https://github.com/alibaba/cluster-contrast-reid) and 
[hhcl](https://github.com/bupt-ai-cz/HHCL-ReID). 
If you find our code useful, please cite their papers.

```
@inproceedings{ge2020selfpaced,
    title={Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID},
    author={Yixiao Ge and Feng Zhu and Dapeng Chen and Rui Zhao and Hongsheng Li},
    booktitle={NeurIPS},
    year={2020}
}
```

```
@inproceedings{arxiv2021Cluster,
    author = {Dai, Zuozhuo and Wang, Guangyuan and Yuan, Weihao and Zhu, Siyu and Tan, Ping},
    title = {Cluster Contrast for Unsupervised Person Re-Identification},
    booktitle = {arXiv:2103.11568},
    year = 2021
}
```

```
@article{hu2021hhcl,
  title={Hard-sample Guided Hybrid Contrast Learning for Unsupervised Person Re-Identification},
  author={Hu, Zheng and Zhu, Chuang and He, Gang},
  journal={arXiv preprint arXiv:2109.12333},
  year={2021}
}
```# DiseSGL
