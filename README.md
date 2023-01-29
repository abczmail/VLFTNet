# VLFTNet
## Table of Contents
- [Environment setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reference](#reference-and-citation)

## Environment setup

Clone this repository and create the `vlftnet` conda environment using the `environment.yml` file:
```
conda env create -f environment.yaml
conda activate vlftnet
```

Then download spacy data by executing the following command:
```
python -m spacy download en_core_web_md
```

**Note:** Python 3 is required to run our code. If you suffer network problems, please download ```en_core_web_md``` library from [here](https://drive.google.com/file/d/1jf6ecYDzIomaGt3HgOqO_7rEL6oiTjgN/view?usp=sharing), unzip and place it to ```/your/anaconda/path/envs/m2release/lib/python*/site-packages/```

## Data Preparation

* **Annotation**. Download the annotation file [annotation.zip](https://drive.google.com/file/d/1Zc2P3-MIBg3JcHT1qKeYuQt9CnQcY5XJ/view?usp=sharing) [1]. Extract and put it in the project root directory.
* **Feature**. Download processed image features [ResNeXt-101](https://pan.baidu.com/s/1lI1i_DQovfze_C7ztwQvfw?pwd=vlft) and [ResNeXt-152](https://pan.baidu.com/s/1XBACq2L-1ax0IacMZg3H7g?pwd=vlft) features [2] (code ```vlft```), put it in the project root directory.
<!-- * **Evaluation**. Download the evaluation tools [here](https://pan.baidu.com/s/1xVZO7t8k4H_l3aEyuA-KXQ). Acess code: jcj6. Extarct and put it in the project root directory. -->


## Training
Run `python train_transformer.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--exp_name` | Experiment name|
| `--batch_size` | Batch size (default: 50) |
| `--workers` | Number of workers, accelerate model training in the xe stage.|
| `--head` | Number of heads (default: 8) |
| `--resume_last` | If used, the training will be resumed from the last checkpoint. |
| `--resume_best` | If used, the training will be resumed from the best checkpoint. |
| `--features_path` | Path to visual features file (h5py)|
| `--annotation_folder` | Path to annotations |
| `--num_clusters` | Number of pseudo regions |

For example, to train the model, run the following command:
```
python train_transformer.py --exp_name VLFTNet --batch_size 50 --m 40 --head 8 --features_path /path/to/features --num_clusters 5
```

## Evaluation
### Offline Evaluation.
Run `python test_transformer.py` to evaluate the model using the following arguments:
```
python test_transformer.py --batch_size 10 --features_path /path/to/features --model_path /path/to/saved_transformer_models/ckpt --num_clusters 5
```
  
**Note:** We have removed the ```SPICE``` evaluation metric during training because it is time-cost. You can add it when evaluate the model: download this [file](https://drive.google.com/file/d/1vEVsbEFjDstmSvoWhu4UdKaJjX1jJXpR/view?usp=sharing) and put it in ```/path/to/evaluation/```, then uncomment codes in [__init__.py](https://github.com/zchoi/S2-Transformer/blob/master/evaluation/__init__.py).

We provide pretrained model [here](https://pan.baidu.com/s/1aAHOfUyN2Je8TT4le5qypg?pwd=vlft)(code ```vlft```), you will get following results (second row) by evaluating the pretrained model:

| Model 	| B@1 	| B@4 	|        M   	| R 	| C 	| S |
|:---------:	|:-------:	|:-:	|:---------------:	|:--------------------------:	|:-------:	| :-------:|
| Our Paper (ResNext101) 	|     81.4   	| 39.7 	|               29.5 	|              59.2             	|    134.7   	|  23.1|



### Online Evaluation
We also report the performance of our model on the online COCO test server with an ensemble of four VLFTNet models. The detailed online test code can be obtained in this [repo](https://github.com/zhangxuying1004/RSTNet).

## Reference
[1] Cornia, M., Stefanini, M., Baraldi, L., & Cucchiara, R. (2020). Meshed-memory transformer for image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.  
[2] Xuying Zhang, Xiaoshuai Sun, Yunpeng Luo, Jiayi Ji, Yiyi Zhou, Yongjian Wu, Feiyue
Huang, and Rongrong Ji. Rstnet: Captioning with adaptive attention on visual and non-visual words. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15465–15474, 2021.
[3] Pengpeng Zeng, Haonan Zhang, Jingkuan Song, and Lianli Gao. S2 transformer for image captioning. In Proceedings of the International Joint Conferences on Artificial Intelligence, volume 5, 2022.

