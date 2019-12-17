
# MVP-Net: Multi-view FPN with Position-aware Attention for Deep Universal Lesion Detection
This is an implementation of MICCAI 2019 paper [MVP-Net: Multi-view FPN with Position-aware Attention for Deep Universal Lesion Detection](https://arxiv.org/abs/1909.04247v3).

## Installation
This code is based on [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch). Please see it for installation.


## Environment
  - Python (Tested on 3.6)
  - PyTorch (Tested on 0.4.1.post2)

## Data preparation
Download DeepLesion dataset [here](https://nihcc.app.box.com/v/deeplesion).

We provide coco-style json annotation files converted from DeepLesion. Unzip Images_png.zip and make sure to put files as following sturcture:

```
data
  ├──DeepLesion
        ├── annotations
        │   ├── deeplesion_train.json
        │   ├── deeplesion_test.json
        │   ├── deeplesion_val.json
        └── Images_png
              └── Images_png
               │    ├── 000001_01_01
               │    ├── 000001_03_01
               │    ├── ...
```

## Training
To train MVP-Net with 9 slices model, run:
```
bash multi_windows_9_slices.sh train
```
We also provide our re-implementation of [3DCE](https://arxiv.org/pdf/1806.09648.pdf), see 3DCE_*.sh for training and testing.

## Testing
After training, put the model path into .sh file, after '--load_ckpt', and run:
```
bash multi_windows_9_slices.sh test
```


## Results on DeepLesion dataset
| FPs per image           | 0\.5   | 1      | 2      | 3      | 4      |
|-------------------------|--------|--------|--------|--------|--------|
| ULDOR                   | 52\.86 | 64\.80 | 74\.84 | \-     | 84\.38 |
| 3DCE, 3 slices          | 55\.70 | 67\.26 | 75\.37 | \-     | 82\.21 |
| 3DCE, 9 slices          | 59\.32 | 70\.68 | 79\.09 | \-     | 84\.34 |
| 3DCE, 27 slices         | 62\.48 | 73\.37 | 80\.70 | \-     | 85\.65 |
| FPN\+3DCE, 3 slices\*   | 58\.06 | 68\.85 | 77\.48 | 81\.03 | 83\.27 |
| FPN\+3DCE, 9 slices\*   | 64\.25 | 74\.41 | 81\.90 | 85\.02 | 87\.21 |
| FPN\+3DCE, 27 slices\*  | 67\.32 | 76\.34 | 82\.90 | 85\.67 | 87\.60 |
| Ours, 3 slices          | 70\.01 | 78\.77 | 84\.71 | 87\.58 | 89\.03 |
| Ours, 9 slices          | 73\.83 | 81\.82 | 87\.60 | 89\.57 | 91\.30 |
| Imp over 3DCE, 27slices | 11\.35 | 8\.45  | 6\.90  | \-     | 5\.65  |

\* indicates our re-implementation of 3DCE with FPN as backbone.
## Contact
If you have questions or suggestions, please open an issue here or send an email to lizihao2018@ia.ac.cn.


