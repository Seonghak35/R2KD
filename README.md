
# R2KD
**Robustness-Reinforced Knowledge Distillation with Correlation Distance and Network Pruning**

Seonghak Kim, Gyeongdo Ham, Yucheol Cho, Daeshik Kim

This provides an implementation of the code for "[Robustness-Reinforced Knowledge Distillation with Correlation Distance and Network Pruning](https://doi.org/10.1109/TKDE.2024.3438074)", as published in the _IEEE Transactions on Knowledge and Data Engineering_.

## Installation

Environments:

- Python 3.8
- PyTorch 1.10.0
- torchvision 0.11.0

Install the package:

```
pip install -r requirements.txt
python setup.py develop
```

## Getting started

1. Evaluation

- You can see `r2kd_eval.sh`.

```bash
# augmentation_type = ["mixup", "cutmix", "cutout", "autoaug", "cutmixpick"]
python tools/eval.py -d tiny_imagenet -m wrn_16_2 -c ./best_results/augmentation_type/wrn402_wrn162_student_best
python tools/eval.py -d tiny_imagenet -m resnet20 -c ./best_results/augmentation_type/res56_res20_student_best
python tools/eval.py -d tiny_imagenet -m resnet8x4 -c ./best_results/augmentation_type/res32x4_res8x4_student_best
python tools/eval.py -d tiny_imagenet -m vgg8 -c ./best_results/augmentation_type/vgg13_vgg8_student_best
python tools/eval.py -d tiny_imagenet -m MobileNetV2 -c ./best_results/augmentation_type/vgg13_mv2_student_best
python tools/eval.py -d tiny_imagenet -m ShuffleV2 -c ./best_results/augmentation_type/res32x4_shuv2_student_best
```

2. Training
- The weights of teacher models can be downloaded via `pretrained_models.sh`. 

```bash
sh pretrained_models.sh
```

- Using these weights, you can train the student models with R2KD, as shown in `r2kd_train.sh`.

```bash
# augmentation_type = ["mixup", "cutmix", "cutout", "autoaug", "cutmixpick"]
python tools/train.py --cfg configs/tiny/r2kd/cutout/wrn40_2_wrn16_2.yaml --pruning -a augmentation_type
python tools/train.py --cfg configs/tiny/r2kd/cutout/res56_res20.yaml --pruning -a augmentation_type
python tools/train.py --cfg configs/tiny/r2kd/cutout/res32x4_res8x4.yaml --pruning -a augmentation_type
python tools/train.py --cfg configs/tiny/r2kd/cutout/vgg13_vgg8.yaml --pruning -a augmentation_type
python tools/train.py --cfg configs/tiny/r2kd/cutout/vgg13_mv2.yaml --pruning -a augmentation_type
python tools/train.py --cfg configs/tiny/r2kd/cutout/res32x4_shuv2.yaml --pruning -a augmentation_type
```

## Citation

Please consider citing **R2KD** in your publications if it helps your research.

```bib
@article{kim2024robustness,
  title={Robustness-reinforced knowledge distillation with correlation distance and network pruning},
  author={Kim, Seonghak and Ham, Gyeongdo and Cho, Yucheol and Kim, Daeshik},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgement

This code is built on [mdistiller](<https://github.com/megvii-research/mdistiller>) and [Multi-Level-Logit-Distillation](<https://github.com/Jin-Ying/Multi-Level-Logit-Distillation>).

Thanks to the contributors of mdistiller and Multi-Level-Logit-Distillation for their exceptional efforts.

