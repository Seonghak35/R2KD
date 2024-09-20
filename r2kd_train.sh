
# cutout 

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/wrn40_2_wrn16_2.yaml --pruning -a cutout

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/res56_res20.yaml --pruning -a cutout

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/res32x4_res8x4.yaml --pruning -a cutout

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/vgg13_vgg8.yaml --pruning -a cutout

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/vgg13_mv2.yaml --pruning -a cutout

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/res32x4_shuv2.yaml --pruning -a cutout


# autoaug

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/wrn40_2_wrn16_2.yaml --pruning -a autoaug

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/res56_res20.yaml --pruning -a autoaug

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/res32x4_res8x4.yaml --pruning -a autoaug

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/vgg13_vgg8.yaml --pruning -a autoaug

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/vgg13_mv2.yaml --pruning -a autoaug

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/res32x4_shuv2.yaml --pruning -a autoaug


# mixup 

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/wrn40_2_wrn16_2.yaml --pruning -a mixup

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/res56_res20.yaml --pruning -a mixup

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/res32x4_res8x4.yaml --pruning -a mixup

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/vgg13_vgg8.yaml --pruning -a mixup

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/vgg13_mv2.yaml --pruning -a mixup

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/res32x4_shuv2.yaml --pruning -a mixup


# cutmix 

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/wrn40_2_wrn16_2.yaml --pruning -a cutmix

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/res56_res20.yaml --pruning -a cutmix

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/res32x4_res8x4.yaml --pruning -a cutmix

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/vgg13_vgg8.yaml --pruning -a cutmix

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/vgg13_mv2.yaml --pruning -a cutmix

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/res32x4_shuv2.yaml --pruning -a cutmix


# cutmixpick

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/wrn40_2_wrn16_2.yaml --pruning -a cutmixpick

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/res56_res20.yaml --pruning -a cutmixpick

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/res32x4_res8x4.yaml --pruning -a cutmixpick

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/vgg13_vgg8.yaml --pruning -a cutmixpick

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/vgg13_mv2.yaml --pruning -a cutmixpick

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/tiny/r2kd/res32x4_shuv2.yaml --pruning -a cutmixpick


