
### cutout ###

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m wrn_16_2 -c ./best_results/cutout/wrn402_wrn162_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m resnet20 -c ./best_results/cutout/res56_res20_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m resnet8x4 -c ./best_results/cutout/res32x4_res8x4_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m vgg8 -c ./best_results/cutout/vgg13_vgg8_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m MobileNetV2 -c ./best_results/cutout/vgg13_mv2_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m ShuffleV2 -c ./best_results/cutout/res32x4_shuv2_student_best


### autoaug ###

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m wrn_16_2 -c ./best_results/autoaug/wrn402_wrn162_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m resnet20 -c ./best_results/autoaug/res56_res20_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m resnet8x4 -c ./best_results/autoaug/res32x4_res8x4_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m vgg8 -c ./best_results/autoaug/vgg13_vgg8_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m MobileNetV2 -c ./best_results/autoaug/vgg13_mv2_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m ShuffleV2 -c ./best_results/autoaug/res32x4_shuv2_student_best


### mixup ###

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m wrn_16_2 -c ./best_results/mixup/wrn402_wrn162_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m resnet20 -c ./best_results/mixup/res56_res20_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m resnet8x4 -c ./best_results/mixup/res32x4_res8x4_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m vgg8 -c ./best_results/mixup/vgg13_vgg8_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m MobileNetV2 -c ./best_results/mixup/vgg13_mv2_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m ShuffleV2 -c ./best_results/mixup/res32x4_shuv2_student_best


### cutmix ###

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m wrn_16_2 -c ./best_results/cutmix/wrn402_wrn162_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m resnet20 -c ./best_results/cutmix/res56_res20_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m resnet8x4 -c ./best_results/cutmix/res32x4_res8x4_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m vgg8 -c ./best_results/cutmix/vgg13_vgg8_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m MobileNetV2 -c ./best_results/cutmix/vgg13_mv2_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m ShuffleV2 -c ./best_results/cutmix/res32x4_shuv2_student_best


### cutmixpick ###

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m wrn_16_2 -c ./best_results/cutmixpick/wrn402_wrn162_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m resnet20 -c ./best_results/cutmixpick/res56_res20_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m resnet8x4 -c ./best_results/cutmixpick/res32x4_res8x4_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m vgg8 -c ./best_results/cutmixpick/vgg13_vgg8_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m MobileNetV2 -c ./best_results/cutmixpick/vgg13_mv2_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -d tiny_imagenet -m ShuffleV2 -c ./best_results/cutmixpick/res32x4_shuv2_student_best

