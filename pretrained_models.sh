# These Tiny ImageNet models are trained by ourselves
if [ ! -d download_ckpts/tiny_teachers/models_tinyimagenet_v2/resnet56_vanilla ]; then
    mkdir -p download_ckpts/tiny_teachers/models_tinyimagenet_v2
    cd download_ckpts/tiny_teachers
    wget https://github.com/MingSun-Tse/Good-DA-in-KD/releases/download/v0.1/teachers_tinyimagenet.zip
    unzip teachers_tinyimagenet.zip
    cd ../..
fi
