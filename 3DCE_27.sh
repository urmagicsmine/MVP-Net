#!/bin/bash
if [ "$1" == "train" ]
then
	CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net_step.py --dataset lesion_train --cfg configs/lesion_baselines/3DCE_27.yaml --bs 4 --nw 16 --use_tfboard
elif [ "$1" == "test" ]
then
	CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/test_net.py --dataset lesion_test --cfg configs/lesion_baselines/3DCE_27.yaml --multi-gpu-testing\
		--load_ckpt Outputs/3DCE_27/Dec11-12-16-27_lung-general-03_step/ckpt/model_step71999.pth # 1212 new, ap50=60.7 pass
else
	echo "choose from [train,test]"
fi
