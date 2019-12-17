#!/bin/bash
if [ "$1" == "train" ]
then
	CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net_step.py --dataset lesion_train --cfg configs/lesion_baselines/multi_windows_9_slices.yaml --bs 4 --nw 4 --use_tfboard
elif [ "$1" == "test" ]
then
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/test_net.py --dataset lesion_test --cfg configs/lesion_baselines/multi_windows_9_slices.yaml --multi-gpu-testing\
		--load_ckpt Outputs/multi_windows/Dec14-00-43-46_lung-general-03_step/ckpt/model_step71999.pth # 1215 new with pos, 9*3 slices
else
	echo "choose from [train,test]"
fi
