#!/bin/bash
if [ "$1" == "train" ]
then
	CUDA_VISIBLE_DEVICES=2,3 python tools/train_net_step.py --dataset lesion_train --cfg configs/lesion_baselines/3DCE_3.yaml --bs 16 --nw 16 --use_tfboard
elif [ "$1" == "test" ]
then
	CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset lesion_test --cfg configs/lesion_baselines/3DCE_3.yaml --multi-gpu-testing\
		--load_ckpt Outputs/3DCE_3/Dec15-22-46-55_lung-general-03_step/ckpt/model_step17999.pth # new 1216 
	# Outputs/3DCE_3/Dec10-17-27-28_lung-general-03_step/ckpt/model_step17999.pth # window(-1024, 1050)
	# Outputs/3DCE_3/Dec10-10-19-38_lung-general-03_step/ckpt/model_step17999.pth # default window
	#Outputs/3DCE_9/Apr27-19-14-49_lung-general-03_step/ckpt/model_step27999.pth # z flip 59.02
else
	echo "choose from [train,test]"
fi
