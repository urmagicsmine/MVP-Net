#!/bin/bash
if [ "$1" == "train" ]
then
	CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net_step.py --dataset lesion_train --cfg configs/lesion_baselines/3DCE_9.yaml --bs 16 --nw 16 --use_tfboard
elif [ "$1" == "test" ]
then
	CUDA_VISIBLE_DEVICES=1 python tools/test_net.py --dataset lesion_test --cfg configs/lesion_baselines/3DCE_9.yaml \
		--load_ckpt Outputs/3DCE_9/Dec15-22-44-06_lung-general-03_step/ckpt/model_step27999.pth # 1216 default setting
	# Outputs/3DCE_9/Apr27-19-14-49_lung-general-03_step/ckpt/model_step27999.pth # z flip 59.02
	# Outputs/3DCE_9/Feb17-15-07-39_cac5_step/ckpt/model_step55999.pth # data aug
	#Outputs/3DCE_9/Jan20-00-23-28_cac5_step/ckpt/model_step87999.pth #1.8,2.1,2.2
	#Outputs/3DCE_9/Jan29-23-23-31_cac5_step/ckpt/model_step43999.pth #slice intv 2.5 64.47,80.42@3.7(nms) 64.53,81.37@14.4(soft_nms)
	# Outputs/3DCE_9/Jan18-11-32-12_cac5_step/ckpt/model_step71999.pth #GN schedule1.5 1.7 1.8
	#	Outputs/3DCE_9/Jan17-10-04-46_cac5_step/ckpt/model_step71999.pth #GN1
	#Outputs/3DCE_9/Dec11-17-04-12_cac5_step/ckpt/model_step71999.pth #OK
	#Outputs/3DCE_9/Dec25-16-11-52_cac5_step/ckpt/model_step71999.pth
else
	echo "choose from [train,test]"
fi
