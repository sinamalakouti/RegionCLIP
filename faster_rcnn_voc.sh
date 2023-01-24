	
	TORCH_DISTRIBUTED_DEBUG=DETAIL python3 ./tools/train_net.py \
	--num-gpus 4 \
	--config-file ./configs/VOC-Experiments/faster_rcnn_CLIP_R_50_C4.yaml \
	MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth
#./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth
	#MODEL.WEIGHTS ./output/model_final.pth
	#MODEL.WEIGHTS ./pretrained_ckpt/clip/teacher_RN50_student_RN50_OAI_CLIP.pth
	#./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth
