#!/bin/bash


while getopts 'c:n:t:r:p' OPT; do
    case $OPT in
        c) cuda=$OPTARG;;
        n) name=$OPTARG;;
		t) task=$OPTARG;;
        r) train="true";;
        p) predict="False";;
        
    esac
done
echo $name	


if ${train}
then
	
	cd /home/wqq/Code_wqq/nnFormer/nnformer/
	CUDA_VISIBLE_DEVICES=${cuda} nnFormer_train 3d_fullres nnFormerTrainerV2_${name} ${task} 0
fi

if ${predict}
then


    cd /home/wqq/Code_wqq/nnFormer/DATASET/nnFormer_raw/nnFormer_raw_data/Task003_tumor
    CUDA_VISIBLE_DEVICES=${cuda} nnFormer_predict -i imagesTs -o inferTs/${name} -m 3d_fullres -t ${task} -f 0 -chk model_best -tr nnFormerTrainerV2_${name}
    python inference_tumor.py ${name}
fi



