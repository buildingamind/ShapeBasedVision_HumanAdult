subdir=V1O1

for subdir in ep0
do
    python3 train_ViTCoT.py \
        --max_epochs 100 \
        --batch_size 128 \
        --data_dir /data/lpandey/UTAustin_EgocentricDataset/outputSamples/output_64x64_Squished/train3_80k/${subdir} \
       	--seed_val 0 \
        --temporal \
        --window_size 3 \
        --head 6 \
        --gpus 1 \
        --val_split 0.1 \
        --transforms transform_gB \
        --resize_dims 64 \
        --exp_name Eyeball_Project/paper_HumanAdults/UTAustinEgocentric/Exp2_GaussianBlurAugmentation/ViT6H/video3/
done

for subdir in ep0
do
    python3 train_ViTCoT.py \
        --max_epochs 100 \
        --batch_size 128 \
        --data_dir /data/lpandey/UTAustin_EgocentricDataset/outputSamples/output_64x64_Squished/train3_80k/${subdir} \
       	--seed_val 10 \
        --temporal \
        --window_size 3 \
        --head 6 \
        --gpus 1 \
        --val_split 0.1 \
        --transforms transform_gB \
        --resize_dims 64 \
        --exp_name Eyeball_Project/paper_HumanAdults/UTAustinEgocentric/Exp2_GaussianBlurAugmentation/ViT6H/video3/
done

for subdir in ep0
do
    python3 train_ViTCoT.py \
        --max_epochs 100 \
        --batch_size 128 \
        --data_dir /data/lpandey/UTAustin_EgocentricDataset/outputSamples/output_64x64_Squished/train3_80k/${subdir} \
       	--seed_val 20 \
        --temporal \
        --window_size 3 \
        --head 6 \
        --gpus 1 \
        --val_split 0.1 \
        --transforms transform_gB \
        --resize_dims 64 \
        --exp_name Eyeball_Project/paper_HumanAdults/UTAustinEgocentric/Exp2_GaussianBlurAugmentation/ViT6H/video3/
done


# NOTES: 
# 1. set the batch size to 128 if single GPU training, else calculate the effective batch size based on num_of_gpus used in distributed training.
# 2. set window_size in the range [1,4]

# extra flags:
#  --shuffle_frames \
#  --shuffle_temporalWindows \
#  --dataset_size 10000 \
#  --dataloader_shuffle \



