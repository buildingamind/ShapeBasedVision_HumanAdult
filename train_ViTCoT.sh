subdir=V1O1


for subdir in samples
do
    python3 train_ViTCoT.py \
        --max_epochs 100 \
        --batch_size 128 \
        --data_dir /data/lpandey/UTAustin_EgocentricDataset/outputSamples/eyeballs/paper_NatHumBeh/train_video2/eyeball1/${subdir} \
       	--seed_val 0 \
        --temporal \
        --window_size 3 \
        --head 1 \
        --gpus 1 \
        --val_split 0.1 \
        --transforms transform_resize \
        --resize_dims 64 \
        --loss_ver v0 \
        --exp_name dummy22/
done



# NOTES: 
# 1. set the batch size to 128 if single GPU training, else calculate the effective batch size based on num_of_gpus used in distributed training.
# 2. set window_size in the range [1,4].
# 3. set loss ver from [v0, v1].
# 4. if dataset_size is not specified, then all the samples from the dataset will be used for training.

# Some Important Flags:
#  --shuffle_frames \
#  --loss_ver v0 \
#  --shuffle_temporalWindows \
#  --dataset_size 10000 \
#  --dataloader_shuffle \
#  --log_path "path_to_save_checkpoints" \

