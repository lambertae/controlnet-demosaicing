export CUDA_VISIBLE_DEVICES=7
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
 --output_dir="model_out" \
 --dataset_name="./flickr8k/new_dataset" \
 --conditioning_image_column=condition \
 --image_column=image \
 --caption_column=caption \
 --resolution=256 \
 --learning_rate=1e-5 \
 --validation_image "./flickr8k/val_img.png" \
 --validation_prompt "the boy laying face down on skateboard is being pushed along the ground by another boy" \
 --train_batch_size=4 \
 --num_train_epochs=100 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=5000 \
 --validation_steps=5000 \