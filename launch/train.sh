#!/usr/bin/env bash

python3 train.py \
--images_folder '../Datasets/Topographies/raw/FiguresStacked Same Size 4X4' \
--label_path '../Datasets/biology_data/TopoChip/AeruginosaWithClass.csv' \
--dataset_name 'biological' \
--n_epochs 200 \
--img_size 224 \
--batch_size 32 \
--num_workers 4 \
--train_dis_freq 1 \
--model_name 'WGANGP' \
--latent_dim 112 \