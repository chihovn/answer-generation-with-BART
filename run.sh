#!/bin/sh
python3 train.py \
	--train_data ./data/processed/ELI5_train_10_doc.json \
	--eval_data ./data/processed/ELI5_val_10_doc.json \
	--seed 0 \
	--is_colab True \
    --name training_1 \
    --logger False \
    --checkpoint_dir checkpoint \
    --model_name facebook/bart \
    --model_size base \
	--num_epochs 2\
	--batch_size 8 \
	--max_input_length 1024 \
	--min_ans_length 64 \
	--max_ans_length 256 \
	--print_freq 1000 \
	--lr 2e-4\
	--backward_freq 16
