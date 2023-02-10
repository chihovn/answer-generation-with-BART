#!/bin/sh
python train.py \
	--train_data ./data/processed/ELI5_train_10_doc.json \
	--eval_data ./data/processed/ELI5_val_10_doc.json \
	--seed 0 \
    --name training_1 \
    --logger True \
    --checkpoint_dir checkpoint \
    --model_name facebook/bart \
    --model_size base \
	--num_epochs 3\
	--batch_size 8 \
	--max_length 1024 \
	--print_freq 1000 \
	--lr 2e-4\
	--backward_freq 16

python train.py --train_data ./data/processed/ELI5_train_10_doc.json --eval_data ./data/processed/ELI5_val_10_doc.json --seed 0 --name training_1 --logger True --checkpoint_dir checkpoint --model_name facebook/bart --model_size base --num_epochs 3 --batch_size 8 --max_length 1024 --print_freq 1000 --lr 2e-4 --backward_freq 16