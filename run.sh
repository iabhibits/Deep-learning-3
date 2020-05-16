#!/bin/bash/
python train.py --model="bilstm" --dataset="snli" --batch_size=64 --embed_dim=300 --d_hidden=512 --epochs=10 --lr=0.001 
