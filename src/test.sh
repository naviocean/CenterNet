# python main.py ctdet --exp_id tinycoco --dataset tinycoco  --num_epoch 600 --lr_step 80,120 --gpus 1 --arch shufflenetv2pdown16v4 --val_intervals 10000 --num_workers 0 --down_ratio 16
#python main.py ctdet --exp_id tinycoco_shufflenet2_16_4 --dataset tinycoco  --num_epoch 600 --lr_step 80,120 --gpus 2 --arch shufflenetv2pdown16v4 --val_intervals 10000 --num_workers 0 --down_ratio 16
python main.py ctdet --exp_id tinycoco_shufflenet2p --dataset tinycoco  --num_epoch 600 --lr_step 80,120 --gpus 0 --arch shufflenetv2p --val_intervals 10000 --num_workers 8
