#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./GAOFENIMG/
mkdir -p ${EXP_DIR}
CUDA_VISIBLE_DEVICES=1 python train.py \
  --dataset GAOFENIMG \  # 数据集
  --cv 0 \
  --arch network.pointflow_resnet_with_max_avg_pool.DeepR2N101_PF_maxavg_deeply \  # 网络结构
  --class_uniform_tile 1024 \
  --max_cu_epoch 64 \  # Class Uniform Max Epochs
  --lr 0.001 \  # 学习率
  --lr_schedule poly \  # 学习率曲线名称-poly
  --poly_exp 0.9 \  # 多项式学习率指数
  --repoly 1.5  \  # Warm Restart new poly exp
  --rescale 1.0 \  # Warm Restarts new learning rate ratio compared to original lr
  --sgd \  # 随机梯度下降
  --aux \  # 辅助损失
  --maxpool_size 14 \  # 最大池化大小
  --avgpool_size 9 \  # 平均池化大小
  --edge_points 128 \  # 
  --match_dim 64 \  # 在pfnet匹配时的维度
  --joint_edge_loss_pfnet \  # jiont loss 的边缘损失权重
  --edge_weight 25.0 \  # 边缘权重
  --ohem \
  --crop_size 512 \  # training crop size
  --max_epoch 64 \  # 总的epoch
  --wt_bound 1.0 \  # Weight Scaling for the losses
  --bs_mult 8 \  # Batch size for training per gpu
  --exp r2n101 \ # experiment directory name  训练后生成的模型保存的目录
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt
