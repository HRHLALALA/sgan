export PYTHONPATH=$PWD:$PYTHONPATH
RUN_NAME=$1
DATASET_NAME=${2:-univ}
echo "Run name: ${RUN_NAME} on Dataset: ${DATASET_NAME}"
DTYPE=global

if [ -p RUN_NAME ]; then
  OUTPUT_DIR=$PWD/checkpoints/${DATASET_NAME}
else
  OUTPUT_DIR=$PWD/checkpoints/${DATASET_NAME}/$RUN_NAME
fi
mkdir -p $OUTPUT_DIR
python scripts/train.py \
  --output_dir $OUTPUT_DIR \
  --encoder_h_dim_d 48 \
  --neighborhood_size 2.0 \
  --clipping_threshold_d 0 \
  --clipping_threshold_g 2.0 \
  --delim tab \
  --print_every 100 \
  --pred_len 12 \
  --loader_num_workers 4 \
  --d_steps 1 \
  --encoder_h_dim_g 32 \
  --batch_size 64 \
  --num_epochs  200 \
  --num_layers 1 \
  --best_k 20 \
  --obs_len 8 \
  --skip 1 \
  --g_steps 1 \
  --g_learning_rate 0.001 \
  --l2_loss_weight 1.0 \
  --grid_size 8 \
  --bottleneck_dim 8 \
  --checkpoint_name checkpoint \
  --gpu_num 0  \
  --restore_from_checkpoint 1 \
  --dropout 0.0 \
  --checkpoint_every 300 \
  --noise_mix_type local \
  --decoder_h_dim_g 32 \
  --pooling_type none \
  --use_gpu 1 \
  --num_iterations 5540 \
  --noise_type gaussian \
  --d_learning_rate 0.001 \
  --checkpoint_start_from None \
  --timing 0 \
  --mlp_dim 64 \
  --num_samples_check 5000 \
  --d_type $DTYPE \
  --dataset_name ${DATASET_NAME} \
  --embedding_dim 16 \
  --noise_dim 8 \
  --pool_every_timestep 0 \
  --variety_loss_mode min
python scripts/evaluate_model.py --model_path ${OUTPUT_DIR}/checkpoint_with_model.pt
