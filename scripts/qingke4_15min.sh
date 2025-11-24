export CUDA_VISIBLE_DEVICES=2

model_name=TimesNet

python -u run.py \
  --gpu_type mps \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ \
  --data_path project4load_short.csv \
  --model_id qingke_960_1 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --target 'sum' \
  --d_model 512 \
  --batch_size 16 \
  --d_ff 512 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 
