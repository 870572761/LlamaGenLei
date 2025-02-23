current_date=$(date +%Y-%m-%d)
log_dir=log/${current_date}/gen_data
mkdir -p $log_dir
CUDA_VISIBLE_DEVICES=0,1,2,3
nohup bash scripts/autoregressive/extract_codes_c2i.sh \
    --vq-ckpt /home/leihaodong/pretrained_models/vq_ds16_c2i.pt \
    --data-path /data2/lei/imagenet/ILSVRC/Data/CLS-LOC/train \
    --code-path /data2/lei/dataset/imagenet_code \
    --flip \
    --image-size 384 > $log_dir/${current_date}_gen_data.log 2>&1 &
echo "Process started with PID $!"