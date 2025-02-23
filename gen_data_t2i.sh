current_date=$(date +%Y-%m-%d)
log_dir=/home/leihaodong/ICCV25/HassLei/log/${current_date}/eval
mkdir -p $log_dir
data_start=0
data_end=-1
# data_path=/data/lei/dataset/laion_coco_100k/laion_coco_100k_0_100000/index_jsonl/image_000000.jsonl
# data_path=/data/lei/dataset/laion_coco_100k/laion_coco_100k_0_100000/index_jsonl/image_200000.jsonl
# start=100000
start=200000
data_path=/data/lei/dataset/laion_coco_100k/laion_coco_100k_0_100000/index_jsonl/image_${start}.jsonl
CUDA_VISIBLE_DEVICES=0,1,2,3 
image_size=256
nohup bash scripts/autoregressive/extract_codes_LACOCO_t2i.sh \
    --vq-ckpt /home/leihaodong/pretrained_models/vq_ds16_t2i.pt \
    --code-path /data/lei/dataset/LACOCO_t2i_img_code/laion_coco${image_size}_codes \
    --data-path  ${data_path} \
    --data-start ${data_start} \
    --data-end ${data_end} \
    --image-size ${image_size} > $log_dir/${current_date}_${start}_t2i.log 2>&1 &