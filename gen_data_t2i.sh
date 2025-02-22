log_dir=/home/leihaodong/ICCV25/HassLei/log
current_date=$(date +%Y-%m-%d)
CUDA_VISIBLE_DEVICES=0,1,2,3 
mkdir -p $log_dir/${current_date}
data_start=0
data_end=-1
# data_path=/data/lei/dataset/laion_coco_100k/laion_coco_100k_0_100000/index_jsonl/jsonl_100000/image_100000.jsonl
data_path=/data/lei/dataset/laion_coco_100k/laion_coco_100k_0_100000/index_jsonl/jsonl_200000/image_200000.jsonl
nohup bash scripts/autoregressive/extract_codes_LACOCO_t2i.sh \
    --vq-ckpt /home/leihaodong/pretrained_models/vq_ds16_t2i.pt \
    --data-path  ${data_path} \
    --code-path /data/lei/dataset/LACOCO_t2i_img_code/laion_coco384_codes \
    --data-start ${data_start} \
    --data-end ${data_end} \
    --image-size 384 > $log_dir/${current_date}/${current_date}_2_t2i.log 2>&1 &