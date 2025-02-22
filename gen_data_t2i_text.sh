log_dir=/home/leihaodong/ICCV25/HassLei/log
current_date=$(date +%Y-%m-%d)
CUDA_VISIBLE_DEVICES=0,2,3 
mkdir -p $log_dir/${current_date}
data_start=0
data_end=-1
data_path=/data/lei/dataset/laion_coco_100k/laion_coco_100k_0_100000/index_jsonl/n2i_000000.jsonl
# data_path=/data/lei/dataset/laion_coco_100k/laion_coco_100k_0_100000/index_jsonl/n2i_100000.jsonl
# data_path=/data/lei/dataset/laion_coco_100k/laion_coco_100k_0_100000/index_jsonl/n2i_200000.jsonl
nohup bash scripts/language/extract_flan_t5_feat_laion_coco_lei.sh \
    --data-path ${data_path} \
    --t5-path /data/lei/dataset/LACOCO_t2i_img_code/laion_coco384_labels \
    --t5-model-path /data/lei/localmodel/ \
    --data-start ${data_start} \
    --data-end ${data_end} > $log_dir/${current_date}/${current_date}_t2i_text1.log 2>&1 &