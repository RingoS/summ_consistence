LOG_PATH=/data/senyang/summ_consistence/log/classify_1
LOG_PATH=/data/senyang/summ_consistence/log/seq2seq_insertion_1

CUDA_VISIBLE_DEVICES=1 python main.py \
--output_dir $LOG_PATH/ \
--do_train True --do_eval True --evaluate_during_training True \
--num_train_epochs 3 \
--logging_dir $LOG_PATH/tensorboad_log/ \
--logging_first_step True \
--logging_steps 2000 \
--save_steps 4000 \
--save_total_limit 30 \
--model_name_or_path /data/senyang/data/bert_model/uncased_L-12_H-768_A-12/ \
--tokenizer_name bert-base-uncased \
--train_data_file /data/senyang/summ_consistence/spacy_ner_data/train/pseudo_data.csv \
--eval_data_file /data/senyang/summ_consistence/spacy_ner_data/val/pseudo_data.csv \
--mlm True \
--fp16 True \
--share_bert_param True \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--classify_or_insertion classify