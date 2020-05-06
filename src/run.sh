LOG_PATH=/data/senyang/summ_consistence/mlm/test_mlm

CUDA_VISIBLE_DEVICES=1 python main.py \
--output_dir $LOG_PATH/ \
--do_train True --do_eval True --evaluate_during_training True \
--num_train_epochs 5 \
--logging_dir $LOG_PATH/tensorboad_log/ \
--logging_first_step True \
--save_steps 1000 \
--save_total_limit 30 \
--model_name_or_path bert-base-uncased \
--config_name bert-base-uncased \
--tokenizer_name bert-base-uncased \
--train_data_file /data/senyang/summ_consistence/spacy_ner_data/train/only_mask.csv \
--eval_data_file /data/senyang/summ_consistence/spacy_ner_data/val/only_mask.csv \
--mlm True \
--fp16 True 