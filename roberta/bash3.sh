python3 finetune/run_classifier.py --pretrained_model_path models/model.bin \
                                   --vocab_path models/google_zh_vocab.txt --config_path models/bert/tiny_config.json \
                                   --train_path datasets/toutiao/data_train.tsv \
                                   --dev_path datasets/toutiao/data_test.tsv \
                                   --test_path datasets/toutiao/data_test.tsv \
                                   --learning_rate 3e-4 --batch_size 64 --epochs_num 8 \
                                   --embedding word_pos_seg --encoder transformer --mask fully_visible
