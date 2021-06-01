python3 inference/run_classifier_infer.py --load_model_path models/model_1_epochs.bin \
                                          --vocab_path models/google_zh_vocab.txt \
                                          --test_path datasets/toutiao/test_nolabel.tsv \
                                          --prediction_path result.tsv \
                                          --labels_num 10 \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible