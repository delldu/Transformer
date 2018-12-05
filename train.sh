# python train.py \
#     -train_atok data/train/train.en-zh.atok \
#     -valid_atok data/valid/valid.en-zh.atok \
#     -save_model trained \
#     -save_mode best \
#     -proj_share_weight \
#     -label_smoothing

python train.py \
    -train_atok data/valid/valid.en-zh.atok \
    -valid_atok data/valid/valid.en-zh.atok \
    -save_model trained \
    -save_mode best \
    -proj_share_weight \
    -label_smoothing
