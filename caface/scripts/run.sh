
DATA_ROOT='../Project-VHT/CAFACE_DATA/'
# PRECOMPUTE_TRAIN_REC='adaface_webface4m_subset_ir101_style35_augmenterv3_fp16'
PRECOMPUTE_TRAIN_REC='AdaFaceWebFace4M_style35'
# BACKBONE_MODEL='../pretrained_models/AdaFaceWebFace4M.ckpt'
BACKBONE_MODEL='../pretrained_models/AdaFaceWebFace4M.ckpt'
CENTER_PATH='../pretrained_models/center_WebFace4MAdaFace_webface4m_subset.pth'

python main.py \
          --prefix hpcc_caface_adaface_catv9_g4_conf512_small \
          --data_root ${DATA_ROOT} \
          --use_precompute_trainrec ${PRECOMPUTE_TRAIN_REC} \
          --start_from_model_statedict ${BACKBONE_MODEL} \
          --center_path ${CENTER_PATH} \
          --train_data_path WebFace4M \
          --gpus 1 \
          --wandb_tags ir_101_arcface \
          --arch ir_101 \
          --tpus 0 \
          --num_workers 16 \
          --batch_size 512 \
          --val_batch_size 64 \
          --num_images_per_identity 32 \
          --freeze_model \
          --aggregator_name style_norm_srm \
          --intermediate_type style \
          --style_index 3,5 \
          --decoder_name catv9_g4_conf512_small \
          --center_loss_lambda 1.0 \
          --limit_train_batches 1.0 \
          --same_aug_within_group_prob 0.75 \
          --datafeed_scheme dual_multi_v1 \
          --epochs 10 \
          --lr 1e-3 \
          --optimizer_type adamw \
          --lr_milestones 6,9 \
          --lr_scheduler step \
          --weight_decay 5e-4 \
          --img_aug_scheme v3 \
          --use_memory \
          --memory_loss_lambda 1.0
#   --head adaface_v3 \
#   --arch ir_101_arcface \