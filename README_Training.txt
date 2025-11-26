Traning from scratch
1. Training data
- Download the webface4m dataset from "https://huggingface.co/datasets/gaunernst/webface4m-wds-gz/tree/main"
- Run /preprocess/precompute_features.py to obtain "Precomputed training data features"
Note: --save_dir parameter should be specified based on "style" define in /caface/trainer_base.py
eg: adaface_webface4m_subset_ir101_style35_augmenterv3_fp16/
2. Validation data
- Download IJB dataset and meta dta from "https://github.com/deepinsight/insightface/tree/master/recognition/_evaluation_/ijb"
- Run align_data.py to align the original data based on this metadata information.
Note: <ROOT_DATA> should follow the caface repo.
3. Download pretrained model and run training script. (Follow the github repo instructions)
