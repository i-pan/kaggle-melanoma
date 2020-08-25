```
# Prepare data 
cd data/isic2019
bash download_isic2019.sh
unzip ISIC_2019_Training_Input.zip
cd ..
kaggle competitions download -c siim-isic-melanoma-classification
unzip siim-isic-melanoma-classification.zip 
cd ../src/etl
python 0_create_isic2019_splits.py
cd ..

# Train ISIC 2019 model
python run.py configs/isic2019/mk001.yaml train --gpu 0,1,2,3 

# Run ISIC 2019 model
# Please change `model_checkpoints` in `src/configs/predict_isic2019.yaml` if using a checkpoint with a different name.
python run.py configs/predict/predict_isic2019.yaml
cd eval
python nevi.py
cd ../etl
python 10_combine_cdeotte_nevi_with_isic2019.py
cd ..

# Train ISIC 2019+2020 model
bash train_kfold.sh

# Get pseudolabels
python run.py configs/predict/predict_bee_nometa.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b6_ns \
    --model-config configs/bee/bee508.yaml \
    --checkpoint-dir ../checkpoints/bee508/tf_efficientnet_b6_ns/ \
    --save-file ../lb-predictions/bee508_5fold.pkl --num-workers 4
cd etl
python 12_make_pseudo_nometa.py
cd ..

# Train with pseudolabels
bash train_kfold_pseudolabel.sh

# Run inference and create submission
bash inference.sh
cd eval
python generate_sub.py
```







