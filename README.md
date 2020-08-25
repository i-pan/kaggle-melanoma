# SIIM-OSIC Melanoma Classification: 2nd Place

Environment:

Python 3.7.7

Anaconda 

PyTorch 1.6

4 NVIDIA Quadro RTX 6000 24GB 


## Setup Python environment
```
conda create -n melanoma python=3.7 pip
pip install -r requirements.txt
```

## Download data
```
cd data/isic2019
bash download_isic2019.sh
unzip ISIC_2019_Training_Input.zip
cd ..
kaggle competitions download -c siim-isic-melanoma-classification
unzip siim-isic-melanoma-classification.zip 
```

## Train ISIC 2019 model
```
cd src/etl
python 0_create_isic2019_splits.py
cd ..
python run.py configs/isic2019/mk001.yaml train --gpu 0,1,2,3 --num-workers 4
```

## Run ISIC 2019 model on ISIC 2020 data
Please change `model_checkpoints` in `src/configs/predict_isic2019.yaml` if using a checkpoint with a different name.
```
cd src
python run.py configs/predict/predict_isic2019.yaml
```

## Assign nevus labels to ISIC 2020 data
```
cd src/eval
python nevi.py
cd ../etl
python 10_combine_cdeotte_nevi_with_isic2019.py
```

## Train ISIC 2019+2020 model
```
cd src
bash train_kfold.sh
```

## Create pseudolabels
```
cd src
python run.py configs/predict/predict_bee_nometa.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b6_ns \
    --model-config configs/bee/bee508.yaml \
    --checkpoint-dir ../checkpoints/bee508/tf_efficientnet_b6_ns/ \
    --save-file ../lb-predictions/bee508_5fold.pkl --num-workers 4
cd etl
python 12_make_pseudo_nometa.py
```

## Train on pseudolabeled data
```
cd src
bash train_kfold_pseudolabel.sh
```

## Inference on test set
```
cd src
bash inference.sh
```

## Generate submission
```
cd eval
python generate_sub.py
```
Final submission CSV will be saved within `eval/` directory as `final_submission.csv`.










