python run.py configs/predict/predict_bee_nometa.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b6_ns \
    --model-config configs/bee/bee508.yaml \
    --checkpoint-dir ../checkpoints/bee508/tf_efficientnet_b6_ns/ \
    --save-file ../lb-predictions/bee508_5fold.pkl --num-workers 4

python run.py configs/predict/predict_bee_nometa.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b7 \
    --model-config configs/bee/bee517.yaml \
    --checkpoint-dir ../checkpoints/bee517/tf_efficientnet_b7/ \
    --save-file ../lb-predictions/bee517_5fold.pkl --num-workers 4

python run.py configs/predict/predict_bee_nometa.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b6_ns \
    --model-config configs/bee/bee608.yaml \
    --checkpoint-dir ../checkpoints/bee608/tf_efficientnet_b6_ns/ \
    --save-file ../lb-predictions/bee568_5fold.pkl --num-workers 4
