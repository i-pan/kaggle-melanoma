# CV 0.9321
pyt run.py configs/predict/predict_bee.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b6_ns \
    --model-config configs/bee/bee008.yaml \
    --checkpoint-dir ../checkpoints/bee008/tf_efficientnet_b6_ns/ \
    --save-file ../lb-predictions/bee008_5fold.pkl --num-workers 4
# CV 0.9247
pyt run.py configs/predict/predict_bee.yaml predict_kfold \
    --gpu 0 --backbone efficientnet_b3_pruned \
    --model-config configs/bee/bee009.yaml \
    --checkpoint-dir ../checkpoints/bee009/efficientnet_b3_pruned/ \
    --save-file ../lb-predictions/bee009_5fold.pkl --num-workers 4
# CV 0.9290
pyt run.py configs/predict/predict_bee.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b4 \
    --model-config configs/bee/bee010.yaml \
    --checkpoint-dir ../checkpoints/bee010/tf_efficientnet_b4/ \
    --save-file ../lb-predictions/bee010_5fold.pkl --num-workers 4
# CV 0.9244
pyt run.py configs/predict/predict_bee.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b8 \
    --model-config configs/bee/bee011.yaml \
    --checkpoint-dir ../checkpoints/bee011/tf_efficientnet_b8/ \
    --save-file ../lb-predictions/bee011_5fold.pkl --num-workers 4
# CV 0.9306
pyt run.py configs/predict/predict_bee.yaml predict_kfold \
    --gpu 0 --backbone efficientnet_b3_pruned \
    --model-config configs/bee/bee013.yaml \
    --checkpoint-dir ../checkpoints/bee013/efficientnet_b3_pruned/ \
    --save-file ../lb-predictions/bee013_5fold.pkl --num-workers 4
# CV 0.9290
pyt run.py configs/predict/predict_bee.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b4 \
    --model-config configs/bee/bee014.yaml \
    --checkpoint-dir ../checkpoints/bee014/tf_efficientnet_b4/ \
    --save-file ../lb-predictions/bee014_5fold.pkl --num-workers 4
# CV 0.9312
pyt run.py configs/predict/predict_bee.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b5 \
    --model-config configs/bee/bee015.yaml \
    --checkpoint-dir ../checkpoints/bee015/tf_efficientnet_b5/ \
    --save-file ../lb-predictions/bee015_5fold.pkl --num-workers 4
# CV 0.9303
pyt run.py configs/predict/predict_bee.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b6 \
    --model-config configs/bee/bee016.yaml \
    --checkpoint-dir ../checkpoints/bee016/tf_efficientnet_b6/ \
    --save-file ../lb-predictions/bee016_5fold.pkl --num-workers 4
# CV 0.9330
pyt run.py configs/predict/predict_bee.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b7 \
    --model-config configs/bee/bee017.yaml \
    --checkpoint-dir ../checkpoints/bee017/tf_efficientnet_b7 \
    --save-file ../lb-predictions/bee017_5fold.pkl --num-workers 4
# CV 0.9263
pyt run.py configs/predict/predict_bee.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b8 \
    --model-config configs/bee/bee018.yaml \
    --checkpoint-dir ../checkpoints/bee018/tf_efficientnet_b8/ \
    --save-file ../lb-predictions/bee018_5fold.pkl --num-workers 4



# CV 0.934
pyt run.py configs/predict/predict_bee2.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b6_ns \
    --model-config configs/bee/bee022.yaml \
    --checkpoint-dir ../checkpoints/bee022/tf_efficientnet_b6_ns/ \
    --save-file ../lb-predictions/bee022_5fold.pkl --num-workers 4
# CV 0.932
pyt run.py configs/predict/predict_bee2.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b7 \
    --model-config configs/bee/bee023.yaml \
    --checkpoint-dir ../checkpoints/bee023/tf_efficientnet_b7/ \
    --save-file ../lb-predictions/bee023_5fold.pkl --num-workers 4


pyt run.py configs/predict/predict_bee.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b6_ns \
    --model-config configs/bee/bee108.yaml \
    --checkpoint-dir ../checkpoints/bee108/tf_efficientnet_b6_ns/ \
    --save-file ../lb-predictions/bee108_5fold.pkl --num-workers 4
# CV 0.932
pyt run.py configs/predict/predict_bee.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b7 \
    --model-config configs/bee/bee117.yaml \
    --checkpoint-dir ../checkpoints/bee117/tf_efficientnet_b7/ \
    --save-file ../lb-predictions/bee117_5fold.pkl --num-workers 4


pyt run.py configs/predict/predict_bee.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b6_ns \
    --model-config configs/bee/bee208.yaml \
    --checkpoint-dir ../checkpoints/bee208/tf_efficientnet_b6_ns/ \
    --save-file ../lb-predictions/bee208_5fold.pkl --num-workers 4


pyt run.py configs/predict/predict_bee.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b7 \
    --model-config configs/bee/bee217.yaml \
    --checkpoint-dir ../checkpoints/bee217/tf_efficientnet_b7/ \
    --save-file ../lb-predictions/bee217_5fold.pkl --num-workers 4

pyt run.py configs/predict/predict_bee_nometa.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b6_ns \
    --model-config configs/bee/bee608.yaml \
    --checkpoint-dir ../checkpoints/bee608/tf_efficientnet_b6_ns/ \
    --save-file ../lb-predictions/bee608_5fold.pkl --num-workers 4


