pyt run.py configs/bee/bee108.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b6_ns --fold 0
pyt run.py configs/bee/bee108.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b6_ns --fold 1
pyt run.py configs/bee/bee108.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b6_ns --fold 2
pyt run.py configs/bee/bee108.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b6_ns --fold 3
pyt run.py configs/bee/bee108.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b6_ns --fold 4

pyt run.py configs/bee/bee117.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b7 --fold 0
pyt run.py configs/bee/bee117.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b7 --fold 1
pyt run.py configs/bee/bee117.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b7 --fold 2
pyt run.py configs/bee/bee117.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b7 --fold 3
pyt run.py configs/bee/bee117.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b7 --fold 4

pyt run.py configs/bee/bee208.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b6_ns --fold 0 --load-previous ../checkpoints/bee108/tf_efficientnet_b6_ns/fold0/EFFNET_005_VM-0.9441.PTH
pyt run.py configs/bee/bee208.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b6_ns --fold 1 --load-previous ../checkpoints/bee108/tf_efficientnet_b6_ns/fold1/EFFNET_001_VM-0.9415.PTH --eps 1.0e-4
pyt run.py configs/bee/bee208.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b6_ns --fold 2 --load-previous ../checkpoints/bee108/tf_efficientnet_b6_ns/fold2/EFFNET_003_VM-0.9459.PTH
pyt run.py configs/bee/bee208.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b6_ns --fold 3 --load-previous ../checkpoints/bee108/tf_efficientnet_b6_ns/fold3/EFFNET_005_VM-0.9320.PTH
pyt run.py configs/bee/bee208.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b6_ns --fold 4 --load-previous ../checkpoints/bee108/tf_efficientnet_b6_ns/fold4/EFFNET_001_VM-0.9270.PTH

pyt run.py configs/bee/bee217.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b7 --fold 0 --load-previous ../checkpoints/bee117/tf_efficientnet_b7/fold0/EFFNET_008_VM-0.9398.PTH
pyt run.py configs/bee/bee217.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b7 --fold 1 --load-previous ../checkpoints/bee117/tf_efficientnet_b7/fold1/EFFNET_005_VM-0.9453.PTH --eps 1.0e-4
pyt run.py configs/bee/bee217.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b7 --fold 2 --load-previous ../checkpoints/bee117/tf_efficientnet_b7/fold2/EFFNET_005_VM-0.9519.PTH
pyt run.py configs/bee/bee217.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b7 --fold 3 --load-previous ../checkpoints/bee117/tf_efficientnet_b7/fold3/EFFNET_005_VM-0.9382.PTH
pyt run.py configs/bee/bee217.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b7 --fold 4 --load-previous ../checkpoints/bee117/tf_efficientnet_b7/fold4/EFFNET_005_VM-0.9357.PTH

pyt run.py configs/bee/bee408.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b6_ns --fold 0 
pyt run.py configs/bee/bee408.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b6_ns --fold 1
pyt run.py configs/bee/bee408.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b6_ns --fold 2 
pyt run.py configs/bee/bee408.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b6_ns --fold 3 
pyt run.py configs/bee/bee408.yaml train --gpu 0,1,2,3 --num-workers 4 --backbone tf_efficientnet_b6_ns --fold 4 

