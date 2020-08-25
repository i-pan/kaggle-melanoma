conda create -y -n shoebox python=3.7 pip
conda activate shoebox
conda install -y pytorch torchvision -c pytorch
conda install -y -c conda-forge opencv
conda install -y scikit-learn scikit-image pandas pyyaml tqdm
conda install -y -c conda-forge gdcm

pip install pydicom timm pretrainedmodels albumentations kaggle iterative-stratification
