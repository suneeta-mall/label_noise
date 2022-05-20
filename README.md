# Labe Noise 

[![website](https://img.shields.io/badge/blog-post-brightgreen.svg?&style=flat-square&logo=Github&logoColor=white&link=https://suneeta-mall.github.io/2022/05/16/confident-learning-clean-data.html)](https://suneeta-mall.github.io/2022/05/16/confident-learning-clean-data.html)

A sample quick-hack multi-label image classification example written using PyTorch ecosystem using PyTorch Lightening. 

This app uses ResNet18 for model and [MLRNet](https://data.mendeley.com/datasets/7j9bv9vwsx/3) dataset. 

By default this sample app trains the ResNet model only 6 classes `['airplane', 'airport', 'buildings', 'cars', 'runway', 'trees']`. These can be changed via config but this sample project only focused on airplanes and airports.


The main intention of this code is to spike out [cleanlab](https://github.com/cleanlab/cleanlab/) project and test how well it does for multi-label problems. [This notebook](https://github.com/suneeta-mall/label_noise/blob/master/label_noise_notebook.ipynb) shows the results of cleanup noise detection. Background on this sample can be found in [this blog post](https://suneeta-mall.github.io/2022/05/16/confident-learning-clean-data.html).


## Data
MLRNet is also available as (7j9bv9vwsx-3.zip) but can also be downloaded from [MLRNet](https://data.mendeley.com/datasets/7j9bv9vwsx/3).
The expected structure of MLRNet is as per standard MLRNet structure is as shown is:
ROOT_DATA_DIR:
- Image:
    All image rar goes here
- Label:
    All CSVs go here



The data module of this app will unpack and handle as long as data is downloaded and provided in the above-specified fashion. 


## Install & other dev
```
docker build -t label_noise .
docker run --rm -it \
  --name label_noise \
  -p 8888:8888 \
  label_noise poetry run jupyter lab \
  --ip=0.0.0.0 --port=8888 --allow-root
```

### Poetry 
Use poetry to create env for this project:
```
poetry install
poetry shell
```

## RUN
To train:
```
python label_noise/app/train.py train  --data-directory MLRNet/7j9bv9vwsx-3/ --output-dir logs --epochs 19
```
