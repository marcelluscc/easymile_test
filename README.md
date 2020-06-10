# Sequential Model on Fashion MNIST

## Usage

### Pre-requisite
1. Docker is installed

### Initial setup
```
docker build -t model .
docker run -it model
```
For the subsequent command, please run it inside the docker container's terminal

### Start Training from Scratch
```
python src/train.py
```

### Start Training from Latest Checkpoint
```
python src/train.py -c
```

### Run prediction from test images
```
python src/predict.py
```

### Run prediction from sample images (sample image file can be found in folder /sample)
```
python src/predict.py -f sample/a.png
```

## Model Summary
- Model type : Sequential
- Stopping time : early stopping, based on the improvement of `validation loss`, with patience of 5, and max epochs of 100 (defined in src/.env)
- Overfitting analysis : based on the plot summary of `accuracy`, `loss`, and `precision` of both training and test dataset
- Hyperparameter choice : `learning_rate` and `momentum` (defined in src/.env)
- Metrics : `accuracy` and `Precision`
- Dependency : listed in requirements.txt