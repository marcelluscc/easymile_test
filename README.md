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