# Sberbank-Russian-Housing-Market_docker
Flask applications on Docker for predicting real estate prices on the Russian market.

#### Stack:
ML: sklearn, pandas, numpy, xgboost

API: Flask

Platform: Docker

Data: from kaggle - https://www.kaggle.com/c/sberbank-russian-housing-market

#### Task: 
prediction of real estate prices, regression task

290 features in use.
The model was fitted on 30471 observations.

Feature transformations: TimestampEncoding, LabelEncoding, SimpleImputer, MinMaxScaler

Model: XGBoostRegressor

### Clone repo and build image
```
$ git clone https://github.com/Krivosheenkova/Sberbank-Russian-Housing-Market_docker.git
$ cd Sberbank-Russian-Housing-Market_docker
$ docker build -t sberbank-housing-market_docker_flask ./docker-flask-sberbank/
```

### Run container

Here you need to create a directory locally and save the pre-trained model there (<your_local_path_to_pretrained_models> you need to replace with the path to this directory)
```
$ docker run -d -p 4140:4140 -v <your_local_path_to_pretrained_models>:/app/app/models sberbank-housing-market_docker_flask
```

### Run get_predictions.py 
```
$ python get_predictions.py <path_to_csv_file> --outfile predictions.csv
```
