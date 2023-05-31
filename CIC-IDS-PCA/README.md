
# Dataset

CICIDS2017 dataset contains benign and the most up-to-date common attacks, which resembles the true real-world data (PCAPs). It also includes the results of the network traffic analysis using CICFlowMeter with labeled flows based on the time stamp, source, and destination IPs, source and destination ports, protocols and attack (CSV files).

Dataset Link: https://www.unb.ca/cic/datasets/ids-2017.html




## Breakdown of all attacks
First, 8 dataset have merged and created the final dataset.

| Class Label             | Number of Sample                                                               |
| ----------------- | ------------------------------------------------------------------ |
|BENIGN	 |1743179|
|DoS Hulk|	231073|
|PortScan|	158930|
|DDoS	|128027|
|DoS GoldenEye|	10293|
|FTP-Patator	|7938|
|SSH-Patator|	5897|
|DoS slowloris	|5796|
|DoS Slowhttptest|	5499|
|Bot	|1966|
|Web Attack � Brute Force|1507|
|Web Attack � XSS|652|
|Infiltration	|36|
|Web Attack � Sql Injection|21|
|Heartbleed	|11|



## Methodology

- Data Standardization have applied  to bring down all the features to a common scale without distorting the differences in the range of the values.

- The dataset have 78 features which drives to high computational complexity. So, Principal Component Analysis (PCA) has used for reduce dimension of the dataframe from 78 to 25.

- The Sequential Deep Learning model consist of CNN-LSTM-Bi-LSTM-GRU

- The metrics which is need to evaluate model performance  have calculated from confusion matrix.

## Model
![image](https://github.com/kowshik14/ML-Project/assets/97826581/131a3fe9-b6ca-4e64-9221-c19a853d3027)


## Authors

- [Kowshik Sankar Roy](https://sites.google.com/view/kowshikroy)

