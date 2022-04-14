# Deep Anomaly Detection and Search

Please copy the user dataset to the location specified in the following, then register environment ad in __ init__.py

    gym             
    ├── envs
           ├── user
                  ├── data
                          ├── annthyroid.csv
                          ├── covertype.csv
                          └── ......
                  ├── __init__.py
                  └── anomaly_detection.py
           ├──  __init__.py (register here)
           └── ......



After registering, to run default test on dataset annthyroid, please follow these steps:

```commandline
git clone https://github.com/shenbiachao/DADS

pip3 install -r requirements.txt

python main.py
```



All configurations are listed in config.py

To change the dataset, please refer to parameter "dataset_name"

To specify other configurations, please see notations in code for detail