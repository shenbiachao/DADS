# Please register ad in gym first!
# file structure:
# gym
# |--- envs
# |    |--- user
# |    |    |--- data
# |    |    |    |--- annthyroid.csv
# |    |    |    |--- civertype.csv
# |    |    |    |--- ...
# |    |    |
# |    |    |--- __init__.py
# |    |    |--- anomaly_detection.py
# |    |--- ...
# |--- ...


import copy
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import random
from sklearn.utils import shuffle
from gym import spaces
import os
import config
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
import umap

all_anomaly_classes = [[1, 2], [4, 6], [2, 3], [2, 3, 4, 5, 6, 7]]
all_normal_classes = [[3], [2], [1], [1]]
all_classes = [[3, 1, 2], [2, 4, 6], [1, 2, 3], [1, 2, 3, 4, 5, 6, 7]]

# load parameters in config.py
train_percentage = config.train_percentage
known_anomaly_num = config.known_anomaly_num
device = config.device
sample_num = config.sample_num
max_trajectory = config.max_trajectory
check_num = config.check_num
reward_list = config.reward_list
strategy_distribution = config.strategy_distribution
sampling_method_distribution = config.sampling_method_distribution
known_anomaly_classes = config.known_anomaly_classes
refresh_interval = config.refresh_interval
dataset_name = config.dataset_name
normalization = config.normalization

def load_data():
    """ Load data
    supported dataset: annthyroid, covertype, cardio, shuttle
    @return dataset_a: initialization of anomaly dataset
    @return dataset_u: initialization of unlabeled dataset (temporary dataset is initialized as empty)
    @return dataset_test: dataset used for evaluation
    @return test_label: true label of dataset_test
    """
    if dataset_name == 'ann':
        source = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/annthyroid.csv"))
        index = 0
    # for the following dataset we need to apply normalization to some columns
    elif dataset_name == 'cov':
        source = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/covertype.csv"))
        if normalization:
            source.iloc[:, :10] = (source.iloc[:, :10] - source.iloc[:, :10].min()) / (
                    source.iloc[:, :10].max() - source.iloc[:, :10].min())
        index = 1
    elif dataset_name == 'car':
        source = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/cardio.csv"))
        if normalization:
            source.iloc[:, :23] = (source.iloc[:, :23] - source.iloc[:, :23].min()) / (
                    source.iloc[:, :23].max() - source.iloc[:, :23].min())
        index = 2
    elif dataset_name == 'shu':
        source = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/shuttle.csv"))
        if normalization:
            source.iloc[:, :9] = (source.iloc[:, :9] - source.iloc[:, :9].min()) / (
                    source.iloc[:, :9].max() - source.iloc[:, :9].min())
        index = 3
    else:
        assert 0, "Dataset not existed."
    known_anomaly_class = known_anomaly_classes[index]
    all_anomaly_class = all_anomaly_classes[index]
    source = shuffle(source)
    width = source.shape[1]
    length = len(source)
    dataset_train = source.iloc[:int(length*train_percentage), :]
    dataset_test = source.iloc[int(length*train_percentage):, :]

    dataset_a = pd.DataFrame(columns=source.columns)
    dataset_u = pd.DataFrame(columns=source.columns)
    for i in range(len(dataset_train)):
        label = dataset_train.iloc[i, width-1]
        if label == known_anomaly_class and len(dataset_a) < known_anomaly_num:
            dataset_a = dataset_a.append(dataset_train.iloc[i, :])
        else:
            if len(dataset_a) < known_anomaly_num:
                dataset_u = dataset_u.append(dataset_train.iloc[i, :])
            else:
                dataset_u = dataset_u.append(dataset_train.iloc[i:, :])
                break
    dataset_a = dataset_a.reset_index(drop=True)
    dataset_u = dataset_u.reset_index(drop=True)

    source = np.concatenate([np.array(dataset_a.values.astype(float)), np.array(dataset_u.values.astype(float))])
    dataset_a = torch.tensor(dataset_a.values.astype(float))[:, :-1].float().to(device)
    dataset_u = torch.tensor(dataset_u.values.astype(float))[:, :-1].float().to(device)
    test_label = torch.tensor(dataset_test.values.astype(float))[:, -1].float().to(device)
    test_label = [1 if i in all_anomaly_class else 0 for i in test_label]
    dataset_test = torch.tensor(dataset_test.values.astype(float))[:, :-1].float().to(device)

    # source and index are used in function plot
    return dataset_a, dataset_u, dataset_test, test_label, source, index


class ad(gym.Env):
    def __init__(self):
        print("Start loading")
        dataset_a, dataset_u, dataset_test, test_label, source, index = load_data()
        print("Finish loading")
        print("Anomaly num: ", len(dataset_a), "   Normal num: ", len(dataset_u))

        # dataset_anomaly, dataset_unlabeled, and dataset_temp are three dataset defined in the paper
        # dataset_test and test_label are used for test
        # tempdata_confidence stores the confidence of each data in dataset_temp
        self.dataset_anomaly = dataset_a
        self.dataset_unlabeled = dataset_u
        self.dataset_anomaly_backup = dataset_a
        self.dataset_unlabeled_backup = dataset_u
        self.dataset_temp = torch.tensor([]).to(device)
        self.dataset_test = dataset_test
        self.test_label = test_label
        self.tempdata_confidence = []

        self.source = source
        self.classes = all_classes[index]

        # initialize current data to be an unlabeled data
        self.current_index = random.randint(0, len(self.dataset_unlabeled) - 1)
        self.current_data = self.dataset_unlabeled[self.current_index]
        self.current_class = "unlabeled"

        self.observation_space = spaces.Discrete(self.current_data.size()[0])
        self.action_space = spaces.Discrete(2)
        self.tot_steps = 0

        # choose the following six methods as unsupervised anomaly datection method
        self.clf_list = [IForest(), OCSVM(), LOF(), KNN(), HBOS(), AutoEncoder(hidden_neurons=[16, 16], verbose=0)]

    def reset(self):
        self.dataset_anomaly = self.dataset_anomaly_backup
        self.dataset_unlabeled = self.dataset_unlabeled_backup
        self.dataset_temp = torch.tensor([]).to(device)

        self.current_index = random.randint(0, len(self.dataset_unlabeled) - 1)
        self.current_data = self.dataset_unlabeled[self.current_index]
        self.current_class = 'unlabeled'

        return self.current_data

    def unsupervised_index(self, type_index, data):
        """ return anomaly index of a data using one of six unsupervised method defined in __init__
        the returned value is normalized to range [0, 1]
        @param type_index: the index of unsupervised detecto, if type_index>=len(self.clf_list), random sampling method will be applied
        @param data: data needs to be calculated, both single data or multiple data is acceptable
        """
        if type_index >= len(self.clf_list):
            if len(data.shape) == 1:
                return random.random()
            else:
                return [random.random() for i in len(data)]

        clf = self.clf_list[type_index]
        if len(data.shape) == 1:
            r = clf.predict_proba(data.cpu().unsqueeze(0))[0][1]
        else:
            r = np.array(clf.predict_proba(data.cpu()))[:, 1]
        data.to(device)

        return r

    def calculate_reward(self, action):
        """ calculate reward based on the class of current data and the action"""
        if self.current_class == 'anomaly':
            if action == 1:
                score = reward_list[0]
            else:
                score = reward_list[1]
        elif self.current_class == 'unlabeled':
            score = 0
        elif self.current_class == 'temp':
            if action == 1 and self.tempdata_confidence[self.current_index] >= check_num:
                score = reward_list[2]
            else:
                score = 0
        else:
            assert 0

        return score

    def sample_method_one(self):
        """ random sampling is used in dataset_anomaly"""
        self.current_class = 'anomaly'
        self.current_index = random.randint(0, len(self.dataset_anomaly) - 1)
        self.current_data = self.dataset_anomaly[self.current_index]

    def sample_method_two(self):
        """ random sampling is used in dataset_temp"""
        self.current_class = 'temp'
        self.current_index = random.randint(0, len(self.dataset_temp) - 1)
        self.current_data = self.dataset_temp[self.current_index]

    def sample_method_three(self, action):
        """ Use unsupervised-based method to sample data in dataset_unlabeled
        Each time sample a certain amount of data, then select the one with the highest unsupervised index
        """
        self.current_class = 'unlabeled'
        candidate_index = np.random.choice([i for i in range(len(self.dataset_unlabeled))], size=config.sample_num,
                                           replace=False)
        candidate = self.dataset_unlabeled[candidate_index]

        choice = np.random.choice([i for i in range(len(sampling_method_distribution))], size=1,
                                  p=sampling_method_distribution)[0]
        score = self.unsupervised_index(choice, candidate)
        self.current_index = np.argmax(score)
        self.current_data = self.dataset_unlabeled[self.current_index]

    def refresh_dataset(self, action):
        """ Refresh three dataset according to the data flow rules"""
        if action == 1 and self.current_class == 'unlabeled':
            self.dataset_unlabeled = torch.cat(
                [self.dataset_unlabeled[:self.current_index], self.dataset_unlabeled[self.current_index+1:]])
            self.tempdata_confidence.append(1)
            self.dataset_temp = torch.cat([self.dataset_temp, self.current_data.unsqueeze(0)])
        elif action == 1 and self.current_class == 'temp':
            if self.tempdata_confidence[self.current_index] >= check_num:
                self.dataset_temp = torch.cat(
                    [self.dataset_temp[:self.current_index], self.dataset_temp[self.current_index + 1:]])
                self.tempdata_confidence = self.tempdata_confidence[:self.current_index] + \
                                           self.tempdata_confidence[self.current_index + 1:]
                self.dataset_anomaly = torch.cat([self.dataset_anomaly, self.current_data.unsqueeze(0)])
            else:
                self.tempdata_confidence[self.current_index] = self.tempdata_confidence[self.current_index] + 1
        elif action == 0 and self.current_class == 'temp':
            self.dataset_temp = torch.cat(
                [self.dataset_temp[:self.current_index], self.dataset_temp[self.current_index + 1:]])
            self.dataset_unlabeled = torch.cat([self.dataset_unlabeled, self.current_data.unsqueeze(0)])
            self.tempdata_confidence = self.tempdata_confidence[:self.current_index] + \
                                       self.tempdata_confidence[self.current_index + 1:]

    def step(self, action):
        """ Environment takes an action, then returns the current data(regarded as state), reward and done flag"""
        reward = self.calculate_reward(action)
        if self.tot_steps % refresh_interval == 0:   # refresh unsupervised detector at regular intervals
            for i in range(len(self.clf_list)):
                if sampling_method_distribution[i] > 0:
                    clf = self.clf_list[i]
                    clf.fit(self.dataset_unlabeled.cpu())

        self.dataset_unlabeled.to(config.device)
        self.tot_steps = self.tot_steps + 1
        self.refresh_dataset(action)

        done = False
        if self.tot_steps % max_trajectory == 0:
            done = True

        while True:   # sample next data according to the probablity distribution
            choice = np.random.choice([0, 1, 2], size=1, p=strategy_distribution)[0]
            if choice == 0 and len(self.dataset_anomaly) != 0:
                self.sample_method_one()
                break
            elif choice == 1 and len(self.dataset_temp) != 0:
                self.sample_method_two()
                break
            elif choice == 2 and len(self.dataset_unlabeled) != 0:
                self.sample_method_three(action)
                break
        return self.current_data, reward, done, " "

    def evaluate(self, net):
        """ Evaluate the agent, return AUC_ROC and AUC_PR"""
        q_values = net(self.dataset_test)
        anomaly_score = q_values[:, 1]
        auc_roc = roc_auc_score(self.test_label, anomaly_score.cpu().detach())
        precision, recall, _thresholds = precision_recall_curve(self.test_label, anomaly_score.cpu().detach())
        auc_pr = auc(recall, precision)

        return auc_roc, auc_pr

    def plot(self):
        """ plot the decomposed data using UMAP
        color_list:
        green: dataset_unlabeled; orange: dataset_temp; red: dataset_anomaly
        shape_list:
        ".":  normal data; "^": class one anomaly_data; "+": class two anomaly data
        """
        anomaly = self.dataset_anomaly.cpu().numpy()
        temp = self.dataset_temp.cpu().numpy()
        unlabeled = self.dataset_unlabeled.cpu().numpy()
        if len(temp) != 0:
            dataset_for_plot = np.vstack([unlabeled, temp, anomaly])
        else:
            dataset_for_plot = np.vstack([unlabeled, anomaly])
        decomposer = umap.UMAP(random_state=0)
        decomposed_data = iter(decomposer.fit_transform(dataset_for_plot))

        color_list = ['green', 'orange', 'red']
        shape_list = [".", "^", "+"]
        for i in range(3):
            color = color_list[i]
            for item in [unlabeled, temp, anomaly][i]:
                line = np.where(np.sum((self.source[:, :-1] - item)**2, axis=1) < 1e-9)[0][0]
                type = self.source[line, -1]
                shape = shape_list[self.classes.index(type)]
                dot = next(decomposed_data)
                plt.scatter(dot[0], dot[1], c=color, marker=shape)
        plt.show()
