# coding=utf-8
# Deep Anomaly Detection and Search

import sys
from os.path import dirname, abspath
from SAC_Discrete import SAC_Discrete
from DQN import DQN
from Trainer import Trainer
from config import Config
from Utility_Functions import set_random_seeds
from anomaly_detection import ad
import warnings

warnings.filterwarnings("ignore")
sys.path.append(dirname(dirname(abspath(__file__))))

if __name__ == "__main__":
    parameter = Config()
    set_random_seeds(parameter.seed)
    environment = ad()

    print(parameter.__dict__)
    AGENT = SAC_Discrete
    trainer = Trainer(parameter, AGENT, environment)
    trainer.run_game_for_agent()
