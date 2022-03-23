import sys
import gym
from os.path import dirname, abspath
from SAC_Discrete import SAC_Discrete
from Trainer import Trainer
from config import Config
from Utility_Functions import set_global_seed
sys.path.append(dirname(dirname(abspath(__file__))))

parameter = Config()
parameter.environment = gym.make("ad-v0")

if __name__ == "__main__":
    set_global_seed(parameter.seed)
    print(parameter.__dict__)
    AGENT = SAC_Discrete
    trainer = Trainer(parameter, AGENT)
    trainer.run_game_for_agent()
