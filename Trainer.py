import copy
import numpy as np
from Utility_Functions import Logger


class Trainer(object):
    """ Train the agent
    @property config: configuration of training
    @property agent: SAC agent to be trained
    @property environment: ad defined in anomaly_detection.py
    """
    def __init__(self, config, agent, environment):
        self.config = config
        self.agent = agent
        self.environment = environment
        env_name = config.dataset_name
        self.logger = Logger("logs", prefix=env_name + "-" + "", print_to_terminal=True)
        self.logger.log_str("logging to {}".format(self.logger.log_path))
        self.logger.log_str(config.__dict__)

    def run_game_for_agent(self):
        """Run the training process several times to calculate the mean and variance of AUC_PR and AUC_ROC"""
        agent_group = "Actor_Critic_Agents"
        agent_round = 1
        auc_roc_list = []
        auc_pr_list = []
        time_list = []
        for run in range(self.config.runs_per_agent):
            print("Run ", run + 1)
            agent_config = copy.deepcopy(self.config)
            agent_config.hyperparameters = agent_config.hyperparameters[agent_group]
            agent = self.agent(agent_config, self.environment)

            time_taken, auc_roc, auc_pr = agent.run_n_episodes(self.logger, run)   # run a single training process

            auc_roc_list.append(auc_roc)
            auc_pr_list.append(auc_pr)
            time_list.append(time_taken)
            time_str = "Time taken: {}".format(time_taken)
            self.logger.log_str(time_str)
            print(time_str, flush=True)
            self.print_two_empty_lines()
            agent_round += 1
        auc_roc_str = "auc_roc: mean {}, variance {}".format(np.mean(auc_roc_list), np.var(auc_roc_list))
        self.logger.log_str(auc_roc_str)
        print(auc_roc_str)
        auc_pr_str = "auc_pr: mean {}, variance {}".format(np.mean(auc_pr_list), np.var(auc_pr_list))
        self.logger.log_str(auc_pr_str)
        print(auc_pr_str)
        time_str = "time: mean {}, variance {}".format(np.mean(time_list), np.var(time_list))
        self.logger.log_str(time_str)
        print(time_str)

    def print_two_empty_lines(self):
        print("-----------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------")
        print(" ")
