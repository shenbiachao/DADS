import gym
import torch
import time
from nn_builder.pytorch.NN import NN
from Utility_Functions import set_random_seeds


class Base_Agent(object):
    """ Base class of agent
    Inherited by SAC_Discrete
    """
    def __init__(self, config, environment):
        self.config = config
        self.environment = environment
        self.action_types = "DISCRETE"
        self.action_size = int(self.environment.action_space.n)
        self.config.action_size = self.action_size

        self.state_size = int(self.environment.reset().size()[0])
        self.hyperparameters = config.hyperparameters
        self.total_episode_score_so_far = 0
        self.episode_number = 0
        self.device = config.device
        self.global_step_number = 0
        self.turn_off_exploration = False
        gym.logger.set_level(40)  # stops it from printing an unnecessary warning

    def step(self):
        """ Take a step in the game. This method must be overriden by any agent"""
        raise ValueError("Step needs to be implemented by the agent")

    def eval(self):
        """ Evaluate the agent itself in the game. This method must be overriden by any agent"""
        raise ValueError("Eval needs to be implemented by the agent")

    def reset_game(self):
        self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False

    def run_n_episodes(self, logger, train_round):
        """ Run game to completion train_round times and then summarises results"""
        start = time.time()
        if self.config.plot_map:
            self.environment.plot()
        while self.episode_number < self.config.num_episodes_to_run:
            self.reset_game()
            self.step()  # here step functon will complete a simgle training episode
            auc_roc, auc_pr = self.eval()
            res = "Episode {}: auc_roc {:.03f} auc_pr {:.03f}".format(self.episode_number, auc_roc,
                                                                      auc_pr) + "\nanomaly: {} temp: {} unlabeled: {}".format(
                len(self.environment.dataset_anomaly), len(self.environment.dataset_temp),
                len(self.environment.dataset_unlabeled))
            print(res)
            logger.log_str(res)
            logger.log_var("round" + str(train_round) + "/auc_roc", auc_roc, self.episode_number)
            logger.log_var("round" + str(train_round) + "/auc_pr", auc_pr, self.episode_number)
        time_taken = time.time() - start
        if self.config.plot_map:
            self.environment.plot()
        return time_taken, auc_roc, auc_pr

    def conduct_action(self, action):
        self.next_state, self.reward, self.done, _ = self.environment.step(action)
        self.total_episode_score_so_far += self.reward

    def enough_experiences_to_learn_from(self):
        return len(self.memory) > self.hyperparameters["batch_size"]

    def save_experience(self, memory=None, experience=None):
        if memory is None: memory = self.memory
        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        if not isinstance(network, list): network = [network]
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm)
        optimizer.step()

    def soft_update_of_target_network(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def create_NN(self, input_dim, output_dim, key_to_use=None, hyperparameters=None):
        """ Create a neural network for the agents to use"""
        if hyperparameters is None: hyperparameters = self.hyperparameters
        if key_to_use: hyperparameters = hyperparameters[key_to_use]

        default_hyperparameter_choices = {"output_activation": None, "hidden_activations": "relu", "dropout": 0.0,
                                          "initialiser": "default", "batch_norm": False,
                                          "columns_of_data_to_be_embedded": [],
                                          "embedding_dimensions": [], "y_range": ()}

        for key in default_hyperparameter_choices:
            if key not in hyperparameters.keys():
                hyperparameters[key] = default_hyperparameter_choices[key]

        return NN(input_dim=input_dim, layers_info=hyperparameters["linear_hidden_units"] + [output_dim],
                  output_activation=hyperparameters["final_layer_activation"],
                  batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
                  hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
                  columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
                  embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"]).to(self.device)

    @staticmethod
    def copy_model_over(from_model, to_model):
        """ Copy model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())