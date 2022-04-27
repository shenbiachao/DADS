import torch
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Base_Agent import Base_Agent
from collections import namedtuple, deque


class Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""

    def __init__(self, buffer_size, batch_size, seed, device=None):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Adds experience(s) into the replay buffer"""
        states, next_states = states.cpu(), next_states.cpu()
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [self.experience(state, action, reward, next_state, done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(experiences)
        else:
            experience = self.experience(states, actions, rewards, next_states, dones)
            self.memory.append(experience)

    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
            return states, actions, rewards, next_states, dones
        else:
            return experiences

    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class Base_Exploration_Strategy(object):
    """Base abstract class for agent exploration strategies. Every exploration strategy must inherit from this class
    and implement the methods perturb_action_for_exploration_purposes and add_exploration_rewards"""
    def __init__(self, config):
        self.config = config

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        raise ValueError("Must be implemented")

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        raise ValueError("Must be implemented")

    def reset(self):
        """Resets the noise process"""
        raise ValueError("Must be implemented")

class Epsilon_Greedy_Exploration(Base_Exploration_Strategy):
    """Implements an epsilon greedy exploration strategy"""
    def __init__(self, config):
        super().__init__(config)
        self.notified_that_exploration_turned_off = False
        if "exploration_cycle_episodes_length" in self.config.hyperparameters.keys():
            print("Using a cyclical exploration strategy")
            self.exploration_cycle_episodes_length = self.config.hyperparameters["exploration_cycle_episodes_length"]
        else:
            self.exploration_cycle_episodes_length = None

        if "random_episodes_to_run" in self.config.hyperparameters.keys():
            self.random_episodes_to_run = self.config.hyperparameters["random_episodes_to_run"]
            print("Running {} random episodes".format(self.random_episodes_to_run))
        else:
            self.random_episodes_to_run = 0

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action_values = action_info["action_values"]
        turn_off_exploration = action_info["turn_off_exploration"]
        episode_number = action_info["episode_number"]
        if turn_off_exploration and not self.notified_that_exploration_turned_off:
            print(" ")
            print("Exploration has been turned OFF")
            print(" ")
            self.notified_that_exploration_turned_off = True
        epsilon = self.get_updated_epsilon_exploration(action_info)


        if (random.random() > epsilon or turn_off_exploration) and (episode_number >= self.random_episodes_to_run):
            return torch.argmax(action_values).item()
        return  np.random.randint(0, action_values.shape[1])

    def get_updated_epsilon_exploration(self, action_info, epsilon=1.0):
        """Gets the probability that we just pick a random action. This probability decays the more episodes we have seen"""
        episode_number = action_info["episode_number"]
        epsilon_decay_denominator = self.config.hyperparameters["epsilon_decay_rate_denominator"]

        if self.exploration_cycle_episodes_length is None:
            epsilon = epsilon / (1.0 + (episode_number / epsilon_decay_denominator))
        else:
            epsilon = self.calculate_epsilon_with_cyclical_strategy(episode_number)
        return epsilon

    def calculate_epsilon_with_cyclical_strategy(self, episode_number):
        """Calculates epsilon according to a cyclical strategy"""
        max_epsilon = 0.5
        min_epsilon = 0.001
        increment = (max_epsilon - min_epsilon) / float(self.exploration_cycle_episodes_length / 2)
        cycle = [ix for ix in range(int(self.exploration_cycle_episodes_length / 2))] + [ix for ix in range(
            int(self.exploration_cycle_episodes_length / 2), 0, -1)]
        cycle_ix = episode_number % self.exploration_cycle_episodes_length
        epsilon = max_epsilon - cycle[cycle_ix] * increment
        return epsilon

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        return reward_info["reward"]

    def reset(self):
        """Resets the noise process"""
        pass


class DQN(Base_Agent):
    """A deep Q learning agent"""
    agent_name = "DQN"
    def __init__(self, config, environment):
        Base_Agent.__init__(self, config, environment)
        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"], config.seed, self.device)
        self.q_network_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyperparameters["learning_rate"], eps=1e-4)
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)

    def reset_game(self):
        super(DQN, self).reset_game()

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.action = self.pick_action()
            self.conduct_action(self.action)
            if self.time_for_q_network_to_learn():
                for _ in range(self.hyperparameters["learning_iterations"]):
                    self.learn()
            self.save_experience()
            self.state = self.next_state  # this is to set the state for the next iteration
            self.global_step_number += 1
        self.episode_number += 1

    def pick_action(self, state=None):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if state is None: state = self.state
        if isinstance(state, np.int64) or isinstance(state, int): state = np.array([state])
        state = state.unsqueeze(0).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)
        self.q_network_local.eval() #puts network in evaluation mode
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train() #puts network back in training mode
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})
        return action

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        if experiences is None: states, actions, rewards, next_states, dones = self.sample_experiences() #Sample experiences
        else: states, actions, rewards, next_states, dones = experiences
        loss = self.compute_loss(states, next_states, rewards, actions, dones)

        actions_list = [action_X.item() for action_X in actions ]

        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss, self.hyperparameters["gradient_clipping_norm"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        Q_targets_next = self.q_network_local(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        Q_targets_current = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        Q_expected = self.q_network_local(states).gather(1, actions.long()) #must convert actions to long so can be used as index
        return Q_expected

    def locally_save_policy(self):
        """Saves the policy"""
        torch.save(self.q_network_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    def time_for_q_network_to_learn(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin and there are
        enough experiences in the replay buffer to learn from"""
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()

    def right_amount_of_steps_taken(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin"""
        return self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def sample_experiences(self):
        """Draws a random sample of experience from the memory buffer"""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones

    def eval(self):
        """ Evaluate the actor"""
        return self.environment.evaluate(self.q_network_local)
