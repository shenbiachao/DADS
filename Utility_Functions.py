import numpy as np
from abc import ABCMeta
import os
from datetime import datetime
from tensorboardX import SummaryWriter
import torch
from torch.distributions import Categorical, normal
from collections import namedtuple, deque
import random
import gym


def abstract(cls):
    return ABCMeta(cls.__name__, cls.__bases__, dict(cls.__dict__))


def create_actor_distribution(action_types, actor_output, action_size):
    if action_types == "DISCRETE":
        assert actor_output.size()[1] == action_size, "Actor output the wrong size"
        action_distribution = Categorical(actor_output)  # This creates a distribution to sample from
    else:
        assert actor_output.size()[1] == action_size * 2, "Actor output the wrong size"
        means = actor_output[:, :action_size].squeeze(0)
        stds = actor_output[:,  action_size:].squeeze(0)
        if len(means.shape) == 2: means = means.squeeze(-1)
        if len(stds.shape) == 2: stds = stds.squeeze(-1)
        if len(stds.shape) > 1 or len(means.shape) > 1:
            raise ValueError("Wrong mean and std shapes - {} -- {}".format(stds.shape, means.shape))
        action_distribution = normal.Normal(means.squeeze(0), torch.abs(stds))
    return action_distribution


class Logger():
    def __init__(self, log_path, prefix="", print_to_terminal=True):
        unique_path = self.make_simple_log_path(prefix)
        log_path = os.path.join(log_path, unique_path)
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.tb_writer = SummaryWriter(log_path)
        self.log_file_path = os.path.join(log_path, "output.txt")
        self.print_to_terminal = print_to_terminal

    def make_simple_log_path(self, prefix):
        now = datetime.now()
        suffix = now.strftime("%d(%H.%M)")
        pid_str = os.getpid()
        return "{}-{}-{}".format(prefix, suffix, pid_str)

    @property
    def log_dir(self):
        return self.log_path

    def log_str(self, content, print_to_terminal=False):
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        if print_to_terminal:
            print("\033[32m{}\033[0m:\t{}".format(time_str, content))
        with open(self.log_file_path, 'a+') as f:
            f.write("{}:\t{}\n".format(time_str, content))

    def log_var(self, name, val, ite):
        """ log variable
        @param name: variable name
        @param val: value of variable
        @param ite: current iteration/step in training
        """
        self.tb_writer.add_scalar(name, val, ite)


class Replay_Buffer(object):
    """
    Replay buffer to store past experiences that the agent can then use for training data
    """
    def __init__(self, buffer_size, batch_size, device=None):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, states, actions, rewards, next_states, dones):
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
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
            return states, actions, rewards, next_states, dones
        else:
            return experiences

    def separate_out_data_types(self, experiences):
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


def set_random_seeds(random_seed):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_seed)
    # tf.set_random_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)
    if hasattr(gym.spaces, 'prng'):
        gym.spaces.prng.seed(random_seed)
