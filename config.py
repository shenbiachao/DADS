# All configurations are listed here

train_percentage = 0.8
known_anomaly_num = 60  # number of known anomalies, the default is he same as DPLAN
device = 'cuda'
sample_num = 50  # number data sampled when sampling dataset_unlabeled,
# then choose the one with the highest unsupervised index (e.g. the one with the highest Iforest anomaly score)
max_trajectory = 5000  # number of steps per each episode
check_num = 4   # threshold of confidence
reward_list = [1, -1, 0.5]   # rewards used in calculating reward
dataset_name = 'ann'
strategy_distribution = [0.3, 0.3, 0.4]   # probability distribution used to choose sampling strategy
sampling_method_distribution = [1, 0, 0, 0, 0, 0]   # probability distribution used to choose unsupervised method
known_anomaly_classes = [1, 4, 2, 2]   # class id of the known anomaly data
refresh_interval = 100   # refresh interval of unsupervised detection method
normalization = True   # whether normalize the data


class Config():
    def __init__(self):
        # env
        # This part is just a copy of the above parameters, for the purpose of save
        self.train_percentage = train_percentage
        self.known_anomaly_num = known_anomaly_num
        self.device = device
        self.sample_num = sample_num
        self.max_trajectory = max_trajectory
        self.check_num = check_num
        self.reward_list = reward_list
        self.strategy_distribution = strategy_distribution
        self.sampling_method_distribution = sampling_method_distribution
        self.known_anomaly_classes = known_anomaly_classes
        self.refresh_interval = refresh_interval
        self.dataset_name = dataset_name
        self.normalization = normalization

        # trainer
        self.num_episodes_to_run = 5   # episodes per training
        self.standard_deviation_results = 1.0
        self.runs_per_agent = 10   # training per agent
        self.plot_map = False   # whether plot the decomposed map of dataset for visualization
        self.seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # random seeds

        # sac agent
        self.hyperparameters = {
            "Actor_Critic_Agents": {
                "Actor": {
                    "learning_rate": 0.005,
                    "linear_hidden_units": [16],
                    "final_layer_activation": "Softmax",
                    "batch_norm": False,
                    "tau": 0.2,
                    "gradient_clipping_norm": 5,
                    "initialiser": "Xavier"
                },
                "Critic": {
                    "learning_rate": 0.005,
                    "linear_hidden_units": [16],
                    "final_layer_activation": None,
                    "batch_norm": False,
                    "buffer_size": 10000,
                    "tau": 0.2,
                    "gradient_clipping_norm": 5,
                    "initialiser": "Xavier"
                },

                "min_steps_before_learning": 1000,
                "batch_size": 64,
                "update_every_n_steps": 64,
                "learning_updates_per_learning_session": 64,
                "automatically_tune_entropy_hyperparameter": True,
                "entropy_term_weight": None,
                "discount_rate": 0.99,
            }
        }
