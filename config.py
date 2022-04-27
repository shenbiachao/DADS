# All configurations are listed here

class Config():
    def __init__(self):
        self.dataset_name = 'har'
        self.manual_dataset = True  # whether manually change the anomaly percentage in unlabeled dataset

        # env
        self.train_percentage = 0.8
        self.known_anomaly_num = 60  # number of known anomalies, default is same as DPLAN
        self.contamination_rate = 0.02  # anomaly percentage in unlabeled dataset, only take effect when manual_dataset=True
        self.device = 'cuda'
        self.sample_num = 50  # number data sampled when sampling dataset_unlabeled,
        # then choose the one with the highest unsupervised index (e.g. the one with the highest Iforest anomaly score)
        self.max_trajectory = 5000  # number of steps per each episode
        self.check_num = 4  # threshold of confidence
        self.reward_list = [1, -2, 0.5]  # rewards used in calculating reward
        self.strategy_distribution = [0.3, 0.3, 0.4]  # probability distribution used to choose sampling strategy
        self.sampling_method_distribution = [0, 0, 0, 1]  # probability distribution used to choose unsupervised method
        self.reward_method_distribution = [0, 0, 0, 0]
        self.extra_reward_ratio = 1
        self.known_anomaly_classes = {'ann': 2, 'cov': 4, 'car': 2, 'shu': 2, 'har': 2}  # class id of the known anomaly data
        self.refresh_interval = 500  # refresh interval of unsupervised detection method
        self.normalization = True  # whether normalize the data


        # trainer
        self.num_episodes_to_run = 10   # episodes per training
        self.standard_deviation_results = 1.0
        self.runs_per_agent = 10   # training per agent
        self.plot_map = False   # whether plot the decomposed map of dataset for visualization
        self.seed = 0   # random seed

        # sac agent
        self.hyperparameters = {
            "Actor_Critic_Agents": {
                "Actor": {
                    "learning_rate": 0.0005,
                    "linear_hidden_units": [32, 16],
                    "final_layer_activation": "Softmax",
                    "batch_norm": False,
                    "tau": 0.2,
                    "gradient_clipping_norm": 5,
                    "initialiser": "Xavier"
                },
                "Critic": {
                    "learning_rate": 0.0005,
                    "linear_hidden_units": [32, 16],
                    "final_layer_activation": None,
                    "batch_norm": False,
                    "buffer_size": 100000,
                    "tau": 0.2,
                    "gradient_clipping_norm": 5,
                    "initialiser": "Xavier"
                },

                "min_steps_before_learning": 10000,
                "batch_size": 128,
                "update_every_n_steps": 128,
                "learning_updates_per_learning_session": 128,
                "automatically_tune_entropy_hyperparameter": True,
                "entropy_term_weight": None,
                "discount_rate": 0.99,
            }
        }
