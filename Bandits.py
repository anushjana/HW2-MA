from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



np.random.seed(1)
NUM_TRIALS = 2000
EPS = 0.1
Bandit_Reward = [1, 2, 3, 4]
TAU = 1 / 5


class Bandit(ABC):
    """ Abstract bandit class"""

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self, x):
        pass

    @abstractmethod
    def experiment(self):
        pass

    def report(self, result_greedy, result_thompson) -> None:
        print('Epsilon Greedy')
        self.plot1(result_greedy)
        self.plot1(result_greedy, reward=False)

        print('average reward is ', np.mean(np.sum(result_greedy['rewards'], axis=1)))
        print('average regret is ', np.mean(np.sum(result_greedy['regrets'], axis=1)))
        print("")

        print('Thompson Sampling')
        self.plot1(result_thompson)
        self.plot1(result_thompson, reward=False)

        print('average reward is ', np.mean(np.sum(result_thompson['rewards'], axis=1)))
        print('average regret is ', np.mean(np.sum(result_greedy['regrets'], axis=1)))

        bandit_data = pd.DataFrame({'Algorithm Used': [], 'Bandit Return': [], 'Reward': []})

        for algorithm_index, algorithm in enumerate([result_greedy, result_thompson]):
            for bandit_index, bandit in enumerate(algorithm['bandits']):
                if algorithm_index == 1:
                    data = pd.Series({'Algorithm Used' :'Thompson Sampling',
                                      'Bandit Return': bandit,
                                      'Reward': np.sum(algorithm['rewards'][:, bandit_index])})
                else:
                    data = pd.Series({'Algorithm Used': 'Epsilon Greedy',
                                      'Bandit Return': bandit,
                                      'Reward': np.sum(algorithm['rewards'][:, bandit_index])})

                bandit_data = bandit_data.append(data, ignore_index=True)

        print(bandit_data)
        bandit_data.to_csv('reports.csv', index=False)
        print("Results saved in bandit_rewards.csv!")


class EpsilonGreedy(Bandit):
    """This is an epsilon greedy class"""

    def __init__(self, mean, epsilon=EPS, tau=TAU):
        self.mean = mean
        self.m = 0
        self.m_estimate = 0
        self.tau = tau
        self.N = 0
        self.eps = epsilon

    def __repr__(self):
        return f'Return {self.mean}'

    def pull(self):
        return (np.random.randn() / np.sqrt(self.tau)) + self.mean

    def update(self, x) -> None:
        self.N += 1
        self.m_estimate = ((self.N - 1) * self.m_estimate + x) / self.N

    def experiment(self, trials=NUM_TRIALS, bandit_rewards=Bandit_Reward):
        bandits = [EpsilonGreedy(m) for m in bandit_rewards]
        rewards = np.zeros((trials, len(bandits)))
        regrets = np.zeros((trials, len(bandits)))

        for i in range(0, trials):
            if np.random.random() < EPS / (i + 1):
                j = np.random.randint(len(bandits))
            j = np.argmax([b.m_estimate for b in bandits])

            x = bandits[j].pull()
            bandits[j].update(x)
            rewards[i, j] = x
            regrets[i, j] = np.max([bandit.m for bandit in bandits]) - x

        result = {'rewards': rewards, 'regrets': regrets, 'bandits': bandits}

        return result

    def plot1(self, result, reward=True) -> None:
        fig, ax = plt.subplots()

        if reward is True:
            cumulative_sum = np.cumsum(result['rewards'], axis=0)
            print("Cumulative sum for rewards: ", cumulative_sum)
        else:
            cumulative_sum = np.cumsum(result['regrets'], axis=0)
            print("Cumulative sum for regrets: ", cumulative_sum)

        for bandit in range(cumulative_sum.shape[1]):
            if reward is True:
                cum_sum = np.log(cumulative_sum)[:, bandit]
            else:
                cum_sum = cumulative_sum[:, bandit]
            ax.plot(np.arange(cumulative_sum.shape[0]), cum_sum, label=result['bandits'][bandit])

        if reward is True:
            ax.set_title("Comparison of bandit cumulative rewards")
        else:
            ax.set_title("Comparison of bandit cumulative regrets")

        plt.legend()
        plt.show()


class ThompsonSampling(Bandit):
    """This is a thompson sampling class"""

    def __init__(self, mean, tau=TAU):

        self.mean = mean
        self.m = 0
        self.t_lambda = 1
        self.tau = tau
        self.N = 0

    def __repr__(self):
        return f'Return {self.mean}'

    def pull(self):
        return (np.random.randn() / np.sqrt(self.tau)) + self.mean

    def update(self, x) -> None:
        self.m = (self.tau * x + self.t_lambda * self.m) / \
                 (self.tau + self.t_lambda)
        self.t_lambda = self.t_lambda + self.tau
        self.N += 1

    def sample(self):
        return np.random.randn() / np.sqrt(self.t_lambda) + self.m

    def experiment(self, trials=NUM_TRIALS, bandit_rewards=Bandit_Reward):

        bandits = [ThompsonSampling(m) for m in bandit_rewards]
        rewards = np.zeros((trials, len(bandits)))
        regrets = np.zeros((trials, len(bandits)))

        for i in range(trials):
            j = np.argmax([b.sample() for b in bandits])
            x = bandits[j].pull()
            bandits[j].update(x)
            rewards[i, j] = x
            regrets[i, j] = np.max([bandit.m for bandit in bandits]) - x

        total_result = {'rewards': rewards, 'regrets': regrets, 'bandits': bandits}
        return total_result

    def plot1(self, result, reward=True):
        fig, ax = plt.subplots()

        if reward:
            cumulative_sum = np.cumsum(result['rewards'], axis=0)
        else:
            cumulative_sum = np.cumsum(result['regrets'], axis=0)

        for bandit in range(cumulative_sum.shape[1]):
            if reward:
                cum_sum = np.log(cumulative_sum)[:, bandit]

            else:
                cum_sum = cumulative_sum[:, bandit]
            ax.plot(np.arange(cumulative_sum.shape[0]), cum_sum, label=result['bandits'][bandit])

        if reward is True:
            ax.set_title("Bandit cumulative reward comparison")
        else:
            ax.set_title("Bandit cumulative regret comparison")

        plt.legend()
        plt.show()

    def plot2(self, epsilon_greedy_results, thompson_results, trials=NUM_TRIALS) -> None:

        fig, ax = plt.subplots(nrows=2, figsize=(10, 10))
        trials = np.arange(0, trials)

        t_rewards = np.cumsum(thompson_results['rewards'])
        e_rewards = np.cumsum(epsilon_greedy_results['rewards'])
        t_regrets = np.cumsum(thompson_results['regrets'])
        e_regrets = np.cumsum(epsilon_greedy_results['regrets'])

        ax[0].plot(trials, t_rewards, label='Thompson Sampling')
        ax[0].plot(trials, e_rewards, label='Epsilon Greedy')
        ax[0].set_title('Total Rewards')

        ax[1].plot(trials, t_regrets, label='Thompson Sampling')
        ax[1].plot(trials, e_regrets, label='Epsilon Greedy')
        ax[1].set_title('Total Regrets')
        plt.show()
