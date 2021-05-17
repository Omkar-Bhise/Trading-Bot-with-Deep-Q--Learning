import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import copy
import chainer
import chainer.functions as funcs
import chainer.links as links
import matplotlib.pyplot as plt


class NN_Q_approximator(chainer.Chain):

    # initialise the network
    def __init__(self, input_size, output_size=3):
        super(NN_Q_approximator, self).__init__(
            fc1=links.Linear(input_size, 128),
            fc2=links.Linear(128, 256),
            fc3=links.Linear(256, 128),
            fc4=links.Linear(128, output_size))

    # add the layers
    def __call__(self, x):
        h = funcs.elu(self.fc1(x))
        h = funcs.elu(self.fc2(h))
        h = funcs.elu(self.fc3(h))
        y = self.fc4(h)
        return y

    # Clears all gradient arrays. This method should be called before the backward computation at every iteration of the optimization.
    def reset(self):
        self.cleargrads()


class Stock_environment:

    # Initialise data, time horizont=history and reset method to wipe it clean
    def __init__(self, data, history_days=90):
        self.data = data
        self.history_days = history_days
        self.reset()

    # Start from beggining, i.e. initialise the values again. This will happen pro training epoch
    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.history = [0 for _ in range(self.history_days)]
        return [self.position_value] + self.history  # history of position values

    # Define what happens when taking action, we know 3 possible actions are available: hold (0) buy (1) and sell (2)
    def step(self, act):
        reward = 0
        # if we choose to buy, than we will save the closing price (that we bought) that day
        # please note that the assumption is that when we buy, we buy "volume of one" for that day of purchasing
        if act == 1:
            self.positions.append(self.data.iloc[self.t, :]['Close'])
        # if we do nothing, nothing changes and there is no reward (hold=1)
        elif act == 0:
            reward = 0
        # and if we (sell=2) and there is nothing to sell we will punish it. Else we will just sum the profit from the previous positions
        elif act == 2:
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0
                # now for everything we bought, when we sell it, we will subtract it from every positions (bought stock)
                for p in self.positions:
                    profits += (self.data.iloc[self.t, :]['Close'] - p)

                reward += profits
                self.profits += profits
                # when we sell we are emptying entire stock that we had (positions is empty list)
                self.positions = []

        # next training day (trading day)
        self.t += 1
        self.position_value = 0
        # is observation of value our position takes for all of the position that we took before
        for p in self.positions:
            self.position_value += (self.data.iloc[self.t, :]['Close'] - p)
        # drop(pop) the first day in the 120 day history (in order to have only 120 days in the history)
        self.history.pop(0)

        self.history.append(self.data.iloc[self.t, :]['Close'] - self.data.iloc[(self.t - 1), :]['Close'])

        return [self.position_value] + self.history, reward, self.done  # history of position values, reward(profits), finished with training (Boolian)


def dqn_training(stock_environment):
    Q_function = NN_Q_approximator(input_size=stock_environment.history_days + 1, output_size=3)
    # Q_max will be the neural network that approximates the Q function FOR the update part i.e. when we use the max(x,.)
    # part of the equation. Do note that here is where these different algorithms (DQN,DDQN) differ.
    Q_max = copy.deepcopy(Q_function)
    # optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(Q_function)

    # how many environments are we going to keep in memory, but not only that but this is influence when do we start training (when we filled the whole memory)
    environment_memory_size = 200
    # how many elements saved in the environment, will be used in training (size of the subset)
    environment_subset_amount = 30

    # We know that we will use the epsilon greedy policy. Having a fix policy is not optimal in every stage. Hence
    # epsilon of 1 in the beggining means more exploration and later on we do more explotation (when we decrease the values)
    epsilon = 1.0
    # how much to decrease epsilon
    epsilon_decrease = 1e-3
    # dont reduce epsilon smaller than smallest_possible_eps (in other words not that much exploitation)
    smallest_possible_eps = 0.001
    # when to start reducing epsilon, after how many actions(steps) we took
    start_reduce_epsilon_after = 150
    show_log_freq = 5
    # how ofter to update the Q-function values
    update_frequency = 6
    # how often to update the Q_max above
    update_Q_max_frequency = 9
    # gamma is parameter is the discount factor, how important past
    gamma = 0.90
    # variable to hold the memory
    memory = []
    # variable to hold all the steps
    total_steps = 0
    # variable to hold all the rewards
    total_rewards = []
    # variable to hold all the losses
    total_losses = []

    start = time.time()
    # how many epochs (periods of training) do we execute. 100
    for epoch in range(150):

        # start with a fresh environment every epoch
        history_pos_value_r = stock_environment.reset()
        step = 0
        done = False
        total_reward = 0
        total_loss = 0
        #print(stock_environment.data)

        while not done and step < (len(stock_environment.data) - 1):

            # choose randomly one of our 3 possible actions, buy hold and sell. Following three lines of code will perform the
            # epsilon greedy policy, i.e. exploration and exploitation and if we are exploiting than we will take the
            # policy out of these 3 that have the maximal Q-value
            max_step = np.random.randint(2)
            if np.random.rand() > epsilon:
                #print(history_pos_value_r)
                max_step = Q_function(np.array(history_pos_value_r, dtype=np.float32).reshape(1, -1))
                max_step = np.argmax(max_step.data)

            # select an action in the environment (the one with maximal Q-value)
            history_pos_value, reward, done = stock_environment.step(max_step)

            # add enironment variables to the memory
            memory.append((history_pos_value_r, max_step, reward, history_pos_value, done))
            # drop the first one we added (last in first out principle)
            if len(memory) > environment_memory_size:
                memory.pop(0)

            # we only start training when the memory is full (so we can sample completely)
            if len(memory) == environment_memory_size:
                # update Q value every 15 steps (i.e. update_frequency)
                if total_steps % update_frequency == 0:
                    # shuffle some random environment values from memory
                    shuffled_memory = np.random.permutation(memory)
                    for j in range(len(shuffled_memory[::environment_subset_amount])):
                        environment_subset = np.array(shuffled_memory[j:j + environment_subset_amount])
                        environment_subset_history_pos_value_r = np.array(environment_subset[:, 0].tolist(),
                                                                          dtype=np.float32).reshape(
                            environment_subset_amount, -1)
                        environment_subset_max_step = np.array(environment_subset[:, 1].tolist(), dtype=np.int32)
                        environment_subset_reward = np.array(environment_subset[:, 2].tolist(), dtype=np.int32)
                        environment_subset_history_pos_value = np.array(environment_subset[:, 3].tolist(),
                                                                        dtype=np.float32).reshape(
                            environment_subset_amount, -1)
                        environment_subset_done = np.array(environment_subset[:, 4].tolist(), dtype=np.bool)

                        Q = Q_function(environment_subset_history_pos_value)
                        maxQ = np.max(Q_max(environment_subset_history_pos_value).data, axis=1)
                        ####################check the Q value from orignal code#####################
                        #print(Q)
                        target = copy.deepcopy(Q.data)
                        # update the Q
                        for i in range(environment_subset_amount):
                            target[i, environment_subset_max_step[i]] = environment_subset_reward[i] + gamma * maxQ[
                                i] * (not environment_subset_done[i])

                        Q_function.reset()
                        # loss function and backpropagation for neural network to learn
                        loss = funcs.mean_squared_error(Q, target)
                        total_loss += loss.data
                        loss.backward()
                        optimizer.update()

                if total_steps % update_Q_max_frequency == 0:
                    Q_max = copy.deepcopy(Q_function)

            # start decreasing epsilon this is where we opt for more exploitation instead of exploration
            if epsilon > smallest_possible_eps and total_steps > start_reduce_epsilon_after:
                epsilon -= epsilon_decrease

            # next step, increase the rewards and steps, history of positional value
            total_reward += reward
            history_pos_value_r = history_pos_value
            step += 1
            total_steps += 1
        print("Epoch : ",epoch)
        print("Total Loss : ",total_loss)
        print("Total profit",stock_environment.profits)
        total_rewards.append(stock_environment.profits)
        total_losses.append(total_loss)

    return total_rewards,total_losses,Q_function


def run_dqn_algorithm(stock, algorithm):
    if algorithm == "DQN":
        return (dqn_training(Stock_environment(stock)))
    else:
        return ("RL algorithm not available")

def test(stock,Q_function):
    stock_environment=Stock_environment(stock)
    #history_pos_value_r= stock_environment.reset()
    #pobs = test_env.reset()
    test_acts = []
    test_rewards = []
    history_pos_value_r = stock_environment.reset()
    step = 0
    done = False
    total_reward = 0
    total_loss = 0
    # print(stock_environment.data)

    while not done and step < (len(stock_environment.data) - 1):
        max_step = Q_function(np.array(history_pos_value_r, dtype=np.float32).reshape(1, -1))
        max_step = np.argmax(max_step.data)
        test_acts.append(max_step.item())

        obs, reward, done = stock_environment.step(max_step)
        test_rewards.append(reward)
        print(stock_environment.profits)
        history_pos_value_r = obs

    test_profits = stock_environment.profits

    print(test_profits)