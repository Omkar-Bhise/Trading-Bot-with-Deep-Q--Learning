"""
Created on Fri Apr  9 14:40:43 2021

@author: omkar
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy

# Intializing Neural Network with ReLU as activation function with following Architecture
# ReLU(input_size*128)->ReLU(128*256)->ReLU(256*256)->ReLU(256*128)->ReLU(128*3)
# Where 3 indicated action space of Agent

class Neural_Network_Q_Learning(nn.Module):

    def __init__(self, input_size, output_size):
        super(Neural_Network_Q_Learning, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        output = self.network(x)
        return (output)


class Stock_Environment:

    # Initialising data for stock enviroment class, time horizon and reset fucntion to clean it for next epoch
    def __init__(self, data, background_days=90):
        self.data = data
        self.background_days = background_days
        self.clear_data()

    # Reset function which will reset all Stock Enviroment parameter before each epoch
    def clear_data(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.history = [0 for _ in range(self.background_days)]
        return [self.position_value] + self.history  # history of position values

    # Step function define what will hapeen when agent take any of the possible three actions : hold(0) buy(1) and sell(2)
    def step(self, act):
        reward = 0
        # We will save closing prise of that day if agent decides to buy
        # Assumption is when we try to buy something , we buy "one volume of stock" for that day
        if act == 1:
            self.positions.append(self.data.iloc[self.t, :]['close'])
        # for hold postion no changes happen
        elif act == 0:
            reward = -10
        # For sell,we will just sum the profit from the previous positions and
        # if agent doesn't have any postion to sell then we punish the agent
        elif act == 2:
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0
                for p in self.positions:
                    profits += (self.data.iloc[self.t, :]['close'] - p)

                reward += profits
                self.profits += profits
                self.positions = []

        # Next traing day
        self.t += 1
        self.position_value = 0
        for p in self.positions:
            self.position_value += (self.data.iloc[self.t, :]['close'] - p)
        # As one training day pass we drop first training day to add new training day
        self.history.pop(0)

        self.history.append(self.data.iloc[self.t, :]['close'] - self.data.iloc[(self.t - 1), :]['close'])
        if (self.t == len(self.data) - 1):
            self.done = True
        # clipping reward
        if reward > 0:
            reward = reward/2
        elif reward < 0:
            reward = -5
        return [self.position_value] + self.history, reward, self.done




def deep_q_network_training(environment):
    Q_approx = Neural_Network_Q_Learning(input_size=environment.background_days + 1, output_size=3)
    # Q_max will be the neural network that approximates the Q function FOR the update part i.e. when we use the max(x,.)
    # part of the equation. Do note that here is where these different algorithms (DQN,DDQN) differ.
    Q_max = copy.deepcopy(Q_approx)
    # optimizer
    Learning_Rate = 0.001
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(list(Q_approx.parameters()), lr=Learning_Rate)

    # how many environments are we going to keep in memory, but not only that but this is influence when do we start training (when we filled the whole memory)

    memory_size = 200

    # how many elements saved in the environment, will be used in training (size of the subset)

    environment_sub_amount = 30
    show_log_freq = 5

    # We know that we will use the epsilon greedy policy. Having a fix policy is not optimal in every stage. Hence
    # epsilon of 1 in the beggining means more exploration and later on we do more explotation (when we decrease the values)

    epsilon = 1.0

    # how much to decrease epsilon

    epsilon_decrease = 1e-3

    # don't reduce epsilon smaller than smallest_possible_eps (in other words not that much exploitation)

    smallest_possible_eps = 0.1

    # when to start reducing epsilon, after how many actions(steps) we took

    reduce_epsilon_after = 150

    show_log_freq = 5

    # how ofter to update the Q-function values

    update_frequency = 6

    # how often to update the Q_max above

    update_Q_max_frequency = 9

    # gamma is parameter is the discount factor, how important past

    gamma = 0.96

    # variable to hold the memory

    memory = []

    # variable to hold all the steps

    max_steps = 0

    # variable to hold all the rewards

    max_rewards = []

    # variable to hold all the losses

    total_losses = []

    start = time.time()
    # how many epochs (periods of training) do we execute. 100
    for epoch in range(100):

        # start with a fresh environment every epoch
        history_pos_value_r = environment.clear_data()
        step = 0
        done = False
        max_reward = 0
        total_loss = 0
        # print(stock_environment.data)

        while not done and step < (len(environment.data) - 1):

            # choose randomly one of our 3 possible actions, buy hold and sell. Following three lines of code will perform the
            # epsilon greedy policy, i.e. exploration and exploitation and if we are exploiting than we will take the
            # policy out of these 3 that have the maximal Q-value
            max_q_step = np.random.randint(2)
            if np.random.rand() > epsilon:
                # print(history_pos_value_r)
                max_q_step = Q_approx(torch.from_numpy(np.array(history_pos_value_r, dtype=np.float32).reshape(1, -1)))
                max_q_step = np.argmax(max_q_step.data.cpu())
                max_q_step = max_q_step.numpy()

            # select an action in the environment (the one with maximal Q-value)
            history_pos_value, reward, done = environment.step(max_q_step)

            # add enironment variables to the memory
            memory.append((history_pos_value_r, max_q_step, reward, history_pos_value, done))
            # drop the first one we added (last in first out principle)
            if len(memory) > memory_size:
                memory.pop(0)

            # we only start training when the memory is full (so we can sample completely)
            if len(memory) == memory_size:
                # update Q value every 15 steps (i.e. update_frequency)
                if max_steps % update_frequency == 0:
                    # shuffle some random environment values from memory
                    shuffled = np.random.permutation(memory)
                    for j in range(len(shuffled[::environment_sub_amount])):
                        environment_subset = np.array(shuffled[j:j + environment_sub_amount])
                        environment_subset_history_pos_value_r = np.array(environment_subset[:, 0].tolist(),
                                                                          dtype=np.float32).reshape(
                            environment_sub_amount, -1)
                        environment_subset_max_step = np.array(environment_subset[:, 1].tolist(), dtype=np.int32)
                        environment_subset_reward = np.array(environment_subset[:, 2].tolist(), dtype=np.int32)
                        environment_subset_history_pos_value = np.array(environment_subset[:, 3].tolist(),
                                                                        dtype=np.float32).reshape(
                            environment_sub_amount, -1)
                        environment_subset_done = np.array(environment_subset[:, 4].tolist(), dtype=np.bool)

                        Q = Q_approx(torch.from_numpy(environment_subset_history_pos_value))
                        Q_ = Q_max(torch.from_numpy(environment_subset_history_pos_value))
                        maxQ = np.max(Q_.data.cpu().numpy(), axis=1)
                        target = copy.deepcopy(Q.data)
                        # update the Q
                        for i in range(environment_sub_amount):
                            target[i, environment_subset_max_step[i]] = environment_subset_reward[i] + gamma * maxQ[
                                i] * (not environment_subset_done[i])

                        Q_approx.zero_grad()
                        # loss function and backpropagation for neural network to learn
                        loss = loss_function(Q, target)
                        total_loss += loss.data.item()
                        loss.backward()
                        optimizer.step()

                if max_steps % update_Q_max_frequency == 0:
                    Q_max = copy.deepcopy(Q_approx)

            # start decreasing epsilon this is where we opt for more exploitation instead of exploration
            if epsilon > smallest_possible_eps and max_steps > reduce_epsilon_after:
                epsilon -= epsilon_decrease

            # next step, increase the rewards and steps, history of positional value
            max_reward += reward
            history_pos_value_r = history_pos_value
            step += 1
            max_steps += 1
        print("Epoch No : ", epoch)
        print("Total Loss : ", total_loss)
        print("Total Profit : ", environment.profits)
        max_rewards.append(environment.profits)
        total_losses.append(total_loss)
    torch.save(Q_approx, "Models/dqn.pt")
    return max_rewards, total_losses


def run_deep_q_network_algorithm(stock,algorithm):
    if algorithm == "Deep_Q_Network":
        return (deep_q_network_training(Stock_Environment(stock)))
    else:
        return ("Given algorithm is not available currently")