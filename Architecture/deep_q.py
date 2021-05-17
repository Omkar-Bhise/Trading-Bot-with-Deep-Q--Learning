import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy

# Intializing Neural Network with ReLU as activation function with follwing Architecture
# ReLU(input_size*128)->ReLU(128*256)->ReLU(256*256)->ReLU(256*128)->ReLU(128*3)
# Where 3 indicated action space of Agent

class NN_Q_approximator(nn.Module):

    def __init__(self, input_size, output_size):
        super(NN_Q_approximator, self).__init__()

        self.fc_val = nn.Sequential(
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
        h = self.fc_val(x)
        return (h)


class Stock_environment:

    # Initialising data for stock enviroment class, time horizon and reset fucntion to clean it for next epoch
    def __init__(self, data, history_days=90):
        self.data = data
        self.history_days = history_days
        self.reset()

    # Reset function which will reset all Stock Enviroment parameter before each epoch
    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.history = [0 for _ in range(self.history_days)]
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




def dqn_training(stock_environment):
    Q_function = NN_Q_approximator(input_size=stock_environment.history_days + 1, output_size=3)
    #device = torch.device("cpu")
    #print(device)
    #Q_function.to(device)

    # Q_max will be the neural network that approximates the Q function FOR the update part i.e. when we use the max(x,.)
    # part of the equation. Do note that here is where these different algorithms (DQN,DDQN) differ.

    Q_max = copy.deepcopy(Q_function)

    # optimizer
    LR = 0.001
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(list(Q_function.parameters()), lr=LR)

    # how many environments are we going to keep in memory, but not only that but this is influence when do we start training (when we filled the whole memory)

    environment_memory_size = 200
    # how many elements saved in the environment, will be used in training (size of the subset)

    environment_subset_amount = 30
    show_log_freq = 5
    # We know that we will use the epsilon greedy policy. Having a fix policy is not optimal in every stage. Hence
    # epsilon of 1 in the beggining means more exploration and later on we do more explotation (when we decrease the values)

    epsilon = 1.0
    # how much to decrease epsilon

    epsilon_decrease = 1e-3

    # dont reduce epsilon smaller than smallest_possible_eps (in other words not that much exploitation)

    smallest_possible_eps = 0.1

    # when to start reducing epsilon, after how many actions(steps) we took

    start_reduce_epsilon_after = 150

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

    total_steps = 0

    # variable to hold all the rewards

    total_rewards = []

    # variable to hold all the losses

    total_losses = []

    start = time.time()
    # how many epochs (periods of training) do we execute. 100
    for epoch in range(5):

        # start with a fresh environment every epoch
        history_pos_value_r = stock_environment.reset()
        step = 0
        done = False
        total_reward = 0
        total_loss = 0
        # print(stock_environment.data)

        while not done and step < (len(stock_environment.data) - 1):

            # choose randomly one of our 3 possible actions, buy hold and sell. Following three lines of code will perform the
            # epsilon greedy policy, i.e. exploration and exploitation and if we are exploiting than we will take the
            # policy out of these 3 that have the maximal Q-value
            max_step = np.random.randint(2)
            if np.random.rand() > epsilon:
                # print(history_pos_value_r)
                max_step = Q_function(torch.from_numpy(np.array(history_pos_value_r, dtype=np.float32).reshape(1, -1)))
                max_step = np.argmax(max_step.data.cpu())
                max_step = max_step.numpy()

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

                        Q = Q_function(torch.from_numpy(environment_subset_history_pos_value))
                        Q_ = Q_max(torch.from_numpy(environment_subset_history_pos_value))
                        maxQ = np.max(Q_.data.cpu().numpy(), axis=1)
                        target = copy.deepcopy(Q.data)
                        # update the Q
                        for i in range(environment_subset_amount):
                            target[i, environment_subset_max_step[i]] = environment_subset_reward[i] + gamma * maxQ[
                                i] * (not environment_subset_done[i])

                        Q_function.zero_grad()
                        # loss function and backpropagation for neural network to learn
                        loss = loss_function(Q, target)
                        total_loss += loss.data.item()
                        loss.backward()
                        optimizer.step()

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
        print("Epoch : ", epoch)
        print("Total Loss : ", total_loss)
        print("Total profit", stock_environment.profits)
        total_rewards.append(stock_environment.profits)
        total_losses.append(total_loss)
    torch.save(Q_function, "Models/dqn.pt")
    return total_rewards, total_losses


def run_dqn_algorithm(stock,algorithm):
    if algorithm == "DQN":
        return (dqn_training(Stock_environment(stock)))
    else:
        return ("RL algorithm not available")