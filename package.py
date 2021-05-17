"""
Created on Fri Apr  9 14:40:43 2021

@author: omkar
"""


import matplotlib.pyplot as plt
import torch
import numpy as np
from Preprocesssing.preprocessing import load_train_data, load_test_data
from Architecture.deep_q import run_deep_q_network_algorithm,Stock_Environment


def trade():
    data = load_train_data()
    data.index.rename("Date")
    print(data)
    returns,loss = run_deep_q_network_algorithm(data,"Deep_Q_Network")
    return returns


def trade_test():
    data = load_test_data()
    data.index.rename("Date")
    test_env = Stock_Environment(data)
    pobs = test_env.reset()
    test_acts = []
    test_rewards = []
    Q_test = torch.load("Models/dqn.pt")

    for _ in range(len(test_env.data) - 1):
        pact = Q_test(torch.from_numpy(np.array(pobs, dtype=np.float32).reshape(1, -1)))
        pact = np.argmax(pact.data)
        test_acts.append(pact.item())

        obs, reward, done = test_env.step(pact.numpy())
        test_rewards.append(reward)

        pobs = obs

    print(test_env.profits)

returns = trade()
trade_test()
plt.plot(range(100), returns, color='g', label='Return')
# plt.plot(range(50), loss, color='r', label='Return')
plt.xlabel('Training epochs')
plt.ylabel('Total Profit in $ per Epoch')
plt.title('Profit')
plt.show()
trade_test()
