import matplotlib.pyplot as plt
import torch
import numpy as np
from Preprocesssing.preprocessing import load_stock_data, test_stock_data
from Architecture.deep_q import run_dqn_algorithm,Stock_environment


def trade():
    data = load_stock_data()
    data.index.rename("Date")
    print(data)
    returns,loss = run_dqn_algorithm(data,"DQN")
    return returns


def trade_test():
    data = test_stock_data()
    data.index.rename("Date")
    test_env = Stock_environment(data)
    pobs = test_env.reset()
    test_acts = []
    test_rewards = []
    Q_test = torch.load("Models/dqn.pt")
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Q_test.to(device)

    for _ in range(len(test_env.data) - 1):
        #pact = Q_test(torch.from_numpy(np.array(pobs, dtype=np.float32).reshape(1, -1)).to(device))
        pact = Q_test(torch.from_numpy(np.array(pobs, dtype=np.float32).reshape(1, -1)))
        #pact = pact.cpu().data.numpy()
        #cpu = torch.device("cpu")
        #pact.to(cpu)
        pact = np.argmax(pact.data)
        test_acts.append(pact.item())

        obs, reward, done = test_env.step(pact.numpy())
        test_rewards.append(reward)

        pobs = obs

    print(test_env.profits)

returns=trade()
plt.plot(range(5), returns, color='g', label='Return')
# plt.plot(range(50), loss, color='r', label='Return')
plt.xlabel('Training epochs')
plt.ylabel('Total Profit in $ per Epoch')
plt.title('Profit')
plt.show()
trade_test()
#print(Q)
#trade_test("AAPl",Q)