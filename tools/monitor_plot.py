import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

'''
Utility Folder in order to visualize Rewards
Usage: place agent folder in logs/agent_xx
args: 
--agent_ids: agents to be plotted (default = None)
--window_size: window size of moving (average default = 100)
example usage:
python reward_plotter.py --agent_ids 26 27 28 --window_size=10
'''



def plot_rewards(args=None):

    data = []

    for i in args.agent_ids:
        data.append(pd.read_csv('logs/agent_{}/monitor.csv'.format(i), skiprows=1))

    average = []
    step_cum = []

    for x in range(len(data)):
        temp = []
        temp_step = []
        sum_ = 0
        
        for i in range(args.window_size-1):
            sum_ += data[x]['l'][i] 
        
        for i in range(args.window_size-1,data[x]['r'].shape[0]):
            temp.append(np.mean(data[x]['r'][i-args.window_size-1:i]))
            sum_ += data[x]['l'][i]
            temp_step.append(sum_)
        
        average.append(temp)
        step_cum.append(temp_step)

    plt.figure(figsize=(12,8))

    lr = args.agent_ids
    colors = [np.random.rand(3,) for x in args.agent_ids]
    for i in range(len(lr)):
        plt.plot(step_cum[i], average[i], '-', color=colors[i])

        plt.title('CARLA')
    plt.xlabel('TimeSteps')
    plt.ylabel('Mean_Reward-{}'.format(args.window_size))
    plt.legend(lr)
    plt.grid()
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_ids', nargs='+',type=int, default=None)
    parser.add_argument('--window_size', type=int, default=100)
    plot_rewards(args = parser.parse_args())


