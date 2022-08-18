
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(x,scores,epsilons,filename):
    fig,(ax1,ax2) = plt.subplots(1,2)

    ax1.plot(x,epsilons,color = "C0")
    ax1.set_title('Epsilon over time')
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Epsilon")
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0,t-100):(t+1)])
    ax2.set_title('Score over time')
    ax2.scatter(x,running_avg,color = "C1")
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel('Score')

    plt.savefig(filename)