import joblib
from matplotlib import pyplot as plt
from collections import Counter


def graph_loss(filename):
    losses, log = joblib.load(filename)
    train_loss = [x['train_loss'] for x in log]
    plt.plot(range(1, len(train_loss) + 1), train_loss)

    graph_title = filename.split('/')[-1]
    plt.title(graph_title)
    plt.show()


def plot_re(filename):
    loss, di = joblib.load(filename)
    label = di['total_label']
    re = di['reconstuction_error']

    divide_idx = label.index(0)
    normal = [round(x, 1) for x in re[:divide_idx]]
    anomaly = [round(x, 1) for x in re[divide_idx:]]

    c_normal = Counter(normal)
    c_anomaly = Counter(anomaly)

    plt.bar(c_normal.keys(), c_normal.values(), color='blue', alpha=0.5, width=0.1)
    plt.bar(c_anomaly.keys(), c_anomaly.values(), color='red', alpha=0.5, width=0.1)
    plt.show()


# plot_re('/home/jieun/Documents/machine_vision/log/191016_01.log')
plot_re('/home/jieun/Documents/machine_vision/log/191016_02.log')