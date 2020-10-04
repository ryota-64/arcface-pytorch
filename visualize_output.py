import numpy as np
import matplotlib.pyplot as plt


# visualize mean of th softmax predict data each known and unkown
def create_mean_graph(data_list):
    sorted_data = [np.sort(data)[::-1] for data in data_list]
    sorted_data = np.array(sorted_data)
    mean_data = np.mean(sorted_data, axis=0)
    print(len(data_list))
    print(mean_data)
    plt.bar(range(len(mean_data)), mean_data)
    plt.ylim(ymax=300, ymin=-300)
    plt.savefig('estimate_visualize/normalize{}.jpg'.format(len(data_list)))
    plt.show()
    for a in sorted_data:
        print(a[0])
    return mean_data


def show_histgram(data_list):
    plt.hist(data_list)
    plt.savefig('estimate_visualize/hist_pos_only{}.jpg'.format(len(data_list)))
    plt.show()

