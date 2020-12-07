import numpy as np
from sklearn.cluster import MeanShift
from matplotlib import pyplot as plt
from path import make_data, make_data_1, get_target_sequential_states

setting = make_data(0.2)
targets_states_sequences = get_target_sequential_states(setting)
states_count = len(targets_states_sequences)
targets_states = []

colors = ['red', 'blue', 'black', 'green', 'cyan', 'magenta', 'yellow']
markers = ['o', '*', 'x', 'v', '^', 's', '+', '>', '<', 's', 'p']
cam_location = np.array([[4.0, 0.], [0.0, 4.], [4.0, 8.], [8.0, 4.]])


def make_figure():
    figure = plt.figure()
    axe = figure.add_subplot(111)
    axe.set_xticks(np.arange(0, 8, 1))
    axe.set_yticks(np.arange(0, 8, 1))
    axe.set_xlim(0, 8)
    axe.set_ylim(0, 8)
    return figure, axe


if __name__ == "__main__":
    # plt.rcParams['font.sans-serif'] = ['SimSun']  # SimHei
    # plt.rcParams['axes.unicode_minus'] = False
    count = 0

    for targets_states in targets_states_sequences:

        fig, ax = make_figure()
        ax.set_xticks(np.arange(0, 8, 1))
        ax.set_yticks(np.arange(0, 8, 1))
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_aspect(1)
        locations = np.asarray(targets_states)[:, 0:2]
        clf = MeanShift(bandwidth=1.5)
        clf.fit(locations)

        labels = clf.labels_
        center_points = clf.cluster_centers_
        print(center_points)

        num_center = len(center_points)

        distance_map = np.zeros((4, num_center), dtype=np.float32)
        for i in range(4):
            distance_map[i] = np.sqrt(np.sum(np.asarray(cam_location[i] - center_points) ** 2, axis=1))

        for target_state in targets_states:
            ax.plot(target_state[0], target_state[1], color=colors[labels[i]],
                    marker=markers[labels[i]], markersize=12)

        for t in range(len(targets_states)):
            ax.plot(targets_states[t][0], targets_states[t][1], color=colors[labels[t] % 7],
                    marker=markers[labels[t] % 11], markersize=12)

            # print(labels[t], center_points[labels[t]])

        plt.scatter(center_points[:, 0], center_points[:, 1], c='red', s=10)
        # # plt.xlabel('带宽h为0.5', fontsize=16)
        plt.tick_params(labelsize=12)
        # # plt.savefig("E:\\scn3\\ms\\" + str(count) + '.png')
        # plt.close()
        count += 1
        plt.show()
        print(num_center)

