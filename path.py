import numpy as np
import matplotlib.pyplot as plt
import math
from config import Config

cfg = Config()


def make_data(dist):
    settings = []
    t1_states = []
    x = np.arange(0.0, 8.0, dist)
    y = x*0 + 3.2
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 1, 0]))

    x = np.arange(8.0, 0.0, -dist)
    y = x * 0 + 2.3
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 1, 0]))
    settings.append(t1_states)

    t2_states = []
    x = np.arange(8.0, 0.0, -dist*0.9)
    y = -0.6*x + 7.5
    step = len(x)
    for i in range(step):
        t2_states.append(np.array([x[i], y[i], 2, 0]))

    x = np.arange(0.0, 8.0, 1.0*dist)
    y = x * -0.75 + 7
    step = len(x)
    for i in range(step):
        t2_states.append(np.array([x[i], y[i], 2, 0]))

    settings.append(t2_states)

    t3_states = []
    x = np.arange(8.0, 0.0, -dist*1.2)
    y = x * 0 + 4
    step = len(x)
    for i in range(step):
        t3_states.append(np.array([x[i], y[i], 3, 0]))

    x = np.arange(0.0, 3.0, dist*1.2)
    y = x * -1.7 + 4.7
    step = len(x)
    for i in range(step):
        t3_states.append(np.array([x[i], y[i], 3, 0]))

    x = np.arange(3.0, 8.0, dist*1.2)
    y = 1.3*x - 3
    step = len(x)
    for i in range(step):
        t3_states.append(np.array([x[i], y[i], 3, 0]))
    settings.append(t3_states)

    t4_states = []
    x = np.arange(0.0, 8.0, dist*1.2)
    step = len(x)
    y = -0.75*x + 6
    for i in range(step):
        t4_states.append(np.array([x[i], y[i], 4, 0]))

    x = np.arange(8.0, 0.0, -dist*1.2)
    y = x * -0.75 + 6
    step = len(x)
    for i in range(step):
        t4_states.append(np.array([x[i], y[i], 4, 0]))
    settings.append(t4_states)

    t5_states = []
    t5_x, t5_y = np.array([0.0, 7.0])
    theta = np.arange(np.pi / 2, 0, -0.5 * dist / 5)
    x = 0 + t5_y * np.cos(theta)
    y = 0 + t5_y * np.sin(theta)
    step = len(x)
    for i in range(step):
        t5_states.append(np.array([x[i], y[i], 5, 0]))
    settings.append(t5_states)

    t6_states = []
    theta = np.arange(0, -np.pi / 2, -0.8 * dist / 5)
    x = 0 + 7.5 * np.cos(theta)
    y = 8 + 7.5 * np.sin(theta)
    step = len(x)
    for i in range(step):
        t6_states.append(np.array([x[i], y[i], 6, 0]))
    settings.append(t6_states)

    t7_states = []
    x = np.arange(11.0, -0.5, -dist * 0.85)
    y = x * 0.7 + 0.4
    step = len(x)
    for i in range(step):
        t7_states.append(np.array([x[i], y[i], 7, 0]))
    settings.append(t7_states)

    t8_states = []
    theta = np.arange(np.pi * 7 / 4, np.pi / 2, -0.7 * dist / 4)
    x = 8 + 7.5 * np.cos(theta)
    y = 8 + 7.5 * np.sin(theta)
    step = len(x)
    for i in range(step):
        t8_states.append(np.array([x[i], y[i], 8, 0]))
    settings.append(t8_states)

    t9_states = []
    theta = np.arange(np.pi * 5 / 4, np.pi / 2, -0.9 * dist / 6)
    x = 8 + 6.5 * np.cos(theta)
    y = 0 + 6.5 * np.sin(theta)
    step = len(x)
    for i in range(step):
        t9_states.append(np.array([x[i], y[i], 9, 0]))
    settings.append(t9_states)

    t10_states = []
    x = np.arange(15.0, -1.0, -dist * 1.1)
    y = x * 0.4 + 1.5
    step = len(x)
    for i in range(step):
        t10_states.append(np.array([x[i], y[i], 10, 0]))
    settings.append(t10_states)

    t11_states = []
    x = np.arange(14.0, -2.0, -0.95 * dist)
    y = x * 1.6 - 2
    step = len(x)
    for i in range(step):
        t11_states.append(np.array([x[i], y[i], 11, 0]))
    settings.append(t11_states)

    t12_states = []
    x = np.arange(20.0, -2.0, -dist * 1.2)
    y = x * -0.7 + 5.8
    step = len(x)
    for i in range(step):
        t12_states.append(np.array([x[i], y[i], 12, 0]))
    settings.append(t12_states)

    t13_states = []
    y = np.arange(19.0, 0.0, -dist)
    x = x * 0 + 6
    step = len(x)
    for i in range(step):
        t13_states.append(np.array([x[i], y[i], 13, 0]))
    settings.append(t13_states)

    t14_states = []
    y = np.arange(-12.0, 8.0, 1.08 * dist)
    x = y*0 + 3.5
    step = len(y)
    for i in range(step):
        t14_states.append(np.array([x[i], y[i], 14, 0]))
    settings.append(t14_states)

    t15_states = []
    x = np.arange(-13.0, 9.0, 1.05 * dist)
    y = x * 0.45 + 1

    step = len(x)
    for i in range(step):
        t15_states.append(np.array([x[i], y[i], 15, 0]))
    settings.append(t15_states)

    return settings


def make_data_1(dist):
    settings = []
    """
    11 people who walk horizontally, use t1 to represent
    """
    t1_states = []
    x = np.arange(-17.0, 8.0, dist)
    y = x * 0 + 5.4
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 1, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(-17.0, 8.0, dist*1.1)
    y = x * 0 + 3.7
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 2, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(-18.0, 7.0, dist)
    y = x * 0 + 1.9
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 3, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(0.0, 9.0, dist)
    y = x * 0 + 1.5
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 4, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(-7.0, 16.0, 1.5*dist)
    y = x * 0 + 1.1
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 5, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(-2.0, 9.0, dist*1.1)
    y = x * 0 + 0.4
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 6, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(-10.0, 11.0, 1.2*dist)
    y = x * 0 + 6.9
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 7, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(-11.0, 16.0, 1.3*dist)
    y = x * 0 + 4.1
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 8, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(-5.0, 9.0, dist*0.8)
    y = x * 0 + 3.4
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 9, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(-2.0, 11.0, dist * 0.7)
    y = x * 0 + 2.8
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 10, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(-10.0, 9.0, dist)
    y = x * 0 + 3.8
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 11, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(-11.0, 9.0, dist*1.1)
    y = x * 0 + 4.3
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 12, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(-10.0, 9.0, dist)
    y = x * 0 + 4.9
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 13, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(-12.0, 10.0, 1.1*dist)
    y = x * 0 + 5.3
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 14, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(-14.0, 12.0, dist*1.2)
    y = x * 0 + 7.2
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 15, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(10.0, -2.0, -dist*0.9)
    y = x * 0 + 7.1
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 16, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(9.0, -2.0, -dist)
    y = x * 0 + 6.6
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 17, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(13.0, -1.0, -dist * 0.9)
    y = x * 0 + 5.8
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 18, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(12.0, -2.0, -dist*1.1)
    y = x * 0 + 2.6
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 19, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(15.0, -3.0, -dist * 1.2)
    y = x * 0 + 1.5
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 20, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(13.0, -3.0, -dist * 0.8)
    y = x * 0 + 2.4
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 21, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(14.0, -2.0, -dist * 0.7)
    y = x * 0 + 1.3
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 22, 0]))
    settings.append(t1_states)

    t1_states = []
    x = np.arange(8.0, -5.0, -dist * 0.9)
    y = x * 0 + 3.9
    step = len(x)
    for i in range(step):
        t1_states.append(np.array([x[i], y[i], 23, 0]))
    settings.append(t1_states)

    """
    6 people who walk slope, use t0 to represent
    """
    t0_states = []
    x = np.arange(-1.0, 12.0, dist)
    y = x * 0.4 + 0.6
    step = len(x)
    for i in range(step):
        t0_states.append(np.array([x[i], y[i], 24, 0]))
    settings.append(t0_states)

    t0_states = []
    x = np.arange(-2.0, 10.0, dist*0.8)
    y = x * 1.4 - 1.6
    step = len(x)
    for i in range(step):
        t0_states.append(np.array([x[i], y[i], 25, 0]))
    settings.append(t0_states)

    t0_states = []
    x = np.arange(-2.0, 9.0, dist)
    y = x * 0.9 + 2.6
    step = len(x)
    for i in range(step):
        t0_states.append(np.array([x[i], y[i], 26, 0]))
    settings.append(t0_states)

    t0_states = []
    x = np.arange(-3.0, 9.0, dist*0.9)
    y = x * 0.6 + 1.3
    step = len(x)
    for i in range(step):
        t0_states.append(np.array([x[i], y[i], 27, 0]))
    settings.append(t0_states)

    t0_states = []
    x = np.arange(-4.0, 10.0, dist)
    y = x * 1.8 - 2.6
    step = len(x)
    for i in range(step):
        t0_states.append(np.array([x[i], y[i], 28, 0]))
    settings.append(t0_states)

    t0_states = []
    x = np.arange(-5.0, 9.0, dist)
    y = -x * 1.5 + 6.6
    step = len(x)
    for i in range(step):
        t0_states.append(np.array([x[i], y[i], 29, 0]))
    settings.append(t0_states)

    t0_states = []
    x = np.arange(-3.0, 9.0, dist*1.2)
    y = -x * 2.3 + 7.6
    step = len(x)
    for i in range(step):
        t0_states.append(np.array([x[i], y[i], 30, 0]))
    settings.append(t0_states)

    t0_states = []
    x = np.arange(10.0, -1.0, -dist*0.9)
    y = -x * 1.5 + 8
    step = len(x)
    for i in range(step):
        t0_states.append(np.array([x[i], y[i], 31, 0]))
    settings.append(t0_states)

    t0_states = []
    x = np.arange(10.0, -2.0, -dist * 0.8)
    y = -x * 1.7 + 9
    step = len(x)
    for i in range(step):
        t0_states.append(np.array([x[i], y[i], 32, 0]))
    settings.append(t0_states)

    t0_states = []
    x = np.arange(12.0, -2.0, -dist*1.1)
    y = x * 1.5 - 2
    step = len(x)
    for i in range(step):
        t0_states.append(np.array([x[i], y[i], 33, 0]))
    settings.append(t0_states)

    t0_states = []
    x = np.arange(-13.0, 9.0, dist)
    y = x * 0.5 + 1
    step = len(x)
    for i in range(step):
        t0_states.append(np.array([x[i], y[i], 34, 0]))
    settings.append(t0_states)

    t0_states = []
    x = np.arange(-12.0, 9.0, dist*0.9)
    y = x * -0.7 + 5
    step = len(x)
    for i in range(step):
        t0_states.append(np.array([x[i], y[i], 35, 0]))
    settings.append(t0_states)

    t0_states = []
    x = np.arange(15.0, -1.0, -dist*1.1)
    y = x * 0.4 + 1.5
    step = len(x)
    for i in range(step):
        t0_states.append(np.array([x[i], y[i], 36, 0]))
    settings.append(t0_states)

    t0_states = []
    x = np.arange(11.0, -0.5, -dist * 0.85)
    y = x * 0.7 + 0.4
    step = len(x)
    for i in range(step):
        t0_states.append(np.array([x[i], y[i], 37, 0]))
    settings.append(t0_states)

    t0_states = []
    x = np.arange(20.0, -2.0, -dist*1.2)
    y = x * -0.7 + 5.8
    step = len(x)
    for i in range(step):
        t0_states.append(np.array([x[i], y[i], 38, 0]))
    settings.append(t0_states)

    t0_states = []
    x = np.arange(17.0, -2.0, -dist * 1.4)
    y = x * -0.5 + 6.8
    step = len(x)
    for i in range(step):
        t0_states.append(np.array([x[i], y[i], 39, 0]))
    settings.append(t0_states)


    """
    4 people who walk along with a circular, use t2 to represent
    """
    t2_x, t2_y = np.array([0.0, 5.0])
    t2_states = []
    theta = np.arange(np.pi / 2, 0, -0.5*dist / t2_y)
    x = 0 + t2_y * np.cos(theta)
    y = 0 + t2_y * np.sin(theta)
    step = len(x)
    for i in range(step):
        t2_states.append(np.array([x[i], y[i], 40, 0]))
    settings.append(t2_states)

    # t2_x, t2_y = np.array([2.0, 0.0])
    t2_states = []
    theta = np.arange(np.pi*3/2, np.pi / 2, -0.9*dist / 6)
    x = 8 + 6 * np.cos(theta)
    y = 0 + 6 * np.sin(theta)
    step = len(x)
    for i in range(step):
        t2_states.append(np.array([x[i], y[i], 41, 0]))
    settings.append(t2_states)

    # t2_x, t2_y = np.array([8.0, 2.0])
    t2_states = []
    theta = np.arange(np.pi*3/2, np.pi / 2, -0.7*dist / 4)
    x = 8 + 4 * np.cos(theta)
    y = 6 + 4 * np.sin(theta)
    step = len(x)
    for i in range(step):
        t2_states.append(np.array([x[i], y[i], 42, 0]))
    settings.append(t2_states)

    t2_x, t2_y = np.array([6.0, 6.0])
    t2_states = []
    theta = np.arange(0, -np.pi / 2, -0.8*dist / 5)
    x = 0 + 5 * np.cos(theta)
    y = 6 + 5 * np.sin(theta)
    step = len(x)
    for i in range(step):
        t2_states.append(np.array([x[i], y[i], 43, 0]))
    settings.append(t2_states)

    return settings


def get_target_sequential_states(settings):
    """
    :param settings: list of list of ndarray [x, y, id, pose=0]
    :return: sequential target states
    """
    target_number = len(settings)
    target_state_lengths = []
    for i in range(target_number):
        target_state_lengths.append(len(settings[i]))
    max_state_length = max(target_state_lengths)
    # print(max_state_length)

    sequential_states = []
    for index in range(max_state_length):
        target_states = []
        for target in range(0, target_number):
            try:
                target_info = settings[target][index]
                if 0.1 <= settings[target][index][0] <= 7.9 and 0.1 <= settings[target][index][1] <= 7.9:
                    if index:
                        t_prev_state = settings[target][index - 1]
                        dx = target_info[0] - t_prev_state[0]
                        dy = target_info[1] - t_prev_state[1]
                        pose = math.atan2(dy, dx) * 57.3
                        target_info[3] = pose
                    target_states.append(target_info)
            except IndexError:
                continue
        if len(target_states):
            sequential_states.append(target_states)
    return sequential_states


def make_site_figure(x_ticks, y_ticks):
    scene_figure = plt.figure()
    scene_axe = scene_figure.add_subplot(111)
    scene_axe.set_xticks(x_ticks)
    scene_axe.set_yticks(y_ticks)
    scene_axe.set_xlim(0, 8)
    scene_axe.set_ylim(0, 8)
    scene_axe.set_aspect(1)
    return scene_figure, scene_axe


if __name__ == "__main__":
    count = 0
    dist = 0.2
    setting = make_data(dist)
    print(len(setting))
    targets_states_sequences = get_target_sequential_states(setting)
    states_count = len(targets_states_sequences)
    print(states_count)
    x_ticks = cfg.x_ticks
    y_ticks = cfg.y_ticks
    colors = cfg.colors
    markers = cfg.markers
    for targets_states in targets_states_sequences:
        print(len(targets_states))
        # figure, ax = make_site_figure(x_ticks, y_ticks)
        # for target_state in targets_states:
        #     ax.plot(target_state[0], target_state[1],
        #             color=colors[int(target_state[2]) % 7],
        #             # marker=markers[int(target_state[2]) % 7],
        #             marker=markers[0],
        #             markersize=14)
        #     plt.text(target_state[0]+0.17, target_state[1]+0.17, str(int(target_state[2])), fontsize=10)
        #
        #     """acquire previous state info to calculate orientation, store it in t[3]"""
        #     if count:
        #         target_prev_state = setting[int(target_state[2]) - 1][count - 1]
        #         delta_x = target_state[0] - target_prev_state[0]
        #         delta_y = target_state[1] - target_prev_state[1]
        #         target_pose = math.atan2(delta_y, delta_x)*57.3
        #         target_state[3] = target_pose
        #         plt.quiver(target_state[0], target_state[1], delta_x, delta_y,
        #                    # scale=0.3, scale_units='xy',
        #                    color=colors[int(target_state[2]) % 7], width=0.008)
        #         # print("pose %f  id %d"%(target_state[3], int(target_state[2])))
        # # plt.savefig("D:\\scn\\update\\" + str(count) + '.png')
        # # plt.close()
        # plt.show()
        # count += 1
