import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from config import Config
import math
from path import make_data, make_data_1, get_target_sequential_states, make_site_figure
from utils import herd_greedy_search, round_robin, cluster_greedy, permutation
from hungarian import hungarians

Cfg = Config()
r = Cfg.radius
fov = Cfg.fov
creation = "new figure"  # "clear_redraw"
cam_id = Cfg.cam_id
cam_focal = Cfg.cam_focal
cam_colors = Cfg.cam_colors

dist = Cfg.sample_time
time_count = Cfg.time_count

x_ticks = Cfg.x_ticks
y_ticks = Cfg.y_ticks
half_fov = fov / 2
cam_location = Cfg.cam_location
cam_limits = np.array([cam_focal - half_fov, cam_focal + half_fov]).T

colors = Cfg.colors
markers = Cfg.markers

setting = make_data(dist)
targets_states_sequences = get_target_sequential_states(setting)
states_count = len(targets_states_sequences)


def draw_layout(cam_locations, cam_focals):
    """
     plot camera states
    """
    fig, axes = make_site_figure(x_ticks, y_ticks)

    limits = np.array([cam_focals - half_fov, cam_focals + half_fov]).T
    monitored_areas = []
    for cid in cam_id:
        wedge = mpatches.Wedge(cam_locations[cid], r, limits[cid][0], limits[cid][1])
        monitored_areas.append(wedge)
        axes.plot(cam_locations[cid][0], cam_locations[cid][1], color='black', marker='x')
    collection = PatchCollection(monitored_areas, cmap=plt.cm.hsv, alpha=0.3)
    collection.set_array(np.array(cam_colors))
    axes.add_collection(collection)

    plt.xlabel(' x-axis (m)')
    plt.ylabel(' y-axis (m)')
    plt.tick_params(labelsize=12)
    plt.text(1, 6, r'R = 4m')
    plt.text(1, 1, r'FOV = 60°')
    # plt.savefig("E:\\scn4\\pycam.pdf")
    plt.show()


def draw_targets_states():
    count = 0
    for targets_states in targets_states_sequences:
        figure, ax = make_site_figure(x_ticks, y_ticks)
        for target_state in targets_states:
            ax.plot(target_state[0], target_state[1],
                    color=colors[int(target_state[2]) % 7],
                    # marker=markers[int(target_state[2]) % 7],
                    marker=markers[0],
                    markersize=12)
            plt.text(target_state[0] + 0.17, target_state[1] + 0.17, str(int(target_state[2])), fontsize=9)

            """acquire previous state info to calculate orientation, store it in t[3]"""
            if count:
                target_prev_state = setting[int(target_state[2]) - 1][count - 1]
                delta_x = target_state[0] - target_prev_state[0]
                delta_y = target_state[1] - target_prev_state[1]
                target_pose = math.atan2(delta_y, delta_x) * 57.3
                target_state[3] = target_pose
                plt.quiver(target_state[0], target_state[1], delta_x, delta_y,
                           # scale=0.3, scale_units='xy',
                           color=colors[int(target_state[2]) % 7], width=0.008)
                # print("pose %f  id %d"%(target_state[3], int(target_state[2])))
        # plt.savefig("D:\\scn\\t15\\" + str(count) + '.png')
        # plt.close()
        # plt.show()
        count += 1


def draw_targets_states_1():
    count = 0
    for targets_states in targets_states_sequences:
        figure, ax = make_site_figure(x_ticks, y_ticks)
        for target_state in targets_states:
            ax.plot(target_state[0], target_state[1],
                    color=colors[int(target_state[2]) % 7],
                    # marker=markers[int(target_state[2]) % 7],
                    marker=markers[0],
                    markersize=4)
        # plt.savefig("D:\\scn\\data1\\" + str(count) + '.png')
        # plt.close()
        # plt.show()
        count += 1


def plot_target_numbers():
    setting_score = make_data(dist)
    tss_score = get_target_sequential_states(setting_score)
    s_count_score = len(tss_score)
    print(s_count_score) # 99

    setting_number = make_data_1(dist)
    tss_number = get_target_sequential_states(setting_number)
    s_count_num = len(tss_number)
    print(s_count_num)   # 125

    x_tick = np.arange(0, 125, 10)
    y_tick = np.arange(0, 30, 3)
    figure = plt.figure()

    axe = figure.add_subplot(111)
    axe.set_xticks(x_tick)
    axe.set_yticks(y_tick)
    axe.set_xlim(0, 125)
    axe.set_ylim(0, 30)

    nt_num = []
    sum_num = 0
    avg_num = 13.1
    n1 = []

    nt_score = []
    sum_score = 0
    avg_score = 7.6
    n2 = []

    for targets_states in tss_number:
        nt_num.append(len(targets_states))
        sum_num += len(targets_states)
        n1.append(avg_num)

    for targets_states in tss_score:
        nt_score.append(len(targets_states))
        sum_score += len(targets_states)
        n2.append(avg_score)

    plt.xlabel('status sequences')
    plt.ylabel('target number')
    plt.tick_params(labelsize=10)
    plt.plot(nt_num, '-', color='k', label="task 1")
    plt.plot(nt_score, '-', color='r', label="task 2")
    plt.plot(n1, '-.', color='c')
    plt.plot(n2, '-.', color='c')
    plt.text(80, 13.5, r'average 13.1')
    plt.text(57, 5.5, r'average 7.6',)

    plt.legend(loc=0, prop=font15)
    # plt.savefig("E:\\scn4\\statecounts.pdf")
    plt.show()


if __name__ == "__main__":
    pre_cam_focal = cam_focal
    pre_cam_location = cam_location

    pre_cam_focal_ms = cam_focal
    pre_cam_location_ms = cam_location

    pre_cam_focal_gs = cam_focal
    pre_cam_location_gs = cam_location

    pre_cam_focal_ip = cam_focal
    pre_cam_location_ip = cam_location

    pre_cam_focal_hun = cam_focal
    pre_cam_location_hun = cam_location

    count = 0
    score_rr = 0
    score_ms = 0
    score_ip = 0
    score_gs = 0
    score_hun = 0
    score_hun_sum = 0
    score_ms_sum = 0
    score_ip_sum = 0

    # assign task
    task_max_number = 0
    task_max_score = 1

    # choose one task each time
    current_task = task_max_number

    # choose one method each time
    draw_baseline = True
    draw_round_robin = False
    draw_hungarian = False
    draw_cluster_greedy = False
    draw_herd_greedy = False

    for targets_states in targets_states_sequences:
        fig, axes = make_site_figure(x_ticks, y_ticks)
        plt.tick_params(labelsize=12)

        Nt = len(targets_states)

        for target_state in targets_states:
            axes.plot(target_state[0], target_state[1],
                      color=colors[int(target_state[2]) % 7],
                      # marker=markers[int(target_state[2]) % 7],
                      marker=markers[0],
                      markersize=10)
            if current_task == task_max_score:
                plt.text(target_state[0] + 0.17, target_state[1] + 0.17, str(int(target_state[2])), fontsize=10)
                """acquire previous state info to calculate orientation, store it in t[3]"""
                if count:
                    target_prev_state = setting[int(target_state[2]) - 1][count - 1]
                    delta_x = target_state[0] - target_prev_state[0]
                    delta_y = target_state[1] - target_prev_state[1]
                    target_pose = math.atan2(delta_y, delta_x) * 57.3
                    target_state[3] = target_pose
                    plt.quiver(target_state[0], target_state[1], delta_x, delta_y,
                               # scale=0.3, scale_units='xy',
                               color=colors[int(target_state[2]) % 7], width=0.008)

        # for the first state, just display original settings
        if not count:
            limits = np.array([cam_focal - half_fov, cam_focal + half_fov]).T
            location2draw = cam_location

        if count:
            if draw_baseline:
                cam_location_ip, cam_focal_ip, score_ip = permutation(targets_states, Cfg,
                                                                      pre_cam_location_ip, pre_cam_focal_ip,
                                                                      num_or_score=current_task)
                score_ip_sum += score_ip
                pre_cam_focal_ip = cam_focal_ip
                pre_cam_location_ip = cam_location_ip
                limits = np.array([cam_focal_ip - half_fov, cam_focal_ip + half_fov]).T
                location2draw = cam_location_ip

            if draw_round_robin:
                cam_location_rr, cam_focal_rr, score_rr = round_robin(Cfg.cam_location, Cfg.cam_focal, Cfg, count,
                                                                      targets_states, num_or_score=current_task)
                limits = np.array([cam_focal_rr - half_fov, cam_focal_rr + half_fov]).T
                location2draw = cam_location_rr

            if draw_hungarian:
                cam_location_hun, cam_focal_hun, score_hun = hungarians(targets_states, Cfg, pre_cam_location_hun,
                                                                        pre_cam_focal_hun, current_task)
                pre_cam_focal_hun = cam_focal_hun
                pre_cam_location_hun = cam_location_hun
                limits = np.array([cam_focal_hun - half_fov, cam_focal_hun + half_fov]).T
                location2draw = cam_location_hun
                score_hun_sum += score_hun

            if draw_cluster_greedy:
                cam_location_ms, cam_focal_ms, score_ms = cluster_greedy(targets_states, Cfg, pre_cam_location_ms,
                                                                         pre_cam_focal_ms, current_task)
                score_ms_sum += score_ms
                pre_cam_focal_ms = cam_focal_ms
                pre_cam_location_ms = cam_location_ms
                limits = np.array([cam_focal_ms - half_fov, cam_focal_ms + half_fov]).T
                location2draw = cam_location_ms

            if draw_herd_greedy:
                cam_location_gs, cam_focal_gs, score_gs = herd_greedy_search(targets_states, Cfg, pre_cam_location_gs,
                                                                             pre_cam_focal_gs)
                pre_cam_focal_gs = cam_focal_gs
                pre_cam_location_gs = cam_location_gs
                limits = np.array([cam_focal_gs - half_fov, cam_focal_gs + half_fov]).T
                location2draw = cam_location_gs
                score_ms_sum += score_gs

        monitored_areas = []
        for cid in cam_id:
            wedge = mpatches.Wedge(location2draw[cid], r, limits[cid][0], limits[cid][1])
            monitored_areas.append(wedge)
            axes.plot(cam_location[cid][0], cam_location[cid][1], color='black', marker='x')
        collection = PatchCollection(monitored_areas, cmap=plt.cm.hsv, alpha=0.3)
        collection.set_array(np.array(cam_colors))
        axes.add_collection(collection)

        plt.xlabel('x-axis(m)')
        plt.ylabel('y-axis(m)')
        plt.tick_params(labelsize=12)

        plt.title("Nt: "+str(Nt)+"  seq: "+str(count) + "  detection: " + str(np.around(score_ip, decimals=1)))

        plt.savefig("E:\\scn4\\ip\\" + str(count) + '.png')
    #     plt.close()
    #     plt.show()
        count += 1

