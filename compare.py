import numpy as np
import matplotlib.pyplot as plt
from config import Config
from path import make_data, make_data_1, get_target_sequential_states
from utils import herd_greedy_search, round_robin, cluster_greedy, permutation
from hungarian import hungarians
import time

# initialize camera configurations
Cfg = Config()
dist = Cfg.sample_time
cam_focal = Cfg.cam_focal
cam_location = Cfg.cam_location

pre_cam_focal_ms = cam_focal.copy()
pre_cam_location_ms = cam_location.copy()

pre_cam_focal_ip = cam_focal.copy()
pre_cam_location_ip = cam_location.copy()

pre_cam_focal_gs = cam_focal.copy()
pre_cam_location_gs = cam_location.copy()

pre_cam_focal_hun = cam_focal.copy()
pre_cam_location_hun = cam_location.copy()

n_ms = []
n_rr = []
n_ip = []
n_gs = []
n_hun = []

count = 0
n_data = []

score_rr_sum = 0
score_ms_sum = 0
score_ip_sum = 0
score_gs_sum = 0
score_hun_sum = 0

"""
assign task here
"""
task_max_number = 0
task_max_score = 1
current_task = task_max_score

if current_task == task_max_number:
    setting_number = make_data_1(dist)
    tss = get_target_sequential_states(setting_number)
    s_count_num = len(tss)
    print(s_count_num)   # 125
    x_tick = np.arange(0, 125, 10)
    y_tick = np.arange(0, 30, 3)
    figure = plt.figure()
    axe = figure.add_subplot(111)
    axe.set_xticks(x_tick)
    axe.set_yticks(y_tick)
    axe.set_xlim(0, 125)
    axe.set_ylim(0, 30)
    n_ms.append(1)
    n_rr.append(1)
    n_ip.append(1)
    n_hun.append(1)

if current_task == task_max_score:
    setting_score = make_data(dist)
    tss = get_target_sequential_states(setting_score)
    s_count_score = len(tss)
    print(s_count_score)  # 99
    x_tick = np.arange(0, 100, 10)
    y_tick = np.arange(0, 14, 2)
    figure = plt.figure()

    axe = figure.add_subplot(111)
    axe.set_xticks(x_tick)
    axe.set_yticks(y_tick)
    axe.set_xlim(0, 100)
    axe.set_ylim(0, 14)
    n_gs.append(0)
    n_rr.append(0)
    n_ip.append(0)
    n_hun.append(0)

    score_rr = 0
    score_ms = 0
    score_ip = 0
    score_gs = 0
    score_hun = 0


if __name__ == "__main__":
    use_baseline = True
    use_round_robin = True
    use_hungarian = True
    use_cluster_greedy = True
    use_herd_greedy = True

    for targets_states in tss:
        Nt = len(targets_states)
        n_data.append(Nt)
        if count:
            # baseline method
            if use_baseline:
                cam_location_ip, cam_focal_ip, score_ip = permutation(targets_states, Cfg, pre_cam_location_ip,
                                                                      pre_cam_focal_ip, current_task)
                score_ip_sum += score_ip
                n_ip.append(score_ip)
                pre_cam_focal_ip = cam_focal_ip
                pre_cam_location_ip = cam_location_ip

            # periodic move
            if use_round_robin:
                cam_location_rr, cam_focal_rr, score_rr = round_robin(Cfg.cam_location, Cfg.cam_focal, Cfg, count,
                                                                      targets_states, current_task)
                score_rr_sum += score_rr
                n_rr.append(score_rr)

            # k_means + hungarian for maximize number
            if use_hungarian:
                cam_location_hun, cam_focal_hun, score_hun = hungarians(targets_states, Cfg, pre_cam_location_hun,
                                                                        pre_cam_focal_hun, current_task)
                score_hun_sum += score_hun
                n_hun.append(score_hun)
                pre_cam_focal_hun = cam_focal_hun
                pre_cam_location_hun = cam_location_hun

        count += 1

    count = 0
    for targets_states in tss:
        if count:
            if use_cluster_greedy:
                cam_location_ms, cam_focal_ms, score_ms = cluster_greedy(targets_states, Cfg, pre_cam_location_ms,
                                                                         pre_cam_focal_ms, current_task)
                score_ms_sum += score_ms
                n_ms.append(score_ms)
                pre_cam_focal_ms = cam_focal_ms
                pre_cam_location_ms = cam_location_ms

            if use_herd_greedy:
                cam_location_gs, cam_focal_gs, score_gs = herd_greedy_search(targets_states, Cfg, pre_cam_location_gs,
                                                                        pre_cam_focal_gs, current_task)
                score_gs_sum += score_gs
                n_gs.append(score_gs)
                pre_cam_focal_gs = cam_focal_gs
                pre_cam_location_gs = cam_location_gs

        count += 1

    print("Baseline:          results for task %d is %.2f" % (current_task, score_ip_sum/(count-1)))
    print("Round Robin:       results for task %d is %.2f" % (current_task, score_rr_sum/(count-1)))
    print("Hungarian greedy:  results for task %d is %.2f" % (current_task, score_hun_sum/(count-1)))
    print("meanShift greedy:  results for task %d is %.2f" % (current_task, score_ms_sum/(count-1)))
    print("herd greedy:       results for task %d is %.2f" % (current_task, score_gs_sum/(count-1)))

    plt.xlabel("state sequence")
    plt.ylabel('total detection number')

    plt.plot(n_data, '--', color='c', label="target number")
    if use_baseline:
        plt.plot(n_ip, '-', color='k', label="baseline")
    if use_cluster_greedy:
        plt.plot(n_ms, '--', color='r', label="meanShift greedy")
    if use_herd_greedy:
        plt.plot(n_gs, '*-', color='y', label="herd greedy")
    if use_round_robin:
        plt.plot(n_rr, ':', color='m', label="periodic move")
    if use_hungarian:
        plt.plot(n_hun, '-.', color='b', label="hungarian greedy")
    plt.tick_params(labelsize=10)

    plt.legend(loc=0)
    # plt.savefig("E:\\scn4\\fullvs1paper.pdf")
    plt.show()
