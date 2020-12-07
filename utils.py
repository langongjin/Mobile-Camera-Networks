import numpy as np
import math
from sklearn.cluster import MeanShift

sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0


def get_location_change(step):
    circle = step % 80+1
    if 1 <= circle <= 20:
        movement = circle * 0.15
    elif 21 <= circle <= 60:
        movement = 6-0.15*circle
    elif 61 <= circle <= 80:
        movement = 0.15*circle - 12
    return movement


def get_focal_change(step):
    global new_angle
    circle = step % 80+1
    if 1 <= circle <= 20:
        new_angle = circle * 2
    elif 21 < circle <= 60:
        new_angle = 80 - 2 * circle
    elif 61 <= circle <= 80:
        new_angle = 2 * circle - 160
    return new_angle


def value_between(value, vmin, vmax):
    if value > vmax:
        value = vmax
    if value < vmin:
        value = vmin
    return value


def relative_pose(tx, ty, tyaw, cx, cy):
    theta_tc = math.atan2(ty - cy, tx - cx) * 57.3
    t2c_relative_pose = tyaw - theta_tc + sgn(theta_tc) * 180
    if t2c_relative_pose > 180:
        t2c_relative_pose -= 360
    if t2c_relative_pose < - 180:
        t2c_relative_pose += 360
    return t2c_relative_pose


def herd_greedy_search(current_targets_states, cfg, pre_cam_location, pre_cam_focal, num_or_score=1):
    r = cfg.radius
    fov = cfg.fov
    cam_location = pre_cam_location
    cam_focal = pre_cam_focal
    cam_number = cfg.cam_number
    kp = 0.12
    target_number = len(current_targets_states)

    # check if targets are within fov
    distance_map = np.zeros((cam_number, target_number), dtype=np.float32)
    for i in range(cam_number):
        distance_map[i] = np.sqrt(np.sum(np.asarray(cam_location[i] -
                                                    np.asarray(current_targets_states)[:, 0:2]) ** 2, axis=1))

    "targets within a camera's fov and relative pose within 90 are used to make decision"
    target_states_copy = current_targets_states.copy()
    targets_subsets_for_cam = []
    closest_target_sets = []
    for c in range(cam_number):
        targets_subsets = []
        for t in range(target_number):
            if 0.1 < distance_map[c][t] < r-0.1:
                poseTC = relative_pose(current_targets_states[t][0], current_targets_states[t][1],
                                       current_targets_states[t][3], cam_location[c][0], cam_location[c][1])
                if abs(poseTC) < 90:
                    target_states_copy[t][3] = poseTC
                    target_states_copy[t][2] = distance_map[c][t]
                    targets_subsets.append(target_states_copy[t])

        # use nearest t
        dist_vector = [i[2] for i in targets_subsets]
        if len(dist_vector):
            index_t = dist_vector.index(min(dist_vector))
            closest_target_sets.append([targets_subsets[index_t]])
        else:
            closest_target_sets.append(targets_subsets)

    targets_subsets_for_cam = closest_target_sets

    for cam in range(cam_number):
        Nct = len(targets_subsets_for_cam[cam])

        if Nct:
            delta_c = 0
            distance = np.zeros(Nct)
            for i in range(Nct):
                pose = targets_subsets_for_cam[cam][i][3]
                distance[i] = targets_subsets_for_cam[cam][i][2]
                if -30 < pose < 30:
                    delta_c += 0
                else:
                    delta_c += kp/60 * pose * np.sqrt(targets_subsets_for_cam[cam][i][2])
                    delta_c = value_between(delta_c, -0.2, 0.2)

            if cam == 0:
                pre_cam_location[cam][0] += delta_c
                pre_cam_location[cam][0] = value_between(pre_cam_location[cam][0], 1.2, 6.8)
                nearest = int(np.argmin(distance))
                theta_tc = math.atan2(targets_subsets_for_cam[cam][nearest][1] - pre_cam_location[cam][1],
                                      targets_subsets_for_cam[cam][nearest][0] - pre_cam_location[cam][0])*57.3
                cam_focal[cam] = theta_tc + (targets_subsets_for_cam[cam][nearest][0] * fov / 8 - 30)*0.8
            elif cam == 1:
                pre_cam_location[cam][1] -= delta_c
                pre_cam_location[cam][1] = value_between(pre_cam_location[cam][1], 1.2, 6.8)
                nearest = int(np.argmin(distance))
                theta_tc = math.atan2(targets_subsets_for_cam[cam][nearest][1] - pre_cam_location[cam][1],
                                      targets_subsets_for_cam[cam][nearest][0] - pre_cam_location[cam][0]) * 57.3
                cam_focal[cam] = theta_tc + ((8-targets_subsets_for_cam[cam][nearest][1]) * fov / 8 - 30)*0.8
            elif cam == 2:
                pre_cam_location[cam][0] -= delta_c
                pre_cam_location[cam][0] = value_between(pre_cam_location[cam][0], 1.2, 6.8)
                nearest = int(np.argmin(distance))
                theta_tc = math.atan2(targets_subsets_for_cam[cam][nearest][1] - pre_cam_location[cam][1],
                                      targets_subsets_for_cam[cam][nearest][0] - pre_cam_location[cam][0]) * 57.3
                cam_focal[cam] = theta_tc + ((8 - targets_subsets_for_cam[cam][nearest][0]) * fov / 8 - 30)*0.8
            elif cam == 3:
                pre_cam_location[cam][1] += delta_c
                pre_cam_location[cam][1] = value_between(pre_cam_location[cam][1], 1.2, 6.8)
                nearest = int(np.argmin(distance))
                theta_tc = math.atan2(targets_subsets_for_cam[cam][nearest][1] - pre_cam_location[cam][1],
                                      targets_subsets_for_cam[cam][nearest][0] - pre_cam_location[cam][0]) * 57.3
                cam_focal[cam] = theta_tc + (targets_subsets_for_cam[cam][nearest][1] * fov / 8 - 30)*0.8

    distance_map = np.zeros((cam_number, target_number), dtype=np.float32)
    for i in range(cam_number):
        distance_map[i] = np.sqrt(np.sum(np.asarray(pre_cam_location[i] -
                                                    np.asarray(current_targets_states)[:, 0:2]) ** 2, axis=1))

    "targets within a camera's fov are used"
    detection_score_map = distance_map.copy()
    detection_vector = np.zeros(target_number, dtype=int)
    for c in range(cam_number):
        focal = cam_focal[c]
        for t in range(target_number):
            if detection_score_map[c][t] >= r - 0.05 or detection_score_map[c][t] <= 0.1:
                detection_score_map[c][t] = 0
            else:
                aim_x = current_targets_states[t][0]
                aim_y = current_targets_states[t][1]
                offset_y = aim_y - pre_cam_location[c][1]
                offset_x = aim_x - pre_cam_location[c][0]
                slope = np.arctan2(offset_y, offset_x) * 180 / np.pi
                dist_focal = focal - slope
                if abs(dist_focal) > 180:
                    dist_focal = 360 - abs(dist_focal)
                if abs(dist_focal) >= fov / 2:
                    detection_score_map[c][t] = 0
                else:
                    poseTC = relative_pose(current_targets_states[t][0], current_targets_states[t][1],
                                           current_targets_states[t][3], pre_cam_location[c][0], pre_cam_location[c][1])
                    detection_score_map[c][t] = (np.cos(poseTC / 57.3) + 0.5) / np.sqrt(distance_map[c][t]) \
                        if abs(poseTC) < 90 else 0
                    detection_vector[t] = 1

    if num_or_score:
        total_score = np.sum(detection_score_map)
    if not num_or_score:
        total_score = np.sum(detection_vector)
    return pre_cam_location, cam_focal, total_score


def round_robin(location, cam_focal, cfg, time_count, current_targets_states, num_or_score):
    r = cfg.radius
    fov = cfg.fov
    cam_number = cfg.cam_number
    target_number = len(current_targets_states)

    # calculate each camera's location and focal at each time stamp
    focal_change = get_focal_change(time_count)
    focals = cam_focal - focal_change

    loc_change = get_location_change(time_count)
    cam_location = location.copy()
    cam_location[0][0] += loc_change
    cam_location[1][1] -= loc_change
    cam_location[2][0] -= loc_change
    cam_location[3][1] += loc_change

    # check if targets are within fov
    distance_map = np.zeros((cam_number, target_number), dtype=np.float32)
    for i in range(cam_number):
        distance_map[i] = np.sqrt(np.sum(np.asarray(cam_location[i] -
                                                    np.asarray(current_targets_states)[:, 0:2]) ** 2, axis=1))

    "targets within a camera's fov are used"
    detection_score_map = distance_map.copy()
    detection_vector = np.zeros(target_number, dtype=int)
    for c in range(cam_number):
        focal = focals[c]
        for t in range(target_number):
            if detection_score_map[c][t] >= r - 0.1 or detection_score_map[c][t] <= 0.1:
                detection_score_map[c][t] = 0
            else:
                aim_x = current_targets_states[t][0]
                aim_y = current_targets_states[t][1]
                offset_y = aim_y - cam_location[c][1]
                offset_x = aim_x - cam_location[c][0]
                slope = np.arctan2(offset_y, offset_x) * 180 / np.pi
                dist_focal = focal - slope
                if abs(dist_focal) > 180:
                    dist_focal = 360 - abs(dist_focal)
                if abs(dist_focal) >= fov/2:
                    detection_score_map[c][t] = 0
                else:
                    poseTC = relative_pose(current_targets_states[t][0], current_targets_states[t][1],
                                           current_targets_states[t][3], cam_location[c][0], cam_location[c][1])
                    detection_score_map[c][t] = (np.cos(poseTC / 57.3) + 0.5)/np.sqrt(distance_map[c][t])\
                        if abs(poseTC) < 90 else 0
                    detection_vector[t] = 1

    if num_or_score:
        total_score = np.sum(detection_score_map)
    if not num_or_score:
        total_score = np.sum(detection_vector)

    return cam_location, focals, total_score


def cluster_greedy(current_targets_states, cfg, pre_cam_location, pre_cam_focal, num_or_score):
    r = cfg.radius
    fov = cfg.fov
    cam_location = pre_cam_location
    cam_focal = pre_cam_focal
    cam_number = cfg.cam_number
    delta_k = cfg.delta_k
    top_k = math.ceil(len(current_targets_states) / 4)
    if top_k > 7:
        top_k = 7
    if top_k <= 5:
        if num_or_score == 0:
            bandwidth = 1.6
        else:
            bandwidth = 1.2
    if top_k > 5:
        bandwidth = 0.5

    select_dist = r - 0.1
    move_factor = 0.12

    # for drawing sector
    target_number = len(current_targets_states)
    cam_angle = np.array([0., 0., 0., 0.])

    locations = np.asarray(current_targets_states)[:, 0:2]
    clusters = MeanShift(bandwidth=bandwidth)
    clusters.fit(locations)

    labels = clusters.labels_
    center_points = clusters.cluster_centers_
    num_center = len(center_points)

    distance_map = np.zeros((cam_number, num_center), dtype=np.float32)
    for i in range(cam_number):
        distance_map[i] = np.sqrt(np.sum(np.asarray(cam_location[i] - center_points) ** 2, axis=1))

    center_points_weights = np.zeros(num_center, dtype=int)
    for label in labels:
        center_points_weights[label] += 1

    detection_focal_config = []
    detection_loc_config = []
    detection_vector = []
    detection_score_vector = []

    for c in range(cam_number):
        center_angle = np.zeros(num_center, dtype=float)
        center_score = np.zeros(num_center, dtype=float)
        center2newlocation = []

        detect_focal_config = []
        detect_loc_config = []

        for cent in range(num_center):
            cam_location_copy = cam_location.copy()
            if 0.1 < distance_map[c][cent] < r:
                # first get each cluster center's weight
                center_score[cent] = center_points_weights[cent] - 0.2 * distance_map[c][cent]

                # then get the center's resultant offset for camera c, get new location and angle
                aim_x = center_points[cent][0]
                aim_y = center_points[cent][1]
                if c == 0:
                    cam_location_copy[c][0] += move_factor * sgn(aim_x-cam_location[c][0])
                elif c == 1:
                    cam_location_copy[c][1] += move_factor * sgn(aim_y - cam_location[c][1])
                elif c == 2:
                    cam_location_copy[c][0] += move_factor * sgn(aim_x - cam_location[c][0])
                elif c == 3:
                    cam_location_copy[c][1] += move_factor * sgn(aim_y - cam_location[c][1])
                center2newlocation.append(cam_location_copy[c])
                slope = np.arctan2(aim_y-cam_location_copy[c][1], aim_x-cam_location_copy[c][0]) * 57.3
                center_angle[cent] = slope
            else:
                center_angle[cent] = cam_focal[c]
                center2newlocation.append(cam_location_copy[c])

        select_top_k = center_score.argsort()[::-1][0:top_k]
        h = len(select_top_k)

        if h == 0:
            detect_focal_config.append(pre_cam_focal[c])
            detect_loc_config.append(cam_location[c])
        else:
            for idx in select_top_k:
                detect_focal_config.append(int(center_angle[idx]))
                detect_loc_config.append(center2newlocation[idx])

        num_config = len(detect_focal_config)
        cam_vector = np.zeros((num_config, target_number), dtype=int)
        cam_score_vector = np.zeros((num_config, target_number), dtype=np.float32)

        for config_id in range(num_config):
            current_focal = detect_focal_config[config_id]
            current_loc = detect_loc_config[config_id]
            dist_ct = np.sqrt(np.sum(np.asarray(current_loc-np.asarray(current_targets_states)[:, 0:2]) ** 2, axis=1))

            for t in range(target_number):
                if dist_ct[t] >= select_dist or dist_ct[t] <= 0.1:
                    cam_vector[config_id][t] = 0
                    cam_score_vector[config_id][t] = 0
                else:
                    aim_x = current_targets_states[t][0]
                    aim_y = current_targets_states[t][1]
                    offset_y = aim_y - current_loc[1]
                    offset_x = aim_x - current_loc[0]
                    slope = np.arctan2(offset_y, offset_x) * 57.3
                    dist_focal = current_focal - slope
                    if -30 <= dist_focal < 30:
                        cam_vector[config_id][t] = 1
                        poseTC = relative_pose(aim_x, aim_y, current_targets_states[t][3],
                                               current_loc[0], current_loc[1])
                        if abs(poseTC) < 90:
                            cam_score_vector[config_id][t] = (np.cos(poseTC / 57.3) + 0.5) / np.sqrt(dist_ct[t])
                        else:
                            cam_score_vector[config_id][t] = 0
                    else:
                        cam_vector[config_id][t] = 0
                        cam_score_vector[config_id][t] = 0

        detection_vector.append(cam_vector)
        detection_score_vector.append(cam_score_vector)
        detection_focal_config.append(detect_focal_config)
        detection_loc_config.append(detect_loc_config)

    len_dc0 = len(detection_focal_config[0])
    len_dc1 = len(detection_focal_config[1])
    len_dc2 = len(detection_focal_config[2])
    len_dc3 = len(detection_focal_config[3])

    detection_vector_sum = []
    detection_config_focal_match = []
    detection_config_loc_match = []
    detection_scores = []
    changes = []
    for id0 in range(len_dc0):
        dc0 = detection_focal_config[0][id0]
        dl0 = detection_loc_config[0][id0]
        dv0 = detection_vector[0][id0]
        ds0 = dv0 * detection_score_vector[0]
        df0 = abs(pre_cam_focal[0] - dc0)/30 + abs(pre_cam_location[0][0] - dl0[0])
        for id1 in range(len_dc1):
            dc1 = detection_focal_config[1][id1]
            dl1 = detection_loc_config[1][id1]
            dv1 = detection_vector[1][id1]
            ds1 = dv1 * detection_score_vector[1]
            df1 = abs(pre_cam_focal[1] - dc1)/30 + abs(pre_cam_location[1][1] - dl1[1])
            for id2 in range(len_dc2):
                dc2 = detection_focal_config[2][id2]
                dl2 = detection_loc_config[2][id2]
                dv2 = detection_vector[2][id2]
                ds2 = dv2 * detection_score_vector[2]
                df2 = abs(pre_cam_focal[2] - dc2) / 30 + abs(pre_cam_location[2][0] - dl2[0])
                for id3 in range(len_dc3):
                    dc3 = detection_focal_config[3][id3]
                    dl3 = detection_loc_config[3][id3]
                    dv3 = detection_vector[3][id3]
                    ds3 = dv3 * detection_score_vector[3]
                    df3 = abs(pre_cam_focal[3] - dc3) / 30 + abs(pre_cam_location[3][1] - dl3[1])

                    dc = [dc0, dc1, dc2, dc3]  # detection config
                    dl = [dl0, dl1, dl2, dl3]
                    dv = dv0 | dv1 | dv2 | dv3  # detection vector
                    ds = ds0 + ds1 + ds2 + ds3
                    sum_dv = dv.sum()  # detection number
                    sum_ds = ds.sum()  # detection score
                    detection_scores.append(sum_ds)
                    detection_vector_sum.append(np.int(sum_dv))
                    detection_config_focal_match.append(dc)
                    detection_config_loc_match.append(dl)
                    changes.append(df0 + df1 + df2 + df3)
    if num_or_score == 0:
        scores = np.asarray(detection_vector_sum) - delta_k * np.asarray(changes)
    elif num_or_score == 1:
        scores = np.asarray(detection_scores) - delta_k * np.asarray(changes)

    scores = np.asarray(detection_vector_sum) - delta_k*np.asarray(changes)

    k_index = int(np.argmax(scores))
    new_focal = detection_config_focal_match[k_index]
    new_loc = detection_config_loc_match[k_index]

    if num_or_score == 1:
        detected_sum = detection_scores[k_index]
    if num_or_score == 0:
        detected_sum = detection_vector_sum[k_index]

    for i in range(cam_number):
        cam_angle[i] = new_focal[i]
        cam_location[i] = new_loc[i]

    return cam_location, cam_angle, detected_sum


def permutation(current_targets_states, cfg, pre_cam_location, pre_cam_focal, num_or_score):
    r = cfg.radius
    fov = cfg.fov
    cam_location = pre_cam_location
    cam_focal = pre_cam_focal
    cam_angle = np.array([0., 0., 0., 0.])
    cam_number = cfg.cam_number
    delta_k = cfg.delta_k
    KC = cfg.KC
    config_number = len(KC[0])
    move = cfg.move
    config_m_number = len(move)
    select_dist = r
    target_number = len(current_targets_states)

    # maps for all camera nodes
    detection_map = []
    detection_score_map = []
    focal_sequences = []
    location_sequences = []

    # fill in the maps
    for c in range(cam_number):
        detect_map = []
        detect_score_map = []
        focal_sequence = []
        location_sequence = []
        for f in KC[c]:
            for m in move:
                location_c = cam_location[c].copy()
                if c == 0:
                    location_c[0] = cam_location[c][0] + m
                    location_c[0] = value_between(location_c[0], 1, 7)
                elif c == 1:
                    location_c[1] = cam_location[c][1] - m
                    location_c[1] = value_between(location_c[1], 1, 7)
                elif c == 2:
                    location_c[0] = cam_location[c][0] - m
                    location_c[0] = value_between(location_c[0], 1, 7)
                elif c == 3:
                    location_c[1] = cam_location[c][1] + m
                    location_c[1] = value_between(location_c[1], 1, 7)

                focal_sequence.append(f)
                location_sequence.append(location_c)

                distance_map = np.sqrt(np.sum(np.asarray(location_c -
                                                         np.asarray(current_targets_states)[:, 0:2]) ** 2, axis=1))
                detect_vector = np.zeros(target_number, dtype=int)
                detect_score_vector = np.zeros(target_number, dtype=np.float32)

                for t in range(target_number):
                    if distance_map[t] >= r - 0.1 or distance_map[t] <= 0.1:
                        detect_vector[t] = 0
                        detect_score_vector[t] = 0
                    else:
                        aim_x = current_targets_states[t][0]
                        aim_y = current_targets_states[t][1]
                        offset_y = aim_y - location_c[1]
                        offset_x = aim_x - location_c[0]
                        slope = np.arctan2(offset_y, offset_x) * 180 / np.pi
                        dist_focal = f - slope
                        if abs(dist_focal) > 180:
                            dist_focal = 360 - abs(dist_focal)
                        if abs(dist_focal) >= fov / 2:
                            detect_vector[t] = 0
                            detect_score_vector[t] = 0
                        if abs(dist_focal) < fov / 2:
                            detect_vector[t] = 1
                            if num_or_score:
                                poseTC = relative_pose(current_targets_states[t][0], current_targets_states[t][1],
                                                       current_targets_states[t][3], location_c[0], location_c[1])
                                if abs(poseTC) < 90:
                                    detect_score_vector[t] = (np.cos(poseTC/57.3)+0.5) / np.sqrt(distance_map[t])
                                else:
                                    detect_score_vector[t] = 0

                detect_map.append(detect_vector)
                detect_score_map.append(detect_score_vector)

        detection_map.append(detect_map)
        detection_score_map.append(detect_score_map)
        focal_sequences.append(focal_sequence)
        location_sequences.append(location_sequence)

    len_dc0 = len(focal_sequences[0])
    len_dc1 = len(focal_sequences[1])
    len_dc2 = len(focal_sequences[2])
    len_dc3 = len(focal_sequences[3])

    detection_vector_sum = []
    detection_config_focal_match = []
    detection_config_loc_match = []
    detection_scores = []
    changes = []

    for id0 in range(len_dc0):
        dc0 = focal_sequences[0][id0]
        dl0 = location_sequences[0][id0]
        dv0 = detection_map[0][id0]
        ds0 = detection_score_map[0][id0]
        df0 = abs(pre_cam_focal[0] - dc0) / 30 + abs(pre_cam_location[0][0] - dl0[0])
        for id1 in range(len_dc1):
            dc1 = focal_sequences[1][id1]
            dl1 = location_sequences[1][id1]
            dv1 = detection_map[1][id1]
            ds1 = detection_score_map[1][id1]
            df1 = abs(pre_cam_focal[1] - dc1) / 30 + abs(pre_cam_location[1][1] - dl1[1])
            for id2 in range(len_dc2):
                dc2 = focal_sequences[2][id2]
                dl2 = location_sequences[2][id2]
                dv2 = detection_map[2][id2]
                ds2 = detection_score_map[2][id2]
                df2 = abs(pre_cam_focal[2] - dc2) / 30 + abs(pre_cam_location[2][0] - dl2[0])
                for id3 in range(len_dc3):
                    dc3 = focal_sequences[3][id3]
                    dl3 = location_sequences[3][id3]
                    dv3 = detection_map[3][id3]
                    ds3 = detection_score_map[3][id3]
                    df3 = abs(pre_cam_focal[3] - dc3) / 30 + abs(pre_cam_location[3][1] - dl3[1])

                    dc = [dc0, dc1, dc2, dc3]
                    dl = [dl0, dl1, dl2, dl3]

                    dv = dv0 | dv1 | dv2 | dv3  # detection vector
                    ds = ds0 + ds1 + ds2 + ds3
                    sum_dv = dv.sum()  # detection number
                    sum_ds = ds.sum()

                    detection_vector_sum.append(np.int(sum_dv))
                    detection_scores.append(sum_ds)

                    detection_config_focal_match.append(dc)
                    detection_config_loc_match.append(dl)
                    changes.append(df0 + df1 + df2 + df3)
    if num_or_score == 0:
        scores = np.asarray(detection_vector_sum) - delta_k*np.asarray(changes)
    if num_or_score == 1:
        scores = np.asarray(detection_scores) - delta_k*np.asarray(changes)

    k_index = int(np.argmax(scores))
    new_focal = detection_config_focal_match[k_index]
    new_loc = detection_config_loc_match[k_index]
    for i in range(cam_number):
        cam_angle[i] = new_focal[i]
        cam_location[i] = new_loc[i]

    if num_or_score:
        detected_results = detection_scores[k_index]
    if not num_or_score:
        detected_results = detection_vector_sum[k_index]

    return cam_location, cam_angle, detected_results
