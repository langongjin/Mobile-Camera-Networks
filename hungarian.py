import numpy as np
from sklearn.cluster import KMeans
from utils import cluster_greedy, value_between, relative_pose

sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0
TOLERANCE = 1e-6  # everything below is considered zero


def hungarians(current_targets_states, cfg, pre_cam_location, pre_cam_focal, num_or_score=0):
    """
    cluster targets into at most 4 groups and use hungarian algorithms to get results
    """
    r = cfg.radius
    cam_location = pre_cam_location
    cam_focal = pre_cam_focal
    cam_number = cfg.cam_number
    k2cluster = cfg.cam_number
    select_dist = r - 0.1
    move_factor = 0.12

    # for drawing sector
    target_number = len(current_targets_states)
    if target_number < 4:
        return cluster_greedy(current_targets_states, cfg, pre_cam_location, pre_cam_focal, 0)

    cam_angle = np.array([0., 0., 0., 0.])

    locations = np.asarray(current_targets_states)[:, 0:2]
    clusters = KMeans(n_clusters=k2cluster, random_state=0).fit(locations)

    labels = clusters.labels_
    center_points = clusters.cluster_centers_

    detection_vector_map = np.zeros((cam_number, target_number), dtype=np.int)
    detection_score_map = np.zeros((cam_number, target_number), dtype=np.float32)

    detection_angle_config = []
    detection_loc_config = []
    biparite_graph_detection_vector = []

    for c in range(cam_number):
        group_location = []
        group_angle = []
        group_detected = []

        for cent in range(k2cluster):
            loc_config = []
            angle_config = []
            detect_num = []
            detect_score_sum = []

            cam_location_copy = cam_location.copy()
            aim_x = center_points[cent][0]
            aim_y = center_points[cent][1]

            if c == 0 or c == 2:
                cam_location_copy[c][0] += move_factor * sgn(aim_x - cam_location[c][0])
                cam_location_copy[c][0] = value_between(cam_location_copy[c][0], 1, 7)
            elif c == 1 or c == 3:
                cam_location_copy[c][1] += move_factor * sgn(aim_y - cam_location[c][1])
                cam_location_copy[c][1] = value_between(cam_location_copy[c][1], 1, 7)

            angle = np.arctan2(aim_y-cam_location_copy[c][1], aim_x-cam_location_copy[c][0]) * 57.3

            loc_config.append(cam_location_copy[c])
            angle_config.append(angle)

            detected = 0
            detect_score = 0
            for index in range(target_number):
                if labels[index] == cent:
                    aim_tx = current_targets_states[index][0]
                    aim_ty = current_targets_states[index][1]
                    dist = np.sqrt((cam_location_copy[c][0] - aim_tx) ** 2 + (cam_location_copy[c][1] - aim_ty) ** 2)
                    if dist >= select_dist:
                        detected += 0
                    else:
                        offset_y = aim_ty - cam_location_copy[c][1]
                        offset_x = aim_tx - cam_location_copy[c][0]
                        slope = np.arctan2(offset_y, offset_x) * 57.3
                        dist_focal = angle - slope
                        if -30 <= dist_focal < 30:
                            detected += 1
                            poseTC = relative_pose(aim_tx, aim_ty, current_targets_states[index][3],
                                                   cam_location_copy[c][0], cam_location_copy[c][1])
                            if abs(poseTC) < 90:
                                detect_score += (np.cos(poseTC / 57.3) + 0.5) / np.sqrt(dist)
                            else:
                                detect_score += 0
                        else:
                            detected += 0
                            detect_score += 0

            detect_num.append(detected)
            detect_score_sum.append(detect_score)

            if not num_or_score:
                current_max_value = max(detect_num)
                current_config_index = detect_num.index(max(detect_num))
            if num_or_score:
                current_max_value = max(detect_score_sum)
                current_config_index = detect_score_sum.index(max(detect_score_sum))

            group_angle.append(angle_config[current_config_index])
            group_location.append(loc_config[current_config_index])
            group_detected.append(current_max_value)
        detection_angle_config.append(group_angle)
        detection_loc_config.append(group_location)
        biparite_graph_detection_vector.append(group_detected)

    matching_map, max_num = maxWeightMatching(biparite_graph_detection_vector)

    for c in range(cam_number):
        c_location = detection_loc_config[c][matching_map[c]]
        c_angle = detection_angle_config[c][matching_map[c]]
        cam_angle[c] = c_angle
        cam_location[c] = c_location

        for t in range(target_number):
            aim_tx = current_targets_states[t][0]
            aim_ty = current_targets_states[t][1]
            dist = np.sqrt((c_location[0] - aim_tx) ** 2 + (c_location[1] - aim_ty) ** 2)
            if dist <= select_dist:
                offset_y = aim_ty - c_location[1]
                offset_x = aim_tx - c_location[0]
                slope = np.arctan2(offset_y, offset_x) * 57.3
                dist_focal = c_angle - slope
                if -30 <= dist_focal < 30:
                    detection_vector_map[c][t] = 1
                    poseTC = relative_pose(aim_tx, aim_ty, current_targets_states[t][3],
                                           c_location[0], c_location[1])
                    if abs(poseTC) < 90:
                        detection_score_map[c][t] = (np.cos(poseTC / 57.3) + 0.5) / np.sqrt(dist)
                    else:
                        detection_score_map[c][t] = 0
                else:
                    detection_vector_map[c][t] = 0
                    detection_score_map[c][t] = 0

    detection_vector = detection_vector_map[0] | detection_vector_map[1] | detection_vector_map[2] | detection_vector_map[3]
    detection_scores = detection_score_map[0] + detection_score_map[1] + detection_score_map[2] + detection_score_map[3]
    sum_dv = detection_vector.sum()
    sum_ds = detection_scores.sum()

    detection_results = 0
    if not num_or_score:
        detection_results = sum_dv
    if num_or_score:
        detection_results = sum_ds

    return cam_location, cam_angle, detection_results


def improveLabels(val):
    """ change the labels, and maintain minSlack.
    """
    for u in S:
        lu[u] -= val
    for v in V:
        if v in T:
            lv[v] += val
        else:
            minSlack[v][0] -= val


def improveMatching(v):
    """ apply the alternating path from v to the root in the tree.
    """
    u = T[v]
    if u in Mu:
        improveMatching(Mu[u])
    Mu[u] = v
    Mv[v] = u


def slack(u, v):
    return lu[u]+lv[v]-w[u][v]


def augment():
    """ augment the matching, possibly improving the lablels on the way.
    """
    while True:
        # select edge (u,v) with u in S, v not in T and min slack
        ((val, u), v) = min([(minSlack[v], v) for v in V if v not in T])
        assert u in S
        assert val > - TOLERANCE
        if val > TOLERANCE:
            improveLabels(val)
        # now we are sure that (u,v) is saturated
        assert abs(slack(u, v)) < TOLERANCE  # test zero slack with tolerance
        T[v] = u                            # add (u,v) to the tree
        if v in Mv:
            u1 = Mv[v]                      # matched edge,
            assert not u1 in S
            S[u1] = True                    # ... add endpoint to tree
            for v in V:                     # maintain minSlack
                if not v in T and minSlack[v][0] > slack(u1, v):
                    minSlack[v] = [slack(u1, v), u1]
        else:
            improveMatching(v)              # v is a free vertex
            return


def maxWeightMatching(weights):
    """ given w, the weight matrix of a complete bipartite graph,
        returns the mappings Mu : U->V ,Mv : V->U encoding the matching
        as well as the value of it.
    """
    global U, V, S, T, Mu, Mv, lu, lv, minSlack, w
    w = weights
    n = len(w)
    U = V = range(n)
    lu = [max([w[u][v] for v in V]) for u in U]  # start with trivial labels
    lv = [0 for v in V]
    Mu = {}                                       # start with empty matching
    Mv = {}
    while len(Mu) < n:
        free = [u for u in V if u not in Mu]      # choose free vertex u0
        u0 = free[0]
        S = {u0: True}                            # grow tree from u0 on
        T = {}
        minSlack = [[slack(u0, v), u0] for v in V]
        augment()
    # val. of matching is total edge weight
    val = sum(lu)+sum(lv)
    # return Mu, Mv, val
    return Mu, val


if __name__ == "__main__":
    max_map, max_v = maxWeightMatching([[8, 3, 5], [7, 6, 9], [8, 7, 4]])
    print(max_map, max_v)
