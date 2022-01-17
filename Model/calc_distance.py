import numpy as np


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if (abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_prev_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    return (pts - pp) / focal


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    return pts * focal + pp


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    em = EM.tolist().EM
    R = em[:3, :3]
    t = em[:3, 3]   # t[0] = t_x, t[1] = t_y, t[2] = t_z
    foe = [t[0]/t[2], t[1]/t[2]]
    return [R, foe, t[2]]


def rotate(pts, R):
    # rotate the points - pts using R
    # ones = np.array([[1, 1, 1]]).T
    ones = np.ones((len(pts[:, 0]), 1), int)
    rotate_matrix = np.dot(R, (np.hstack([pts, ones])).T)
    first_two_rows = rotate_matrix[:2]
    third_row = rotate_matrix[2]
    new_mat = first_two_rows / third_row
    return new_mat.T


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    m = (foe[1] - p[1])/(foe[0] - p[0])
    n = (p[1] * foe[0] - foe[1] * p[0]) / (foe[0] - p[0])
    # run over all norm_pts_rot and find the one closest to the epipolar line
    distances = []  # list of tuples - [0] = distance, [1] = point, [2] = index
    for index, norm_pt in enumerate(norm_pts_rot):
        d = abs((m*norm_pt[0] + n - norm_pt[1])/((m**2 + 1)**0.5))
        distances.append((d, norm_pt, index))
    min_dist = min(distances, key=lambda x: x[0])
    # return the closest point and its index
    return min_dist[2], min_dist[1]


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    z_x = tZ * (foe[0] - p_rot[0]) / (p_curr[0] - p_rot[0])
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    z_y = tZ * (foe[1] - p_rot[1]) / (p_curr[1] - p_rot[1])
    # combine the two estimations and return estimated Z - distance between them
    snr = abs(p_curr[0] - p_rot[0]) / abs(p_curr[1] - p_rot[1])
    if snr > 1:
        return z_x
    return (1 - snr) * z_y + snr * z_x