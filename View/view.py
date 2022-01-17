from Model.calc_distance import prepare_3D_data, rotate, unnormalize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib._png as png


def visualizer(auxiliary_stage1, auxiliary_stage2, curr_container, prev_container=None, focal=None, pp=None):

    fig, (candidates, tfl, dist) = plt.subplots(3, 1, figsize=(12, 6))
    candidates.set_title("part 1 - finding traffic lights")
    tfl.set_title("part 2 - reducing candidates using cnn ")
    dist.set_title("part 3 - calculating distance")
    candidates.imshow(png.read_png_int(curr_container.img_p))
    candidates.plot(np.array(auxiliary_stage1["red"])[:, 1], np.array(auxiliary_stage1["red"])[:, 0],
                    'rx', markersize=8)
    candidates.plot(np.array(auxiliary_stage1["green"])[:, 1], np.array(auxiliary_stage1["green"])[:, 0],
                    'g+', markersize=8)

    tfl.imshow(png.read_png_int(curr_container.img_p))
    tfl.plot(np.array(auxiliary_stage2["red"])[:, 1], np.array(auxiliary_stage2["red"])[:, 0], 'rx',
             markersize=8)
    tfl.plot(np.array(auxiliary_stage2["green"])[:, 1], np.array(auxiliary_stage2["green"])[:, 0], 'g+',
             markersize=8)
    if prev_container is not None:
        norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
        norm_rot_pts = rotate(norm_prev_pts, R)
        rot_pts = unnormalize(norm_rot_pts, focal, pp)
        foe = np.squeeze(unnormalize(np.array([norm_foe]), focal, pp))

        dist.imshow(curr_container.img_p)
        prev_p = prev_container.traffic_light
        dist.plot(prev_p[:, 0], prev_p[:, 1], 'b+')

        curr_p = curr_container.traffic_light
        dist.plot(curr_p[:, 0], curr_p[:, 1], 'b+')

        for i in range(len(curr_p)):
            if curr_container.valid[i]:
                dist.text(curr_p[i, 0], curr_p[i, 1],
                              r'{0:.1f}'.format(curr_container.traffic_lights_3d_location[i, 2]), color='r')
        dist.plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')
        plt.show()
