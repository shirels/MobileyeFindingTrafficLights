from Model.find_tfll_p1 import find_tfl
from Model.predict_with_cnn import prediction
from Model.calc_distance import calc_TFL_dist
import Model.calc_distance
from View.view import visualizer
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib._png as png




class FrameContainer(object):
    def __init__(self, img_path, index):
        self.img = png.read_png_int(img_path)
        self.img_p = img_path
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []
        self.index = index


class TFLMan:

    def __init__(self, pkl_path: str):
        """
        input:
        pkl path
        :return:
        """
        with open(pkl_path, 'rb') as pkl:
            self.pkl_data = pickle.load(pkl, encoding='latin1')
        self.auxiliary_stage1 = {}
        self.auxiliary_stage2 = {}
        self.prev_container = None
        self.distances = []


    def on_frame(self, i, current_frame, prev_frame, pkl):

        self.auxiliary_stage1 = self.part_1(current_frame)
        self.auxiliary_stage2 = self.part_2(current_frame, self.auxiliary_stage1)
        if prev_frame is None:
            self.prev_container = FrameContainer(current_frame, i)
            self.prev_container.traffic_light = np.array(self.auxiliary_stage2['red'] + self.auxiliary_stage2['green'])
            focal = None
            pp = None
            curr_container = self.prev_container
            prev_container = None
        if prev_frame is not None:
            prev_container = self.prev_container
            curr_container = FrameContainer(current_frame, i)
            curr_container.traffic_light = np.array(self.auxiliary_stage2['red'] + self.auxiliary_stage2['green'])
            focal = self.pkl_data['flx']
            pp = self.pkl_data['principle_point']
            EM = np.eye(4)
            for i in range(prev_container.index, curr_container.index):
                EM = np.dot(self.pkl_data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
            curr_container.EM = EM
            self.distances = self.part_3(prev_container=prev_container, curr_container=curr_container,
                                    tfl_prev=prev_container.traffic_light,
                                    tfl_curr=curr_container.traffic_light,
                                    EM=EM, pp=pp, focal=focal)
        visualizer(self.auxiliary_stage1, self.auxiliary_stage2, curr_container, prev_container, focal, pp)



    def part_1(self, frame):
        """

        :param frame: get a frame image
        :return: vector of colors
        """
        self.auxiliary_stage1 = find_tfl(frame)
        return self.auxiliary_stage1


    def part_2(self, current_frame, auxiliary):
        """
        reduces candidates through cnn
        :param current_frame:
        :param auxiliary:
        :return: traffic light - vector of K<=position
        """
        self.auxiliary_stage2 = prediction(current_frame=current_frame, auxiliary=self.auxiliary_stage1)
        return self.auxiliary_stage2


    def part_3(self, prev_container=None, curr_container=None, tfl_prev=None, tfl_curr=None,
               EM=None, pp=None, focal=None):
        """
        :param prev_frame:
        :param current_frame:
        :param tfl_prev:
        :param tfl_curr:
        :param EM:
        :param pp:
        :param focal:
        :return:
        """
        curr_container = calc_TFL_dist(prev_container, curr_container, focal, pp)
        self.prev_container = curr_container
        return curr_container.traffic_lights_3d_location[:, 2]


   
