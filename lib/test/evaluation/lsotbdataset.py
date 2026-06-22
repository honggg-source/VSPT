import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
import os
import glob
import pdb

def LSOTBDataset():
    return LSOTBDataset().get_sequence_list()


class LSOTBDataset(BaseDataset):
    """VOTRGBT2018 dataset

    Publication:
        The sixth Visual Object Tracking VOTRGBT2018 challenge results.
        Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pfugfelder, Luka Cehovin Zajc, Tomas Vojir,
        Goutam Bhat, Alan Lukezic et al.
        ECCV, 2018
        https://prints.vicos.si/publications/365

    Download the dataset from http://www.votchallenge.net/vot2019rgbt/dataset.html"""
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.lsotb_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name        

        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        end_frame = ground_truth_rect.shape[0]

        imgv_dir = os.path.join(self.base_path, sequence_path, 'img')   
        imgv_list = glob.glob(imgv_dir + "/*.jpg")
        imgv_list.sort()
        framesv = [os.path.join(imgv_dir, x) for x in imgv_list]        

#        return Sequence(sequence_name, framesv, ground_truth_rect)
        return Sequence(sequence_name, framesv, 'lsotb', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list=['bus_V_002',
'person_V_007',
'person_V_008',
'bus_V_003',
'bus_V_004',
'bus_V_005',
'car_V_001',
'car_V_003',
'car_V_009',
'car_V_010',
'car_V_014',
'person_V_002',
'car_D_002',
'dog_D_001',
'car_D_004',
'car_D_005',
'car_D_007',
'car_D_009',
'dog_D_002',
'person_D_003',
'person_D_004',
'person_D_006',
'person_D_009',
'person_D_011',
'person_D_014',
'person_D_015',
'person_D_016',
'person_D_019',
'person_D_020',
'person_D_022',
'hog_D_001',
'person_D_023',
'drone_D_001',
'cat_D_001',
'airplane_H_001',
'airplane_H_002',
'badger_H_001',
'bat_H_001',
'bird_H_001',
'bird_H_002',
'bird_H_003',
'boat_H_001',
'cat_H_001',
'cat_H_002',
'cow_H_001',
'deer_H_001',
'dog_H_001',
'fox_H_001',
'helicopter_H_001',
'helicopter_H_002',
'hog_H_001',
'hog_H_002',
'hog_H_003',
'hog_H_004',
'person_H_002',
'person_H_003',
'person_H_004',
'person_H_006',
'person_H_008',
'person_H_011',
'person_H_012',
'person_H_013',
'face_H_001',
'leopard_H_001',
'owl_H_001',
'airplane_H_003',
'mouse_H_001',
'head_H_002',
'boy_S_001',
'boy_S_002',
'bus_S_004',
'pickup_S_001',
'car_S_001',
'car_S_003',
'helicopter_S_001',
'hog_S_001',
'street_S_001',
'street_S_002',
'street_S_003',
'street_S_004',
'street_S_005',
'person_S_001',
'person_S_003',
'person_S_006',
'person_S_007',
'person_S_008',
'person_S_010',
'person_S_011',
'person_S_012',
'person_S_014',
'person_S_015',
'person_S_016',
'person_S_017',
'person_S_018',
'person_S_019',
'truck_S_001',
'coyote_S_001',
'crowd_S_001',
'head_S_001',
'hand_S_001']
        return sequence_list
