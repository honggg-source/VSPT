import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
import os
import glob
import pdb

def VTUAVDataset():
    return VTUAVDataset().get_sequence_list()


class VTUAVDataset(BaseDataset):
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
        self.base_path = self.env_settings.vtuav_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name

        anno_path = '{}/{}/ir.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        end_frame = ground_truth_rect.shape[0]

        imgv_dir = os.path.join(self.base_path, sequence_path, 'ir')
        imgv_list = glob.glob(imgv_dir + "/*.jpg")
        imgv_list.sort()
        framesv = [os.path.join(imgv_dir, x) for x in imgv_list]

#        return Sequence(sequence_name, framesv, ground_truth_rect)
        return Sequence(sequence_name, framesv, 'vtuav', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        seq_home = 'E:/tracker_data/VTUAV'
        sequence_list = [f for f in os.listdir(seq_home) if os.path.isdir(os.path.join(seq_home, f))]
        return sequence_list
