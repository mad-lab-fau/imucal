import h5py
import numpy as np


class calibration_matrices(object):

    def __init__(self, K_a, R_a, b_a, K_g, R_g, K_ga, b_g):
        """
        Contructor for calibration matrices class
        :param K_a: Scaling matrix acceleration (numpy array)
        :param R_a: Rotation matrix acceleration (numpy array)
        :param b_a: Bias vector acceleration (numpy array)
        :param K_g: Scaling matrix gyroscope (numpy array)
        :param R_g: Rotation matrix gyroscope (numpy array)
        :param K_ga: Effect matrix of acceleration on gyroscope (numpy array)
        :param b_g: Bias vector gyroscope (numpy array)
        """

        self.K_a = K_a
        self.R_a = R_a
        self.b_a = b_a
        self.K_g = K_g
        self.R_g = R_g
        self.K_ga = K_ga
        self.b_g = b_g

    def print(self):
        print('K_a:')
        print(self.K_a)
        print('')
        print('R_a:')
        print(self.R_a)
        print('')
        print('b_a:')
        print(self.b_a)
        print('')
        print('K_g:')
        print(self.K_g)
        print('')
        print('R_g:')
        print(self.R_g)
        print('')
        print('k_g:')
        print(self.K_ga)
        print('')
        print('b_g:')
        print(self.b_g)
        print('')

    def save_to_hdf5(self, filename):
        """
        Saves calibration matrices to hdf5 fileformat
        :param filename: filename (including h5 at end)
        """

        with h5py.File(filename, 'w') as hdf:
            hdf.create_dataset('K_a', data=self.K_a)
            hdf.create_dataset('R_a', data=self.K_a)
            hdf.create_dataset('b_a', data=self.b_a)
            hdf.create_dataset('K_g', data=self.K_g)
            hdf.create_dataset('R_g', data=self.R_g)
            hdf.create_dataset('K_ga', data=self.K_ga)
            hdf.create_dataset('b_g', data=self.b_g)


def read_from_hdf5(filename):
    """
    Reads calibration data stored in hdf5 fileformat (created by calibration_matrices save_to_hdf5)
    :param filename: filename
    :return: calibration_matrices object
    """

    with h5py.File(filename, 'r') as hdf:
        K_a = np.array(hdf.get('K_a'))
        R_a = np.array(hdf.get('R_a'))
        b_a = np.array(hdf.get('b_a'))
        K_g = np.array(hdf.get('K_g'))
        R_g = np.array(hdf.get('R_g'))
        K_ga = np.array(hdf.get('K_ga'))
        b_g = np.array(hdf.get('b_g'))

        calib_mat = calibration_matrices(K_a, R_a, b_a, K_g, R_g, K_ga, b_g)

        return calib_mat
