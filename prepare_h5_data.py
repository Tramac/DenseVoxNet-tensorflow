#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import time
import h5py

import SimpleITK as sitk
import numpy as np

from utils import pre_process_isotropic, create_ROIs


def prepare_h5_data(data_folder, h5_save_folder, h5_save_list, **kwargs):
    """Generate the hdf5 file and a file list used to train the network"""

    # re-sample the data into same resolution, default is False.
    use_isotropic = 0
    patchSize = 64
    if len(kwargs) < 3:
        data_folder = "./data"
        h5_save_folder = "./h5_data"
        h5_save_list = "./train.txt"

    if not os.path.exists(h5_save_folder):
        os.mkdir(h5_save_folder)

    fid = open(h5_save_list, 'w')

    # generate the hdf5 file
    for id in range(10):
        tic = time.time()

        img_filename = "training_axial_crop_pat" + str(id) + ".nii.gz"
        seg_filename = "training_axial_crop_pat" + str(id) + "-label.nii.gz"
        img_path = os.path.join(data_folder, img_filename)
        seg_path = os.path.join(data_folder, seg_filename)

        img_nii = sitk.GetArrayFromImage(sitk.ReadImage(img_path))  # numpy array
        seg_nii = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        img_nii = np.transpose(img_nii, [0, 2, 1])  # keep same with matlab
        seg_nii = np.transpose(seg_nii, [0, 2, 1])

        # pre-process the images (intensity and resize)
        img, seg = pre_process_isotropic(img_nii, seg_nii, use_isotropic, id)

        # crop the heart patches (ROI) from whole image and randomly crop patchs
        img, seg, rimgs, rsegs = create_ROIs(img, seg, patchSize)

        # Do data augmentation (permute & rotate & flip)
        # Only do augmentation at axial plane
        # Can do augmentation in all three planes by setting it_p = 0:3
        pp = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
        for it_p in range(1):
            permute_img = np.transpose(img, pp[it_p])
            permute_seg = np.transpose(seg, pp[it_p])
            for it_r in range(4):
                # rotate
                rotate_img = np.rot90(permute_img, it_r)
                rotate_seg = np.rot90(permute_seg, it_r)
                for it_f in range(2):
                    if it_f == 0:
                        flip_img = rotate_img
                        flip_seg = rotate_seg
                    else:
                        flip_img = np.flip(rotate_img, axis=it_f - 1)
                        flip_seg = np.flip(rotate_seg, axis=it_f - 1)

                    # h5 file path
                    h5_path = os.path.join(h5_save_folder,
                                           str(id) + "_p" + str(it_p) + "_r" + str(it_r) + "_f" + str(it_f) + ".h5")
                    if os.path.exists(h5_path):
                        os.remove(h5_path)

                    d1, d2, d3 = flip_img.shape
                    data_dims = [d1, d2, d3]
                    d1, d2, d3 = flip_seg.shape
                    seg_dims = [d1, d2, d3]

                    # store as h5 format
                    f = h5py.File(h5_path, 'w')
                    f.create_dataset('data', data_dims, dtype=None, data=flip_img)
                    f.create_dataset('label', seg_dims, dtype='uint8', data=np.uint8(flip_seg))
                    f.close()
                    fid.write(h5_path + '\n')

        # save randomly cropped patches
        # Default, do not use them to train the model
        for i in range(0):
            h5_path = os.path.join(h5_save_folder, str(id) + '_random_' + str(i) + '.h5')
            if os.path.exists(h5_path):
                os.remove(h5_path)

            write_img = rimgs[i]
            write_seg = rsegs[i]
            d1, d2, d3 = write_img.shape
            data_dims = [d1, d2, d3]
            d1, d2, d3 = write_seg.shape
            seg_dims = [d1, d2, d3]

            # store as h5 format
            f = h5py.File(h5_path, 'w')
            f.create_dataset('data', data_dims, dtype=None, data=write_img)
            f.create_dataset('label', seg_dims, dtype='uint8', data=np.uint8(write_seg))
            f.close()
            fid.writelines(h5_path)

        toc = time.time()
        print("Time: %gs" % (toc - tic))

    fid.close()

if __name__ == '__main__':
    prepare_h5_data(data_folder='/data', h5_save_folder='/h5_data', h5_save_list='train.txt')
