import random

import h5py
import numpy as np


class BatchDataset:
    images = []
    labels = []
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, data_list):
        """
        Initialize a generic file reader with batching for list of files
        :param data_list: list of file to read
        """
        print("Initializing Batch Dataset Reader...")
        self.files = data_list
        self.read_images()

    def read_images(self):
        self.images = np.array([self.load_image(filename) for filename in self.files])
        self.labels = np.array([self.load_label(filename) for filename in self.files])

    def load_image(self, filename):
        """
        Load image from h5 file
        :param filename:
        :return: image, label
        """
        f = h5py.File(filename, 'r')
        image = f['data']
        image = np.squeeze(image)
        # image = np.transpose(image, [2, 1, 0])

        return image

    def load_label(self, filename):
        f = h5py.File(filename, 'r')
        label = f['label']
        label = np.squeeze(label)
        # label = np.transpose(label, [2, 1, 0])

        return label

    def random_crop(self, images, labels, size):
        crop_images = []
        crop_labels = []
        for i in range(images.shape[0]):
            x, y, z = images[i].shape
            if size > x or size > y or size > z:
                raise IndexError("Please input the right size")
            random_center_x = random.randint(size / 2, x - size / 2)
            random_center_y = random.randint(size / 2, y - size / 2)
            random_center_z = random.randint(size / 2, z - size / 2)
            crop_image = images[i][random_center_x - size / 2: random_center_x + size / 2,
                         random_center_y - size / 2: random_center_y + size / 2,
                         random_center_z - size / 2: random_center_z + size / 2]
            crop_label = labels[i][random_center_x - size / 2: random_center_x + size / 2,
                         random_center_y - size / 2: random_center_y + size / 2,
                         random_center_z - size / 2: random_center_z + size / 2]

            crop_images.append(crop_image)
            crop_labels.append(crop_label)

        crop_images = np.array(crop_images)
        crop_labels = np.array(crop_labels)

        return crop_images, crop_labels

    def next_batch(self, batch_size, crop=True, crop_size=64):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finish epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        batch_images = self.images[start: end]
        batch_labels = self.labels[start: end]
        if crop:
            batch_images, batch_labels = self.random_crop(batch_images, batch_labels, crop_size)

        batch_images = np.transpose(batch_images, [0, 3, 2, 1])
        batch_labels = np.transpose(batch_labels, [0, 3, 2, 1])
        batch_images = np.expand_dims(batch_images, axis=4)
        batch_labels = np.expand_dims(batch_labels, axis=4)

        return batch_images, batch_labels
