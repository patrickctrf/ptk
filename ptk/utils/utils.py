import os
from queue import Queue
from threading import Thread

import numpy as np
import torch
from numpy import load, array, int64, absolute
from torch import from_numpy
from torch.utils.data import Dataset


def split_into_chunks(lista, split_dims):
    """
Split a list into evenly sized chunks. The last chunk will be smaller if the
original list length is not divisible by 'split_dims'.

    :param lista: List to be split.
    :param split_dims: Length of each split chunk.
    """
    aux_list = []
    # For item i in a range that is a length of l,
    for i in range(0, len(lista), split_dims):
        # Create an index range for l of n items:
        aux_list.append(lista[i:i + split_dims])

    return aux_list


def find_nearest(array_to_search, value):
    """
This function takes 1 array as first argument and a value to find the element
in array whose value is the closest. Returns the closest value element and its
index in the original array.

    :param array_to_search: Reference array.
    :param value: Value to find closest element.
    :return: Tuple (Element value, element index).
    """
    idx = (ggitabsolute(array_to_search - value)).argmin()
    return array_to_search[idx], idx


def axis_angle_into_quaternion(normalized_axis, angle):
    """
Takes an axis-angle rotation and converts into quaternion rotation.
https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    :param normalized_axis: Axis of rotation (3-element array).
    :param angle: Simple rotation angle (float or 1-element array).
    :return: 4-element array, containig quaternion (q0,q1,q2,q3).
    """
    # From axis-angle notation into quaternion notation.
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    quaternion_orientation_r = np.zeros((4,))
    quaternion_orientation_r[0] = np.cos(angle / 2)
    quaternion_orientation_r[1] = np.sin(angle / 2) * normalized_axis[0]
    quaternion_orientation_r[2] = np.sin(angle / 2) * normalized_axis[1]
    quaternion_orientation_r[3] = np.sin(angle / 2) * normalized_axis[2]

    return quaternion_orientation_r


def quaternion_into_axis_angle(quaternion):
    """
Takes an quaternion rotation and converts into axis-angle rotation.
https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    :param quaternion: 4-element tensor, containig quaternion (q0,q1,q2,q3).
    :return: (Axis of rotation (3-element tensor), Simple rotation angle (float or 1-element tensor))
    """
    # Simple rotation angle
    angle = torch.acos(quaternion[0]) * 2

    # Avoids recalculating this sin.
    sin_angle_2 = torch.sin(angle / 2)

    # Rotation axis
    normalized_axis = torch.zeros((3,))
    normalized_axis[0] = quaternion[1] / sin_angle_2
    normalized_axis[1] = quaternion[2] / sin_angle_2
    normalized_axis[2] = quaternion[3] / sin_angle_2

    return normalized_axis, angle


def skew_matrix_from_array(x):
    """
Receives a 3-element array and return its respective skew matrix.

    :param x: 3-element array.
    :return: Respective skew-matrix (3x3)
    """
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0],
    ])


def array_from_skew_matrix(x):
    """
Receives a skew matrix and returns its associated 3-element vector (array).

    :param x: Skew matrix (3x3)
    :return: Associated array (3-element).
    """
    return np.array([x[2][1], x[0][2], x[1][0]])


def exp_matrix(skew_matrix):
    norma = np.linalg.norm(skew_matrix)

    return np.eye(N=3, M=3) + \
           (np.sin(norma) / norma) * skew_matrix + \
           (1 - np.cos(norma)) / (norma ** 2) * np.matmul(skew_matrix, skew_matrix)


def rotation_matrix_into_axis_angle(r_matrix):
    """
Converts a 3x3 rotation matrix into equivalent axis-angle rotation.

    :param r_matrix: 3x3 rotation matrix (array).
    :return: Tuple -> (normalized_axis (3-element array), rotation angle)
    """
    # Converts R orientation matrix into equivalent skew matrix. SO(3) -> so(3)
    # phi is a simple rotation angle (the value in radians of the angle of rotation)
    phi = np.arccos((np.trace(r_matrix) - 1) / 2)

    # Skew "orientation" matrix into axis-angles tensor (3-element).
    # we do not multiply by phi, so we have a normalized rotation AXIS (in a SKEW matrix yet)
    # normalized because we didnt multiply the axis by the rotation angle (phi)
    return array_from_skew_matrix((r_matrix - r_matrix.T) / (2 * np.sin(phi))), phi


def axis_angle_into_rotation_matrix(normalized_axis, angle):
    return exp_matrix(skew_matrix_from_array(normalized_axis * angle))


class GenericDatasetFromFiles(Dataset):
    def __init__(self, data_path="", convert_first=False, mmap_mode=None, transform=None, device=torch.device("cpu")):
        """
Dataset class loader in PyTorch pattern. It expects data in the NPZ file format
(numpy), with 2 files called 'x_data.npz' and 'y_data.npz'.

        :param data_path: Directory where data is located.
        :param convert_first: If arrays must be converted into Tensors.
        :param mmap_mode: Numpy memmap mode. Keeps arrays on disk if they are
        too big for memory. More info {‘r+’, ‘r’, ‘w+’, ‘c’}: https://numpy.org/doc/stable/reference/generated/numpy.memmap.html#numpy.memmap
        :param transform: PyTorch transform.
        :param device: If converting data into tensors, already put those in
        PyTorch's device (torch.device("cuda:0"), for example). Default: torch.device("cpu")
        """
        super().__init__()
        self.x_dictionary = load(os.path.join(data_path, "x_data.npz"), mmap_mode=mmap_mode)
        self.y_dictionary = load(os.path.join(data_path, "y_data.npz"), mmap_mode=mmap_mode)
        self.length = len(self.x_dictionary)
        self.convert_first = convert_first

        self.x_dictionary_files = array(self.x_dictionary.files)
        self.y_dictionary_files = array(self.y_dictionary.files)

        self.device = device

    def __len__(self):
        """
How many samples in this dataset.

        :return: Length of this dataset.
        """
        return self.length

    def __iter__(self):
        """
Returns an iterable of itself.

        :return: Iterable around this class.
        """
        self.counter = 0
        return self

    def __next__(self):
        """
Intended to be used as iterator.

        :return: Next iteration element.
        """
        self.counter = self.counter + 1
        if self.counter > self.length:
            raise StopIteration()
        return self[self.counter - 1]

    def __getitem__(self, idx):
        """
Get itens from dataset according to idx passed. The return is in numpy arrays.

        :param idx: Index or slice to return.
        :return: 2 elements os 2 lists (x,y) values, according to idx.
        """
        if self.convert_first is True:
            return self.__get_as_tensor__(idx)
        # If we receive an index, return the sample.
        # Else, if receiving an slice or array, return an slice or array from the samples.
        if isinstance(idx, int) or isinstance(idx, int64):
            return self.x_dictionary["arr_" + str(idx)].astype("float32"), self.y_dictionary["arr_" + str(idx)].astype("float32")
        else:
            return [self.x_dictionary[file_name].astype("float32") for file_name in self.x_dictionary_files[idx]], \
                   [self.y_dictionary[file_name].astype("float32") for file_name in self.y_dictionary_files[idx]]

    def __get_as_tensor__(self, idx):
        """
Same as __getitem__, but converting the return into torch.Tensor.

        :param idx: Index or slice to return.
        :return: 2 elements or 2 lists (x,y) values, according to idx.
        """
        # If we receive an index, return the sample.
        # Else, if receiving an slice or array, return an slice or array from the samples.
        if isinstance(idx, int):
            return from_numpy(self.x_dictionary["arr_" + str(idx)].astype("float32")).to(self.device), from_numpy(self.y_dictionary["arr_" + str(idx)].astype("float32")).to(self.device)
        else:
            return [from_numpy(self.x_dictionary[file_name].astype("float32")).to(self.device) for file_name in self.x_dictionary_files[idx]], \
                   [from_numpy(self.y_dictionary[file_name].astype("float32")).to(self.device) for file_name in self.y_dictionary_files[idx]]


class DataManager(Thread):
    def __init__(self, data_loader, buffer_size=3, device=torch.device("cpu"), data_type=torch.float32):
        """
This manager intends to load a PyTorch dataloader like from disk into memory,
reducing the acess time. It does not easily overflow memory, because we set a
buffer size limiting how many samples will be loaded at once. Everytime a sample
is consumed by the calling thread, another one is replaced in the
buffer (unless we reach the end of dataloader).

A manger may be called exactly like a dataloader, an it's based in an internal
thread that loads samples into memory in parallel. This is specially useful
when you are training in GPU and processor is almost idle.

        :param data_loader: Base dataloader to load in parallel.
        :param buffer_size: How many samples to keep loaded (caution to not overflow RAM). Must be integer > 0. Default: 3.
        :param device: Torch device to put samples in, like torch.device("cpu") (default). It saves time by transfering in parallel.
        :param data_type: Automatically casts tensor type. Default: torch.float32.
        """
        super().__init__()
        self.buffer_queue = Queue(maxsize=buffer_size)
        self.data_loader = data_loader
        self.buffer_size = buffer_size
        self.device = device
        self.data_type = data_type

        self.dataloader_finished = False

    def run(self):
        """
Runs the internal thread that iterates over the dataloader until fulfilling the
buffer or the end of samples.
        """
        for i, (x, y) in enumerate(self.data_loader):
            # Important to set before put in queue to avoid race condition
            # would happen if trying to get() in next() method before setting this flag
            if i >= len(self) - 1:
                self.dataloader_finished = True

            self.buffer_queue.put([x.to(dtype=self.data_type, device=self.device),
                                   y.to(dtype=self.data_type, device=self.device)])

    def __iter__(self):
        """
Returns an iterable of itself.

        :return: Iterable around this class.
        """
        self.start()
        self.dataloader_finished = False
        return self

    def __next__(self):
        """
Intended to be used as iterator.

        :return: Next iteration element.
        """
        if self.dataloader_finished is True and self.buffer_queue.empty():
            raise StopIteration()

        return self.buffer_queue.get()

    def __len__(self):
        return len(self.data_loader)

# def tensorboard_and_callbacks(batch_size, log_dir="./logs", model_checkpoint_file="best_weights.{val_loss:.4f}-{epoch:05d}.hdf5", csv_file_path="loss_log.csv"):
#     """
# Utility function to generate Keras tensorboard and others callbacks, deal with
# directory needs and keep the code clean.
#
#     :param batch_size: batch size in training data (needed for compatibility).
#     :param log_dir: Where to save the logs files.
#     :param model_checkpoint_file: File to save weights that resulted best (smallest) validation loss.
#     :param csv_file_path: CSV file path to save loss and validation loss values along the training process.
#     :return: tesnorboard_callback for keras callbacks.
#     """
#     # We need to exclude previous tensorboard and callbacks logs, or it is gone
#     # produce errors when trying to visualize it.
#     try:
#         shutil.rmtree(log_dir)
#     except OSError as e:
#         print("Aviso: %s : %s" % (log_dir, e.strerror))
#
#     try:
#         # Get a list of all the file paths with first 8 letters from model_checkpoint_file.
#         file_list = glob.glob(model_checkpoint_file[:8] + "*")
#
#         # Iterate over the list of filepaths & remove each file.
#         for file_path in file_list:
#             os.remove(file_path)
#     except OSError as e:
#         print("Error: %s : %s" % (model_checkpoint_file, e.strerror))
#
#     try:
#         os.remove(csv_file_path)
#     except OSError as e:
#         print("Error: %s : %s" % (csv_file_path, e.strerror))
#
#     tensorboard_callback = TensorBoard(log_dir=log_dir,
#                                        histogram_freq=1,
#                                        batch_size=batch_size,
#                                        write_grads=True,
#                                        write_graph=True,
#                                        write_images=True,
#                                        update_freq="epoch",
#                                        embeddings_freq=0,
#                                        embeddings_metadata=None)
#
#     model_checkpoint_callback = ModelCheckpoint(filepath=model_checkpoint_file,
#                                                 save_weights_only=True,
#                                                 monitor='val_loss',
#                                                 mode='min',
#                                                 save_best_only=True)
#
#     csv_logger_callback = CSVLogger(csv_file_path, separator=",", append=True)
#
#     return [model_checkpoint_callback, csv_logger_callback]
