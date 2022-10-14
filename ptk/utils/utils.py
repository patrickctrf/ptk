import os
from queue import Queue
from threading import Thread

import numpy as np
import torch

__all__ = ["split_into_chunks", "GenericDatasetFromFiles",
           "DataManager"]


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


class GenericDatasetFromFiles(torch.utils.data.Dataset):
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
        self.x_dictionary = np.load(os.path.join(data_path, "x_data.npz"), mmap_mode=mmap_mode)
        self.y_dictionary = np.load(os.path.join(data_path, "y_data.npz"), mmap_mode=mmap_mode)
        self.length = len(self.x_dictionary)
        self.convert_first = convert_first

        self.x_dictionary_files = np.array(self.x_dictionary.files)
        self.y_dictionary_files = np.array(self.y_dictionary.files)

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
        if isinstance(idx, int) or isinstance(idx, np.int64):
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
            return torch.from_numpy(self.x_dictionary["arr_" + str(idx)].astype("float32")).to(self.device), torch.from_numpy(self.y_dictionary["arr_" + str(idx)].astype("float32")).to(self.device)
        else:
            return [torch.from_numpy(self.x_dictionary[file_name].astype("float32")).to(self.device) for file_name in self.x_dictionary_files[idx]], \
                   [torch.from_numpy(self.y_dictionary[file_name].astype("float32")).to(self.device) for file_name in self.y_dictionary_files[idx]]


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
        for i, sample in enumerate(self.data_loader):
            # Important to set before put in queue to avoid race condition
            # would happen if trying to get() in next() method before setting this flag
            if i >= len(self) - 1:
                self.dataloader_finished = True

            self.buffer_queue.put([x.to(dtype=self.data_type, device=self.device, non_blocking=True) for x in sample])

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
