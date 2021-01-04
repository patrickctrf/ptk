import glob
import os
import shutil
from queue import Queue
from threading import Thread, Event

import torch
from numpy import load, arange, random, array, int64, absolute, asarray
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from torch import from_numpy
from torch.nn.utils.rnn import pack_sequence
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
    array_to_search = asarray(array_to_search)
    idx = (absolute(array_to_search - value)).argmin()
    return array_to_search[idx], idx


def tensorboard_and_callbacks(batch_size, log_dir="./logs", model_checkpoint_file="best_weights.{val_loss:.4f}-{epoch:05d}.hdf5", csv_file_path="loss_log.csv"):
    """
Utility function to generate Keras tensorboard and others callbacks, deal with
directory needs and keep the code clean.

    :param batch_size: batch size in training data (needed for compatibility).
    :param log_dir: Where to save the logs files.
    :param model_checkpoint_file: File to save weights that resulted best (smallest) validation loss.
    :param csv_file_path: CSV file path to save loss and validation loss values along the training process.
    :return: tesnorboard_callback for keras callbacks.
    """
    # We need to exclude previous tensorboard and callbacks logs, or it is gone
    # produce errors when trying to visualize it.
    try:
        shutil.rmtree(log_dir)
    except OSError as e:
        print("Aviso: %s : %s" % (log_dir, e.strerror))

    try:
        # Get a list of all the file paths with first 8 letters from model_checkpoint_file.
        file_list = glob.glob(model_checkpoint_file[:8] + "*")

        # Iterate over the list of filepaths & remove each file.
        for file_path in file_list:
            os.remove(file_path)
    except OSError as e:
        print("Error: %s : %s" % (model_checkpoint_file, e.strerror))

    try:
        os.remove(csv_file_path)
    except OSError as e:
        print("Error: %s : %s" % (csv_file_path, e.strerror))

    tensorboard_callback = TensorBoard(log_dir=log_dir,
                                       histogram_freq=1,
                                       batch_size=batch_size,
                                       write_grads=True,
                                       write_graph=True,
                                       write_images=True,
                                       update_freq="epoch",
                                       embeddings_freq=0,
                                       embeddings_metadata=None)

    model_checkpoint_callback = ModelCheckpoint(filepath=model_checkpoint_file,
                                                save_weights_only=True,
                                                monitor='val_loss',
                                                mode='min',
                                                save_best_only=True)

    csv_logger_callback = CSVLogger(csv_file_path, separator=",", append=True)

    return [model_checkpoint_callback, csv_logger_callback]


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


class PackingSequenceDataloader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        """
Utility class for loading data with input already formated as packed sequences
(for PyTorch LSTMs, for example).

        :param dataset: PyTorch-like Dataset to load.
        :param batch_size: Mini batch size.
        :param shuffle: Shuffle data.
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Esta equacao apenas ARREDONDA a quantidade de mini-batches para CIMA
        # Quando o resultado da divisao nao eh inteiro (sobra resto), significa
        # apenas que o ultimo batch sera menor que os demais
        self.length = len(dataset) // self.batch_size + (len(dataset) % self.batch_size > 0)

        self.shuffle_array = arange(self.length)
        if shuffle is True: random.shuffle(self.shuffle_array)

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

        :return: Tuple containing (input packed sequence, output targets)
        """
        if self.counter >= self.length:
            raise StopIteration()

        mini_batch_input = pack_sequence(self.dataset[self.shuffle_array[self.counter:self.counter + self.batch_size]][0], enforce_sorted=False)
        mini_batch_output = torch.stack(self.dataset[self.shuffle_array[self.counter:self.counter + self.batch_size]][1])
        self.counter = self.counter + 1

        return mini_batch_input, mini_batch_output

    def __len__(self):
        return self.length


class DataManager(Thread):
    def __init__(self, data_loader, buffer_size=3, device=torch.device("cpu"), data_type=torch.float32, stop_event=Event(), update_rate=1.0):
        super().__init__()
        self.buffer_queue = Queue(maxsize=buffer_size)
        self.data_loader = data_loader
        self.buffer_size = buffer_size
        self.device = device
        self.data_type = data_type
        self.stop_event = stop_event
        self.update_rate = update_rate

        self.dataloader_finished = False

    def run(self):
        for i, (x, y) in enumerate(self.data_loader):
            # Important to set before put in queue to avoid race condition
            # would happen if trying to get() in next() method before setting this flag
            if i >= len(self) - 1:
                self.dataloader_finished = True

            self.buffer_queue.put([x.to(self.data_type).to(self.device),
                                   y.to(self.data_type).to(self.device)])

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
