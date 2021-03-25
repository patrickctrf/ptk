import glob
import os
import shutil
from queue import Queue
from threading import Thread

import torch
from numpy import load, arange, random, array, int64, absolute, asarray, ones, argwhere, cumsum, hstack, save, zeros, where, array_split
from pandas import read_csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from torch import from_numpy, cat
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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
        :param buffer_size: How many samples to keep loaded (caution to not overflow RAM). Default: 3.
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


class AsymetricalTimeseriesDataset(Dataset):
    def __init__(self, x_csv_path, y_csv_path, max_window_size=200, min_window_size=10, convert_first=False, device=torch.device("cpu"), shuffle=True):
        super().__init__()
        self.input_data = read_csv(x_csv_path).to_numpy()
        self.output_data = read_csv(y_csv_path).to_numpy()
        self.min_window_size = min_window_size
        self.convert_first = convert_first
        self.device = device

        # =========SCALING======================================================
        # features without timestamp (we do not scale timestamp)
        input_features = self.input_data[:, 1:]
        output_features = self.output_data[:, 1:]

        # Scaling data
        self.input_scaler = StandardScaler()
        input_features = self.input_scaler.fit_transform(input_features)
        self.output_scaler = MinMaxScaler()
        output_features = self.output_scaler.fit_transform(output_features)

        # Replacing scaled data (we kept the original TIMESTAMP)
        self.input_data[:, 1:] = input_features
        self.output_data[:, 1:] = output_features
        # =========end-SCALING==================================================

        # Save timestamps for syncing samples.
        self.input_timestamp = self.input_data[:, 0]
        self.output_timestamp = self.output_data[:, 0]

        # Throw out timestamp, we are not going to RETURN this.
        self.input_data = self.input_data[:, 1:]
        self.output_data = self.output_data[:, 1:]

        # We are calculating the number of samples that each window size produces.
        # We discount the window SIZE from the total number of samples (timeseries).
        # P.S.: It MUST use OUTPUT shape, because unlabeled data doesnt not help us.
        # P.S. 2: The first window_size is min_window_size, NOT 1.
        n_samples_per_window_size = ones((max_window_size - min_window_size,)) * self.output_data.shape[0] - arange(min_window_size + 1, max_window_size + 1)

        # Now, we know the last index where we can sample for each window size.
        # Concatenate element [0] in the begining to avoid error on first indices.
        self.last_window_sample_idx = hstack((array([0]), cumsum(n_samples_per_window_size))).astype("int")

        self.length = int(n_samples_per_window_size.sum())
        self.indices = arange(self.length)

        self.shuffle_array = arange(self.length)
        if shuffle is True: random.shuffle(self.shuffle_array)

        return

    def __getitem__(self, idx):
        """
Get itens from dataset according to idx passed. The return is in numpy arrays.

        :param idx: Index or slice to return.
        :return: 2 elements or 2 lists (x,y) values, according to idx.
        """

        # If we receive an index, return the sample.
        # Else, if receiving an slice or array, return an slice or array from the samples.
        if isinstance(idx, int) or isinstance(idx, int64):

            if idx >= len(self) or idx < 0:
                raise IndexError('Index out of range')

            # shuffling indices before return
            idx = self.shuffle_array[idx]

            argwhere_result = argwhere(self.last_window_sample_idx < idx)
            window_size = self.min_window_size + (argwhere_result[-1][0] if argwhere_result.size != 0 else 0)

            window_start_idx = idx - self.last_window_sample_idx[(argwhere_result[-1][0] if argwhere_result.size != 0 else 0)]

            _, x_start_idx = find_nearest(self.input_timestamp, self.output_timestamp[window_start_idx])
            _, x_finish_idx = find_nearest(self.input_timestamp, self.output_timestamp[window_start_idx + window_size])

            x = self.input_data[x_start_idx: x_finish_idx + 1]
            y = self.output_data[window_start_idx + window_size] - self.output_data[window_start_idx]

            # If we want to convert into torch tensors first
            if self.convert_first is True:
                return from_numpy(x.astype("float32")).to(self.device), \
                       from_numpy(y.astype("float32")).to(self.device)
            else:
                return x, y
        else:
            # If we received a slice(e.g., 0:10:-1) instead an single index.
            return self.__getslice__(idx)

    def __getslice__(self, slice_from_indices):
        return list(zip(*[self[i] for i in self.indices[slice_from_indices]]))

    def __len__(self):
        return self.length


class BatchTimeseriesDataset(Dataset):
    def __init__(self, x_csv_path, y_csv_path, max_window_size=200, min_window_size=10, convert_first=False, device=torch.device("cpu"), shuffle=True, batch_size=1):
        super().__init__()
        self.batch_size = batch_size

        self.base_dataset = \
            AsymetricalTimeseriesDataset(x_csv_path=x_csv_path,
                                         y_csv_path=y_csv_path,
                                         max_window_size=max_window_size,
                                         min_window_size=min_window_size,
                                         convert_first=convert_first,
                                         device=device,
                                         shuffle=False)

        try:
            tabela = load("tabela_elementos_dataset.npy")

        except FileNotFoundError as e:
            tabela = zeros((len(self.base_datasetdataset),))
            i = 0
            for element in tqdm(self.base_dataset):
                tabela[i] = element[0].shape[0]
                i = i + 1
            save("tabela_elementos_dataset.npy", tabela)

        # dict_count = Counter(tabela)
        # ocorrencias = array(list(dict_count.values()))

        # Os arrays nesta lista contem INDICES para elementos do dataset com
        # mesmo comprimento.
        self.lista_de_arrays_com_mesmo_comprimento = []

        # grupos de arrays com mesmo comprimento.
        # Eles serao separados em batches, entao alguns
        # batches sao de arrays com mesmo comprimento que outros. Alguns
        # batches serao um pouco maiores, pois a quantidade de elementos com o
        # mesmo tamanho talvez nao seja um multiplo inteiro do batch_size
        # escolhido
        for i in range(tabela.min().astype("int"), tabela.max().astype("int") + 1):
            self.lista_de_arrays_com_mesmo_comprimento.extend(
                array_split(where(tabela == i)[0],
                            where(tabela == i)[0].shape[0] // self.batch_size + (where(tabela == i)[0].shape[0] % self.batch_size > 0))
            )

        self.length = len(self.lista_de_arrays_com_mesmo_comprimento)

        self.shuffle_array = arange(self.length)

        if shuffle is True:
            random.shuffle(self.shuffle_array)

        return

    def __getitem__(self, idx):
        """
Get itens from dataset according to idx passed. The return is in numpy arrays.

        :param idx: Index to return.
        :return: 2 elements (batches), according to idx.
        """

        # If we are shuffling indices, we do it here. Else, we'll just get the
        # same index back
        idx = self.shuffle_array[idx]

        # concatenate tensor in order to assemble batches
        x_batch = \
            cat([self.base_dataset[dataset_element_idx][0].unsqueeze(0)
                 for dataset_element_idx
                 in self.lista_de_arrays_com_mesmo_comprimento[idx]], 0)
        y_batch = \
            cat([self.base_dataset[dataset_element_idx][1].unsqueeze(0)
                 for dataset_element_idx
                 in self.lista_de_arrays_com_mesmo_comprimento[idx]], 0)

        return x_batch, y_batch

    def __len__(self):
        return self.length


class CustomDataLoader(object):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.dataloader = DataLoader(*args, **kwargs)
        self.iterable = None

    def __iter__(self):
        self.iterable = iter(self.dataloader)
        return self

    def __len__(self):
        return len(self.dataloader)

    def __next__(self):
        next_sample = next(self.iterable)
        return next_sample[0].view(next_sample[0].shape[1:]), \
               next_sample[1].view(next_sample[1].shape[1:])
