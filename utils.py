def split_into_chunks(list, split_dims):
    """
Split a list into evenly sized chunks. The last chunk will be smaller if the original list length is not divisible by 'split_dims'.

    :param list: List to be split.
    :param split_dims: Length of each split chunk.
    """
    aux_list = []
    # For item i in a range that is a length of l,
    for i in range(0, len(list), split_dims):
        # Create an index range for l of n items:
        aux_list.append(list[i:i + split_dims])

    return aux_list


def tensorboard_and_callbacks(batch_size, log_dir="./logs", model_checkpoint_file="best_weights.{val_loss:.4f}-{epoch:05d}.hdf5", csv_file_path="loss_log.csv"):
    """
Utility function to generate Keras tensorboard and others callbacks, deal with directory needs and keep the code clean.

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
