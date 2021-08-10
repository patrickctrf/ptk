from numpy import ones, where, array, array_split, hstack, vstack

__all__ = ["timeseries_split", "TimeSeriesSplitCV"]


def timeseries_split(data_x, data_y=None, sampling_window_size=10, n_steps_prediction=1, stateful=False, enable_asymetrical=False, is_classifier=False, threshold=1):
    """
Split the given time series into input (X) and observed (y) data.
There are 3 principal modes of splitting data with this function: Stateful univariate series, non stateful univariate series, and non stateful multivariate series.

When you have the input data and the output data inside the same array, we consider it as a univariate series and all data is passed through the data_x parameters.
If the input data does not contain your observed information (y), the input array is given in the data_x parameter and the prediction target goes to the data_y parameter.

It's also available a stateful splitting mode if you are working with a stateful LSTM, for example.

There is no stateful multivariate series option because, in this case, your input array or matrix (X) and your observed array or matrix data (y) would be already given to you by definition.

    :param data_x: array-like input data from your series.
    :param data_y: array-like observed data (target) for your series. Let it be 'None' if dealing with a single component (univariate) series.
    :param sampling_window_size: Size of window sampling (W) or time steps entering your network for each prediction. Must be positive integer. Ignored if enable_asymetrical is True.
    :param n_steps_prediction: How many steps it is going to predict ahead. Must be positive integer.
    :param stateful: True or False, indicating whether your network are suposedto work statefully or not, respectively.
    :param enable_asymetrical: Whenever to return asymetrical sequences in X output data.
    :param is_classifier: If True, the 'y' output data is transformed into +1 or 0, according to 'threshold' selected by user. Useful for non quantitative prediction (classification, not regression).
    :param threshold: Threshold for 'is_classifier' parameter. Float in the interval of your observed data 'data_y'.
    :return: X input and y observed data, formatted according to the given function parameters.
    """

    # Se estivermos trabalhando com o caso asimetrico, nao faz sentido falar em
    # janela de amostragem, portanto zeramos para nao precisar alterar o resto
    # do algoritmo.
    if enable_asymetrical is True:
        sampling_window_size = 0

    if data_y is None:
        data_y = data_x

    # If we receive arrays of shape N, it should acttually be shape (N,1).
    # Matrix arrays are already properly formatted.
    if len(data_x.shape) == 1:
        data_x = data_x.reshape(-1, 1)
    if len(data_y.shape) == 1:
        data_y = data_y.reshape(-1, 1)

    # Converte os dados para array numpy, para podermos utilizar a API dos arrays.
    data_x = array(data_x)
    data_y = array(data_y)

    if stateful is False or enable_asymetrical is True:
        # arrays para armazenar as saidas
        # Teremos a quantidade de samples - quantos steps prevemos a frente - o tamanho da janela de predicao + 1 (porque a predicao eh feita sobre ultimo elemento da janela).
        X = ones((1 + data_x.shape[0] - n_steps_prediction - sampling_window_size, sampling_window_size) + data_x.shape[1:])
        y = ones((1 + data_y.shape[0] - n_steps_prediction - sampling_window_size, n_steps_prediction) + data_y.shape[1:])

        # Para o caso assimetrico, X tera arrays de tamanhos diferentes e
        # precisamos portnto de uma lista para armazena-los antes de converter
        # em uma matriz assimetrica numpy.
        X_asymmetric = (1 + data_x.shape[0] - n_steps_prediction - sampling_window_size) * [0]

        i = 0
        # Precisamos de N amostras (sampling_window_size) e as amostras de
        # ground truth (n_steps_prediction) para fazer mais um split.
        while i < 1 + data_x.shape[0] - n_steps_prediction - sampling_window_size:
            X[i] = (data_x[i:i + sampling_window_size])
            y[i] = (data_y[i + sampling_window_size:(i + sampling_window_size) + n_steps_prediction])

            # Sequencias assimetricas sao compostas do elemento atual a ser
            # adicionado concatenado com os anteriores.
            if i == 0:
                X_asymmetric[i] = data_x[i:i + 1]
            else:
                X_asymmetric[i] = vstack((X_asymmetric[i - 1], data_x[i:i + 1]))

            i += 1

        # Apos acabar o loop while, convertemos a LISTA X_asymmetric em uma
        # matriz ASSIMETRICA (cada elemento eh um array de tamanho diferente).
        X_asymmetric = array(X_asymmetric, dtype=object)

    # Se estivermos trabalhando com uma serie stateful simetrica, a unica
    # demanda eh deslocar a entrada e a saida em uma unidade (a entrada eh tudo
    # o que vimos ate agora e a saida eh o proximo step, que sera conhecido no
    # passo seguinte).
    else:
        X = data_x[:-1]
        y = data_y[1:]

    if is_classifier is True:
        y = where(y > threshold, 1.0, 0.0)

    if enable_asymetrical is True:
        return X_asymmetric, y

    return X, y


class TimeSeriesSplitCV(object):
    def __init__(self, n_splits=5, training_percent=0.7, sampling_window_size=10, n_steps_prediction=1, stateful=False, is_classifier=False, threshold=1, blocking_split=False):
        """
Time series split with cross validation separation as a compatible sklearn-like splitter.
There are 3 principal modes of splitting data with this function: Stateful univariate series, non stateful univariate series, and non stateful multivariate series.

When you have the input data and the output data inside the same array, we consider it as a univariate series and all data is passed through the data_x parameters.
If the input data does not contain your observed information (y), the input array is given in the data_x parameter and the prediction target goes to the data_y parameter.

It's also available a stateful splitting mode if you are working with a stateful LSTM, for example.

There is no stateful multivariate series option because, in this case, your input array or matrix (X) and your observed array or matrix data (y) would be already given to you by definition.


        :param n_splits: Like k-folds split, how many sub series to split.
        :param training_percent: Ratio between train and validation data for cross validation.
        :param sampling_window_size: Size of window sampling (W) or time steps entering your network for each prediction. Must be positive integer.
        :param n_steps_prediction: How many steps it is going to predict ahead. Must be positive integer.
        :param stateful: True or False, indicating whether your network are suposedto work statefully or not, respectively.
        :param is_classifier: If True, the 'y' output data is transformed into +1 or 0, according to 'threshold' selected by user. Useful for non quantitative prediction (classification, not regression).
        :param threshold: Threshold for 'is_classifier' parameter. Float in the interval of your observed data 'data_y'.
        :return: X input and y observed data, formatted according to the given function parameters.
        """
        super().__init__()
        self.n_splits = n_splits
        self.training_percent = training_percent
        self.sampling_window_size = sampling_window_size
        self.n_steps_prediction = n_steps_prediction
        self.stateful = stateful
        self.is_classifier = is_classifier
        self.threshold = threshold
        self.blocking_split = blocking_split

    def get_n_splits(self, X=None, y=None, groups=None):
        """
Returns the number of splitting iterations in the cross-validator

        :param X: Always ignored, exists for compatibility.
        :param y: Always ignored, exists for compatibility.
        :param groups: Always ignored, exists for compatibility.
        :return: Number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
Generate indices to split data into training and test set.

        :param X: array-like input data from your series.
        :param y: array-like observed data (target) for your series. Let it be 'None' if dealing with a single component series.
        :param groups: Always ignored, exists for compatibility.
        """
        X = array(X)
        y = array(y)

        # A API do numpy ja providencia uma funcao para realizarmos o split de
        # um array em partes iguais com N splits. Se nao for um array que se
        # divide em N partes iguais, a ultima sera menor que as demais.
        splits_indices = array_split(range(X.shape[0]), self.n_splits)

        # listas para retorno
        train_list = []
        test_list = []

        # A lista de acumulada somente sera utilizada em casa de cross
        # validation nao blocking.
        accumulate_list = []

        for i, single_split in enumerate(splits_indices):
            train = single_split[:int(single_split.shape[0] * self.training_percent)]
            test = single_split[int(single_split.shape[0] * self.training_percent):]

            train_list.append(train)
            test_list.append(test)

            # Se esivermos trabalhando a forma classica (nao blocking) da cross
            # validation, temos que fazer uma list com o acumulado do treino ate entao
            if self.blocking_split is False:
                if i > 0:
                    accumulate_list.append(hstack((accumulate_list[i - 1], test, train)))
                else:
                    accumulate_list.append(train)

        # Ha duas principais maneiras de fazer cross validation em series
        # temporais, sendo a primeira classica e a segunda usando tamanhos
        # iguais para todos os splits (blocking).
        # https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/
        if self.blocking_split is False:
            return zip(accumulate_list, test_list)
        else:
            return zip(train_list, test_list)
