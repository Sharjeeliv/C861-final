from .base import TextCNN, BasicRNN, BasicLSTM
from .hybrid import CNN_RNN, RNN_CNN, Parallel_CNN_RNN, BiLSTM_CNN

MODELS = {
    "RNN": BasicRNN,
    "LSTM": BasicLSTM,
    "CNN": TextCNN,
    "H1": CNN_RNN,
    "H2": RNN_CNN,
    "H3": Parallel_CNN_RNN,
    "H4": BiLSTM_CNN,
}