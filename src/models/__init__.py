from .base import BaseCNN, BaseRNN, BaseLSTM
from .hybrid import Hybrid1, Hybrid2, Hybrid3, Hybrid4

MODELS = {
    "RNN":  BaseRNN,
    "LSTM": BaseLSTM,
    "CNN":  BaseCNN,
    "H1":   Hybrid1,
    "H2":   Hybrid2,
    "H3":   Hybrid3,
    "H4":   Hybrid4,
}