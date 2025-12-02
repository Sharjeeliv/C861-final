from .models import TextCNN, BasicRNN, BasicLSTM

MODELS = {
    "RNN": BasicRNN,
    "LSTM": BasicLSTM,
    "CNN": TextCNN,
}