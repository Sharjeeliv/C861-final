# General utility methods
import time

def time_execution(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        m, s = divmod(execution_time, 60)
        ms = (execution_time - int(execution_time)) * 1000
        print(f"\033[91;1mEXECUTION TIME: {int(m):02}:{int(s):02}.{int(ms):03}\033[0m")
        return result
    return wrapper

# ********************************
# EARLY STOPPING IMPLEMENTATION
# ********************************
class EarlyStopping:
    def __init__(self, patience=5, delta=1e-3):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter < self.patience: return
            self.early_stop = True

        else:
            self.best_score = score
            self.counter = 0