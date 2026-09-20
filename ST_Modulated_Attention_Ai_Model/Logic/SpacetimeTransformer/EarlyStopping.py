class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.counter = 0
        elif val_loss < self.best_loss - self.min_delta:
            # 有改善，重置计数器
            self.best_loss = val_loss
            self.counter = 0
        else:
            # 无改善，计数器加1
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop