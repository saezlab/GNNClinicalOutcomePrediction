from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage, Precision, Recall
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar