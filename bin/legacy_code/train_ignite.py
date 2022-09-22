import torch
import pprint
from model import GCN
from dataset import TissueDataset
from torch_geometric.loader import DataLoader
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage, Precision, Recall, RootMeanSquaredError
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar
from torch.utils.tensorboard import SummaryWriter

# https://www.kaggle.com/protan/ignite-example

dataset = TissueDataset("../data")

model = GCN(dataset.num_node_features, hidden_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()


print(dataset.raw_file_names)
print(len(dataset))

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:170]
validation_dataset = dataset[170:205]
test_dataset = dataset[205:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def process_function(engine, batch):
    
    model.train()
    optimizer.zero_grad()
    y_pred = model(batch.x, batch.edge_index, batch.batch).type(torch.DoubleTensor)
    loss = criterion(y_pred.squeeze(), batch.y)
    loss.backward()
    optimizer.step()
    return loss.item()

def eval_function(engine, batch):
    model.eval()
    with torch.no_grad():
        
        y_pred = model(batch.x, batch.edge_index, batch.batch).type(torch.DoubleTensor)
        return y_pred, batch.y


trainer = Engine(process_function)
train_evaluator = Engine(eval_function)
validation_evaluator = Engine(eval_function)

pbar = ProgressBar(persist=True, bar_format="")
pbar.attach(trainer, ['RMSE'])

def score_function(engine):
    val_loss = engine.state.metrics['RMSE']
    return -val_loss
    
handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
validation_evaluator.add_event_handler(Events.COMPLETED, handler)


RunningAverage(output_transform=lambda x: x).attach(trainer, 'RMSE')

rmse = RootMeanSquaredError()

rmse.attach(train_evaluator, "RMSE")
rmse.attach(validation_evaluator, "RMSE")

writer = SummaryWriter(log_dir="../logs")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    pbar.log_message(
        "Training Results - Epoch: {} \nMetrics\n{}"
        .format(engine.state.epoch, pprint.pformat(metrics)))
    writer.add_scalar("training/loss", metrics["RMSE"], engine.state.epoch)

    


def log_validation_results(engine):
    validation_evaluator.run(validation_loader)
    metrics = validation_evaluator.state.metrics
    pbar.log_message(
        "Validation Results - Epoch: {} \nMetrics\n{}"
        .format(engine.state.epoch, pprint.pformat(metrics)))
    pbar.n = pbar.last_print_n = 0
    writer.add_scalar("validation/loss", metrics["RMSE"], engine.state.epoch)

trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

trainer.run(train_loader, max_epochs=100)