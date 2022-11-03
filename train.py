import torch
import optuna
import random
import argparse
import numpy as np

from copy import deepcopy
from torch_geometric.nn import (
    GCN, 
    GraphSAGE,
    GAT,
    GraphUNet
)
from utils.custom_dataset import CustomDataset
from training.engine_gnn import train, test
from utils.metrics import accuracy, roc_auc



def parse_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='dataset .npz')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.') #Default seed same as GCNII
    parser.add_argument('--steps', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--num-layers-min', type=int, default=2, help='Number of layers.')
    parser.add_argument('--num-layers-max', type=int, default=5, help='Number of layers.')
    parser.add_argument('--hidden-dim-min',  type=int, default=32, help='min hidden dimensions.')
    parser.add_argument('--hidden-dim-max',  type=int, default=128, help='min hidden dimensions.')
    parser.add_argument('--hidden-dim-step', type=int, default=32, help='min hidden dimensions.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--lr-min', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-max', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--weight-decay-min', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--weight-decay-max', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of optuna trials')
    args = parser.parse_args()
    return args


def run_on_split(
    data,
    model,
    optimizer,
    train_mask,
    val_mask,
    test_mask,
    args,
    metric=accuracy
):
    features = data.x
    labels = data.y

    num_features = features.shape[1]
    num_labels = len(torch.unique(labels))

    best = torch.inf
    best_params = None
    for step in range(args.steps):
        loss_train, metric_train = train(model, optimizer, data, train_mask, metric)
        loss_val, metric_val = test(model, data, val_mask, metric)

        if step % args.log_freq == 0 and args.log_freq > 0:
            print(f'Train acc {metric_train:.3f} / Val acc {metric_val:.3f}')

        if loss_val < best:
            best = loss_val
            bad_counter = 0
            best_params = deepcopy(model.state_dict())
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    # load best params
    model.load_state_dict(best_params)
    loss_test, metric_test = test(model, data, test_mask, metric)
    # return test accuracy
    return metric_test



if __name__ == '__main__':
    args = parse_args()
    args.log_freq = -1
    # fix seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    dataset = CustomDataset(args.data_path)
    data = dataset[0].to(device)
    num_classes = len(torch.unique(data.y))

    metric = accuracy if len(torch.unique(data.y)) > 2 else roc_auc

    def objective(trial: optuna.Trial):
        num_layers = trial.suggest_int(
            'num_layers', 
            args.num_layers_min, 
            args.num_layers_max
        )

        hidden_channels = trial.suggest_int(
            'hidden_dim', 
            low=args.hidden_dim_min, 
            high=args.hidden_dim_max, 
            step=args.hidden_dim_step,
        )

        lr = trial.suggest_float(
            'lr',
            low=args.lr_min,
            high=args.lr_max,
            log=True,
        )

        weight_decay = trial.suggest_float(
            'weight_decay',
            low=args.weight_decay_min,
            high=args.weight_decay_max,
            log=True,
        )

        activation = trial.suggest_categorical('activation', ['relu', 'gelu', 'tanh'])
        
        num_splits = data.train_mask.shape[1]
        test_accs = []
        for i in range(num_splits):
            print(f'Split [{i+1}/{num_splits}]')
            # create model
            model = GraphSAGE(
                in_channels=data.x.shape[1],
                hidden_channels=hidden_channels, 
                num_layers=num_layers,
                out_channels=num_classes,
                act=activation
            ).to(device)
            # create optimizer
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
            train_mask, val_mask, test_mask = \
                data.train_mask[:, i], data.val_mask[:, i], data.test_mask[:, i]
            test_acc = run_on_split(data, model, optimizer, train_mask, val_mask, test_mask, args, metric)
            print(f'Test accuracy {test_acc:.3f}')
            test_accs.append(100 * test_acc)
        # return mean test accuracy among splits
        return np.mean(test_accs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)
    
    best_trial = study.best_trial

    print('Accuracy: {}'.format(best_trial.value))
    print("Best hyperparameters: {}".format(best_trial.params))
