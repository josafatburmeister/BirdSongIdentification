from torch.utils.data import DataLoader
from torch import Tensor
import torch
import time
import copy
from training import metrics


def train_model(train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, number_classes,
                model, criterion, optimizer, scheduler, num_epochs: int = 25, multi_label_classification=True, multi_label_classification_threshold: float =0.5):
    since: float = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    print('amount of species', len(train_loader.dataset.class_to_idx))
    different_species_count: int = len(train_loader.dataset.class_to_idx)
    best_avg_f1: Tensor = torch.zeros(different_species_count)
    best_min_f1: Tensor = torch.zeros(different_species_count)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('----------')

        # Training Phase
        model.train()
        best_acc = 0.0

        train_metrics = metrics.Metrics(number_classes, multi_label=multi_label_classification)

        for images_train, labels_train in train_loader:
            images_train: Tensor = images_train.to(device)
            labels_train: Tensor = labels_train.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(images_train)

                if multi_label_classification:
                    predictions = torch.sigmoid(outputs).map(lambda x: 0 if x < multilabel_threshold else 1)
                else:
                    _, predictions = torch.max(outputs, 1)

                loss = criterion(outputs, labels_train)

                loss.backward()
                optimizer.step()

            train_metrics.update(predictions, labels_train)

            scheduler.step()

        # Evaluate Model
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        val_metrics = metrics.Metrics(number_classes, multi_label=multi_label_classification)

        for images_val, labels_val in val_loader:
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)

            optimizer.zero_grad()

            outputs = model(images_val)
            if multi_label_classification:
                predictions = torch.sigmoid(outputs).map(lambda x: 0 if x < multi_label_classification_threshold else 1)
            else:
                _, predictions = torch.max(outputs, 1)

            loss = criterion(outputs, labels_train)

            running_loss += loss.item() * images_val.size(0)
            running_corrects += torch.sum(predictions == labels_val.data)
            val_metrics.update(predictions, labels_val)

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('val', epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if torch.min(val_metrics.f1_score()) > torch.min(best_min_f1):
            best_min_f1 = val_metrics.f1_score()
            best_model_min_f1 = copy.deepcopy(model.state_dict())

        if torch.mean(val_metrics.f1_score()) > torch.mean(best_avg_f1):
            best_avg_f1 = val_metrics.f1_score()
            best_model_avg_f1 = copy.deepcopy(model.state_dict())

        print('f1 score:', val_metrics.f1_score())
        print('f1 avg:', torch.mean(val_metrics.f1_score()))
        print('f1 min:', torch.min(val_metrics.f1_score()))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model
