import torch
import time
import copy
from torch.nn.functional import softmax


def train_model(train_loader, val_loader, test_loader,
                model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    print('amount of species', len(train_loader.dataset.class_to_idx))
    different_species_count = len(train_loader.dataset.class_to_idx)
    best_avg_f1 = torch.zeros(different_species_count)
    best_min_f1 = torch.zeros(different_species_count)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('----------')

        # Training Phase
        model.train()
        running_loss = 0.0
        best_acc = 0.0
        train_tp = torch.zeros(different_species_count)
        train_fp = torch.zeros(different_species_count)
        train_fn = torch.zeros(different_species_count)

        for images_train, labels_train in train_loader:
            images_train = images_train.to(device)
            labels_train = labels_train.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                images_out = model(images_train)
                _, preds = torch.max(images_out, 1)
                loss = criterion(images_out, labels_train)

                loss.backward()
                optimizer.step()

            # running_loss += loss.item() * images_train.size(0)
            # tp += torch.sum(preds == labels_train.data)
            for p, l in zip(preds, labels_train.data):
                if p == l:
                    train_tp += torch.zeros(different_species_count).scatter_(0, p, 1)
                else:
                    train_fp += torch.zeros(different_species_count).scatter_(0, p, 1)
                    train_fn += torch.zeros(different_species_count).scatter_(0, l, 1)

            scheduler.step()

            # epoch_loss = running_loss / len(train_set)
            # epoch_acc = running_corrects.double() / len(train_set)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     'train', epoch_loss, epoch_acc))

        # Evaluate Model
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        val_tp = torch.zeros(different_species_count)
        val_fp = torch.zeros(different_species_count)
        val_fn = torch.zeros(different_species_count)

        for images_val, labels_val in val_loader:
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)

            optimizer.zero_grad()

            outputs = model(images_val)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels_val)

            running_loss += loss.item() * images_val.size(0)
            running_corrects += torch.sum(preds == labels_val.data)
            for p, l in zip(preds, labels_val.data):
                if p == l:
                    val_tp += torch.zeros(different_species_count).scatter_(0, p, 1)
                else:
                    val_fp += torch.zeros(different_species_count).scatter_(0, p, 1)
                    val_fn += torch.zeros(different_species_count).scatter_(0, l, 1)

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
           'val', epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        precision = val_tp / (val_tp + val_fp)
        recall = val_tp / (val_tp + val_fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

        if torch.min(f1_score) > torch.min(best_min_f1):
            best_min_f1 = f1_score
            best_model_min_f1 = copy.deepcopy(model.state_dict())

        if torch.mean(f1_score) > torch.mean(best_avg_f1):
            best_avg_f1 = f1_score
            best_model_avg_f1 = copy.deepcopy(model.state_dict())

        print('f1 score:', f1_score)
        print('f1 avg:', torch.mean(f1_score))
        print('f1 min:', torch.min(f1_score))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model



