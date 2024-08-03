import torch
import torch.utils.data as data_utils
from tqdm.auto import tqdm
from torchmetrics import Accuracy
from sklearn.utils.multiclass import type_of_target
def train_data(model:nn.Module,
               trainloader:data_utils.DataLoader,
               optimizer:torch.optim,
               loss_func:torch.nn
               ):
    class_type = str(type_of_target(next(iter(trainloader))[1]))  
    train_loss = 0
    train_accuracy = 0
    for X, y in trainloader:
        model.train()

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_func(pred,y.unsqueeze(dim=-1))
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            match(class_type):
                case 'binary':
                    eval_predict = pred.round()
                    train_acc = Accuracy(task=class_type)
                    train_accuracy += train_acc(eval_predict,y.unsqueeze(dim=-1)).item()
                case 'multiclass':
                    eval_predict = pred.argmax(dim=-1)
                    train_acc = Accuracy(task=class_type)
                    train_accuracy += train_acc(eval_predict,y.unsqueeze(dim=-1)).item()
                case _:
                    raise TypeError(f'type {class_type} error')
    train_loss /= len(trainloader)
    train_accuracy /= len(trainloader)

    return train_loss, train_accuracy
def test_data(model:nn.Module,
              testloader:data_utils.DataLoader,
              loss_func:torch.nn):
    test_loss = 0
    test_accuracy = 0
    class_type = type_of_target(next(iter(testloader))[1])
    model.eval()
    with torch.inference_mode():
        for X, y in testloader:
            eval_predict = model(X)
            test_loss += loss_func(eval_predict,y.unsqueeze(dim=-1)).item()
            match(class_type):
                case 'binary':
                    test_acc = Accuracy(task=class_type)
                    test_accuracy += test_acc(eval_predict.round(),y.unsqueeze(dim=-1)).item()
                case 'multiclass':
                    test_acc = Accuracy(task=class_type)
                    test_accuracy += test_acc(eval_predict.argmax(dim=-1),y.unsqueeze(dim=-1)).item()
                case _:
                    raise TypeError(f'type {class_type} error')
    test_loss /= len(testloader)
    test_accuracy /= len(testloader)
    return test_loss, test_accuracy