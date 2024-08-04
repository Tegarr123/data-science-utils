import torch
import torch.utils.data as data_utils
from tqdm.auto import tqdm
from torchmetrics import Accuracy
from sklearn.utils.multiclass import type_of_target
from torch import nn
import copy
import matplotlib.pyplot as plt

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


def train_and_test(model:nn.Module,
                   train_loader:data_utils.DataLoader,
                   test_loader:data_utils.DataLoader,
                   optimizer:torch.optim,
                   loss_fun:nn,
                   epoch:int):
    bestmodel = None
    test_loss_min = 999999
    df_loss_acc = pd.DataFrame(columns=['Train Loss',
                                        'Test Loss',
                                        'Train Accuracy',
                                        'Test Accuracy'])

    for e in range(epoch):
        print(f"EPOCH : {e}")
        train_loss, train_acc = train_data(model=model,
                                            trainloader=train_loader,
                                            optimizer=optimizer,
                                            loss_func=loss_fun)
        test_loss, test_acc = test_data(model=model,
                                        testloader=test_loader,
                                        loss_func=loss_fun)
        
        df_loss_acc.loc[e, ['Train Loss',
                            'Test Loss',
                            'Train Accuracy',
                            'Test Accuracy']] = (train_loss,
                                                 test_loss,
                                                 train_acc,
                                                 test_acc)

        if test_loss <= test_loss_min:
            test_loss_min = test_loss
            bestmodel = copy.deepcopy(model)
    
    fig, ax = plt.subplots(nrows=2,ncols=1)
    ax[0].plot(list(range(1,epoch+1)), df_loss_acc['Train Loss'],label='Train Loss')
    ax[0].plot(list(range(1,epoch+1)), df_loss_acc['Test Loss'],label='Test Loss')
    ax[0].set_title('Loss Function')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(list(range(1,epoch+1)), df_loss_acc['Train Accuracy'],label='Train Accuracy')
    ax[1].plot(list(range(1,epoch+1)), df_loss_acc['Test Accuracy'],label='Test Accuracy')
    ax[1].set_title('Accuracy Metrics')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    return bestmodel, df_loss_acc
            