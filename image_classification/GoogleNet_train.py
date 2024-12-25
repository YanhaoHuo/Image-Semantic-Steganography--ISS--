import torch
from torch import nn
import torch.utils.data
from torchvision.datasets import CIFAR10
from torch.utils.data import ConcatDataset
from Models.DataTransform import data_tr_1, data_transform_1
from Models.GoogleNet import googlenet
from datetime import datetime
import pandas as pd


train_set = CIFAR10('./data', train=True, transform=data_tr_1, download=True) #train_set.data[0].shape (32, 32, 3)
train_set_tr = CIFAR10('./data', train=True, transform=data_transform_1, download=True)
concat_dataset = ConcatDataset([train_set, train_set_tr])
train_data = torch.utils.data.DataLoader(concat_dataset, batch_size=64, shuffle=True)
test_set = CIFAR10('./data', train=False, transform=data_tr_1, download=True)   #image = Image.fromarray(train_set.data[0],'RGB')
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)   #image.save('image.png')

net = googlenet(3, 10, aux_logits=True, training=True, verbose=False)


criterion = nn.CrossEntropyLoss()  #loss function


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def train(net, train_data, valid_data, num_epochs, criterion):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)

    prev_time = datetime.now()
    df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Train Acc', 'Valid Loss', 'Valid Acc', 'Time'])
    #epoch
    lr = 0.01
    for epoch in range(num_epochs):
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        if lr>1e-5 and (epoch+1)%16 == 0:
            lr = lr/10
            print(lr)
        train_loss = 0
        train_acc = 0
        net.train()
        for im, label in train_data:
            with torch.no_grad():
                im = im.to(device)
                label = label.to(device)
            # forward
            output = net(im)

            if net.aux_logits:
                loss1 = criterion(output[0], label)
                loss2 = criterion(output[1], label)
                loss3 = criterion(output[2], label)
                loss = loss1 + 0.1 * loss2 + 0.1 * loss3
            else:
                loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if net.aux_logits:
                train_acc += get_acc(output[0], label)
            else:
                train_acc += get_acc(output, label)
        cur_time = datetime.now()
        # print
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        #val
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net.eval()
            for im, label in valid_data:
                with torch.no_grad():
                    im = im.to(device)
                    label = label.to(device)
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)

            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data)))
            new_row = [epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data), "%02d:%02d:%02d" % (h, m, s)]
            df.loc[epoch] = new_row
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
            new_row = [epoch, train_loss / len(train_data),
                       train_acc / len(train_data), 0, 0, "%02d:%02d:%02d" % (h, m, s)]
            df.loc[epoch] = new_row

        prev_time = cur_time
        print(epoch_str + time_str)

        torch.save(net.state_dict(), 'google_net_add.pkl')
        #df.to_csv('epoch_str.csv')


train(net, train_data, test_data, 100, criterion)

