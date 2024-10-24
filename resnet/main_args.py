import torch.nn as nn
import torch.optim as optim
from data.data_loader import get_data_loader
from model.ResNet import Bottleneck, ResNet, ResNet50
from train import train
from test import test
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', '--gpu', dest='gpu', type=int, default=0)
parser.add_argument('-m', '--momentum', dest='momentum', type=float, default=0.9)
parser.add_argument('-w', '--weight_decay', dest='weight_decay', type=float, default=0.0001)
parser.add_argument('-f', '--factor', dest='factor', type=float, default=0.1)
parser.add_argument('-lr', '--lr', dest='lr', type=float, default=0.1)
parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=128)
parser.add_argument('-p', '--patience', dest='patience', type=int, default=5)
parser.add_argument('-e', '--max_epoch', dest='max_epoch', type=int, default=100)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    trainloader, testloader = get_data_loader(batch_size=args.batch_size)
    net = ResNet50(10).to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
    EPOCHS = args.max_epoch

    for epoch in range(EPOCHS):
        print('epoch:{}'.format(epoch))
        train_loss = train(trainloader, optimizer, net, criterion, scheduler)
        accuracy = test(testloader, net)
        print('Train Loss: {}, Test accuracy: {}%'.format(train_loss, 100 * accuracy))