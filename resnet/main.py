import torch.nn as nn
import torch.optim as optim
from data.data_loader import get_data_loader
from model.ResNet import Bottleneck, ResNet, ResNet50
from train import train
from test import test

if __name__ == "__main__":
    trainloader, testloader = get_data_loader(batch_size=128)
    net = ResNet50(10).to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)
    EPOCHS = 5

    for epoch in range(EPOCHS):
        print('epoch:{}'.format(epoch))
        train_loss = train(trainloader, optimizer, net, criterion, scheduler)
        accuracy = test(testloader, net)
        print('Train Loss: {}, Test accuracy: {}%'.format(train_loss, 100 * accuracy))