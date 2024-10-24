
def train(trainloader, optimizer, net, criterion, scheduler):
    losses = []
    running_loss = 0

    for i, inp in enumerate(trainloader):
        inputs, labels = inp
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0 and i > 0:
            print(f'Loss: ', running_loss / 100)
            running_loss = 0.0

    avg_loss = sum(losses) / len(losses)
    scheduler.step(avg_loss)
    return avg_loss