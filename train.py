import torch

from constants import device


def test(model, loader):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for b_imgs, b_labels in loader:
            total += len(b_labels)
            b_out = model(b_imgs)
            b_pred = torch.argmax(b_out, dim=1)
            correct += (b_pred == b_labels).sum()

        return correct / total


def train_epochs(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, verbose=True):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for b_i, (b_imgs, b_labels) in enumerate(train_loader):
            b_imgs = b_imgs.to(device)
            b_labels = b_labels.to(device)

            b_outputs = model(b_imgs)
            loss = loss_fn(b_outputs, b_labels)

            if verbose and b_i % 20 == 0:
                print(f'Epoch {epoch} out of {n_epochs}, Batch {b_i} out of {len(train_loader)}, Loss: {loss}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss

        if verbose:
            print(f'--Epoch: {epoch}\n'
                  f'Train loss: {total_loss / len(train_loader)}\n'
                  f'Validation accuracy: {test(model, val_loader)}--')
