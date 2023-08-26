import torch

from constants import device


def test(model, train_loader, val_loader):
    model.eval()
    accuracies = []
    with torch.no_grad():
        for loader in [train_loader, val_loader]:
            total = 0
            correct = 0
            for b_imgs, b_labels in loader:
                total += len(b_labels)
                b_out = model(b_imgs)
                b_pred = torch.argmax(b_out, dim=1)
                correct += (b_pred == b_labels).sum()

            accuracies.append(correct / total)
        return accuracies


def test_verbose(model, train_loader, val_loader):
    test_acc = test(model, train_loader, val_loader)
    print(f'Training accuracy: {test_acc[0]}\n'
          f'Validation accuracy: {test_acc[1]}\n-----')

    return test_acc


def _train_epoch(optimizer, model, loss_fn, train_loader, verbose=True, epoch_n=None):
    total_loss = 0.0
    for b_i, (b_imgs, b_labels) in enumerate(train_loader):
        b_imgs = b_imgs.to(device)
        b_labels = b_labels.to(device)

        b_outputs = model(b_imgs)
        loss = loss_fn(b_outputs, b_labels)

        if verbose and b_i % 30 == 0:
            print(f'Epoch {epoch_n}, Batch {b_i} out of {len(train_loader)}, Loss: {loss}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss

    if verbose:
        print(f'Average training loss: {total_loss / len(train_loader)}')


def train_epochs(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, verbose=True):
    model.train()
    for epoch in range(1, n_epochs + 1):
        _train_epoch(optimizer, model, loss_fn, train_loader, val_loader, verbose, epoch)

        if verbose:
            test_verbose(model, train_loader, val_loader)


def _get_interactive_n_epochs():
    while True:
        try:
            n_epochs = int(input('Enter the number of more epochs to try: '))
            break
        except ValueError as e:
            continue

    return n_epochs


def train_interactive(optimizer, model, loss_fn, train_loader, val_loader, n_epochs_to_pause=100, target_val_acc=None,
                      verbose=True):
    if target_val_acc is None and n_epochs_to_pause is None:
        raise ValueError('At least one boundary should be specified on training')

    epoch_cnt = 1
    while epoch_cnt <= n_epochs_to_pause:
        _train_epoch(optimizer, model, loss_fn, train_loader, verbose, epoch_cnt)
        acc = test_verbose(model, train_loader, val_loader) if verbose else test(model, train_loader, val_loader)

        if acc[1] >= target_val_acc:
            break;

        if epoch_cnt == n_epochs_to_pause:
            new_epochs = _get_interactive_n_epochs()
            n_epochs_to_pause += new_epochs
            
        epoch_cnt += 1
