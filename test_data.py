from dataset import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders()
signals, labels = next(iter(train_loader))

print('Signal shape:', signals.shape)
print('Label shape: ', labels.shape)
print('Signal range:', round(signals.min().item(), 2), 'to', round(signals.max().item(), 2))
print('Label sample:', labels[0].tolist())
