import torch
from torchvision import transforms
import torchvision as tv
from torch.utils.data import DataLoader

transform_test = transforms.Compose([transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_set = tv.datasets.CIFAR10(
    root='set',
    train=False,
    download=False,
    transform=transform_test)

test_loader = DataLoader(
    test_set,
    batch_size=1,
    shuffle=False,
    num_workers=2)


def test(epoch):
    mobile_net = torch.load(f'pkl/mobile_net{epoch}.pkl')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = mobile_net(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_acc = 100 * correct / total
    print(f'accuracy on test set: {100 * correct / total}%')
    return test_acc


if __name__ == '__main__':
    test(49)
