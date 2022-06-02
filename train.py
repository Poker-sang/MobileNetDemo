import torch
import torchvision
from torchvision import transforms
import torch.optim as optim
from main import MobileNet
from torch.utils.data import DataLoader

# viz = Visdom(env='Mobilenet')
# viz.line(np.array([[0., 0.]]), np.array([0]), win='train', opts=dict(title='loss&acc'))
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(
    root='set',
    train=True,
    download=False,
    transform=transform_train)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=8,
    shuffle=True,
)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MobileNet().cuda()  # .to(device)
criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=pow(10, -9), last_epoch=-1)


def train(num_epoch):
    for epoch in range(num_epoch):
        for i, (img, label) in enumerate(train_loader):
            img = img.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch}/{num_epoch}],loss: {loss.data:.6f}')
        torch.save(model, f'pkl/mobile_net{epoch}.pkl')
        # testacc = test.test(epoch)
        # viz.line(np.array([[float(loss), float(testacc)]]), np.array([epoch]), win='train', update='append')


if __name__ == '__main__':
    train(50)
