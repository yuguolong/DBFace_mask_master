import torch
import torchvision.models as models
import torchvision
from torch import nn,optim
from torchvision import transforms

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return float(num_correct)/float(total)

#加载数据
train_dir = './mobilenet_train_data/train'
test_dir = './mobilenet_train_data/test'

data_transform = transforms.Compose([
        transforms.Resize(160),
        transforms.CenterCrop(160),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

train_data = torchvision.datasets.ImageFolder(train_dir,transform=data_transform)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_data = torchvision.datasets.ImageFolder(test_dir,transform=data_transform)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

# 加载模型
mobilenet_v2 = models.mobilenet_v2(pretrained=True)

fix_param = True
if fix_param:
    for param in mobilenet_v2.parameters():
        param.requires_grad = False
dim_in = mobilenet_v2.classifier[1].in_features  ############################ 重写全连接层
mobilenet_v2.classifier[1] = nn.Linear(dim_in, 2)  # 增加线性 分类
# 定义优化函数和损失函数
if fix_param:
    optimizer = optim.Adam(mobilenet_v2.classifier[1].parameters(), lr=1e-4)
else:
    optimizer = optim.Adam(mobilenet_v2.classifier[1].parameters(), lr=1e-5)

valid_loss = 0
valid_acc = 0
valid_acc1 = 0
valid_loss1 = 0
criterion = nn.CrossEntropyLoss()

for epoch in range(4):
    mobilenet_v2.train()
    for train_image, train_label in train_data_loader:
        train_image = torch.Tensor(train_image)
        train_label = torch.tensor(train_label)
        train_label = train_label.long()
        output = mobilenet_v2(train_image)
        loss = criterion(output, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        valid_loss = loss.item()
        valid_acc = get_acc(output, train_label)

    for test_image, test_label in test_data_loader:
        test_image = torch.Tensor(test_image)
        test_label = torch.tensor(test_label)
        test_label = test_label.long()
        output = mobilenet_v2(test_image)
        loss = criterion(output, test_label)

        valid_loss1 = loss.item()
        valid_acc1 = get_acc(output, test_label)

    print('训练损失：',valid_loss,'训练准确率：',valid_acc)
    print('测试损失：',valid_loss1,'测试准确率：',valid_acc1)
torch.save(mobilenet_v2, './logs/params.pth')