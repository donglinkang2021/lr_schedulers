import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 初始化网络和参数
num_epochs = 40
initial_lr = 0.1

def rate(step:int, model_size:int, factor:int, warmup:int):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

model = SimpleNet()
def get_optimizer():
    return optim.SGD(model.parameters(), lr=initial_lr)

# 定义不同的学习率调度器
schedulers = {
    'LambdaLR': optim.lr_scheduler.LambdaLR(get_optimizer(), lr_lambda=lambda epoch: 0.95 ** epoch),
    'LambdaLR2': optim.lr_scheduler.LambdaLR(get_optimizer(), lr_lambda=lambda step: rate(step, 10, factor=1, warmup=5)),
    'MultiplicativeLR': optim.lr_scheduler.MultiplicativeLR(get_optimizer(), lr_lambda=lambda epoch: 0.95),
    'StepLR': optim.lr_scheduler.StepLR(get_optimizer(), step_size=10, gamma=0.1),
    'MultiStepLR': optim.lr_scheduler.MultiStepLR(get_optimizer(), milestones=[20, 30], gamma=0.1),
    'ConstantLR': optim.lr_scheduler.ConstantLR(get_optimizer()),
    'LinearLR': optim.lr_scheduler.LinearLR(get_optimizer(), start_factor=1.0, end_factor=0.1, total_iters=40),
    'ExponentialLR': optim.lr_scheduler.ExponentialLR(get_optimizer(), gamma=0.9),
    'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR(get_optimizer(), T_max=20),
    'CyclicLR': optim.lr_scheduler.CyclicLR(get_optimizer(), base_lr=0.001, max_lr=0.1, step_size_up=10, mode='triangular'),
    'CosineAnnealingWarmRestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts(get_optimizer(), T_0=10, T_mult=2),
    'OneCycleLR': optim.lr_scheduler.OneCycleLR(get_optimizer(), max_lr=0.1, total_steps=num_epochs),
    'PolynomialLR': optim.lr_scheduler.PolynomialLR(get_optimizer(), total_iters=num_epochs, power=2),
    'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau(get_optimizer(), factor=0.1, patience=10),
}


# 保存学习率变化的曲线
lr_curves = {name: [] for name in schedulers.keys()}

# 训练过程
for name, scheduler in schedulers.items():
    optimizer = scheduler.optimizer
    for epoch in range(num_epochs):
        optimizer.step()  # 进行一次优化步骤
        lr_curves[name].append(optimizer.param_groups[0]['lr'])  # 记录当前学习率
        
        # 处理ReduceLROnPlateau需要的指标
        if name == 'ReduceLROnPlateau':
            metrics = torch.tensor(1.0)  # 模拟一个指标
            scheduler.step(metrics)  # 更新学习率
        else:
            scheduler.step()  # 更新学习率

# 可视化学习率曲线
plt.figure(figsize=(20, 15))
for i, (name, lr) in enumerate(lr_curves.items()):
    plt.subplot(4, 4, i + 1)
    plt.plot(lr)
    plt.title(name)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid()

plt.tight_layout()
# plt.show()
plt.savefig('lr_schedulers.png')