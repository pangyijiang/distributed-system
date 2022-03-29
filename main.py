
from pre_data import pre_dataloader
from net import Net, Train_Base
from matplotlib import pyplot as plt

num_class = 10

train_dataloader, test_dataloader = pre_dataloader(train_rate = 0.8, num_class = num_class)
model = Net(input_dim = num_class)
trainer = Train_Base(model, train_dataloader, test_dataloader, num_epochs = 200)

trainer.train()
y_all, y_hat_all = trainer.test()

plt.plot(range(100), y_all[:100], label = "Ground Truth")
plt.plot(range(100), y_hat_all[:100], label = "Predicted Value")
plt.legend()
plt.show()

