import torch.nn as nn
from tqdm import tqdm
import torch
import numpy as np
import os


class Net(nn.Module):
    def __init__(self, input_dim = 10, out_range = 5.0):
        super().__init__()
        self.input_dim = input_dim
        self.out_range = torch.tensor(out_range, dtype=torch.float)
        self.net = nn.Sequential(nn.Linear(self.input_dim, 512),
                                       nn.ReLU(),
                                       nn.Dropout(p = 0.3),
                                       nn.Linear(512, 256),
                                       nn.ReLU(),
                                       nn.Dropout(p = 0.3),
                                       nn.Linear(256, 64),
                                       nn.ReLU(),
                                       nn.Dropout(p = 0.3),
                                       nn.Linear(64, 32),
                                       nn.ReLU(),
                                       nn.Dropout(p = 0.3),
                                       nn.Linear(32, input_dim),
                                       nn.Sigmoid())

    def forward(self, x):
        return self.net(x)*self.out_range


class Train_Base:
    
    def __init__(self, model, train_dataloader, test_dataloader, num_epochs = 100):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.num_epochs = num_epochs

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            model = model.to( self.device) 

        self.optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2, momentum = 0.9, weight_decay = 2.0e-4, nesterov=True)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer =  self.optimizer, T_max = len( self.train_dataloader)* self.num_epochs, eta_min = 5e-4)     

    def train(self):
        pbar = tqdm(initial=0, total = self.num_epochs)
        pbar.set_description("epoch = 0, loss = None")
        for epoch in range(0, self.num_epochs, 1):
            self.model.train()       #model.train() doesn’t change param.requires_grad
            loss_task_epoch = []
            for batch_idx, (x, y) in enumerate(self.train_dataloader):
                x = x.to(self.device).float()
                y = y.to(self.device).float()
                yhat = self.model(x) 
                loss_task = self.loss_func(y, yhat, x)
                loss_task_epoch.append(loss_task.item())  
                self.optimizer.zero_grad()
                loss_task.backward()  
                self.optimizer.step()        
                self.lr_scheduler.step()
            loss_epoch_avg = np.mean(loss_task_epoch)
            pbar.update(1)
            str_record = "epoch = %d, loss = %.3f"%(epoch + 1, loss_epoch_avg)
            pbar.set_description(str_record)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            x_all = []
            y_all = []
            y_hat_all = []
            for id_batch, (x, y) in enumerate(self.test_dataloader):
                x = x.to(self.device).float()
                y = y.to(self.device).float()
                yhat = self.model(x)  
                yhat = torch.where(x >= 0, y, yhat)
                x_all = x_all + x.detach().cpu().tolist()
                y_all = y_all + y.detach().cpu().tolist()
                y_hat_all = y_hat_all + yhat.detach().cpu().tolist()
                
        return x_all, y_all, y_hat_all

    def loss_func(self, y, yhat, x):
        yhat = torch.where(x >= 0, y, yhat)
        return torch.nn.functional.mse_loss(y ,yhat)
