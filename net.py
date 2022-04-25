import torch.nn as nn
from tqdm import tqdm
import torch.distributed as dist
import psutil
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
    
    def __init__(self, model, train_dataloader, test_dataloader, argv):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.argv = argv

        os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % self.argv.local_rank
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device) 
        if torch.cuda.is_available():
            self.ddp_model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.argv.local_rank], output_device= self.argv.local_rank)
        else:
            self.ddp_model = torch.nn.parallel.DistributedDataParallel(self.model)

        self.optimizer = torch.optim.SGD(self.ddp_model.parameters(), lr = 1e-2, momentum = 0.9, weight_decay = 2.0e-4, nesterov=True)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer =  self.optimizer, T_max = len( self.train_dataloader)* self.argv.num_epochs, eta_min = 5e-4)     

        self.fea_in_hook = None
        self.fea_out_hook = None

    def train(self):
        print("\r" + "epoch = 0, loss = None", end = "")
        for epoch in range(0, self.argv.num_epochs, 1):
            self.ddp_model.train()       #model.train() doesn’t change param.requires_grad
            loss_task_epoch = []
            for batch_idx, (x, y) in enumerate(self.train_dataloader):
                x = x.to(self.device).float()
                y = y.to(self.device).float()
                yhat = self.ddp_model(x) 
                loss_task = self.loss_func(y, yhat, x)
                loss_task_epoch.append(loss_task.item())  
                self.optimizer.zero_grad()
                loss_task.backward()  
                self.optimizer.step()        
                self.lr_scheduler.step()
            loss_epoch_avg = np.mean(loss_task_epoch)
            if torch.distributed.get_rank() == 0:
                str_record = "Master node - epoch = %d\%d, loss = %.3f"%(epoch + 1, self.argv.num_epochs, loss_epoch_avg)
                info_dic = self.rec_info()
                info_dic = ["Node%d_ram:%.1f%%_cpu:%.1f%%"%(k, info_dic[k][0], info_dic[k][1]) for k in info_dic]
                info_dic = '  '.join(info_dic)
                print("\r" + str_record + ",  Slave info - " + info_dic, end = "")
            else:
                str_record = "Slaver node - epoch = %d\%d, loss = %.3f"%(epoch + 1, self.argv.num_epochs, loss_epoch_avg)
                print("\r" + str_record, end = "")
                self.send_info()
        print("\n")
                

    def test(self):
        self.ddp_model.eval()
        with torch.no_grad():
            x_all = []
            y_all = []
            y_hat_all = []
            for id_batch, (x, y) in enumerate(self.test_dataloader):
                x = x.to(self.device).float()
                y = y.to(self.device).float()
                yhat = self.ddp_model(x)  
                yhat = torch.where(x >= 0, y, yhat)
                x_all = x_all + x.detach().cpu().tolist()
                y_all = y_all + y.detach().cpu().tolist()
                y_hat_all = y_hat_all + yhat.detach().cpu().tolist()
                
        return x_all, y_all, y_hat_all


    def send_info(self, target_rank = 0):
        ram_info = psutil.virtual_memory().percent
        cpu_info = psutil.cpu_percent()
        info_send = torch.tensor([ram_info, cpu_info]) 
        dist.send(info_send, dst = target_rank)

    def rec_info(self):
        world_size = dist.get_world_size()
        info_dic = {k+1:None for k in range(world_size - 1)}
        for r in range(world_size - 1):
            info_rec = torch.tensor([0.0, 0.0]) 
            dist.recv(info_rec, src = r + 1)
            info_rec = info_rec.tolist()
            info_dic[r + 1] = info_rec
        return info_dic

    def loss_func(self, y, yhat, x):
        yhat = torch.where(x >= 0, y, yhat)
        return torch.nn.functional.mse_loss(y ,yhat)

    def enable_hook(self, model, hook_func, n_layer = "layer3.4.bn2"):
        for n, m in model.named_modules():
            if n == n_layer:
                m.register_forward_hook(hook = hook_func)

    def hook_representation(self, module, fea_in, fea_out):
        self.fea_in_hook = fea_in
        self.fea_out_hook = fea_out
        return None
