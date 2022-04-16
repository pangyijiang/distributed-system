
from cProfile import label
from turtle import color
from pre_data import pre_dataloader
from net import Net, Train_Base
from matplotlib import pyplot as plt
import numpy as np
import argparse

def main():

    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int, default= 0, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--world_size", type=int, default= 2, help="world size.")
    parser.add_argument("--num_class", type=int, default= 10, help="Number of class.")
    parser.add_argument("--num_epochs", type=int, default= 200, help="Number of training epoches.")
    argv = parser.parse_args()

    model = Net(input_dim = argv.num_class)
    train_dataloader, test_dataloader = pre_dataloader(train_rate = 0.8, num_class = argv.num_class, argv = argv)
    
    trainer = Train_Base(model, train_dataloader, test_dataloader, argv)

    trainer.train()
    x_all, y_all, y_hat_all = trainer.test()

    num_show = 3
    color_known = "black"; color_unknow = "blue"; color_prd = "red"
    fig, axs = plt.subplots(nrows = 1, ncols = num_show)
    margin = 0
    for i in range(num_show):
        x, y, y_hat = x_all[margin+i], y_all[margin+i], y_hat_all[margin+i]
        id_unknown = [id_x for id_x, x_i in enumerate(x) if x_i < 0]
        id_known = [id_x for id_x, x_i in enumerate(x) if x_i >= 0]

        x_known = [x_j for j, x_j in enumerate(x) if j in id_known]
        axs[i].scatter(id_known, x_known, c = [color_known for j in range(len(x_known))], label = "Input")

        y_known = [y_j for j, y_j in enumerate(y) if j in id_unknown]
        axs[i].scatter(id_unknown, y_known, c = [color_unknow for j in range(len(y_known))], label = "Ground Truth")

        y_hat_hid = [y_hat_j for j, y_hat_j in enumerate(y_hat) if j in id_unknown]
        axs[i].scatter(id_unknown, y_hat_hid, c = [color_prd for j in range(len(y_hat_hid))], label = "Prediction", marker = "+")
        axs[i].set_ylim([0, 5])
        axs[i].set_xticks([j for j in range(argv.num_class)]) 
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

