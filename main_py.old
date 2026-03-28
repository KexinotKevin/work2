import argparse
import os
import time
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataset import *
from metrics import *
from model_r import LGUNet_rela

def recon_loss(new_x, final_x):
    return 1 - F.cosine_similarity(new_x, final_x, dim=0).mean()

# train & validation
def train(args):
    train_loss = []
    val_loss = []
    model.train()
    for epoch in range(args.num_epochs):
        t = time.time()
        train_loss_tmp = []
        for g_data, lb_data in trainloader:
            g_data = g_data.to(device)
            lb_data = lb_data.to(device)
            optimizer.zero_grad()
            # new_x, new_edge_attr, final_x, final_edge_index, final_edge_weight, q = model(g_data, lb_data, g_data.batch)
            # loss = criterion(new_x, final_x).mean() + criterion(new_edge_attr[:,0], final_edge_weight).mean()
            # loss = recon_loss(new_x, final_x).mean() + recon_loss(new_edge_attr[:,0], final_edge_weight).mean()
            # lb_pred = model(g_data, lb_data, g_data.batch).mean()
            lb_pred= model(g_data, lb_data, g_data.batch)
            # loss = torch.sqrt(criterion(lb_pred, lb_data)).mean()    #RMSE
            loss = abs(lb_pred - lb_data).mean()
            # loss = 1 - r2_score(lb_data, lb_pred)
            loss.backward()
            optimizer.step()
            train_loss_tmp.append(loss.item())
        scheduler.step()
        train_loss_v = sum(train_loss_tmp) / len(train_loss_tmp)

        # validation
        model.eval()
        val_loss_tmp = []
        with torch.no_grad():
            for g_data, lb_data in valloader:
                g_data = g_data.to(device)
                lb_data = lb_data.to(device)
                # lb_pred= model(g_data, lb_data, g_data.batch).mean()
                lb_pred= model(g_data, lb_data, g_data.batch)
                # loss = torch.sqrt(criterion(lb_pred, lb_data)).mean()   #RMSE
                loss = abs(lb_pred - lb_data).mean()
                val_loss_tmp.append(loss.item())
            val_loss_v = sum(val_loss_tmp) / len(val_loss_tmp)

        train_loss.append(train_loss_v)
        val_loss.append(val_loss_v)
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(train_loss_v),
              'loss_val: {:.4f}'.format(val_loss_v),
              'time: {:.4f}s'.format(time.time() - t))

        if train_loss_v <= min(train_loss):
            best_train = epoch
            os.system("rm ./params/bt_*")
            torch.save(model, "./params/bt_E{}.pth".format(epoch, train_loss_v))
        if val_loss_v <= min(val_loss):
            best_val = epoch
            os.system("rm ./params/bv_*")
            torch.save(model, "./params/bv_E{}.pth".format(epoch, val_loss_v))

    os.system('mv ./params/bt_* ./params/best_train_{}.pth'.format(args.label_type))
    os.system('mv ./params/bv_* ./params/best_validation_{}.pth'.format(args.label_type))

    df = pd.DataFrame(columns=['train_loss', 'val_loss'])
    df['train_loss'] = train_loss
    df['val_loss'] = val_loss
    df.to_csv("HCD_loss_{}.csv".format(args.label_type), index=False)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.set_default_dtype(torch.float64)

    parser = argparse.ArgumentParser(description='LG-BrainUNet')
    for kind in ['CogFluidComp_Unadj', 'CogCrystalComp_Unadj','CogTotalComp_Unadj']:
        print(kind)
        parser = argparse.ArgumentParser(description=kind)
        parser.add_argument('--dataset', type=str, default='HCD', help='dataset to train', choices=['HCP', 'HCD'])
        parser.add_argument('--label_type', type=str, default=kind, help='label type',
                            choices=['CogFluidComp_Unadj', 'CogCrystalComp_AgeAdj', 'CogCrystalComp_Unadj',
                                    'CogCrystalComp_AgeAdj', 'CogTotalComp_Unadj'])
        parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs')
        parser.add_argument('--batch', type=int, default=16, help='number of batch size')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--l2_penalty', type=float, default=0.001)
        parser.add_argument('--input_dimension', type=int, default=246, help='input dimension')
        parser.add_argument('--hidden_dimension', type=int, default=246, help='hidden dimension')
        parser.add_argument('--output_dimension', type=int, default=1, help='output dimension')
        parser.add_argument('--depth', type=float, default=3, help='depth of net')
        parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
        parser.add_argument('--pool_ratio', type=float, default=[0.5, 0.8, 0.5], help='dropout rate')

        args = parser.parse_args()
        print(args)

        dt = dataset(dsType=args.dataset, labelType=args.label_type)
        dt.setsubset(labelType=args.label_type, labeldim=args.hidden_dimension, split_ratio=[0.7, 0.15, 0.15], create_val=True)  # [train, val, test]
        trainloader, testloader, valloader = dt.train_dataloader(batchsize=args.batch), dt.test_dataloader(), dt.val_dataloader()
        print("dataset is okay.")

        model = LGUNet_rela(args).to(device)
        criterion = nn.MSELoss()
        
        from torch.optim import lr_scheduler, Adam
        optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_penalty)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[3, 5, 10, 20, 30], gamma=0.6)
        
        """
        train
        """
        train(args=args)

        """
        test with RMSE and R-square
        """
        model_t = torch.load('./params/best_validation_{}.pth'.format(args.label_type)).to(device)
        # model_t = model
        # model_t = torch.load('./params/best_validation_CogFluidComp_Unadj.pth').to(device)
        model_t.eval()

        from sklearn.metrics import r2_score, root_mean_squared_error
        from scipy.stats import pearsonr

        total_rmse=[]
        total_r2=[]
        total_p = []
        for i in range(11):         # run 10 times
            test_r2 = []
            test_rmse = []
            with torch.no_grad():
                lb_p=[]
                lb_t=[]
                for g_test, lb_test in testloader:
                    g_test = g_test.to(device)
                    lb_test = lb_test.to(device)
                    lb_pred = model_t(g_test, lb_test, g_test.batch)
                    # lb_pred = model_t(g_test, lb_test,g_test.batch).mean()     # not recon
                    # rmse_score = rmse(lb_test, lb_pred)
                    # r2_score = r_squared(lb_test, lb_pred)
                    lb_test = lb_test.cpu().numpy()
                    lb_pred = lb_pred.cpu().numpy()
                    lb_t.append(lb_test)
                    lb_p.append(lb_pred)
                lb_test = np.array(lb_t).squeeze()
                lb_pred = np.array(lb_p).squeeze()
                rmse_score = root_mean_squared_error(lb_test, lb_pred)
                r_square = r2_score(lb_test, lb_pred)
                p_corr = pearsonr(lb_test, lb_pred)

                total_rmse.append(rmse_score)
                total_r2.append(r_square)
                total_p.append(p_corr)
        df2 = pd.DataFrame(columns=['repeat_rmse', 'repeat_r2', 'pearson_corr'])
        df2['repeat_rmse'] = total_rmse
        df2['repeat_r2'] = total_r2
        df2['pearson_corr'] = total_p
        df2.to_csv("./params/HCD_test_my_{}.csv".format(args.label_type), index=False)


        # print('test for {} is done!'.format(args.label_type),
        #  'rmse: {:.4f}'.format(rmse_v),
        #  'r_square: {:.4f}'.format(r_v))
