import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataset import *
from vis import *
import os


def load_results():
    dtname = "HCP"
    for kind in ['CogFluidComp_Unadj','CogCrystalComp_Unadj','CogTotalComp_Unadj']:
        print(kind)
        dt = dataset(dsType='HCP', labelType=kind)
        dt.setsubset(labelType=kind, labeldim=None, split_ratio=[1, 0, 0], create_val=False)  # [train, val, test]
        dtloader = dt.train_dataloader(batchsize=1)

        model = torch.load('./params/{}/best_train_{}.pth'.format(dtname, kind))
        model.eval()
        all_map=[]
        all_lb=[]
        all_sc=[]
        all_fc=[]
        with torch.no_grad():
            for g_test, lb_test in dtloader:
                g_test = g_test.to(device)
                lb_test = lb_test.to(device)
                lb_pred, new_x, new_edge_attr, final_x, final_edge_index, final_edge_weight = model(g_test, lb_test, g_test.batch)
                idmap = select_sig_attr_per_edge(final_edge_index, final_edge_weight)
                scmat, fcmat = make_sym_connectome(final_edge_index, final_edge_weight)
                all_map.append(idmap)
                all_lb.append(lb_pred.item())
                all_sc.append(scmat)
                all_fc.append(fcmat)
        np.save("idmap_{}.npy".format(kind), np.array(all_map))
        np.save("lbs_{}.npy".format(kind), np.array(all_lb))
        np.save("scmat_{}.npy".format(kind), np.array(all_sc))
        np.save("fcmat_{}.npy".format(kind), np.array(all_fc))
        print("************{} is done.**************".format(kind))


if '__main__' == __name__:

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(678)

    # load_results()

    dtname = 'HCP'
    atlasDir = "/media/shulab/WD_10T/datasets/atlas/"
    if dtname == 'HCP':
        atlasL = os.path.join(atlasDir, 'aal90.nii')
    elif dtname == 'HCD':
        atlasL = os.path.join(atlasDir, 'bna246.nii')

    import nibabel as nib
    from nilearn.plotting import find_xyz_cut_coords, plot_connectome
    atlas = nib.load(atlasL)
    coords = get_coords(atlas)
    coords = coords[0:90]

    for kind in ['CogFluidComp_Unadj','CogCrystalComp_Unadj','CogTotalComp_Unadj']:
        idmap = np.load("idmap_{}.npy".format(kind))
        lbs = np.load("lbs_{}.npy".format(kind))
        scmat = np.load("scmat_{}.npy".format(kind))
        fcmat = np.load("fcmat_{}.npy".format(kind))
        
        # plot_distribution(idmap, kind)

        fc_mean = fcmat.mean(axis=0)
        fc_pos_masked = np.zeros((90, 90))
        for k in range(878):
            fc_pos_masked += fc_mean * (idmap[k,:, :]==1)
        plot_connectome(fc_pos_masked, coords, title="High-influenced Positive Functional Connectivity for {}".format(kind), output_file="fcpos_{}.pdf".format(kind))
        plot_connectome(fc_pos_masked, coords, title="Top 10% High-influenced Positive Functional Connectivity for {}".format(kind), output_file="fcpos_10_{}.pdf".format(kind), edge_threshold="90%")


        fc_neg_masked = np.zeros((90, 90))
        for k in range(878):
            fc_neg_masked += fc_mean * (idmap[k,:, :]==2)
        plot_connectome(fc_neg_masked, coords, title="High-influenced Negative Functional Connectivity for {}".format(kind), output_file="fcneg_{}.pdf".format(kind))
        plot_connectome(fc_neg_masked, coords, title="Top 10% High-influenced Negative Functional Connectivity for {}".format(kind), output_file="fcneg_10_{}.pdf".format(kind), edge_threshold="90%")


        sc_mean = scmat.mean(axis=0)
        sc_masked = np.zeros((90,90))
        for k in range(878):
            sc_masked += sc_mean * (idmap[k,:, :]==0)
        plot_connectome(sc_masked, coords, title="High-influenced Structual Connectivity for {}".format(kind), output_file="sc_{}.pdf".format(kind))
        plot_connectome(sc_masked, coords, title="Top 10% High-influenced Positive Functional Connectivity for {}".format(kind), output_file="sc_10_{}.pdf".format(kind), edge_threshold="90%")









