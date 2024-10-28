'''
@PackageName: TE-GCN-main - ronghe_gpu_6.py
@author: Weizhetao
@since 2024/10/12 17:20
'''

import torch
import pickle
import argparse
import numpy as np
import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser(description='multi-stream ensemble')
    parser.add_argument(
        '--ctr_joint',
        type=str,
        default='all_mix_pred_data/ctr_J.pkl'),  # ctr_joint
    parser.add_argument(
        '--ctr_bone',
        type=str,
        default='all_mix_pred_data/ctr_B.pkl'),  # ctr_bone
    parser.add_argument(
        '--ctr_joint_bone',
        type=str,
        default='all_mix_pred_data/ctr_JB.pkl'),    # ctr_joint_bone
    parser.add_argument(
        '--td_joint',
        type=str,
        default='all_mix_pred_data/td_j_B.npy'),  # td_joint
    parser.add_argument(
        '--td_bone',
        type=str,
        default='all_mix_pred_data/td_b_B.npy'),  # td_bone
    parser.add_argument(
        '--td_joint_bone',
        type=str,
        default='all_mix_pred_data/td_jb_B.npy'),  # td_joint_bone
    parser.add_argument(
        '--te_joint',
        type=str,
        default='all_mix_pred_data/te_j_B.npy'),  # te_joint
    parser.add_argument(
        '--te_bone',
        type=str,
        default='all_mix_pred_data/te_b_B.npy'),  # te_bone
    parser.add_argument(
        '--te_joint_bone',
        type=str,
        default='all_mix_pred_data/te_jb_B.npy'),   # te_joint_bone

    parser.add_argument(
        '--val_sample',
        type=str,
        default='test_B_label.npy'),

    return parser


def Cal_Score(File, Rate, ntu60XS_num, Numclass):
    final_score = torch.zeros(ntu60XS_num, Numclass).cuda()
    for idx, file in enumerate(File):
        #关于te的npy需要特殊处理
        if file.find('te') == -1:
            fr = open(file, 'rb')
            inf = pickle.load(fr)
            df = pd.DataFrame(inf)
            df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
            score = torch.tensor(data=df.values, device='cuda')
        else:
            inf = np.load(file, allow_pickle=True)
            score = torch.tensor(data=inf, device='cuda')
        final_score += Rate[idx] * score
    return final_score


def Cal_Acc(final_score, true_label):
    wrong_index = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label!= true_label[index]:
            wrong_index.append(index)

    wrong_num = np.array(wrong_index).shape[0]
    total_num = true_label.shape[0]
    Acc = (total_num - wrong_num) / total_num
    return Acc


def gen_label(val_npy_path):
    true_label = np.load(val_npy_path)
    return torch.from_numpy(true_label).cuda()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Mix_GCN Score File
    ctr_j = args.ctr_joint
    ctr_b = args.ctr_bone
    ctr_jb = args.ctr_joint_bone
    td_j = args.td_joint
    td_b = args.td_bone
    td_jb = args.td_joint_bone
    te_j = args.te_joint
    te_b = args.te_bone
    te_jb = args.te_joint_bone


    val_npy_path = args.val_sample

    File = [ctr_j, ctr_b, ctr_jb,
            td_j, td_b, td_jb,
            te_j, te_b, te_jb]

    best_acc=0

    Numclass = 155
    Sample_Num = 4599
    most_acc = 0.01
    best_rate = None

    Rate = [1, 1, 1,
            1, 1, 1,
            1, 1, 1]
    final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
    true_label = gen_label(val_npy_path)
    Acc = Cal_Acc(final_score, true_label)
    np.save('pred.npy', final_score.detach().cpu().numpy())
    print(Acc)