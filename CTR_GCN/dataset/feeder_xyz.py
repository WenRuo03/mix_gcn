import torch
import numpy as np
from torch.utils.data import Dataset
from . import tools

coco_pairs = [(1, 6), (2, 1), (3, 1), (4, 2), (5, 3), (6, 7), (7, 1), (8, 6), (9, 7), (10, 8), (11, 9),
              (12, 6), (13, 7), (14, 12), (15, 13), (16, 14), (17, 15)]


class Feeder(Dataset):
    def __init__(self, data_path: str, label_path: str, p_interval: list = [0.95],window_size: int = 64, bone: bool = False, vel: bool = False,
                 debug: bool = False):
        super(Feeder, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.p_interval = p_interval
        self.window_size = window_size
        self.bone = bone
        self.vel = vel
        self.debug = debug
        self.load_data()

    def load_data(self):
        # 加载.npy文件，确保形状为(N, C, T, V, M)
        self.data = np.load(self.data_path)  # 假设数据已经是(N, C, T, V, M)的格式
        self.label = np.load(self.label_path)

        if self.debug:
            self.data = self.data[:100]
            self.label = self.label[:100]

        N, C, T, V, M = self.data.shape
        self.sample_name = [f'{self.data_path.split("/")[-1].split(".")[0]}_{i}' for i in range(N)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        # 取得第 idx 个样本数据
        data_numpy = self.data[idx]  # N, C, T, V, M
        label = self.label[idx]
        # data_numpy = np.array(data_numpy)

        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        if valid_frame_num == 0:
            return np.zeros((3, self.window_size, 17, 2)), label, idx

        # 调整数据，确保输入是 (C, T, V, M)
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)

        # 计算骨架数据（骨）
        if self.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in coco_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy

        # 计算速度数据
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        # 去掉0关节点的平移影响
        data_numpy = data_numpy - np.tile(data_numpy[:, :, 0:1, :], (1, 1, 17, 1))  # (C, T, V, M)

        return data_numpy, label, idx  # C T V M

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


# if __name__ == "__main__":
#     # Debug
#     data_path=r'save_2d_pose/2d_test_A_bone.npy'
#     label_path=r'save_2d_pose/test_A_label.npy'
#     data_np=np.load(label_path)
#     print(data_np)
#     train_loader = torch.utils.data.DataLoader(
#         dataset=Feeder(data_path=data_path,label_path=label_path),
#         batch_size=4,
#         shuffle=True,
#         num_workers=2,
#         drop_last=False)
#
#     # val_loader = torch.utils.data.DataLoader(
#     #     dataset=Feeder(data_path='/data-home/liujinfu/MotionBERT/pose_data/V1.npz'),
#     #     batch_size=4,
#     #     shuffle=False,
#     #     num_workers=2,
#     #     drop_last=False)
#     datalodaer_num=len(train_loader)
#     print(datalodaer_num)
#     for batch_idx, (data, label,index) in enumerate(train_loader):
#         if batch_idx == 0:
#             print(data)
#             print(label)
#             break
#     # data = data.float()  # B C T V M
#     # label = label.long()  # B 1
#     print("pasue")