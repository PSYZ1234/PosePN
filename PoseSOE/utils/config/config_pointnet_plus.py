# coding:utf-8

import os
import os.path as osp


class Config:
    # -------------------- data config --------------------#

    num_classes = 40
    classes = []
    totality = 0
    num_point = 0

    save_path = ''

    # -------------------- model config --------------------#

    learning_rate = 1e-3
    batch_size = 8
    decay_rate = 0.7
    decay_step = 200000
    end_point = {}


config = Config()

if __name__ == '__main__':
    print(config.test_files)
