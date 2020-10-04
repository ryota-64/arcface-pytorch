import os
import pathlib

import numpy as np

limit_n = 10
train_num_rate = 0.7


def main(data_set_path, test=False):
    criteria_dir_list = [pathlib.Path(data_set_path).joinpath(dir_name) for dir_name in os.listdir(data_set_path)
                          if pathlib.Path(data_set_path).joinpath(dir_name).is_dir()]

    if test:
        # criteria_list を取得
        criteria_list = load_criteria_list(data_set_path)
        print(criteria_list)
        if os.path.exists(os.path.join(data_set_path, 'test_filenames.txt')):
            os.remove(os.path.join(data_set_path, 'test_filenames.txt'))
        for criteria_dir in criteria_dir_list:
            image_list = [image_name for image_name in os.listdir(str(criteria_dir))
                          if pathlib.Path(image_name).suffix in ['.jpg', '.jpeg']]
            if len(image_list) >= 2:

                if pathlib.Path(criteria_dir).stem in criteria_list:
                    criteria_i = criteria_list.index(pathlib.Path(criteria_dir).stem)
                else:
                    criteria_i = len(criteria_list)
                    criteria_list.append(pathlib.Path(criteria_dir).stem)

                with open(os.path.join(data_set_path, 'test_filenames.txt'), mode='a') as f:
                    save_filenams(data_set_path, criteria_i, criteria_dir, image_list, f)
        print(criteria_list)
        save_criteria_list(data_set_path, criteria_list, mode='w')

    else:
        if os.path.exists(os.path.join(data_set_path, 'train_filenames.txt')):
            os.remove(os.path.join(data_set_path, 'train_filenames.txt'))
        if os.path.exists(os.path.join(data_set_path, 'val_filenames.txt')):
            os.remove(os.path.join(data_set_path, 'val_filenames.txt'))
        criteria_list = []
        criteria_count = 0
        for criteria_i, criteria_dir in enumerate(criteria_dir_list):
            image_list = [image_name for image_name in os.listdir(str(criteria_dir))
                          if pathlib.Path(image_name).suffix in ['.jpg', '.jpeg']]
            print(criteria_dir)
            # 枚数を下限を設定
            if len(image_list) >= 3:

                criteria_list += [criteria_dir]
                print('s')
                train_num = int(len(image_list) * train_num_rate)
                train_list = np.random.choice(image_list, size=train_num)
                val_list = [image for image in image_list if image not in train_list]
                with open(os.path.join(data_set_path, 'train_filenames.txt'), mode='a') as f:
                    save_filenams(data_set_path, criteria_count, criteria_dir, train_list, f)
                with open(os.path.join(data_set_path, 'val_filenames.txt'), mode='a') as f:
                    save_filenams(data_set_path, criteria_count, criteria_dir, val_list, f)
                criteria_count += 1
        print(len(criteria_list))
        for i, criteria in enumerate(criteria_list):
            print(i, criteria)
        save_criteria_list(data_set_path, criteria_list)


def save_filenams(data_set_path, criteria_i, criteria_dir, image_list, f):
    # 枚数を上限を設定
    for image_name in image_list[:limit_n]:
        image_path = criteria_dir.joinpath(image_name)
        relative_image_path = image_path.relative_to(pathlib.Path(data_set_path))
        # print(relative_image_path, relative_image_path.parent.stem)
        f.writelines('{0} {1}\n'.format(relative_image_path, criteria_i))


def save_criteria_list(data_set_path, criteria_dir_list, mode='w'):
    with open(str(pathlib.Path(data_set_path).parents[0].joinpath('criteria_list.txt')), mode=mode) as f:
        for criteria_i, criteria_dir in enumerate(criteria_dir_list):
            f.writelines('{0} {1}\n'.format(pathlib.Path(criteria_dir).stem, criteria_i))


def load_criteria_list(data_set_path):
    criteria_list = []
    with open(str(pathlib.Path(data_set_path).parents[0].joinpath('criteria_list.txt')), mode='r')as f:
        for line in f:
            criteria_list.append(line.rstrip('\n').split()[0])
    return criteria_list




if __name__ == '__main__':
    main('Datasets/invoices/train')
    main('Datasets/invoices/test', test=True)
