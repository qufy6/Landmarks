import os, cv2
import hdf5storage
import numpy as np
import sys
import _root


def process_wflw(anno, target_size):
    image_name = anno[-1]
    image_path = os.path.join(root, 'WFLW', 'WFLW_images', image_name)
    print(image_path)
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    lms = anno[:196]
    lms = [float(x) for x in lms]
    lms_x = lms[0::2]
    lms_y = lms[1::2]
    # get coo of lmks x and y
    lms_x = [x if x >= 0 else 0 for x in lms_x]
    lms_x = [x if x <= image_width else image_width for x in lms_x]
    lms_y = [y if y >= 0 else 0 for y in lms_y]
    lms_y = [y if y <= image_height else image_height for y in lms_y]
    # keep positive and in boundary
    lms = [[x, y] for x, y in zip(lms_x, lms_y)]
    lms = [x for z in lms for x in z]
    bbox = anno[196:200]
    bbox = [float(x) for x in bbox]
    attrs = anno[200:206]
    attrs = np.array([int(x) for x in attrs])
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox

    width = bbox_xmax - bbox_xmin
    height = bbox_ymax - bbox_ymin
    scale = 1.2  # scale the images with some boundary
    bbox_xmin -= width * (scale - 1) / 2
    bbox_ymin -= height * (scale - 1) / 2
    bbox_xmax += width * (scale - 1) / 2
    bbox_ymax += height * (scale - 1) / 2
    bbox_xmin = max(bbox_xmin, 0)
    bbox_ymin = max(bbox_ymin, 0)
    bbox_xmax = min(bbox_xmax, image_width - 1)
    bbox_ymax = min(bbox_ymax, image_height - 1)
    width = bbox_xmax - bbox_xmin
    height = bbox_ymax - bbox_ymin
    image_crop = image[int(bbox_ymin):int(bbox_ymax), int(bbox_xmin):int(bbox_xmax), :]
    image_crop = cv2.resize(image_crop, (target_size, target_size))

    tmp1 = [bbox_xmin, bbox_ymin] * 98
    tmp1 = np.array(tmp1)
    tmp2 = [width, height] * 98
    tmp2 = np.array(tmp2)
    lms = np.array(lms) - tmp1
    lms = lms / tmp2
    lms = lms.tolist()
    # print('lms', np.array(lms).shape)=196
    lms = zip(lms[0::2], lms[1::2])

    return image_crop, list(lms)


def gen_data_wflw(root_folder, data_name, target_size):
    if not os.path.exists(os.path.join(root_folder, data_name, 'images_train')):
        os.mkdir(os.path.join(root_folder, data_name, 'images_train'))
    if not os.path.exists(os.path.join(root_folder, data_name, 'images_test')):
        os.mkdir(os.path.join(root_folder, data_name, 'images_test'))

    # train //change the format of dataset
    train_file = 'list_98pt_rect_attr_train.txt'
    with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_rect_attr_train_test', train_file),
              'r') as f:
        annos_train = f.readlines()
    annos_train = [x.strip().split() for x in annos_train]
    print(len(annos_train))  # 7500
    count = 1
    with open(os.path.join(root_folder, 'WFLW', 'train.txt'), 'w') as f:
        for anno_train in annos_train:
            image_crop, anno = process_wflw(anno_train, target_size)
            pad_num = 4 - len(str(count))
            image_crop_name = 'wflw_train_' + '0' * pad_num + str(count) + '.jpg'
            print(image_crop_name)
            cv2.imwrite(os.path.join(root_folder, 'WFLW', 'images_train', image_crop_name), image_crop)
            f.write(image_crop_name + ' ')
            for x, y in anno:
                f.write(str(x) + ' ' + str(y) + ' ')
            f.write('\n')
            count += 1

    test_file = 'list_98pt_rect_attr_test.txt'
    with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_rect_attr_train_test', test_file),
              'r') as f:
        annos_test = f.readlines()
    annos_test = [x.strip().split() for x in annos_test]
    names_mapping = {}
    count = 1
    with open(os.path.join(root_folder, 'WFLW', 'test.txt'), 'w') as f:
        for anno_test in annos_test:
            image_crop, anno = process_wflw(anno_test, target_size)
            pad_num = 4 - len(str(count))
            image_crop_name = 'wflw_test_' + '0' * pad_num + str(count) + '.jpg'
            print(image_crop_name)
            names_mapping[anno_test[0] + '_' + anno_test[-1]] = [image_crop_name, anno]
            cv2.imwrite(os.path.join(root_folder, 'WFLW', 'images_test', image_crop_name), image_crop) # write in the folder
            f.write(image_crop_name + ' ')
            for x, y in list(anno):
                f.write(str(x) + ' ' + str(y) + ' ')
            f.write('\n')
            count += 1
    # pose---326
    test_pose_file = 'list_98pt_test_largepose.txt'
    with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_test', test_pose_file), 'r') as f:
        annos_pose_test = f.readlines()
    names_pose = [x.strip().split() for x in annos_pose_test]
    names_pose = [x[0] + '_' + x[-1] for x in names_pose]
    with open(os.path.join(root_folder, 'WFLW', 'test_pose.txt'), 'w') as f:
        for name_pose in names_pose:
            if name_pose in names_mapping:
                image_crop_name, anno = names_mapping[name_pose]
                f.write(image_crop_name + ' ')
                for x, y in anno:
                    f.write(str(x) + ' ' + str(y) + ' ')
                f.write('\n')
            else:
                print('error!')
                exit(0)
    # expr---314
    test_expr_file = 'list_98pt_test_expression.txt'
    with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_test', test_expr_file), 'r') as f:
        annos_expr_test = f.readlines()
    names_expr = [x.strip().split() for x in annos_expr_test]
    names_expr = [x[0] + '_' + x[-1] for x in names_expr]
    with open(os.path.join(root_folder, 'WFLW', 'test_expr.txt'), 'w') as f:
        for name_expr in names_expr:
            if name_expr in names_mapping:
                image_crop_name, anno = names_mapping[name_expr]
                f.write(image_crop_name + ' ')
                for x, y in anno:
                    f.write(str(x) + ' ' + str(y) + ' ')
                f.write('\n')
            else:
                print('error!')
                exit(0)
    # illu---698
    test_illu_file = 'list_98pt_test_illumination.txt'
    with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_test', test_illu_file), 'r') as f:
        annos_illu_test = f.readlines()
    names_illu = [x.strip().split() for x in annos_illu_test]
    names_illu = [x[0] + '_' + x[-1] for x in names_illu]
    with open(os.path.join(root_folder, 'WFLW', 'test_illu.txt'), 'w') as f:
        for name_illu in names_illu:
            if name_illu in names_mapping:
                image_crop_name, anno = names_mapping[name_illu]
                f.write(image_crop_name + ' ')
                for x, y in anno:
                    f.write(str(x) + ' ' + str(y) + ' ')
                f.write('\n')
            else:
                print('error!')
                exit(0)
    # mu---206
    test_mu_file = 'list_98pt_test_makeup.txt'
    with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_test', test_mu_file), 'r') as f:
        annos_mu_test = f.readlines()
    names_mu = [x.strip().split() for x in annos_mu_test]
    names_mu = [x[0] + '_' + x[-1] for x in names_mu]
    with open(os.path.join(root_folder, 'WFLW', 'test_mu.txt'), 'w') as f:
        for name_mu in names_mu:
            if name_mu in names_mapping:
                image_crop_name, anno = names_mapping[name_mu]
                f.write(image_crop_name + ' ')
                for x, y in anno:
                    f.write(str(x) + ' ' + str(y) + ' ')
                f.write('\n')
            else:
                print('error!')
                exit(0)
    # occu---736
    test_occu_file = 'list_98pt_test_occlusion.txt'
    with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_test', test_occu_file), 'r') as f:
        annos_occu_test = f.readlines()
    names_occu = [x.strip().split() for x in annos_occu_test]
    names_occu = [x[0] + '_' + x[-1] for x in names_occu]
    with open(os.path.join(root_folder, 'WFLW', 'test_occu.txt'), 'w') as f:
        for name_occu in names_occu:
            if name_occu in names_mapping:
                image_crop_name, anno = names_mapping[name_occu]
                f.write(image_crop_name + ' ')
                for x, y in anno:
                    f.write(str(x) + ' ' + str(y) + ' ')
                f.write('\n')
            else:
                print('error!')
                exit(0)

    # blur ---773
    test_blur_file = 'list_98pt_test_blur.txt'
    with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_test', test_blur_file), 'r') as f:
        annos_blur_test = f.readlines()
    names_blur = [x.strip().split() for x in annos_blur_test]
    names_blur = [x[0] + '_' + x[-1] for x in names_blur]
    with open(os.path.join(root_folder, 'WFLW', 'test_blur.txt'), 'w') as f:
        for name_blur in names_blur:
            if name_blur in names_mapping:
                image_crop_name, anno = names_mapping[name_blur]
                f.write(image_crop_name + ' ')
                for x, y in anno:
                    f.write(str(x) + ' ' + str(y) + ' ')
                f.write('\n')
            else:
                print('error!')
                exit(0)
    gen_meanface(root_folder, data_name)


if __name__ == '__main__':
    root = _root.gen_data_wflw_root
    gen_data_wflw(root, 'WFLW', 256)
