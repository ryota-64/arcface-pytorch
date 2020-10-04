##########################################################################################################################################################################
# This file is adapted from Caffe's classify demo found at                                                                                                               #
# https://github.com/BVLC/caffe/blob/master/python/classify.py                                                                                                           #
# The original file was Caffe: a fast open framework for deep learning. http://caffe.berkeleyvision.org/                                                                 #
# For original license please check https://github.com/BVLC/caffe                                                                                                        #
# If you use this file, please consider citing                                                                                                                           #
# @article{jia2014caffe,                                                                                                                                                 #
#   Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor}, #
#   Journal = {arXiv preprint arXiv:1408.5093},                                                                                                                          #
#   Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},                                                                                              #
#   Year = {2014}                                                                                                                                                        #
# }                                                                                                                                                                      #
##########################################################################################################################################################################

import scipy as sp
import os, sys, glob
import os.path as path
import pathlib
import shutil
# import caffe
import argparse, time
import numpy as np
from scipy.io import savemat

import multiprocessing as mp
import torch
from torch.utils import data
from torch.nn import DataParallel

from config.config import Config
from data.dataset import Dataset
from models import *
from OSDN.openmax_utils import getlabellist, get_train_labels


opt = Config()
opt.num_classes = len(get_train_labels(opt.train_root, opt.criteria_list))
opt.metric = 'liner'
NPROCESSORS  = 31


# def runClassifierTest(args):
#     """ Given list of arguments, run classifier
#     """
#
#     image_dims = [int(s) for s in args.images_dim.split(',')]
#     if args.force_grayscale:
#       channel_swap = None
#       mean_file = None
#     else:
#       channel_swap = [int(s) for s in args.channel_swap.split(',')]
#       mean_file = args.mean_file
#
#     # Make classifier.
#     classifier = caffe.Classifier(args.model_def, args.pretrained_model,
#             image_dims=image_dims, gpu=args.gpu, mean_file=mean_file,
#             input_scale=args.input_scale, channel_swap=channel_swap)
#
#     if args.gpu:
#         print 'GPU mode'
#
#     # Load numpy array (.npy), directory glob (*.jpg), or image file.
#     args.input_file = os.path.expanduser(args.input_file)
#     if args.input_file.endswith('npy'):
#         inputs = np.load(args.input_file)
#     elif os.path.isdir(args.input_file):
#         inputs =[caffe.io.load_image(im_f)
#                  for im_f in glob.glob(args.input_file + '/*.' + args.ext)]
#     else:
#         inputs = [caffe.io.load_image(args.input_file)]
#
#     if args.force_grayscale:
#       inputs = [rgb2gray(input) for input in inputs];
#
#     print "Classifying %d inputs." % len(inputs)
#
#     # Classify.
#     start = time.time()
#     scores = classifier.predict(inputs, not args.center_only).flatten()
#     print "Done in %.2f s." % (time.time() - start)
#
#     if args.print_results:
#         with open(args.labels_file) as f:
#           labels_df = pd.DataFrame([
#                {
#                    'synset_id': l.strip().split(' ')[0],
#                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
#                }
#                for l in f.readlines()
#             ])
#         labels = labels_df.sort('synset_id')['name'].values
#
#         indices = (-scores).argsort()[:5]
#         predictions = labels[indices]
#
#         meta = [
#                    (p, '%.5f' % scores[i])
#                    for i, p in zip(indices, predictions)
#                ]
#
#         print meta
#
#     # Save
#     np.save(args.output_file, scores)


# todo like pytorch, batch extract and concatenate those by catgegory
def extractFeatures(args):
    """ Loop through files, extract caffe features, and save them to file
    """
    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train',
                            input_shape=opt.input_shape, random_erase=False)
    train_loader = data.DataLoader(train_dataset,
                                  # batch_size=opt.train_batch_size,
                                  batch_size=4,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()
    else:
        raise TypeError('not match the config backbone')

    model = DataParallel(model)
    # load_model(model, opt.test_model_path)
    if device == 'cpu':
        model.load_state_dict(torch.load(opt.test_model_path, map_location={'cuda:0': 'cpu'}))
    # todo else: (gpu)
    else:
        model.load_state_dict(torch.load(opt.test_model_path))
    model.to(torch.device(device))
    model.eval()

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)
    if device == 'cuda':
        metric_fc.load_state_dict(torch.load(opt.test_metric_fc_path))
    else:
        metric_fc.load_state_dict(torch.load(opt.test_metric_fc_path, map_location={'cuda:0': 'cpu'}))

    metric_fc.eval()
    features_list = torch.Tensor().cuda() if device == 'cuda' else torch.Tensor()
    labels = []
    # todo pytorch で　特定の層の出力はどう出すのか調べる
    for i, (imgs, pids) in enumerate(train_loader):
        features = model(imgs)
        # scores = metric_fc(features, pids)
        scores = metric_fc(features)
        temp_labels = [int(pid.data) for pid in pids]
        features_list = torch.cat((features_list, scores.data), 0)
        labels += temp_labels

    # todo 各カテゴリごとに分ける　→　そもそもtrain_loaderの段階から分ける？
    criteria_list = get_train_labels(opt.train_root, opt.criteria_list)
    for i, criteria in enumerate(criteria_list):
        category_features = [feature for k, feature in enumerate(features_list) if labels[k] == i]
        if category_features:
            print(criteria)
            print(len(category_features))
            for img_no, feature in enumerate(category_features):
                feature_dict = {
                    'IMG_NAME': '{}_{}'.format(criteria, img_no),
                    'feature': sp.asarray(feature.data.cpu())
                }
                # feature_dict['prob'] = sp.asarray(classifier.blobs['prob'].data.squeeze(axis=(2, 3)))
                # feature_dict['scores'] = sp.asarray(scores)
                output_dir = pathlib.Path(opt.feature_path).joinpath(criteria)
                output_dir.mkdir(exist_ok=True, parents=True)
                outfname = str(output_dir.joinpath('{}_{}.mat'.format(criteria, img_no)))
                savemat(outfname, feature_dict)
                from scipy.io import loadmat
                ff = loadmat(outfname)




    # todo mat で保存　→　下にその例あり 保存先のディレクトリを調べる　imgname はどこから？
    #
    # # pool = mp.Pool(processes=NPROCESSORS)
    # # arg_l = []
    st = time.time()
    # imglistFeatures = imglist[imagelist_args[0]:imagelist_args[1]]
    # for imgname in imglistFeatures:
    #     arg_l += [(imgname, args)]
    # pool.map(compute_features_multiproc, arg_l)
    # print("Time taken for extracting features from %s images %s secs with %s Processors" %(len(imglistFeatures), time.time() - st, NPROCESSORS))
    #

# def compute_features(img, model, args):
#     """
#     Instantiate a classifier class, pass the images through the network and save features.
#     Features are saved in .mat format
#     """
#     image_dims = [int(s) for s in args.images_dim.split(',')]
#     image_shape = opt.input_shape
#
#     # todo what is it ?
#     if args.force_grayscale:
#       channel_swap = None
#       mean_file = None
#     else:
#       channel_swap = [int(s) for s in args.channel_swap.split(',')]
#       mean_file = args.mean_file
#
#
#     # Make classifier.
#     classifier = caffe.Classifier(args.model_def, args.pretrained_model,
#             image_dims=image_dims, gpu=args.gpu, mean_file=mean_file,
#             input_scale=args.input_scale, channel_swap=channel_swap)
#
#     if args.gpu:
#         print 'GPU mode'
#
#
#

#     print outfname
#     if not path.exists(path.dirname(outfname)):
#         os.makedirs(path.dirname(outfname))
#
#     inputs = [caffe.io.load_image(imgname)]
#
#     if args.force_grayscale:
#         inputs = [rgb2gray(input) for input in inputs];
#
#     print "Classifying %d inputs." % len(inputs)
#
#     scores = classifier.predict(inputs, not args.center_only)
        # Now save features


# def compute_features_multiproc(params):
#     """ Multi-Processing interface for extarcting features
#     """
#     return compute_features(*params)

def main(argv):

    train_loader = ''


    # pycaffe_dir = os.path.dirname(__file__)
    #
    parser = argparse.ArgumentParser()
    # # Required arguments: input and output files.
    # parser.add_argument(
    #     "input_file",
    #     help="Input image, directory, or npy."
    # )
    # parser.add_argument(
    #     "output_file",
    #     help="Output npy filename."
    # )
    # # Optional arguments.
    # parser.add_argument(
    #     "--model_def",
    #     default=os.path.join(pycaffe_dir,
    #             "../data/caffe_train_data/imagenet_deploy.prototxt"),
    #     help="Model definition file."
    # )
    # parser.add_argument(
    #     "--pretrained_model",
    #     default=os.path.join(pycaffe_dir,
    #             "../data/caffe_train_data/caffe_reference_imagenet_model"),
    #     help="Trained model weights file."
    # )
    # parser.add_argument(
    #     "--gpu",
    #     action='store_true',
    #     help="Switch for gpu computation."
    # )
    # parser.add_argument(
    #     "--center_only",
    #     action='store_true',
    #     help="Switch for prediction from center crop alone instead of " +
    #          "averaging predictions across crops (default)."
    # )
    #
    # parser.add_argument(
    #     "--mean_file",
    #     default=os.path.join(pycaffe_dir,
    #                          '../data/caffe_train_data/ilsvrc_2012_mean.npy'),
    #     help="Data set image mean of H x W x K dimensions (numpy array). " +
    #          "Set to '' for no mean subtraction."
    # )
    # parser.add_argument(
    #     "--input_scale",
    #     type=float,
    #     default=255,
    #     help="Multiply input features by this scale before input to net"
    # )
    # parser.add_argument(
    #     "--channel_swap",
    #     default='2,1,0',
    #     help="Order to permute input channels. The default converts " +
    #          "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    #
    # )
    # parser.add_argument(
    #     "--ext",
    #     default='jpg',
    #     help="Image file extension to take as input when a directory " +
    #          "is given as the input file."
    # )
    # parser.add_argument(
    #     "--labels_file",
    #     default=os.path.join(pycaffe_dir,
    #             "../data/caffe_train_data/synset_words_caffe_ILSVRC12.txt"),
    #     help="Readable label definition file."
    # )
    # parser.add_argument(
    #     "--print_results",
    #     action='store_true',
    #     help="Write output text to stdout rather than serializing to a file."
    # )
    # parser.add_argument(
    #     "--force_grayscale",
    #     action='store_true',
    #     help="Converts RGB images down to single-channel grayscale versions," +
    #          "useful for single-channel networks like MNIST."
    # )
    #
    # parser.add_argument(
    #     "--run_quick_test",
    #     action='store_true',
    #     help="Switch for gpu computation."
    # )
    #
    # parser.add_argument(
    #     "--extract_features",
    #     action='store_true',
    #     help="Switch for gpu computation."
    # )
    #
    
    args = parser.parse_args()
    
    # if args.run_quick_test:
    #     runClassifierTest(args)

    if pathlib.Path(opt.feature_path).exists():
        shutil.rmtree(opt.feature_path)
    extractFeatures(args)


if __name__ == "__main__":
    main(sys.argv)
