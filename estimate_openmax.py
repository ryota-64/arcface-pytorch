import numpy as np
from torch.utils import data
from torch.nn import DataParallel

from OSDN.evt_fitting import *
from OSDN.openmax_utils import *
from config.config import Config
from data.dataset import Dataset
from models import *
from visualize_output import create_mean_graph, show_histgram


def openmax(input_score, weibull_model, categories, eu_weight=5e-2, alpharank=10, distance_type='eucos'):
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    nb_classes = len(categories)
    # take index in descending order ( list[::-1] means revers the list)
    ranked_list = input_score.argsort().ravel()[::-1][:alpharank]
    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]
    omega = np.zeros(nb_classes)
    omega[ranked_list] = alpha_weights
    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        dist_l = []
        wscore_l = []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
            channel_dist = calc_distance(input_score_channel, mav[category_name], eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)
            dist_l.append(channel_dist)
            wscore_l.append(wscore)

            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)
        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_score.ravel()))
    return openmax_prob, softmax_prob


def calc_distance(query_score, mcv, eu_weight, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + \
                         spd.cosine(mcv, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mcv, query_score)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mcv, query_score)
    else:
        raise TypeError("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance


def compute_openmax_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su))

        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    # Take channel mean
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    return modified_scores


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def main():
    opt = Config()
    opt.num_classes = len(get_train_labels(opt.train_root, opt.criteria_list))
    opt.metric = 'liner'

    distance_path = opt.distance_path
    mean_path = opt.mean_files_path
    alpha_rank = opt.ALPHA_RAN

    labellist = getlabellist(opt.criteria_list)
    train_labels = get_train_labels(opt.train_root, opt.criteria_list)

    # recreate or first create
    weibull_model = weibull_tailfitting(mean_path, distance_path, train_labels,
                                        tailsize=opt.WEIBULL_TAIL_SIZE, distance_type=opt.distance_type)

    # data loader
    test_dataset = Dataset(opt.test_root, opt.test_list, phase='test', input_shape=opt.input_shape)

    test_loader = data.DataLoader(test_dataset,
                                  batch_size=opt.test_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    # load model , both of feature, fc_modeal
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()
    else:
        raise TypeError('backbone: {} is not expected'.format(opt.backbone))

    model = DataParallel(model)
    model.to(device)
    if device == 'cuda':
        model.load_state_dict(opt.test_model_path)
    else:
        model.load_state_dict(torch.load(opt.test_model_path, map_location={'cuda:0': 'cpu'}))
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
        metric_fc.load_state_dict(opt.test_metric_fc_path)
    else:
        metric_fc.load_state_dict(torch.load(opt.test_metric_fc_path, map_location={'cuda:0': 'cpu'}))

    metric_fc.eval()
    print(labellist)
    openmax_preds_list = []
    softmax_preds_list = []
    ans_preds_list = []
    softmax_data_list_known = []
    softmax_data_list_unknown = []

    # # data loader
    # test_dataset = Dataset('estimate_visualize', 't-SNE_test1568010965.919611.png', phase='test', input_shape=opt.input_shape)
    #
    # test_loader = data.DataLoader(test_dataset,
    #                               batch_size=1,
    #                               shuffle=True,
    #                               num_workers=opt.num_workers)
    # # from PIL import Image
    # # img = Image.open('estimate_visualize/t-SNE_test1568010965.919611.png', )
    # # img = np.array(img)
    # # img = img[np.newaxis,:,:]
    # # print(img.shape)
    # # tt = test_dataset.transforms
    # # # for t in tt:
    # # img = tt(img)
    # # # img = img.resize(256,256)
    # # # img = torch.Tensor(img)
    # for i, (imgs, label_ids) in enumerate(test_loader):
    #     # compute feature and estimate score →　create img_preds that contains feature, score
    #     imgs_feature = model(imgs)
    #     # scores = metric_fc(imgs_feature, label_ids)
    #     scores = metric_fc(imgs_feature)
    #     scores = scores.detach().numpy()
    #     print(scores)
    # # ff = model(img)
    # # score = metric_fc(ff)
    # print(scores)
    # print(softmax(scores[0]))

    for i, (imgs, label_ids) in enumerate(test_loader):
        # compute feature and estimate score →　create img_preds that contains feature, score
        imgs_feature = model(imgs)
        # scores = metric_fc(imgs_feature, label_ids)
        scores = metric_fc(imgs_feature)
        scores = scores.detach().numpy()
        scores = np.array(scores)[:, np.newaxis, :]
        temp_labels = [labellist[pid] for pid in label_ids]
        for ii, (score, label) in enumerate(zip(scores, temp_labels)):
            openmax_predict, softmax_predict = openmax(score, weibull_model, train_labels, eu_weight=opt.euc_scale,
                                                       alpharank=alpha_rank, distance_type=opt.distance_type)

            softmax_ans = labellist[np.argmax(softmax_predict)]
            # type 1
            # openmax_ans = labellist[np.argmax(openmax_predict)] if np.argmax(openmax_predict) < len(
            #     train_labels) else 'unknown'

            # type2
            # openmax_ans = softmax_ans if np.sort(score, axis=1)[0][::-1][0] > opt.SCORE_THRESHOLD else 'unknown'

            # type3
            openmax_ans = softmax_ans if np.sort(score, axis=1)[0][::-1][0] / np.linalg.norm(score,
                                                                                             ord=2) > opt.SCORE_NORMALIZED else 'unknown'

            # type4
            openmax_ans = softmax_ans if np.sort(score, axis=1)[0][::-1][0] / np.linalg.norm(score[score > 0],
                                                                                             ord=2) > opt.SCORE_NORMALIZED else 'unknown'

            ans_label = label if labellist.index(label) < len(train_labels) else 'unknown'
            if ans_label == 'unknown':
                softmax_data_list_unknown.append(np.sort(score, axis=1)[0][::-1][0] / np.linalg.norm(score[score > 0],
                                                                                                     ord=2))
                if np.sort(score, axis=1)[0][::-1][0] / np.linalg.norm(score[score > 0],ord=2) > 0.7:
                    import matplotlib.pyplot as plt
                    print(label)
                    plt.imshow(np.array(imgs[ii][0]))
                    plt.savefig('estimate_visualize/{}_{}.jpg'.format(i, label))
                    plt.show()
            else:
                softmax_data_list_known.append(np.sort(score, axis=1)[0][::-1][0] / np.linalg.norm(score[score > 0],
                                                                                                   ord=2))

            # if ans_label == 'unknown':
            #     softmax_data_list_unknown.append(score[0])
            # else:
            #     softmax_data_list_known.append(score[0])
            openmax_preds_list.append(openmax_ans)
            softmax_preds_list.append(softmax_ans)
            ans_preds_list.append(ans_label)
            print('predict_softmax: {}, predict_openmax: {}, answer: {}'.format(softmax_ans, openmax_ans, ans_label))

    # create_mean_graph(softmax_data_list_known)
    # create_mean_graph(softmax_data_list_unknown)
    show_histgram(softmax_data_list_unknown)
    show_histgram(softmax_data_list_known)

    # accuracy check
    soft_acc = accuracy(softmax_preds_list, ans_preds_list)
    open_acc = accuracy(openmax_preds_list, ans_preds_list)
    print('softmax:', soft_acc / len(ans_preds_list))
    print('openmax:', open_acc / len(ans_preds_list))


def accuracy(preds, labels):
    accs = list()
    for i in range(len(preds)):
        accs.append(preds[i] == labels[i])
    print(accs.count(True))
    return accs.count(True)


if __name__ == '__main__':
    main()
