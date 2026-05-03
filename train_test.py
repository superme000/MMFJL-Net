import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ["OMP_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "32"
from numpy import random
import logging
from logging.handlers import RotatingFileHandler
import sys
from torch.utils.data import TensorDataset, DataLoader,Subset
from dataset.dataset_load import TestDataset,TrainDataset
from model.MMFJL_Net import generate_model
import seaborn as sns
import torch
torch.cuda.empty_cache()
torch.set_num_threads(32)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, manifold
from configs import load_config
import time
from datetime import datetime

n_classes = 2
count = 1
c1 = 'MCI'
c2 = 'AD'
depth = 'train_test_resnet10_conmix_100epoch'
path = '{0}_vs._{1}_{2}/i={3}-{4}.pth'.format(
    c1, c2,depth, i, count)

def save_checkpoint(best_acc, model, optimizer, args, epoch):
    global path
    if not os.path.isdir(path):
        path = '{0}_vs._{1}_{2}/i={3}-{4}.pth'.format(
        c1, c2,depth, i, count)
    logger.info('Best Model Saving...')
    save_dir = os.path.join('checkpoints', os.path.dirname(path))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)  # 使用 exist_ok=True 避免目录已存在时抛出异常
    if args.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, os.path.join(save_dir,os.path.basename(path)))
    logger.info('Model saved successfully.')

def setup_logging(log_dir='logs', log_file='training.log'):
    """配置同时输出到终端和文件的日志系统"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, log_file)
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 防止重复添加handler
    if logger.handlers:
        logger.handlers.clear()
    
    # 文件handler（带滚动备份）
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 添加handler
    logger.addHandler(file_handler)  # 添加文件handler
    logger.addHandler(console_handler)  # 添加控制台handler
    
    return logger
logger = setup_logging(log_dir=f'result/{c1}_vs._{c2}_{depth}/logs')

from torchinfo import summary

def log_model_architecture(model, input_shape, logger):
    """
    使用torchinfo生成详细的模型结构报告
    参数:
        model: 要分析的模型
        input_shape: 输入张量的形状 (不包括batch_size)
        logger: 日志记录器对象
    """
    # 生成模型摘要
    model_stats = summary(
        model,
        input_size=(1, *input_shape),  # 添加batch维度
        verbose=0,  # 不自动打印
        col_names=[
            "input_size",
            "output_size", 
            "num_params",
            "kernel_size",
            "trainable"
        ],
        depth=4,  # 显示子模块层级
        col_width=20,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 记录模型结构
    logger.info("\n" + "=" * 80)
    logger.info("Model Architecture Summary")
    logger.info("=" * 80)
    
    # 逐行记录摘要信息
    for line in str(model_stats).split('\n'):
        logger.info(line)

from sklearn.model_selection import StratifiedKFold
def load_confusion_matrix(c1, c2, depth, i,cm,classes):
    """
    Load the confusion matrix from the saved .npy file.
    :param c1: First class label
    :param c2: Second class label
    :param depth: Depth parameter
    :param i: Iteration number
    :return: Loaded confusion matrix
    """
    row_sums = cm.sum(axis=1, keepdims=True)
    confusion_matrix = (cm / row_sums)*100
    cm_normal = cm / row_sums
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normal, annot=confusion_matrix, fmt='.2f', cmap='Blues', vmin=0, vmax=1,
                   xticklabels=classes, 
                   yticklabels=classes)
    for t in plt.gca().texts:
        t.set_text(t.get_text() + '%')
    plt.title('Confusion Matrix(%)')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(ticks=np.arange(len(classes)) + 0.5, labels=classes)
    plt.yticks(ticks=np.arange(len(classes)) + 0.5, labels=classes)
    plt.tight_layout()
    plt.savefig(f'result/{c1}_vs._{c2}_{depth}/confusion_matrix_(i={i}).jpg', dpi=500)
    plt.close()

def plot_roc_curve(c1, c2, depth, i,fpr, tpr, roc_auc):
    """
    Load the fpr and tpr data and plot the ROC curve.
    :param c1: First class label
    :param c2: Second class label
    :param depth: Depth parameter
    :param i: Iteration number
    """
    base_fpr = np.linspace(0, 1, 100)
    # 插值所有TPR到统一FPR轴
    interp_tprs = []
    for fpr, tpr in zip(fpr, tpr):
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0  # 强制从(0,0)开始
        interp_tprs.append(tpr_interp)
    
    # 计算平均TPR和标准差
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0  # 强制结束于(1,1)
    mean_auc = metrics.auc(base_fpr, mean_tpr)
    # Plot ROC curve
    plt.figure()
    plt.plot(base_fpr,mean_tpr, color='red', lw=2, label=f'ROC curve (area = {mean_auc:.2f})'.format(mean_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Iteration {i})')
    plt.legend(loc="lower right")
    plt.savefig(f'result/{c1}_vs._{c2}_{depth}/roc_curve_i={i}.jpg')

    
# 输入数据推荐使用numpy数组，使用list格式输入会报错
def K_Flod_spilt(K, fold, data_indices, label, random_state=4):
    '''
    :param K: The number of parts into which the dataset is to be divided.
    :param fold: To fetch the data of the first fold. If you want to take the 5th fold then fold=5.
    :param data_indices: Indices of the data (not the actual data).
    :param label: Labels corresponding to the data.
    :param random_state: Random seed for reproducibility.
    '''
    # 打乱数据索引
    data_indices = np.array(data_indices)
    # 使用 KFold 划分数据
    splittrain_epoch_list = []
    split_test_list = []
    kf = StratifiedKFold(n_splits=K,shuffle=True, random_state=random_state)
    for train, test in kf.split(data_indices,label):
        splittrain_epoch_list.append(data_indices[train].tolist())
        split_test_list.append(data_indices[test].tolist())

    # 返回指定 fold 的索引
    train, test = splittrain_epoch_list[fold], split_test_list[fold]
    return train, test


def t_sne(x, label):
    tsne = manifold.TSNE(n_components=2)  # dataset [N, dim]
    X_tsne = tsne.fit_transform(x)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure()
    True_labels = label.reshape((-1, 1))

    S_data = np.hstack((X_norm, True_labels))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})

    colors = ['#8e6fad', '#cf2f2f']
    l = ['{}'.format(c2), '{}'.format(c1)]
    marker = ['o', 'o']
    for index in range(2):
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=5, marker=marker[index], c=colors[index])

    plt.legend(l)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('result/{0}_vs._{1}_{2}/t-sne_(i={3}).jpg'.format(c1, c2,depth, i), dpi=500)


def train_epoch(epoch, train_loader, model, optimizer, criterion_cls, args):
    model.train()
    losses = 0.
    losses_cls = 0.
    acc = 0.
    total = 0.
    n = 0.
    for idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.long().cuda()
        output= model(data)
        _, pred = F.softmax(output, dim=-1).max(1)
        acc += pred.eq(target).sum().item()
        total += target.size(0)

        loss_cls = criterion_cls(output, target)
        loss = loss_cls

        losses_cls += loss_cls
        losses += loss

        loss.backward()

        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
        optimizer.step()
        n = idx
    logger.info(
        '[{0}][Epoch: {1:4d}], Loss: {2:.3f}, Loss_cls: {3:.3f}, Acc: {4:.3f}, Correct {5} / Total {6}'.format(
            count, epoch, losses / (n + 1), losses_cls / (n + 1),
                          acc / total * 100., acc, total))
    r_loss = losses_cls / (n+1)
    r_loss = r_loss.cpu().detach().numpy()
    return r_loss,acc / total * 100.

def test_epoch(epoch, test_loader, model, args):
    model.eval()

    acc = 0.
    pred_matrix = []
    target_matrix = []
    output_tsne = []
    target_tsne = []
    score = np.zeros(shape=[1, 2])
    TP = 0.
    FN = 0.
    FP = 0.
    TN = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            if args.cuda:
                data, target = data.cuda(), target.long().cuda()

            output = model(data)
            if idx == 0:
                output_tsne = output.cpu().numpy()
                target_tsne = target.cpu().numpy()

            else:
                output_tsne = np.vstack([output_tsne, output.cpu().numpy()])
                target_tsne = np.concatenate([target_tsne, target.cpu().numpy()], axis=0)

            _, pred = F.softmax(output, dim=-1).max(1)
            out = F.softmax(output, dim=-1)
            if idx == 0:
                score[0][0] = out[0][0].cpu().numpy()
                score[0][1] = out[0][1].cpu().numpy()
            else:
                score = np.vstack([score, out.cpu().numpy()])
            for k in range(len(pred)):
                pred_matrix.append(pred[k].cpu())
                target_matrix.append(target[k].cpu())
            acc += pred.eq(target).sum().item()
            bio_onehot = np.empty(shape=[0, 2])
            for i, value in enumerate(target_matrix):
                if value == 0:
                    bio_onehot = np.concatenate((bio_onehot, [[1, 0]]), 0)
                if value == 1:
                    bio_onehot = np.concatenate((bio_onehot, [[0, 1]]), 0)
            label = bio_onehot
            matrix = metrics.confusion_matrix(target_matrix, pred_matrix)

        TP += matrix[0, 0]
        FN += matrix[0, 1]
        FP += matrix[1, 0]
        TN += matrix[1, 1]

        precision = TP / (TP + FP)
        Sen = TP / (TP + FN)
        recall = TP / (TP + FN)
        Spe = TN / (TN + FP)
        f1_score = 2 * precision * recall / (precision + recall)
        fpr, tpr, _ = metrics.roc_curve(label.ravel(), score.ravel(), pos_label=1, drop_intermediate=False)
        auc = metrics.auc(fpr, tpr)

        logger.info('[{0}][Epoch: {1:4d}]'.format(count, epoch))
        logger.info('cls: Acc: {0:.3f}, Sen: {1:.4f}, Spe: {2:.4f}, F1: {3:.4f}, BAC: {4:.4f}'.format(
            acc / len(test_loader.dataset) * 100., Sen, Spe, f1_score, (Sen + Spe) / 2))

    return acc / len(test_loader.dataset) * 100., Sen, Spe, f1_score, (
            Sen + Spe) / 2, auc, fpr, tpr, output_tsne, target_tsne, matrix

from sklearn.model_selection import train_test_split

def stratified_split(data_indices, label, test_size=0.2, random_state=4):

    label = np.array(label)
    train_indices, test_indices = train_test_split(
        data_indices,
        test_size=test_size,
        stratify=label,
        shuffle=True,
        random_state=random_state
    )
    
    return train_indices, test_indices
def main(args):
    global count, c1, c2, i
    k = 10
    dataset = TrainDataset(included_labels=(c1,c2))
    # 获取所有数据的索引
    data_indices = list(range(len(dataset)))
    labels = dataset.labels
    total_start_time = datetime.now()
    random_seeds = []  # 用于存储每次迭代的随机种子
    seed_value = 42  
    random.seed(seed_value)
    np.random.seed(seed_value)
    random_number = 42
    i = 7
    train_loss = []
    test_acc = []
    test_sen = []
    test_spe = []
    test_bac = []
    test_f1 = []
    test_auc = []
    test_fpr = []
    test_tpr = []
    test_feature = []
    test_target = []
    test_matrix = []
    count = 1
    for ii in range(k):
        random_seeds.append((ii, random_number))

        # train_indices, test_indices = stratified_split(data_indices, labels,random_state=random_number)
        train_indices, test_indices = K_Flod_spilt(k, ii, data_indices, labels,random_state=random_number)
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)

        torch.manual_seed(1)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        model = generate_model(model_depth=10, in_planes=1, num_classes=2)
        print(args.epochs)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                momentum=args.momentum)

        start_epoch = 1
        if args.cuda:
            model = model.cuda()
        criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.2)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=3, eta_min=0.00001)
        best_acc = 0.
        best_loss = float('inf')
        best_epoch = 0
        begin_time = datetime.now()
        start_time = time.asctime(time.localtime(time.time()))
        logger.info('开始时间：{}'.format(start_time))
        for epoch in range(start_epoch, epoch + 1):
            loss,acc = train_epoch(epoch, train_loader, model, optimizer, criterion_cls, args)
            train_loss.append(loss)
            # 保存最佳模型
            if acc > best_acc:
                best_acc = acc
                best_loss = loss  
                best_epoch = epoch
                save_checkpoint(best_acc, model, optimizer, args, best_epoch)                 
            elif acc == best_acc and loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                save_checkpoint(best_acc, model, optimizer, args, best_epoch)                           
            lr_scheduler.step()
            os.makedirs('result/{0}_vs._{1}_{2}/'.format(c1, c2,depth), exist_ok=True)
            logger.info('Current Learning Rate: {}'.format(lr_scheduler.get_last_lr()))
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        checkpoints = torch.load(os.path.join('checkpoints', path))
        model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        start_epoch = checkpoints['global_epoch']
        acc, sen, spe, f1_score, bac, auc, fpr, tpr, feature, target, matrix = test_epoch(
            start_epoch, test_loader, model, args)
        logger.info('The shape of feature:{}'.format(feature.shape))
        logger.info('The shape of target:{}'.format(target.shape))

        test_acc.append(acc)
        test_sen.append(sen)
        test_spe.append(spe)
        test_f1.append(f1_score)
        test_bac.append(bac)
        test_auc.append(auc)
        test_fpr.append(fpr)
        test_tpr.append(tpr)
        if isinstance(feature, torch.Tensor):
            feature = feature.cpu().numpy()
        elif not isinstance(feature, np.ndarray):
            raise ValueError("Feature must be a PyTorch tensor or a NumPy array.")
        if count == 1:
            test_feature = feature
            test_target = target
            test_matrix = matrix
        else:
            test_feature = np.vstack([test_feature, feature])
            test_target = np.concatenate([test_target, target], axis=0)
            test_matrix = test_matrix + matrix
        count = count + 1
        os.makedirs('result/{0}_vs._{1}_{2}'.format(c1, c2,depth), exist_ok=True)
        np.save('result/{0}_vs._{1}_{2}/losstrain_epoch(i={3}, k={4}).npy'.format(c1, c2,depth, i, ii), train_loss)
        train_loss = []
        last_time = datetime.now()
        end_time = time.asctime(time.localtime(time.time()))
        k_time = last_time - begin_time
        logger.info('结束时间：{}'.format(end_time))
        logger.info('所需时间：{}'.format(k_time))

        logger.info(
            'cls: Acc: {0:.3f}±{1:.3f}, Sen: {2:.4f}±{3:.4f}, Spe: {4:.4f}±{5:.4f}, F1: {6:.4f}±{7:.4f}, BAC: {8:.4f}±{9:.4f}, AUC: {10:.4f}±{11:.4f}'.format(
                np.mean(test_acc), np.std(test_acc), np.mean(test_sen), np.std(test_sen), np.mean(test_spe),
                np.std(test_spe), np.mean(test_f1), np.std(test_f1), np.mean(test_bac), np.std(test_bac),
                np.mean(test_auc), np.std(test_auc)))
        
        plot_roc_curve(c1, c2, depth, i, test_fpr, test_tpr, np.mean(test_auc))
        np.save(
            'result/{0}_vs._{1}_{2}/feature_(i={3}).npy'.format(
                c1, c2,depth, i), test_feature)
        np.save(
            'result/{0}_vs._{1}_{2}/label_(i={3}).npy'.format(
                c1, c2,depth, i), test_target)
        t_sne(test_feature, test_target)

        result_cls = {'ACC': test_acc, 'SEN': test_sen, 'SPE': test_spe, 'F1': test_f1, 'BAC': test_bac,
                      'AUC': test_auc}
        

        np.save('result/{0}_vs._{1}_{2}/cls_result.npy'.format(
            c1, c2,depth), result_cls)
        output_file_path = 'result/{0}_vs._{1}_{2}/random_seed.txt'.format(
            c1, c2,depth)
        with open(output_file_path, 'w') as f:
            f.write("Iteration\tRandom_Seed\n")  
            for iteration, seed in random_seeds:
                f.write(f"{iteration}\t{seed}\n")  
        np.save(
            'result/{0}_vs._{1}_{2}/confusion_matrix_(i={3}).npy'.format(
                c1, c2,depth, i), test_matrix)
        total_end_time = datetime.now()
        total_time = total_end_time - total_start_time
        load_confusion_matrix(c1, c2, depth, i, test_matrix, classes=[c1, c2])
        logger.info('总时间: {}'.format(total_time))

        np.save('result/{0}_vs._{1}_{2}/i={3}_fpr.mat'.format(c1, c2,depth, i), np.array(test_fpr, dtype=object))
        np.save('result/{0}_vs._{1}_{2}/i={3}_tpr.mat'.format(c1, c2,depth, i), np.array(test_tpr, dtype=object))

if __name__ == '__main__':
    args = load_config()
    model = generate_model(model_depth=10, in_planes=1, num_classes=2)
    log_model_architecture(model=model,input_shape=(2, 152, 182, 152),logger=logger)
    main(args)

