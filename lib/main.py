import os, time
import numpy as np
import torch
import torch.nn as nn
from model import Cosine_feature_merge_with_skip_connection, ResNet_18, ResNet_50
from arguments import opt
import dataloader
from torchvision.utils import save_image
import math
# import loss
import torchvision.transforms as transforms
from model import LeNetEncoder, DTNEncoder, feat_classifier
import torch.nn.functional as F
import collections

use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:{}'.format(opt['gpu'])) if use_cuda else torch.device('cpu')

if use_cuda:
    torch.cuda.manual_seed(opt['seed'])

# Loss functions
adversarial_loss = torch.nn.BCELoss().to(device)
auxiliary_loss = torch.nn.CrossEntropyLoss().to(device)
l1loss = nn.L1Loss().to(device)
MSELoss = nn.MSELoss().to(device)
cosineloss=nn.CosineEmbeddingLoss(reduction='mean').to(device)


#-----------------set up source and target task--------------------------
source_domains =opt['source_tasks'].split(',')
source_dataset_name = opt['source_tasks'].split(',')[0]
target_dataset_name = opt['target_task']
if source_dataset_name in ['cifar','stl']:
    opt['source_classes'] = 9
    opt['target_classes'] = 9
elif source_dataset_name in ['mnist', 'usps', 'svhn']:
    opt['source_classes'] = 10
    opt['target_classes'] = 10
elif source_dataset_name in ['amazon', 'dslr', 'webcam']:
    if opt['DA_type'] =='CDA':
        opt['source_classes'] = 31
        opt['target_classes'] = 31
    elif opt['DA_type'] == 'PDA':
        opt['source_classes'] = 31
        opt['target_classes'] = 16
    elif opt['DA_type'] == 'ODA':
        opt['source_classes'] = 16
        opt['target_classes'] = 31
else:
    print('Warning: Unknown source task or DA_type!')


data_sets=['mnist', 'usps', 'svhn', 'cifar', 'stl', 'amazon', 'dslr', 'webcam']
st_set={source_dataset_name, target_dataset_name}
for s in ['mnist', 'usps', 'svhn']:
    if s not in st_set:
        second_target = s
        break
if source_dataset_name == 'cifar':
    second_target='stl'
if source_dataset_name == 'stl':
    second_target='cifar'
for s in ['amazon', 'dslr', 'webcam']:
    if s not in st_set:
        second_target = s
        break

#-----------------Load source models--------------------------
encoders = []
classifiers = []
source_model_number = opt['source_model_number']
for s in source_domains:
    if s in data_sets:
        # for r in range(0, source_model_number):
        for r in range(4, source_model_number):
            encoder_dir = './{}/{}_{}/source_encoder.pt'.format(opt['model_dir'], source_dataset_name, r+1)
            classifier_dir = './{}/{}_{}/source_classifier.pt'.format(opt['model_dir'], source_dataset_name, r+1)
            if source_dataset_name == 'mnist' or source_dataset_name == 'usps':
                encoder = LeNetEncoder(output_dim=256, type='bn')
            elif source_dataset_name == 'svhn':
                encoder = DTNEncoder(output_dim=256, type='bn')
            elif source_dataset_name == 'cifar' or source_dataset_name == 'stl':
                encoder = ResNet_18(image_channels=3)
            elif source_dataset_name in ['amazon', 'dslr', 'webcam']:
                encoder = ResNet_50()

            classifier = feat_classifier(feature_dim=encoder.output_dim, class_num=opt['source_classes'], type='wn')

            encoder.load_state_dict(torch.load(encoder_dir, map_location=lambda storage, loc: storage.cuda(device)))
            encoder = encoder.to(device)
            classifier.load_state_dict(torch.load(classifier_dir, map_location=lambda storage, loc: storage.cuda(device)))
            classifier = classifier.to(device)
            encoders.append(encoder)
            classifiers.append(classifier)
    else:
        print('Warning: Unknown source task: {}'.format(s))


if target_dataset_name in data_sets:
    test_dataloader = dataloader.get_dataset_testloader(target_dataset_name, batch_size=opt['batch_size'],
                                                        data_size=opt['data_size'], resize_size=opt['resize_size'],
                                                        data_channel=opt['data_channel'], da_type=opt['DA_type'])
else:
    print('Warning: Unknown target task!')



if source_dataset_name == 'mnist' or source_dataset_name == 'usps':
    target_encoder = LeNetEncoder(output_dim=256, type='bn')
elif source_dataset_name == 'svhn':
    target_encoder = DTNEncoder(output_dim=256, type='bn')
elif source_dataset_name == 'cifar' or source_dataset_name == 'stl':
    target_encoder = ResNet_18(image_channels=3)
elif source_dataset_name in ['amazon', 'dslr', 'webcam']:
    target_encoder = ResNet_50()

target_encoder = target_encoder.to(device)
target_classifier = feat_classifier(feature_dim=target_encoder.output_dim, class_num=opt['target_classes'], type='wn')
target_classifier = target_classifier.to(device)


#------------------Feature aggregation net-------------------------
Cosine_feature_merge_with_skip_connection_net = Cosine_feature_merge_with_skip_connection(num_models=source_model_number,
    input_features=target_encoder.output_dim)
Cosine_feature_merge_with_skip_connection_net = Cosine_feature_merge_with_skip_connection_net.to(device)


cmd = "mkdir -p {}".format(opt['save_dir'])
os.system(cmd)

#------------------Sample target data--------------------------
data_size = opt['data_size']
X_t,Y_t = dataloader.create_target_samples_v2(n=opt['n_target_samples'], dataset=target_dataset_name,
                                              clss=opt['target_classes'], random=opt['random_sample'],
                                              data_size=data_size, resize_size=opt['resize_size'],
                                              data_channel=opt['data_channel'], da_type=opt['DA_type'])
X_t=X_t.to(device)
Y_t=Y_t.to(device)


train_transform = [transforms.RandomCrop(data_size, padding=4,padding_mode='reflect'),
                  transforms.RandomRotation(5, center=(0, 0), fill=-1),
                  transforms.RandomPerspective(distortion_scale=0.02, p=0.2, fill=-1),
                  transforms.RandomAffine(degrees=(-5, 5), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=-1)]
if source_dataset_name in ['cifar', 'stl', 'amazon', 'dslr', 'webcam']:
    train_transform.append(transforms.RandomHorizontalFlip())
transform_applier = transforms.RandomApply(transforms=train_transform, p=0.5)


#for vis
#indexs = torch.argsort(Y_t)
#save_image(X_t[indexs].data, "%s/sample_data.jpg" % (opt['save_dir']), nrow=opt['n_target_samples'], normalize=True)

# for initialize model
def integration(models, model_weights=[1, 1]):
    worker_state_dict = [x.state_dict() for x in models]
    weight_keys = list(worker_state_dict[0].keys())
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for i in range(len(models)):
            mw = model_weights[i]
            key_sum = key_sum+worker_state_dict[i][key]*mw
        fed_state_dict[key] = key_sum / sum(model_weights)
    #### update fed weights to fl model
    return fed_state_dict
    # fl_model.load_state_dict(fed_state_dict)


def loss_fn_kd(outputs, labels, teacher_outputs, temperature=6, alpha=0.95):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    # alpha = params.alpha
    # alpha = 0.95
    # T = temperature
    T = temperature
    KD_loss = F.cross_entropy(outputs, labels) * (1. - alpha)+ \
            nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return KD_loss


def test_acc(encoder, classifier, test_dataloader):
    classifier.eval()
    encoder.eval()
    acc = 0
    for data, labels in test_dataloader:
        data = data.to(device)
        labels = labels.to(device)
        y_test_pred = classifier(encoder(data))
        acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()
    accuracy = round(acc / float(len(test_dataloader)), 3)
    return 100*accuracy

def test_acc_ens(encoders, classifiers, test_dataloader):
    acc = 0
    for data, labels in test_dataloader:
        data = data.to(device)
        labels = labels.to(device)

        for i in range(len(encoders)):
            classifiers[i].eval()
            encoders[i].eval()
            if i==0:
                y_test_pred = classifiers[i](encoders[i](data))
            else:
                y_test_pred += classifiers[i](encoders[i](data))

        acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()
    accuracy = round(acc / float(len(test_dataloader)), 3)
    return 100*accuracy

def train_feature_merge_teacher_model(model_epoch=1000):
    global encoders, target_encoder, target_classifier, Cosine_feature_merge_with_skip_connection_net

    encoders_params = list()
    # for i in range(len(encoders)):
    #     encoders_params += list(encoders[i].parameters())
    encoders_params += list(encoders[len(encoders)-1].parameters())
    target_classifier_params = list(target_classifier.parameters())

    wd1, wd2, nv, mt = 0.1, 0.1, True, 0.9

    parameters =[
        {'params': encoders_params, 'lr': opt['t_lr']*opt['lr_decay'], 'weight_decay': wd1, 'momentum': mt, 'nesterov':nv},
        {'params': list(Cosine_feature_merge_with_skip_connection_net.parameters()), 'lr': opt['t_lr'], 'weight_decay': wd2, 'momentum': mt, 'nesterov':nv}]
    if opt['DA_type'] in ['PDA', 'ODC'] :
        parameters.append({'params': target_classifier_params, 'lr': opt['t_lr'], 'weight_decay': wd2, 'momentum': mt, 'nesterov':nv})

    optimizer = torch.optim.SGD(parameters)

    max_iter = math.ceil(X_t.shape[0] / opt['batch_size'])

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(model_epoch * max_iter * 0.34), gamma=0.1)

    for epoch in range(1, int(model_epoch) + 1):
        number = X_t.shape[0]
        # index = torch.randperm(X_t.shape[0])
        max_iter = math.ceil(number/opt['batch_size'])
        for iter in range(0, max_iter):
            Cosine_feature_merge_with_skip_connection_net.train()
            for i in range(len(encoders)):
                encoders[i].train()
            if opt['DA_type'] in ['PDA', 'ODA']:
                target_classifier.train()

            optimizer.zero_grad()

            X_batch = X_t[iter * opt['batch_size']:min(number, (iter + 1) * opt['batch_size'])]
            Y_batch = Y_t[iter * opt['batch_size']:min(number, (iter + 1) * opt['batch_size'])]
            if opt['apply_transform']==1:
                X_batch = transform_applier(X_batch)

            fs = []

            for num in range(len(encoders)-1):
                encoders[num].train()
                # encoders[num].eval()
                f = encoders[num](X_batch)
                fs.append(f.detach())

            fs.append(encoders[len(encoders)-1](X_batch))

            final_feature_dist_loss, merged_features = Cosine_feature_merge_with_skip_connection_net(fs)

            all_features = merged_features[-1:]

            loss_str = ['{:.7f}'.format(opt['beta']**final_feature_dist_loss.item())]
            inter_loss = 0
            for i in range(len(all_features)):
                lossi = auxiliary_loss(target_classifier(all_features[i]), Y_batch)
                loss_str.append('{:.7f}'.format(10*lossi.item()))
                inter_loss += 10*lossi

            loss = (1-opt['beta'])*inter_loss + opt['beta']*(final_feature_dist_loss)

            loss.backward()
            optimizer.step()
            scheduler.step()

        if epoch%opt['print_freq'] == 0:

            # acc =[0 for k in range(len(new_fs) + len(merged_features))]
            acc = [0 for k in range(len(all_features))]
            for data, labels in test_dataloader:
                data = data.to(device)
                labels = labels.to(device)
                test_fs = []
                for num in range(len(encoders)):
                    encoders[num].eval()
                    f = encoders[num](data)
                    test_fs.append(f)

                _, merged_features = Cosine_feature_merge_with_skip_connection_net(test_fs)
                all_features = merged_features[-1:]

                for i in range(len(all_features)):
                    pred = target_classifier(all_features[i])
                    acc[i] += (torch.max(pred, 1)[1] == labels).float().mean().item()

            acc = ['{:.3f}'.format(100* round(acc[j] / float(len(test_dataloader)), 3)) for j in range(len(all_features))]
            print("iter  Epoch {}/{}  lr: {:.7f} loss:{} acc: {}".format(
                epoch, model_epoch, optimizer.param_groups[0]['lr'], loss_str, acc[0]))


def train_target_encoder(model_epoch=1000):
    global encoders, target_encoder, target_classifier, Cosine_feature_merge_with_skip_connection_net

    lr = opt['s_lr']
    if source_dataset_name in ['cifar', 'stl']:
        wd, lr, mt = 0.1, lr, 0.9
    else:
        wd, lr, mt = 0.001, lr, 0.9
    
    
    if source_dataset_name in []:
        optimizer = torch.optim.SGD([{'params': list(target_encoder.parameters()),
                                       'lr':  lr, 'weight_decay': wd, 'momentum': mt, 'nesterov': True}])
    else:
        optimizer = torch.optim.Adam([{'params': list(target_encoder.parameters()), 'lr': lr, 'weight_decay': wd}])

    epoch_iters = math.ceil(X_t.shape[0] / opt['batch_size'])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(model_epoch * 0.34), gamma=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(model_epoch * epoch_iters * 0.5), gamma=0.1)

    for epoch in range(1, int(model_epoch) + 1):
        number = X_t.shape[0]
        # index = torch.randperm(X_t.shape[0])
        epoch_iters = math.ceil(number / opt['batch_size'])
        for iter in range(0, epoch_iters):
            target_classifier.eval()
            # Cosine_feature_merge_net.eval()
            Cosine_feature_merge_with_skip_connection_net.eval()
            # target_classifier.train()
            target_encoder.train()
            optimizer.zero_grad()

            X_batch = X_t[iter * opt['batch_size']:min(number, (iter + 1) * opt['batch_size'])]
            Y_batch = Y_t[iter * opt['batch_size']:min(number, (iter + 1) * opt['batch_size'])]

            # transform
            if opt['apply_transform']==1:
                X_batch = transform_applier(X_batch)

            tf = target_encoder(X_batch)
            output = target_classifier(tf)
            if opt['lambda'] == 0:
                loss = auxiliary_loss(output, Y_batch)
                loss_str = ['cn_loss:{:.5f}'.format(loss.item())]
            else:
                fs = []
                for num in range(len(encoders)):
                    # encoders[num].train()
                    encoders[num].eval()
                    f = encoders[num](X_batch)
                    fs.append(f.detach())

                _, merged_features = Cosine_feature_merge_with_skip_connection_net(fs)
                layer_merge_features = merged_features

                merged_features = [f.detach() for f in merged_features]
                all_features = merged_features[-1:]

                teacher_outputs = target_classifier(all_features[-1])
                loss = loss_fn_kd(output, Y_batch, teacher_outputs, temperature=6, alpha=opt['lambda'])

                loss_str = ['teacher_loss:{:.5f}'.format(loss.item())]

            loss.backward()
            optimizer.step()
            scheduler.step()

        if epoch % (opt['print_freq']/2) == 0 or epoch == int(model_epoch):

            acc = test_acc(target_encoder, target_classifier, test_dataloader)
            
            print("Epoch {}/{}  lr: {:.9f}  loss:{}  acc: {:.3f} ".format(
                epoch, model_epoch, optimizer.param_groups[0]['lr'],
               loss_str, acc))
    return


if __name__ == '__main__':

    model_weights = np.zeros(len(classifiers))
    model_weights[-1] = 1
    if opt['DA_type'] == 'CDA':
        target_classifier.load_state_dict(integration(classifiers, model_weights=model_weights))
    else:
        target_classifier.init()


    target_encoder.load_state_dict(integration(encoders, model_weights=model_weights))


    train_feature_merge_teacher_model(model_epoch=opt['t_epoch'])


    if opt['DA_type'] == 'CDA':
        target_encoder.load_state_dict(integration(encoders, model_weights=model_weights))

    train_target_encoder(model_epoch=int(opt['s_epoch']))
