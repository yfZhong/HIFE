import torch
import copy
import os
from arguments import opt
import dataloader
from model import LeNetEncoder, DTNEncoder, feat_classifier, ResNet_18, ResNet_50
from train_source_models import test_acc, test_acc_ens
import torchvision.transforms as transforms
import math
use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:{}'.format(opt['gpu'])) if use_cuda else torch.device('cpu')

if use_cuda:
    torch.cuda.manual_seed(opt['seed'])


source_dataset_name = opt['source_tasks'].split(',')[0]
target_dataset_name = opt['target_task']

if source_dataset_name in ['cifar','stl']:
    classes = 9
elif source_dataset_name in ['mnist', 'usps', 'svhn']:
    classes = 10
elif source_dataset_name in ['amazon', 'dslr', 'webcam']:
    classes = 31
else:
    print('Warning: Unknown source task!')
opt['num_classes'] = classes
#-------------------model & dir---------------------------------
#get source and target dataset name
# data_sets={'m':'mnist', 's':'svhn', 'u':'usps', 'c':'cifar', 't':'stl'}
#
data_sets=['mnist', 'usps', 'svhn', 'cifar', 'stl', 'amazon', 'dslr', 'webcam']

def finetune_models():
    encoders=[]
    classifiers=[]
    source_model_number = opt['source_model_number']
    for r in range(0, source_model_number):
    #for r in range(5, 6):
    # for r in [7]:
        print('-------------------%d----- \nFinetune source model'%(r+1))

        if source_dataset_name in data_sets and target_dataset_name in data_sets:

            test_dataloader = dataloader.get_dataset_testloader(target_dataset_name, batch_size=opt['batch_size'],
                                                                data_size=opt['data_size'],resize_size=opt['resize_size'], data_channel=opt['data_channel'])
            encoder_dir = '{}/{}_{}/source_encoder.pt'.format(opt['model_dir'], source_dataset_name, r+1)
            classifier_dir = '{}/{}_{}/source_classifier.pt'.format(opt['model_dir'], source_dataset_name, r+1)
        else:
            print('Warning: Unknown task!')


        #------------------Encoder and classifier--------------------------------------------
        if source_dataset_name == 'mnist' or source_dataset_name == 'usps':
            encoder = LeNetEncoder(output_dim=256, type='bn')
        elif source_dataset_name == 'svhn':
            encoder = DTNEncoder(output_dim=256, type='bn')
        elif source_dataset_name == 'cifar' or source_dataset_name == 'stl':
            encoder = ResNet_18(image_channels=3, num_classes=classes)
        elif source_dataset_name in ['amazon', 'dslr', 'webcam']:
            # encoder = ResNet_18(image_channels=3, num_classes=classes)
            encoder = ResNet_50()

        classifier = feat_classifier(feature_dim=encoder.output_dim, class_num=classes, type='wn')

        #-------------------load source model--------------------------
        encoder.load_state_dict(torch.load(encoder_dir, map_location=lambda storage, loc: storage.cuda(device)))
        encoder = encoder.to(device)
        classifier.load_state_dict(torch.load(classifier_dir, map_location=lambda storage, loc: storage.cuda(device)))
        classifier = classifier.to(device)

        # ------------------Sample target data--------------------------
        data_size = opt['data_size']
        X_t, Y_t = dataloader.create_target_samples_v2(n=opt['n_target_samples'], dataset=target_dataset_name,
                                                       clss=classes, random=opt['random_sample'],
                                                       data_size=data_size, resize_size=opt['resize_size'],
                                                       data_channel=opt['data_channel'])
        X_t = X_t.to(device)
        Y_t = Y_t.to(device)


        wd=0.001
        # if source_dataset_name == 'cifar' or source_dataset_name == 'stl':
        #     optimizer = torch.optim.SGD([{'params': list(encoder.parameters()),
        #                                   'lr': opt['lr'], 'weight_decay': wd, 'momentum': 0.5, 'nesterov': True}])
        # else:
        #     optimizer = torch.optim.Adam([{'params': list(encoder.parameters()), 'lr': opt['lr'], 'weight_decay': wd}])

        # optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=opt['lr'], weight_decay=0.01)

        optimizer = torch.optim.Adam(list(encoder.parameters()), lr=opt['s_lr'], weight_decay=wd)

        max_iter = math.ceil(X_t.shape[0] / opt['batch_size'])

        # optimizer = torch.optim.Adam(list(classifier.parameters()), lr=opt['lr'], weight_decay=0.01)
        # optimizer = torch.optim.SGD(list(encoder.parameters()) + list(classifier.parameters()), lr=opt['lr'],  momentum=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int((opt['s_epoch']*max_iter)*0.5), gamma=0.1)


        train_transform =[transforms.RandomRotation(5, center=(0, 0), fill=-1),
                            transforms.RandomPerspective(distortion_scale=0.02, p=0.2, fill=-1),
                            transforms.RandomAffine(degrees=(-5, 5), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=-1)]
        #
        # train_transform = transforms.Compose([transforms.RandomRotation(15, center=(0, 0), fill=-1),
        #                                       transforms.RandomPerspective(distortion_scale=0.2, p=1.0, fill=-1),
        #                                       transforms.RandomAdjustSharpness(sharpness_factor=2),
        #                                       transforms.GaussianBlur(kernel_size=(5, 5), sigma=0.5)])

        transform_applier = transforms.RandomApply(transforms=train_transform, p=0.5)

        indexs = torch.argsort(Y_t)

        save_acc =[]

        # Loss functions
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        ckp_epoch = 0

        acc0 = test_acc(encoder, classifier, test_dataloader)

        print("Initial accuracy: %.3f " % (acc0))

        if source_dataset_name == 'cifar' or source_dataset_name == 'stl':
            preprocess = transforms.Compose(
                [transforms.RandomCrop(opt['data_size'], padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip()])

        for epoch in range(opt['s_epoch']):
            number = X_t.shape[0]
            max_iter = math.ceil(number / opt['batch_size'])
            for iter in range(0, max_iter):
                # classifier.train()
                classifier.eval()
                encoder.train()
                # encoder.eval()
                optimizer.zero_grad()

                X_batch = X_t[iter * opt['batch_size']:min(number, (iter + 1) * opt['batch_size'])]
                Y_batch = Y_t[iter * opt['batch_size']:min(number, (iter + 1) * opt['batch_size'])]

                if source_dataset_name == 'cifar' or source_dataset_name == 'stl':
                    X_batch = preprocess(X_batch)

                # X_t_tf = transform_applier(X_batch)
                outputs = classifier(encoder(X_batch))
                loss = loss_fn(outputs, Y_batch)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss = loss.item()
                train_acc = (torch.max(outputs, 1)[1] == Y_batch).float().mean().item()
                # print("Epoch %d/%d  lr:%5f  train_loss: %.3f  " % (
                #     epoch, opt['model_epoch'], scheduler.get_last_lr()[0], train_loss))

            if epoch % opt['print_freq'] == 0 or epoch == opt['s_epoch'] -1:
                #testing
                accuracy = test_acc(encoder, classifier, test_dataloader)
                print("Epoch %d/%d  lr:%5f   train_loss: %.3f    accuracy: %.3f " % (
                    epoch, opt['s_epoch'], optimizer.param_groups[0]['lr'], train_loss, accuracy))
                # save_image(X_t_tf[indexs].data, "{}/sample_data_tf_{}.jpg".format(opt['save_dir'], epoch),
                #            nrow=opt['n_target_samples'],
                #            normalize=True)

        encoders.append(encoder)
        classifiers.append(classifier)
        # save_dir = "{}/{}/FT/{}".format(opt['model_dir'], opt['task'], opt['n_target_samples'])
        # cmd = "mkdir -p {}".format(save_dir)
        # os.system(cmd)
        # torch.save(classifier.state_dict(), "{}/classifier_{}.pt".format(save_dir, r+1))
        # torch.save(encoder.state_dict(), "{}/encoder_{}.pt".format(save_dir, r+1))

    ens_accuracy = test_acc_ens(encoders, classifiers, test_dataloader)
    print("Final Ensamble Accuracy: %.3f " % (ens_accuracy))


def test_ft_ens():
    encoders=[]
    classifiers=[]
    source_dataset_name = data_sets[opt['task'][0]]
    target_dataset_name = data_sets[opt['task'][-1]]
    test_dataloader = dataloader.get_dataset_testloader(target_dataset_name, batch_size=opt['batch_size'],
                                                        data_size=opt['data_size'], data_channel=opt['data_channel'])
    source_model_number = opt['source_model_number']
    for r in range(0, source_model_number):

        # ------------------Encoder and classifier--------------------------------------------
        if source_dataset_name == 'mnist' or source_dataset_name == 'usps':
            encoders.append(LeNetEncoder(output_dim=256, type='bn'))
        elif source_dataset_name == 'svhn':
            encoders.append(DTNEncoder(output_dim=256, type='bn'))
        elif source_dataset_name == 'cifar' or source_dataset_name == 'stl':
            encoders.append(ResNet_18(image_channels=3, num_classes=classes))

        classifiers.append(feat_classifier(feature_dim=encoders[r].output_dim, class_num=classes, type='wn'))

    for r in range(0, source_model_number):

        model_dir = "{}/{}/FT/{}".format(opt['model_dir'], opt['task'], opt['n_target_samples'])
        encoder_dir = "{}/encoder_{}.pt".format(model_dir, r+1)
        classifier_dir = "{}/classifier_{}.pt".format(model_dir, r+1)


        # -------------------load source model--------------------------
        encoders[r].load_state_dict(torch.load(encoder_dir, map_location=lambda storage, loc: storage.cuda(device)))
        encoders[r] = encoders[r].to(device)
        classifiers[r].load_state_dict(torch.load(classifier_dir, map_location=lambda storage, loc: storage.cuda(device)))
        classifiers[r] = classifiers[r].to(device)

        accuracy = test_acc(encoders[r], classifiers[r], test_dataloader)
        print("Model %d accuracy: %.3f " % (r+1, accuracy))


    ens_accuracy = test_acc_ens(encoders, classifiers, test_dataloader)
    print("Final Ensamble Accuracy: %.3f " % (ens_accuracy))


if __name__ == '__main__':
    finetune_models()
    # test_ft_ens()
