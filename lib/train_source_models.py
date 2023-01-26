import os
import torch
from arguments import opt
import dataloader
from model import LeNetEncoder, DTNEncoder, feat_classifier, ResNet_18, ResNet_50

# from torchvision.models import resnet18
import torchvision.transforms as transforms
import random
use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:{}'.format(opt['gpu'])) if use_cuda else torch.device('cpu')



if use_cuda:
    torch.cuda.manual_seed(opt['seed'])

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


train_dataloader = dataloader.get_trainloader(source_dataset_name, batch_size=opt['batch_size'],
                                              data_size=opt['data_size'], resize_size=opt['resize_size'], data_channel=opt['data_channel'])
source_testdata = dataloader.get_dataset_testloader(source_dataset_name, batch_size=opt['batch_size'],
                                                    data_size=opt['data_size'], resize_size=opt['resize_size'], data_channel=opt['data_channel'])
target_testdata = dataloader.get_dataset_testloader(target_dataset_name, batch_size=opt['batch_size'],
                                                    data_size=opt['data_size'], resize_size=opt['resize_size'], data_channel=opt['data_channel'])
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

train_dataloader2 = dataloader.get_trainloader(target_dataset_name, batch_size=opt['batch_size'],
                                              data_size=opt['data_size'],resize_size=opt['resize_size'], data_channel=opt['data_channel'])
train_dataloader3 = dataloader.get_trainloader(second_target, batch_size=opt['batch_size'],
                                              data_size=opt['data_size'],resize_size=opt['resize_size'], data_channel=opt['data_channel'])

target_testdata2 = dataloader.get_dataset_testloader(second_target, batch_size=opt['batch_size'],
                                                     data_size=opt['data_size'],resize_size=opt['resize_size'], data_channel=opt['data_channel'])
loss_fn = torch.nn.CrossEntropyLoss().to(device)

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


if source_dataset_name == 'svhn':
    acc_ranges = [[55, 60],[60, 65], [65, 70], [70, 75], [75, 80], [82, 85], [87, 90], [90, 95]]
    # evaluate_start_iter = [0, 20, 60, 100, 160, 400, 1100, 15000]  # for svhn
    evaluate_start_iter = [60, 60, 100, 100, 300, 400, 1100, 15000]
    test_frequency = 10  #for svhn
elif source_dataset_name == 'cifar':
    acc_ranges = [[50, 55], [55, 60], [60, 65], [65, 70], [70, 75], [77, 80], [80, 85], [85, 90]]
    evaluate_start_iter = [150, 300, 560, 600, 800, 1200, 4000, 15000]
    test_frequency = 50  # for cifar
elif source_dataset_name == 'stl':
    acc_ranges = [[47.5, 50], [50, 52.5], [52.5, 55], [55, 57.5], [57.5, 60], [60, 62.5], [62.5, 65], [65, 100]]
    evaluate_start_iter = [10, 20, 30, 100, 100, 400, 600, 650]
    test_frequency = 10  # for stl
elif source_dataset_name in ['amazon', 'webcam']:
    # acc_ranges = [[50, 55], [55, 60], [60, 65], [65, 70], [70, 75], [77, 80], [80, 85], [85, 90]]
    acc_ranges = [[60, 65], [65, 70], [70, 75], [77, 80],[80, 85], [85, 90], [90, 95], [95, 100]]
    evaluate_start_iter = [20, 20, 20, 20, 20, 80, 100, 1500]
    test_frequency = 5
elif source_dataset_name in ['dslr']:
    # acc_ranges = [[50, 55], [55, 60], [60, 65], [65, 70], [70, 75], [77, 80], [80, 85], [85, 90]]
    acc_ranges = [[60, 65], [65, 70], [70, 75], [75, 80],[80, 85], [85, 90], [90, 95], [95, 100]]
    evaluate_start_iter = [10, 15, 15, 20, 20, 20, 20, 1500]
    test_frequency = 1
else:
    acc_ranges = [[60, 65], [65, 70], [70, 75], [75, 80], [80, 85], [85, 90], [90, 95], [95, 100]]
    # evaluate_start_iter = [0,5,10,10,20,20,60,10000] # for mnist or usps
    evaluate_start_iter = [20, 20, 20, 20, 20, 20, 60, 10000]  # for mnist or usps
    test_frequency = 4  # for mnist and usps
    # evaluate_start_iter = [100, 100, 200, 200, 200, 200, 250, 10000]  # for mnist or usps
    # test_frequency = 5  # for mnist and usps

def get_itr_pct(itr, pct):
    num = len(itr)
    i = 0
    for v in itr:
        i+=1
        if i>num*pct:
            break
        yield v

def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v

def train_source_models():
    global acc_ranges
    source_model_number = opt['source_model_number']
    for r in reversed(range(0, source_model_number)):

        if r == source_model_number-1:
            print('------------Train strong source model {} ----------------'.format(source_dataset_name))
        else:
            print('------------Train weak source model {} at source accuracy interval [{}, {}]---------'.format(
                source_dataset_name, acc_ranges[r][0], acc_ranges[r][1]))


        # save_dir = '{}/{}_source_acc_{}'.format(opt['save_dir'], source_dataset_name, acc_l)
        # if r == len(acc_ranges)-1:
        #     save_dir = '{}/{}_source_acc_best'.format(opt['save_dir'], source_dataset_name)
        prefix=''
        if
        save_dir = '{}/{}_{}'.format(opt['save_dir'], source_dataset_name, r+1)
        cmd = "mkdir -p {}".format(save_dir)
        os.system(cmd)

        saved = False
        while True:
            if source_dataset_name == 'mnist' or source_dataset_name == 'usps':
                encoder = LeNetEncoder(output_dim=256, type='bn')
            # elif source_dataset_name=='svhn' or source_dataset_name=='cifar':
            elif source_dataset_name == 'svhn':
                encoder=DTNEncoder(output_dim=256, type='bn')
                # encoder = LeNetEncoder(output_dim=256, type='bn')
            elif source_dataset_name in ['cifar', 'stl']:
                encoder = ResNet_18(image_channels=3)
            elif source_dataset_name in ['amazon', 'dslr', 'webcam']:
                encoder = ResNet_50()
            # elif source_dataset_name=='cifar':
            #     encoder = DTNEncoder(output_dim=256, hidden_feature_dim=256*12*12, type='bn')

            # classifier = feat_classifier(feature_dim=encoder.output_dim, class_num=opt['num_classes'], type='wn')
            classifier = feat_classifier(feature_dim=encoder.output_dim, class_num=opt['source_classes'], type='wn')
            encoder = encoder.to(device)
            classifier = classifier.to(device)

            optimizer = torch.optim.SGD(list(encoder.parameters()) + list(classifier.parameters()), lr=opt['lr'],  momentum=0.5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int((opt['model_epoch']*len(train_dataloader))*0.34), gamma=0.1)

            max_acc = 0
            need_repeat = False
            for epoch in range(opt['model_epoch']):

                train_loss=0
                train_acc=0
                for i, (images, labels) in enumerate(train_dataloader):

                    classifier.train()
                    encoder.train()
                    # Move tensors to the configured device
                    if source_dataset_name == 'cifar' or source_dataset_name == 'stl':
                        preprocess = transforms.Compose([transforms.RandomCrop(opt['data_size'], padding=4, padding_mode='reflect'),
                                                         transforms.RandomHorizontalFlip()])
                        images = preprocess(images)
                    images = images.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = classifier(encoder(images))
                    loss = loss_fn(outputs, labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    train_loss += loss.item()
                    train_acc += (torch.max(outputs, 1)[1] == labels).float().mean().item()
                    # save_image(images.data, "%s/%04d.jpg" % (opt['save_dir'], i), nrow=8, normalize=True)

                    curr_iter = epoch*len(train_dataloader) + i
                    if r == source_model_number - 1:
                        continue
                    elif curr_iter >= evaluate_start_iter[r] and curr_iter % test_frequency == 0:

                        with torch.no_grad():
                            accuracy1 = test_acc(encoder, classifier, source_testdata)
                            acc_l, acc_r = acc_ranges[r][0], acc_ranges[r][1]
                            if accuracy1 > acc_l and accuracy1 < acc_r:
                                accuracy2 = test_acc(encoder, classifier, target_testdata)
                                accuracy3 = test_acc(encoder, classifier, target_testdata2)

                                torch.save(classifier.state_dict(), '{}/source_classifier.pt'.format(save_dir))
                                torch.save(encoder.state_dict(), '{}/source_encoder.pt'.format(save_dir))

                                print("Save Model %d----Epoch %d/%d Iter %d/%d lr: %.6f train_loss: %.6f source accuracy: %.4f target accuracy: %.4f target accuracy2: %.4f" % (
                                    r+1, epoch, opt['model_epoch'], i, len(train_dataloader),  optimizer.param_groups[0]['lr'],
                                    train_loss, accuracy1, accuracy2, accuracy3))
                                saved = True
                                break

                            elif accuracy1 > acc_r:
                                print("Over Trained---Need_repeat----Epoch %d/%d Iter %d/%d  lr: %.6f train_loss: %.6f source accuracy: %.4f " % (
                                        epoch, opt['model_epoch'], i, len(train_dataloader), optimizer.param_groups[0]['lr'], train_loss, accuracy1))
                                need_repeat = True
                                break
                            else:
                                print("Under trained---Go on----Epoch %d/%d Iter %d/%d  lr: %.6f train_loss: %.6f source accuracy: %.4f " % (
                                        epoch, opt['model_epoch'], i, len(train_dataloader),  optimizer.param_groups[0]['lr'], train_loss, accuracy1))
                                continue

                if r == source_model_number - 1 and epoch > 2:
                # if r == source_number - 1 and epoch>1 and epoch %1 == 0 or epoch == (opt['model_epoch']-1):
                    accuracy1 = test_acc(encoder, classifier, source_testdata)
                    accuracy2 = test_acc(encoder, classifier, target_testdata)
                    accuracy3 = test_acc(encoder, classifier, target_testdata2)
                    torch.save(classifier.state_dict(), '{}/source_classifier.pt'.format(save_dir))
                    torch.save(encoder.state_dict(), '{}/source_encoder.pt'.format(save_dir))
                    print("Train best model----Epoch %d/%d  lr: %.6f train_loss: %.6f source accuracy: %.4f target accuracy: %.4f target accuracy2: %.4f" % (
                            epoch, opt['model_epoch'], optimizer.param_groups[0]['lr'],
                            train_loss, accuracy1, accuracy2, accuracy3))
                    if epoch == (opt['model_epoch']-1):
                        max_source_acc = accuracy1
                        min_source_acc = opt['min_source_acc']
                        interval = (max_source_acc - min_source_acc)/source_model_number
                        acc_ranges = [[min_source_acc+k*interval, min_source_acc+(k+1)*interval] for k in range(source_model_number)]
                        print("Acc_ranges for weak source hypothesis: {}".format(acc_ranges))
                        saved = True
                if need_repeat == True or saved == True:
                    break
            if saved==True:
               break

if __name__ == '__main__':
    train_source_models()
