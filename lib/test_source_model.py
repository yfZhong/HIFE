import os
import torch
from arguments import opt
import dataloader
from models import LeNetEncoder, DTNEncoder, feat_classifier, ResNet_18, ResNet_50

cmd = "mkdir -p {}".format(opt['save_dir'])
os.system(cmd)

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
    # print("Iter{}--------Init acc: %.5f" % (iter, 100 * accuracy))
    return 100*accuracy

source_model_number = opt['source_model_number']
for r in range(0, source_model_number):
    encoder_dir='./{}/{}_{}/source_encoder.pt'.format(opt['model_dir'], source_dataset_name, r+1)
    classifier_dir='./{}/{}_{}/source_classifier.pt'.format(opt['model_dir'], source_dataset_name, r+1)
    if not os.path.exists(encoder_dir):
        print('Warning: No model!')

    source_dataloader = dataloader.get_dataset_testloader(source_dataset_name, batch_size=opt['batch_size'],
                                                        data_size=opt['data_size'], resize_size=opt['resize_size'], data_channel=opt['data_channel'])

    test_dataloader = dataloader.get_dataset_testloader(target_dataset_name, batch_size=opt['batch_size'],
                                                        data_size=opt['data_size'],resize_size=opt['resize_size'], data_channel=opt['data_channel'])

    target_testdata2 = dataloader.get_dataset_testloader(second_target, batch_size=opt['batch_size'],
                                                         data_size=opt['data_size'],resize_size=opt['resize_size'], data_channel=opt['data_channel'])

    if source_dataset_name == 'mnist' or source_dataset_name == 'usps':
        encoder = LeNetEncoder(output_dim=256, type='bn').to(device)
    elif source_dataset_name == 'svhn':
        encoder = DTNEncoder(output_dim=256, type='bn').to(device)
    elif source_dataset_name == 'cifar':
        encoder = ResNet_18(image_channels=3, num_classes=classes)
    elif source_dataset_name in ['amazon', 'dslr', 'webcam']:
        # encoder = ResNet_18(image_channels=3, num_classes=classes)
        encoder = ResNet_50().to(device)
        # encoder = LeNetEncoder(output_dim=256, type='bn')
    classifier = feat_classifier(feature_dim=encoder.output_dim, class_num=classes, type='wn').to(device)

    encoder.load_state_dict(torch.load(encoder_dir, map_location=lambda storage, loc: storage.cuda(device)))
    classifier.load_state_dict(torch.load(classifier_dir, map_location=lambda storage, loc: storage.cuda(device)))
    # encoder=encoder.to(device)
    # classifier=classifier.to(device)

    acc0 = test_acc(encoder, classifier, source_dataloader)
    acc1 = test_acc(encoder, classifier, test_dataloader)
    acc2 = test_acc(encoder, classifier, target_testdata2)
    print("Model {} Acc on source {}: {:.5f}\nAcc on {}: {:.5f}\n Acc on {}: {:.5f}".format(
        r+1,source_dataset_name, acc0, target_dataset_name, acc1, second_target, acc2))

