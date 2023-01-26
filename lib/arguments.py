import argparse
parser=argparse.ArgumentParser()

parser.add_argument('--source_tasks', type=str, default='mnist')
parser.add_argument('--source_model_number', type=int,default=8)
parser.add_argument('--target_task', type=str, default='usps')

parser.add_argument('--s_epoch',type=int,default=500)
parser.add_argument('--t_epoch',type=int,default=200)
parser.add_argument('--n_target_samples',type=int,default=7)
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--t_lr',type=float,default=0.005)
parser.add_argument('--lr_decay',type=float,default=0.1)
parser.add_argument('--s_lr',type=float,default=0.0001)
parser.add_argument('--data_size',type=int,default=28)
parser.add_argument('--resize_size',type=int,default=32)
parser.add_argument('--data_channel',type=int,default=1)
parser.add_argument('--source_classes',type=int,default=10)
parser.add_argument('--target_classes',type=int,default=10)

parser.add_argument('--beta',type=float,default=0.1)
parser.add_argument('--lambda',type=float,default=0.5)
parser.add_argument('--seed',type=int,default=1)
parser.add_argument('--gpu',type=str,default='0')

parser.add_argument('--print_freq',type=int,default=50)
parser.add_argument('--model_dir',type=str,default='model')
parser.add_argument('--save_dir',type=str,default='log/')
parser.add_argument('--random_sample', action='store_true', help='sample data randomly or not')
parser.add_argument('--apply_transform',type=int,default=1)
parser.add_argument('--seed_id',type=int,default=0)

parser.add_argument('--DA_type', type=str, default='CDA', choices=["CDA", "PDA", "ODA"])

opt=vars(parser.parse_args())
