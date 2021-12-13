import argparse

parser = argparse.ArgumentParser(description='AHDR')

# Training specifications
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--test_each', action='store_true',
                    help='set this option to test each model')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--ep', type=float, default=0, 
                    help='test epoch')
parser.add_argument('--log_interval', type=int, default=8)
parser.add_argument('--val_interval', type=int, default=20)
parser.add_argument('--val', type=bool, default=True)

# Data specifications
parser.add_argument('--logger_file', type=str, default='./train.log',
                    help='training log file name')
parser.add_argument('--dir_train', type=str, default='../Kalantari_dataset/Training/',
                    help='training dataset directory')
parser.add_argument('--dir_test', type=str, default='../Kalantari_dataset/Test/',
                    help='test dataset directory')
parser.add_argument('--model_path', type=str, default='ckp/',
                    help='trained model directory')
parser.add_argument('--model', type=str, default='latest.pth',
                    help='model name')
parser.add_argument('--ext', type=list, default=['.png', '.jpg', '.tif', '.bmp'],
                    help='extension of image files')
parser.add_argument('--batch_size', type=int, default=8,
                    help='training batch size')
parser.add_argument('--patch_size', type=int, default=256,
                    help='input patch size')
parser.add_argument('--save_dir', type=str, default='./results/',
                    help='test results directory')
parser.add_argument('--use_transform', action='store_false',
                    help='transform input data')

# Network specifications
parser.add_argument('--act_type', type=str, default='prelu',
                    help='type of activation function')
parser.add_argument('--norm_type', type=str, default='bn',
                    help='type of normalization')
parser.add_argument('--num_channels', type=int, default=3,
                    help='number of channels of input color images')
parser.add_argument('--num_features', type=int, default=64,
                    help='number of features')
parser.add_argument('--num_layers', type=int, default=6,
                    help='number of dense layers in RDBlock')
parser.add_argument('--growth', type=int, default=32,
                    help='growth number of channels in dense layers')

args = parser.parse_args()
