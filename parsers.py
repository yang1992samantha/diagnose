import argparse
import datetime
import os

HOME_DIR = '/home/lixin/diagnose2/'
def get_args():
    parser = argparse.ArgumentParser(description='Baseline')
    
    parser.add_argument('--data_version', type=str, default="")
    parser.add_argument('--use_wandb', type=bool, default=False)

    parser.add_argument('--dataset',type=str,default='DILHDataset')
    parser.add_argument('--trainer',type=str,default='DILHTrainer')
    parser.add_argument('--model_name', type=str, default='DILH')
    parser.add_argument('--config',type=str,default='MIMICConfig')

    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--hidden_size',type=int,default=512)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--accumulation_steps',type=int ,default = 1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--warmup_rate', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--result_path', type=str, default="result")
    parser.add_argument('--loss_fn',type=str,default='BCE')

    parser.add_argument('--test_freq',type=int,default=1)
    # parser.add_argument('--test_only',action="store_true", default=False)
    parser.add_argument('--test_only', type=bool, default=False)
    parser.add_argument('--test_model_path',type=str,default='')

    # 标签平滑
    # parser.add_argument('--label_smooth_lambda',type=float,default=0.02)
    parser.add_argument('--use_pretrain_embed_weight',type=bool,default=True)
    parser.add_argument('--sample_radio',type=int,default=2)
    parser.add_argument('--local_rank',type=int)

    args = parser.parse_args()

    save_model_names = [args.model_name, "seed", str(args.seed),
                        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")]
    save_model_path = os.path.join(HOME_DIR + "logs/checkpoints", '_'.join(save_model_names) + ".pth")     # best model path
    result_path = os.path.join(HOME_DIR + "logs/results", '_'.join(save_model_names))         # the report of test dataset path
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    args.__setattr__('save_model_path',save_model_path)
    args.__setattr__('result_path',result_path)
    return args
