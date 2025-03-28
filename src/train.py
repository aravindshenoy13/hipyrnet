import argparse
from utils.trainer import train_model
import wandb


def main(args):
    hp = True
    if args.use_hypernet == 'False':
        hp = False
    config = {
        'batch_size': args.batch_size,
        'root_dir' : args.root_dir,
        'epochs': args.epochs,
        'dset' : args.dset,
        'device':args.device,
        'kernel_loss_weight':args.kernel_loss_weight,
        'lr': args.lr,
        'loss_weight':args.loss_weight,
        'gan_type':args.gan_type,
        'use_hypernet': hp,
        'nrb_high' : args.nrb_high,
        'nrb_low' : args.nrb_low,
    }

    wandb.login(key=args.key)
    wandb.init(
        project="HyperLPTN",
          config={
            'use_hypernet': args.use_hypernet, 
            'lr': args.lr, 
            'max_ssim': 0, 
            'max_psnr': 0, 
            'test_psnr': 0, 
            'test_ssim': 0, 
            'best_epoch': 0, 
            'loss_weight': args.loss_weight, 
            'gan_type': args.gan_type
        }, allow_val_change=True
    )
    train_model(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=8)
    parser.add_argument('--epochs', type=int, required=False, default=300)
    parser.add_argument('--root_dir', type=str, required=False)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument('--dset', type=str, required=False, default='grad')
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--loss_weight', type=float, required=False, default=3000)
    parser.add_argument('--kernel_loss_weight', type=float, required=False, default=0.05)
    parser.add_argument('--gan_type', type=str, required=False)
    parser.add_argument('--use_hypernet', type=str, required=False, default='True')
    parser.add_argument('--key', type=str, required=False, default='')
    parser.add_argument('--nrb_low', type=int, required=False)
    parser.add_argument('--nrb_high', type=int, required=False)
    arguments = parser.parse_args()
    main(arguments)
