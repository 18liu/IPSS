import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random
from config import config
from IPSS import IPSS
from dataset import MyDataset
import time
import numpy as np
import logging
from utils import train_epoch, eval_epoch

def set_logging(config):
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(20)
    config = config()

    config.log_file = config.model_name + ".log"
    config.ckpt_path = os.path.join(config.ckpt_path, config.model_name)
    config.log_path = os.path.join(config.log_path, config.type_name)

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)

    set_logging(config)
    logging.info(config)


    dataset = MyDataset

    train_dataset=dataset( 
            csv_path = config.train_csv,
            transform=transforms.Compose([              
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]),
            seed=80
        )
    test_dataset=dataset(
            csv_path = config.test_csv,
            transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]),
            seed=80
        )
    print("model:", config.model_name,  "| device:", config.device)
    logging.info('number of train data: {}'.format(len(train_dataset)))
    logging.info('number of test data: {}'.format(len(test_dataset)))

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,drop_last=False, num_workers=config.num_workers,shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, drop_last=False,num_workers=config.num_workers,shuffle=False)


    model = IPSS().to(device=config.device)
    logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), model.parameters())) / 10 ** 6))

    # loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)
    
    losses, scores = [], []
    best_srcc = 0
    best_plcc = 0
    main_score = 0
    for epoch in range(0, config.epoch):
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, srcc, plcc, rmse = train_epoch(epoch, model, criterion, optimizer, scheduler, train_loader, config)
        print("[train epoch %d/%d] loss: %.6f, srcc: %.4f, plcc: %.4f, rmse: %.4f, lr: %.6f, time: %.2f min" % \
                (epoch+1, config.epoch, loss_val, srcc, plcc, rmse, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))

        if (epoch + 1) % config.val_freq == 0:
            start_time = time.time()
            logging.info('Starting eval...')
            logging.info('Running val {} in epoch {}'.format(config.dataset_name, epoch + 1))
            loss, srcc_v, plcc_v, rmse_v = eval_epoch(config, epoch, model, criterion, test_loader)
            print("[val epoch %d/%d] loss: %.6f, srcc_v: %.4f, plcc_v: %.4f, rmse_v: %.4f, lr: %.6f, time: %.2f min" % \
                (epoch+1, config.epoch, loss, srcc_v, plcc_v, rmse_v, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))

            if plcc_v + srcc_v > main_score:
                main_score = plcc_v + srcc_v
        
                best_srcc = srcc_v
                best_plcc = plcc_v

                # save weights
                ckpt_name = "epoch{}.pt".format(epoch + 1)
                model_save_path = os.path.join(config.ckpt_path, ckpt_name)
                torch.save(model.state_dict(), model_save_path)
    print("Best validation performance: SRCC: {:.4f}, PLCC: {:.4f}".format(best_srcc, best_plcc))





