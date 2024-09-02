import serial
import hashlib
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import *
from src.tool import *
from src.utils import *
from bioaug import GaussianNoise, SignalDrift, LocalJittering, Distortion, \
    RandomCutout, Scaling, TimeWarping, Permutation, MagnitudeWarping, ImpedanceVariation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    # Parameter settings
    seed_everything(42)
    batch_size = 128
    learning_rate = 0.0001
    max_epoch = 500

    milestones = [20, 40, 60, 80]
    sample_rate = 1000
    data_length = 4 * sample_rate
    window_size = 200 * 2
    step_size = 25
    augmentation_p = 0.1

    num_classes = int(input("Enter number of classes: "))
    labels = [f"label{i + 1}" for i in range(num_classes)]

    ser = serial.Serial('/dev/ttyUSB0', 921600)
    train_data, test_data = makedataset(ser, labels, data_length, window_size, step_size)
    torch.cuda.set_device(0)
    # ======================== step 1/5 Data ==============================

    train_transform = transforms.Compose([
        GaussianNoise(p=augmentation_p, SNR=(20, 30)),
        SignalDrift(p=augmentation_p, drift_rate=(0.05, 0.2), func=['linear', 'exp']),
        LocalJittering(p=augmentation_p, alpha=(0.5, 1.5), frequency=(1, 1000), duration=(5, 50), num_jitters=[1, 2, 3]),
        Distortion(p=augmentation_p, harmonic_degree=(0, 0.02), phase_shift=(0, 0.05), distortion_type=['harmonic', 'phase']),
        RandomCutout(p=augmentation_p, area=(10, 60), num=(1, 8), default=0),
        #Scaling(p=augmentation_p, sigma=(1, 10)),
        TimeWarping(p=augmentation_p, sigma=(0.1, 1), knot=(1, 10)),
        #Permutation(p=augmentation_p, nPerm=(2, 4), minSegLength=(10, 100)),
        MagnitudeWarping(p=augmentation_p, sigma=(0.1, 0.2), knot=(1, 5)),
        ImpedanceVariation(p=augmentation_p, amplitude=(-0.5, 1), frequency=(0, 1.0), func=['linear', 'sin', 'exp'])
    ])
    #
    # valid_transform = transforms.Compose([
    #     uLawNormalization(p=1, u=256)
    # ])

    # 构建Dataset实例
    TRAIN_DATASET = OnlineDataset(data_source=train_data, transforms=train_transform)
    TEST_DATASET = OnlineDataset(data_source=test_data, transforms=None)

    # 构建DataLoader
    train_loader = DataLoader(dataset=TRAIN_DATASET, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=TEST_DATASET, batch_size=batch_size, shuffle=True, num_workers=8)

    # ======================== step 2/5 Model ==============================
    model = ResNet(num_classes, window_size).to(device)

    # ======================== step 3/5 Loss function ==============================
    main_criterion = nn.CrossEntropyLoss()
    # ======================== step 4/5 Optimizers ==============================
    main_optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(main_optimizer, gamma=0.8, milestones=milestones)

    # ======================== step 5/5 Train ==============================
    loss_rec = {"trian": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_f1, best_epoch = 0, 0, 0

    for epoch in range(max_epoch):

        # train
        loss_train, acc_train, f1_train = ModelTrainer.train(train_loader, model, main_criterion, main_optimizer, epoch, device, max_epoch, num_classes)
        loss_val, acc_valid, f1_valid = ModelTrainer.valid(test_loader, model, main_criterion, device, num_classes)

        if acc_valid > best_acc:
            best_acc = acc_valid
            best_f1 = f1_valid
            torch.save(model.state_dict(), 'trained/' + 'model_' + str(epoch) + '_' + str(round(best_acc, 4)) + '.pt')

        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc: {:.2%} "
              "Train loss:{:.4f} Valid loss:{:.4f} "
              "Train F1: {:.2%} Valid F1: {:.2%} "
              "Best Acc: {:.2%} Best F1: {:.2%}".format(
            epoch + 1, max_epoch, acc_train, acc_valid, loss_train, loss_val, f1_train, f1_valid, best_acc, best_f1
        ))

        scheduler.step()

