"""
Authors:
Nawras Abbas    315085043
Michael Bikman  317920999
Module is used for debug - to plot results from logged test run.
Used for debug and reports only.
"""
import matplotlib.pyplot as plt

# log file path - to parse and print plots from it
FILE = r'D:\GIT\DL_HW_21\project\reports' \
       r'\!2021_08_27-17_09_45_report_resnet12_fix_dropout_64masks_5ways_step50_drop0.1_train\2021_08_27-17_09_45.log'


def main():
    """
    Will parse log file and print plots af accuracy and loss per epoch
    """
    with open(FILE) as f:
        train_acc_per_epoch = []
        train_loss_per_epoch = []
        valid_acc_per_epoch = []
        valid_loss_per_epoch = []
        for line in f:
            if line.startswith('--------Run result'):
                break
            if line.startswith('train acc:'):
                t_acc = float(line.split(':')[1].strip())
                train_acc_per_epoch.append(t_acc)
            if line.startswith('train loss:'):
                t_loss = float(line.split(':')[1].strip())
                train_loss_per_epoch.append(t_loss)
            if line.startswith('validation acc:'):
                v_acc = float(line.split(':')[1].strip())
                valid_acc_per_epoch.append(v_acc)
            if line.startswith('validation loss:'):
                v_loss = float(line.split(':')[1].strip())
                valid_loss_per_epoch.append(v_loss)

        if len(train_acc_per_epoch) != len(train_loss_per_epoch) or \
                len(valid_acc_per_epoch) != len(valid_loss_per_epoch):
            raise ValueError('Inconsistent data!')

        xs = range(1, len(train_acc_per_epoch) + 1)

        fig, axes = plt.subplots(2)

        axes[0].plot(xs, train_acc_per_epoch, 'r', label='train')
        axes[0].plot(xs, valid_acc_per_epoch, 'g', label='valid')
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("accuracy")
        axes[0].legend(loc='upper left')

        axes[1].plot(xs, train_loss_per_epoch, 'r', label='train')
        axes[1].plot(xs, valid_loss_per_epoch, 'g', label='valid')
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("loss")
        axes[1].legend(loc='upper right')
        plt.subplots_adjust(hspace=0.45)
        plt.show()


if __name__ == '__main__':
    main()
    print('OK')
