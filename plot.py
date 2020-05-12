'''
Usage: python plot.py --l InnerEye-classifier-style-content-separation-v3-concatenated.log --t predictions_loss --v val_predictions_loss --s A-6-loss --label loss
'''

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(args):
    data_file_path = os.path.join(args.d, args.l)
    figure_file_path = os.path.join(args.d, args.s + '.png')
    width, height = args.w, args.h

    df = pd.read_csv(data_file_path).head(175)
    train_acc = df[args.t]
    val_acc = df[args.v]

    if args.g:
        sns.set(style="darkgrid")

    plt.rcParams['figure.figsize'] = width, height
    fig = plt.figure()
    plt.plot(train_acc, label='Training {}'.format(args.label))
    plt.plot(val_acc, label='Validation {}'.format(args.label))
    plt.xlabel('epoch')
    plt.ylabel(args.label)
    plt.legend()
    if args.s != '':
        fig.savefig(figure_file_path)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=str, default='/home/adnan/Desktop/InnerEye-logs', help='data_root_path')
    parser.add_argument('--l', type=str, default='Log-InnerEye-sequential-final.log', help='log_file_path')
    parser.add_argument('--w', type=int, default=6, help='width of the plot')
    parser.add_argument('--h', type=int, default=4, help='height of the plot')
    parser.add_argument('--g', type=bool, default=True, help='grid plot')
    parser.add_argument('--e', type=str, default='epoch', help='epoch column')
    parser.add_argument('--t', type=str, default='acc', help='train accuracy column')
    parser.add_argument('--v', type=str, default='val_acc', help='valid accuracy column')
    parser.add_argument('--s', type=str, default='', help='Figure save path')
    parser.add_argument('--label', type=str, default='accuracy', help='Label of y axis')
    parser_args = parser.parse_args()

    main(parser_args)
