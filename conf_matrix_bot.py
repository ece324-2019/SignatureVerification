import argparse
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sigverDataset import Triplet_Eval_Dataset


def make_conf_matrix(testloader, net, eval_margin):
    pred_list = []
    labels_list = []
    for data in testloader:
        img0, img1, img2, label = data
        img0, img1, img2, label = img0.cuda(), img1.cuda(), img2.cuda(), label.cuda()
        output1, output2, output3 = net(img0, img1, img2)

        dist = torch.nn.PairwiseDistance(p=2)
        dist_pos = dist(output1, output2)
        dist_neg = dist(output1, output3)

        for j in range(output1.shape[0]):
            labels_list.append(label[j])
            if dist_neg[j] - dist_pos[j] > eval_margin:
                pred_list.append(1)
            else:
                pred_list.append(0)

    return confusion_matrix(labels_list, pred_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--valid_size', type=int, default=4)
    parser.add_argument('--split_coefficient', type=int, default=0.2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--loss_type', choices=['mse', 'ce'], default='ce')
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--if_batch', type=bool, default=False)
    parser.add_argument('--num_kernel', type=int, default=30)
    parser.add_argument('--model_type', choices=['small', 'test', 'best', 'best_small'], default='test')
    parser.add_argument('--baseline_margin', type=float, default=0.75)
    parser.add_argument('--triplet_margin', type=float, default=2)
    parser.add_argument('--triplet_eval_margin', type=float, default=0.8)
    parser.add_argument('--computer', type=str, default='google')
    args = parser.parse_args()

    if args.computer == 'terry':
        # terry
        data_base_dir = 'D:/1_Study/EngSci_Year3/ECE324_SigVer_project/'

        baseline_train_csv = "20_overfit_list.csv"
        baseline_valied_csv = "20_overfit_list.csv"
        baseline_test_csv = "20_overfit_list.csv"

        # triplet_train_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_train_triplet_list.csv"
        # triplet_valid_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_valid_triplet_list.csv"
        # triplet_test_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_test_triplet_list.csv"

        triplet_train_csv = "20_train_triplet_list.csv"
        triplet_valid_csv = "20_valid_triplet_list.csv"
        # triplet_test_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_test_triplet_list.csv"

    elif args.computer == 'google':
        # google
        data_base_dir = '/content/'

        baseline_train_csv = "20_overfit_list.csv"
        baseline_valied_csv = "20_overfit_list.csv"
        baseline_test_csv = "20_overfit_list.csv"

        # triplet_train_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_train_triplet_list.csv"
        # triplet_valid_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_valid_triplet_list.csv"
        # triplet_test_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_test_triplet_list.csv"

        triplet_train_csv = "/content/50k_train_triplet_list.csv"
        triplet_valid_csv = "/content/500_test_triplet_list_dutch.csv"
        triplet_test_csv = "20_valid_triplet_list_diff.csv"

    elif args.computer == 'yize':
        # yize
        data_base_dir = '/Users/yizezhao/PycharmProjects/ece324/sigver/'

        baseline_train_csv = "20_overfit_list.csv"
        baseline_valied_csv = "20_overfit_list.csv"
        baseline_test_csv = "20_overfit_list.csv"

        # triplet_train_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_train_triplet_list.csv"
        # triplet_valid_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_valid_triplet_list.csv"
        # triplet_test_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_test_triplet_list.csv"

        triplet_train_csv = "20_train_triplet_list.csv"
        triplet_valid_csv = "20_valid_triplet_list.csv"
        triplet_test_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_test_triplet_list.csv"

    sig_transformations = transforms.Compose([
        transforms.Resize((200, 300)),
        transforms.ToTensor()
    ])

    triplet_valid_dataset = Triplet_Eval_Dataset(csv=triplet_valid_csv, dir=data_base_dir,
                                                 transform=sig_transformations)
    triplet_valid_dataloader = DataLoader(triplet_valid_dataset,
                                          shuffle=True,
                                          batch_size=args.batch_size)
    model = torch.load('/content/models/triplet_sigVerNet_ep1_step441.pt')

    print(make_conf_matrix(triplet_valid_dataloader, model, eval_margin=0.8))


if __name__ == "__main__":
    main()
