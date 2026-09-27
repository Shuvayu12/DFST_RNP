"""Sanity check before running RNP: does the converted DFST model + generator behave like a backdoor?

Reports clean accuracy under the three plausible input normalizations and the attack success
rate (ASR) of generator-styled images, so a normalization mismatch or a wrong target label is
caught before spending an hour on unlearning.

    python check_dfst.py --n 2000
"""
import argparse
import collections

import numpy as np
import torch
from torchvision import datasets

import models
from models.dfst_generator import load_dfst_generator

NORMS = {
    'repo   mean/std=(0.4914,0.4822,0.4465)/(0.2023,0.1994,0.2010)': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'alt    mean/std=(0.4914,0.4822,0.4465)/(0.247,0.243,0.261)':    ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    'none   raw [0,1]':                                              ((0, 0, 0), (1, 1, 1)),
}


def normalize(t, ms):
    m, s = ms
    return (t - torch.tensor(m).view(1, 3, 1, 1)) / torch.tensor(s).view(1, 3, 1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backdoor_model_path', default='weights/dfst_resnet18.tar')
    p.add_argument('--dfst_generator_path', default='weights/dfst_generator.pt')
    p.add_argument('--arch', default='resnet18')
    p.add_argument('--n', type=int, default=2000, help='number of test images to use')
    p.add_argument('--target_label', type=int, default=0)
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    net = getattr(models, args.arch)(num_classes=10, norm_layer=None)
    ck = torch.load(args.backdoor_model_path, map_location='cpu')
    net.load_state_dict(ck['state_dict'] if 'state_dict' in ck else ck)
    net = net.to(device).eval()
    G = load_dfst_generator(args.dfst_generator_path, device=device)

    ds = datasets.CIFAR10(root='data/CIFAR10', train=False, download=True)
    idx = torch.randperm(len(ds))[:args.n].numpy()
    x = torch.tensor(ds.data[idx]).permute(0, 3, 1, 2).float() / 255.0
    y = torch.tensor(np.array(ds.targets)[idx])

    @torch.no_grad()
    def predict(t):
        return torch.cat([net(t[i:i + 256].to(device)).argmax(1).cpu() for i in range(0, len(t), 256)])

    print(f'--- clean accuracy on {args.n} test images ---')
    for name, ms in NORMS.items():
        print(f'{name:70s} acc={float((predict(normalize(x, ms)) == y).float().mean()):.4f}')

    with torch.no_grad():
        xt = torch.cat([G(x[i:i + 200].to(device)).cpu() for i in range(0, args.n, 200)])
    print(f'--- generator: output range [{float(xt.min()):.3f}, {float(xt.max()):.3f}], '
          f'mean |x_trig - x| = {float((xt - x).abs().mean()):.4f} ---')
    non_target = y != args.target_label
    for name, ms in NORMS.items():
        pr = predict(normalize(xt, ms))
        hist = dict(sorted(collections.Counter(pr.tolist()).items()))
        asr = float((pr[non_target] == args.target_label).float().mean())
        print(f'{name:70s} ASR(target={args.target_label})={asr:.4f}  pred histogram={hist}')
    print('Expect one normalization row with clean acc ~0.9x and ASR ~0.9x. That is the one main.py '
          'uses if it is the "repo" row; otherwise edit MEAN/STD in data/poison_tool_cifar.py.')


if __name__ == '__main__':
    main()
