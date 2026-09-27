"""Convert the two DFST checkpoints (pickled whole models from the original DFST repo) into
plain state dicts that this repo can load without the DFST source code on the path.

    python convert_dfst.py --clf cifar10_resnet18_dfst.pt --gen cifar10_resnet18_dfst_generator.pt --out weights/

Produces:
    weights/dfst_resnet18.tar   -> {'state_dict': ...}   pass as --backdoor_model_path to main.py
    weights/dfst_generator.pt   -> generator state dict   pass as --dfst_generator_path to main.py
"""
import argparse
import os
import pickle
import types

import torch
import torch.nn as nn

import models
from models.dfst_generator import CycleGenerator


class _PlaceholderUnpickler(pickle.Unpickler):
    """The checkpoints reference models.resnet / models.cyclegan / models.custom_modules from the
    original DFST repo. Substitute an empty nn.Module subclass for each so the object graph can be
    rebuilt and its parameters read out; forward() is never called on these placeholders."""

    def find_class(self, module, name):
        if module.startswith('models.'):
            return type(name, (nn.Module,), {})
        return super().find_class(module, name)


def _load_pickled_model(path):
    pm = types.ModuleType('placeholder_pickle')
    pm.Unpickler = _PlaceholderUnpickler
    pm.load = pickle.load
    obj = torch.load(path, map_location='cpu', weights_only=False, pickle_module=pm)
    return obj.module if isinstance(obj, nn.DataParallel) else obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clf', default='cifar10_resnet18_dfst.pt')
    p.add_argument('--gen', default='cifar10_resnet18_dfst_generator.pt')
    p.add_argument('--out', default='weights/')
    p.add_argument('--arch', default='resnet18')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    clf_sd = _load_pickled_model(args.clf).state_dict()
    ref = getattr(models, args.arch)(num_classes=10, norm_layer=None)
    ref.load_state_dict(clf_sd, strict=True)          # fails loudly on any name/shape mismatch
    clf_out = os.path.join(args.out, 'dfst_resnet18.tar')
    torch.save({'state_dict': clf_sd}, clf_out)
    print(f'classifier: {len(clf_sd)} tensors -> {clf_out}')

    gen_sd = _load_pickled_model(args.gen).state_dict()
    CycleGenerator().load_state_dict(gen_sd, strict=True)
    gen_out = os.path.join(args.out, 'dfst_generator.pt')
    torch.save(gen_sd, gen_out)
    print(f'generator: {len(gen_sd)} tensors -> {gen_out}')


if __name__ == '__main__':
    main()
