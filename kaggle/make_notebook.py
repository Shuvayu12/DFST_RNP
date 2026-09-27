"""Regenerates rnp_dfst.ipynb. Shell cells are kept to one physical line each because IPython's
`!` magic does not reliably honour backslash line continuations."""
import json
import os

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text})


md("# RNP vs DFST on CIFAR-10\n\n"
   "Runs Reconstructive Neuron Pruning (ICML 2023) against a DFST-backdoored ResNet-18.\n\n"
   "**Settings -> Accelerator: GPU (T4/P100)**. Internet must be ON for the git clone and the CIFAR-10 download.")

code("REPO_URL = 'https://github.com/Shuvayu12/DFST_RNP.git'\n"
     "TARGET_LABEL = 0   # DFST target class\n")

code("import torch\n"
     "print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),\n"
     "      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')\n")

code("!git clone $REPO_URL RNP\n%cd RNP\n!ls -la\n")

md("## 1. Convert the DFST checkpoints\n\n"
   "The two `.pt` files are whole pickled models from the original DFST code. "
   "`convert_dfst.py` turns them into plain state dicts under `weights/`.")

code("!mkdir -p weights logs\n"
     "!python convert_dfst.py --clf cifar10_resnet18_dfst.pt --gen cifar10_resnet18_dfst_generator.pt --out weights/\n"
     "!ls -la weights/\n")

md("## 2. Sanity check the backdoor\n\n"
   "One normalization row should show clean accuracy around 0.9 **and** ASR around 0.9+. "
   "If that row is not the `repo` one, edit `MEAN_CIFAR10 / STD_CIFAR10` in `data/poison_tool_cifar.py` before continuing.")

code("!python check_dfst.py --n 2000 --target_label $TARGET_LABEL\n")

md("## 3. Run RNP\n\n"
   "Unlearn -> recover masks -> prune. Same hyper-parameters as the paper's BadNets run; "
   "only the trigger type and model path change. Roughly 10-15 minutes on a T4.")

code("RNP_ARGS = ' '.join([\n"
     "    '--arch resnet18',\n"
     "    '--backdoor_model_path weights/dfst_resnet18.tar',\n"
     "    '--trigger_type dfstTrigger',\n"
     "    '--dfst_generator_path weights/dfst_generator.pt',\n"
     "    f'--target_label {TARGET_LABEL}',\n"
     "    '--target_type all2one',\n"
     "    '--output_weight weights/',\n"
     "    '--log_root logs/',\n"
     "])\n"
     "print(RNP_ARGS)\n")

code("!python main.py $RNP_ARGS\n")

md("## 4. Results\n\n"
   "`logs/pruning_by_threshold.txt` lists, for each mask threshold, how many neurons were pruned and the resulting "
   "attack success rate (PoisonACC) and clean accuracy (CleanACC).")

code("import pandas as pd\n"
     "df = pd.read_csv('logs/pruning_by_threshold.txt', sep=r'\\s*\\t\\s*', engine='python')\n"
     "df\n")

code("import matplotlib.pyplot as plt\n"
     "fig, ax = plt.subplots(figsize=(7, 4))\n"
     "ax.plot(df['Mask'], df['PoisonACC'], marker='o', label='Attack success rate')\n"
     "ax.plot(df['Mask'], df['CleanACC'], marker='s', label='Clean accuracy')\n"
     "ax.set_xlabel('pruning threshold on mask value'); ax.set_ylabel('accuracy')\n"
     "ax.set_title('RNP on DFST (ResNet-18, CIFAR-10)'); ax.legend(); ax.grid(alpha=.3); plt.show()\n")

md("## 5. Keep the outputs\n\n"
   "Copy logs and the unlearned checkpoint to `/kaggle/working` so they survive *Save & Run All*.")

code("!mkdir -p /kaggle/working/rnp_dfst_out\n"
     "!cp -r logs /kaggle/working/rnp_dfst_out/\n"
     "!cp weights/unlearned_model_last.tar /kaggle/working/rnp_dfst_out/ 2>/dev/null || true\n"
     "!ls -la /kaggle/working/rnp_dfst_out\n")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rnp_dfst.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('wrote', out, 'with', len(cells), 'cells')
