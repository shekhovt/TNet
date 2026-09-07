# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
import sys
sys.path.append ('.')
sys.path.append ('..')

import train
import utilities.imagenet_class_descriptions as imagenet_class_descriptions
import torchvision
from torchview import draw_graph
from torchviz import make_dot

from torchvision.utils import save_image
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

def evaluate_m(net, loader, method, variant):
    root_dir = train.find_root(o)
    net.eval()

    with train.torch.no_grad():
        for i, (data, target) in enumerate (loader):
            targetGPU = target.to(train.dev)
            dataGPU = data.to(train.dev)
            assert (dataGPU.device == train.dev)

            scoresGPU = net.forward (dataGPU, method=method)
            g = draw_graph (net, input_data=dataGPU, expand_nested=True, depth=32, method=method, hide_module_functions = False)
            g.visual_graph.render (root_dir + o.net_name + '_model_visualization')

            break





def run_test(data, o, variant):
    net = train.current_setup.create_net(o)
    try:
        state = train.load_state(o, net, variant)
        print(variant)
    except FileNotFoundError as e:
        print('File not found')
        print(e)
#        return

    train.data_sweep_accumulate_BN(net, data.train_loader_test, o.m_eval_t_det) # flat average should not depend on order
    evaluate_m(net, data.test_loader, o.m_eval_t_det, variant)


# main


args_str = ' '.join (sys.argv[1:])
o = train.o_from_str (args_str)
train.setup (o)
root_dir = train.find_root(o)
print(root_dir)

data = train.current_setup.create_data(o)
subclasses = imagenet_class_descriptions.get_subclasses (o.num_classes)
subclasses_reverse_index = [ x for x in range (1000) ]
if (o.num_classes < 1000):
    subclasses_reverse_index = [ subclasses.index(x) if x in set (subclasses) else -1 for x in np.arange (1000) ]

# for x in imagenet_class_descriptions.desc_subclasses (subclasses): print (x)

with train.torch.cuda.device (train.dev):
    run_test(data, o, 'best_val_A1')
#    run_test(data, o, 'final')
