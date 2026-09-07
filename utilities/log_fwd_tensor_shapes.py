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
sys.path = sorted (sys.path, key = lambda x: x.find (sys.prefix) >= 0, reverse=True)    # move the virtual env to the top of the module search paths --> venv pacpages will take precedence over system ones

import numpy as np
import pickle

import torch
import torch.nn as nn
import torchvision
from torchview import draw_graph
from torchviz import make_dot
from torchvision.utils import save_image
import torchvision.models as models

import torchscope

from transformers import AutoImageProcessor, AutoModelForImageClassification, MobileNetV2ForImageClassification, MobileNetV1ForImageClassification



modelName = "resnet-50"
#modelName = "mobilenet_v2_1.0_224"
#modelName = "mobilenet_v1_1.0_224"
dataSize = [ 1, 3, 224, 224 ];

log = []




def hook (layer, input, output):
    try:
        m = type (layer).__name__;
        i = [list (x.size ()) for x in input];
        o = [ list (output.size ()) ];
        p = [list (x.size ()) for x in layer.parameters ()];

        print (m, type (input).__name__, type (output).__name__);
        print (f"    Inputs: {i}")
        print (f"    Outputs: {o}")
        print (f"    Parameters: {p}")

        log.append ([ m, i, o, p ]);
    except:
        pass



def installHook (net):
    if (hasattr (net, '_modules')):
       for name, layer in net._modules.items():
           layer.register_forward_hook (hook);
           installHook (layer);






model = AutoModelForImageClassification.from_pretrained ("microsoft/" + modelName);
#model = MobileNetV2ForImageClassification.from_pretrained ("google/" + modelName);
#model = MobileNetV1ForImageClassification.from_pretrained ("google/" + modelName);
installHook (model)
testInput = torch.randn (dataSize);
model.eval ();
model (testInput);
print (f"N of parameters: {sum (p.numel () for p in model.parameters ())}")
print (f"N of trainable parameters: { sum (p.numel () for p in model.parameters () if p.requires_grad) }")
#torchscope.scope (model, input_size = tuple (dataSize [1:]));
#g = draw_graph (model, input_data = testInput, expand_nested = True, depth=32, hide_module_functions = True)
#g.visual_graph.render (modelName + '_model_visualization')


with open (modelName + '_fwd_log.pkl', 'wb') as f:
    pickle.dump (log, f)


