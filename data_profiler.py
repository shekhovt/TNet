# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
# Data logging for post analysis
# The functionality is implemented in the DataProfiler class, LoggedVarsConfiguration is used for configurationof what is logged
# The logging itself occurs in Method::dispatch, and is controlled in the train script

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict
from typing import Tuple
from typing import List
from typing import Optional
from enum import Enum
from dataclasses import dataclass
import json

class LoggedFormat (str, Enum):   
    RAW = 'RAW'                                         # typically a tensor, no information is lost, but memory/storage inefficient
    HISTOGRAM = 'HISTOGRAM'                             # sufficient for most visualizations
    PER_CHANNEL_HISTOGRAM = 'PER_CHANNEL_HISTOGRAM'     # required for per-channel stats of the activations
#    MEAN_VARIANCE = 'MEAN_VARIANCE'
#    PER_CHANNEL_MEAN_VARIANCE = 'PER_CHANNEL_MEAN_VARIANCE'

@dataclass
class LoggedVarSpecs:
    layer_class_name : str;                  # matched against layer.__class__.__name__
    description: str;                        # unique description of the logged variable that differentiates variables of the same module, will be shown in titles of the output plots
    log_forward_input: bool = False;         # log the input to the forward call
    log_forward_output: bool = False;        # log the result of the forward call
    log_expression: str | None = None;       # log an expression that evaluates to a tensor. Typically a member variable to be logged, eg. 'layer.Bias'. Available vars: layer: nn.module, method: Method, input: Tensor, output: Tensor, *args, **kwargs: additional args passed to dispatch
    condition: str | None = None;            # logged only if eval(condition) is True. Available vars: layer: nn.module, method: Method, input: Tensor, output: Tensor, *args, **kwargs: additional args passed to dispatch
    discretization_K: str | None = None;     # expression evaluating to an int, nonzero if the results shall be shown as discretized variables (bars instead of histograms)
    is_activation: str | None = None;        # if None: False for log_expressions. If not None: expression evaluating to bool
    logged_format: LoggedFormat = LoggedFormat.RAW;

class LoggedVarsConfiguration (List [LoggedVarSpecs]):
# JSON seriazable list of tensors to be logged

    def __init__ (self):
        self.groups = []
        self.filter = None
        self.skip_activations = False
        self.skip_weights = False
        self.discretized_only = False

    def setDefaults (self):
        self.append (LoggedVarSpecs ('ScaleBias', 'ScaleBias input',  log_forward_input = True, logged_format = LoggedFormat.HISTOGRAM))
        self.append (LoggedVarSpecs ('ScaleBias', 'ScaleBias scale',  log_expression = 'layer.weight', condition = 'layer.is_activation', logged_format = LoggedFormat.HISTOGRAM))
        self.append (LoggedVarSpecs ('ScaleBias', 'ScaleBias bias',   log_expression = 'layer.bias', condition = 'layer.is_activation and layer.bias is not None', logged_format = LoggedFormat.HISTOGRAM))
        self.append (LoggedVarSpecs ('ScaleBias', 'ScaleBias output', log_forward_output = True, logged_format = LoggedFormat.HISTOGRAM))

        self.append (LoggedVarSpecs ('QConv2d', 'QConv2D input', log_forward_input = True, logged_format = LoggedFormat.HISTOGRAM))
#        self.append (LoggedVarSpecs ('QConv2d', 'QConv2D Weight', log_expression = 'layer.weight'))  # discretized weights are only a temp var, not available :-(
        self.append (LoggedVarSpecs ('QConv2d', 'QConv2D output', log_forward_output = True, logged_format = LoggedFormat.HISTOGRAM))

        self.append (LoggedVarSpecs ('Gate', 'Gate input', log_forward_input = True, logged_format = LoggedFormat.HISTOGRAM))
        self.append (LoggedVarSpecs ('Gate', 'Gate output', log_forward_output = True, logged_format = LoggedFormat.HISTOGRAM))

        self.append (LoggedVarSpecs ('QReLU', 'QReLU input', log_forward_input = True, is_activation = 'layer.is_activation', logged_format = LoggedFormat.HISTOGRAM))
        self.append (LoggedVarSpecs ('QReLU', 'QReLU output', log_forward_output = True, is_activation = 'layer.is_activation', logged_format = LoggedFormat.HISTOGRAM))

        # self.append (LoggedVarSpecs ('Squash', 'Squash input', log_forward_input = True, is_activation = 'layer.is_activation', logged_format = LoggedFormat.HISTOGRAM))
        # self.append (LoggedVarSpecs ('Squash', 'Squash output', log_forward_output = True, is_activation = 'layer.is_activation', logged_format = LoggedFormat.HISTOGRAM))

        self.append (LoggedVarSpecs ('Quant', 'Quant input', log_forward_input = True, is_activation = 'layer.is_activation', logged_format = LoggedFormat.HISTOGRAM))
        self.append (LoggedVarSpecs ('Quant', 'Quant output (A)', log_forward_output = True, is_activation = 'layer.is_activation', condition = 'layer.is_activation', discretization_K = 'layer.K', logged_format = LoggedFormat.PER_CHANNEL_HISTOGRAM))
        self.append (LoggedVarSpecs ('Quant', 'Quant output (W)', log_forward_output = True, is_activation = 'layer.is_activation', condition = 'not layer.is_activation', discretization_K = 'layer.K', logged_format = LoggedFormat.HISTOGRAM))

        # self.append (LoggedVarSpecs ('BatchNorm2d', 'BatchNorm2D input', log_forward_input = True, logged_format = LoggedFormat.HISTOGRAM))
        self.append (LoggedVarSpecs ('BatchNorm2d', 'BatchNorm2D Weight', log_expression = 'layer.weight', condition = 'layer.affine'))
        self.append (LoggedVarSpecs ('BatchNorm2d', 'BatchNorm2D Bias', log_expression = 'layer.bias', condition = 'layer.affine'))
        # self.append (LoggedVarSpecs ('BatchNorm2d', 'BatchNorm2D output', log_forward_output = True, logged_format = LoggedFormat.HISTOGRAM))

        self.groups = [ 'QReLU' ]
        self.filter = None
        self.skip_activations = False
        self.skip_weights = False
        self.discretized_only = False


    def load (self, path: str):
        with open (path, 'r') as f:
            v = json.load (f, object_hook = lambda d: LoggedVarSpecs (layer_class_name = d ['layer_class_name'], log_forward_input = d ['log_forward_input'], log_forward_output = d ['log_forward_output'], log_expression = d ['log_expression'], condition = d ['condition'], discretization_K = d ['discretization_K'], is_activation = d ['is_activation'], logged_format = d ['logged_format'], description = d ['description']) if 'layer_class_name' in d else d)

        vars = v.get ('logged_variables')
        if (vars is not None and len (vars) > 0):
            self.clear ();
            self.extend (vars);
        self.groups = v.get ('groups', [])
        self.filter = v.get ('filter', None)
        self.skip_activations = v.get ('skip_activations', True)
        self.skip_weights = v.get ('skip_weights', True)
        self.discretized_only = v.get ('discretized_only', False)

    def save (self, path: str):
        with open (path, 'w') as f:
            f.writelines (json.dumps ({"skip_activations": self.skip_activations, "skip_weights": self.skip_weights, "discretized_only": self.discretized_only, "groups": self.groups, "filter": self.filter, "logged_variables": self}, default = vars, indent = 1))

    groups: List [str]
    filter: str | None
    skip_activations: bool
    skip_weights: bool
    discretized_only: bool





class _LoggedVariable (List [Dict [str, int | float | Tuple | torch.Tensor]]):   # 
   
    def __init__ (self, layer_name: str, desc: str, is_activation: bool):
        self.layer_name = layer_name            
        self.desc = desc                        
        self.is_activation = is_activation
        self.nSaved = 0
        self.size = []
        self.K = 0
        self.logNextPass = False
        self.counter : int = 0
    
    def acceptsData (self, loggingFrequency: int) -> bool:

        return self.logNextPass or (loggingFrequency != 0 and self.counter % loggingFrequency == loggingFrequency - 1)

    def log (self, loggingFrequency: int, data: Tuple | List | torch.Tensor, K: int = 0):
    
        if self.acceptsData (loggingFrequency):
            processTimeMS = time.process_time_ns () / 1e6

            if isinstance (data, Tuple):
                self.append ({ "counter": self.counter, "ts": processTimeMS, "data": data })

            if isinstance (data, List):
                self.log (loggingFrequency, tuple (data), K)

            if isinstance (data, torch.Tensor):
                self.append ({ "counter": self.counter, "ts": processTimeMS, "data": data.detach ().cpu () })
                    
        self.counter = self.counter + 1
        self.logNextPass = False


    layer_name: str
    desc: str
    is_activation: bool
    nSaved : int
    size: List [int]
    K: int
    format: LoggedFormat

# end class _LoggedVariable



class DataProfiler:

    @classmethod
    def configure (cls, varConfig, net: torch.nn.Module, loggingFrequency = 0, logFirstRun = True):
        cls.loggingFrequency = loggingFrequency
        cls.logFirstRun = logFirstRun
        cls.enabled = False
        cls.varConfig = varConfig

        cls._moduleLayerNames = {}    # Module names can be queried from a constructed net. Used in the visualization in chart labels 
        if net is not None:
            for name, module in net.named_modules (): 
                if isinstance (module, DataProfilerModule):
                    cls._moduleLayerNames [id (module)] = name [ : name.rfind ('.') ]
                else:
                    cls._moduleLayerNames [id (module)] = name
                for childName, childModule in module.named_children (): 
                    cls._moduleParents [id (childModule)] = module

    enabled: bool = False          # enable/disable logging functionality, e.g. to isolate training from interleaved validations, or to log only selected epochs. Tested as the very first thing in logging calls, for performance reasons if logging off
    loggingFrequency: int = 0      # 0 - logging off, 1 - every frame logged, ...
    logFirstRun: bool = False      # can be set before the first logData calls, ie. before the logging is configured
    varConfig : LoggedVarsConfiguration = LoggedVarsConfiguration () # logging configuration

    @classmethod
    def logNextForwardPass (cls, comment: str = ""):   # instructs each logged variable to store the next value passing through
        for v in cls._loggedData.values ():
            v.logNextPass = True

    @classmethod
    def saveCheckpoint (cls, dirPath:str, comment: str = ""):  
        cls._nSaved = cls._nSaved + 1
        path = f"{dirPath}dp{cls._dateTimeString}_{cls._nSaved:05}.pkl"
        d = {}
        for k, v in cls._loggedData.items ():
            if (len (v) > 0): 
                d [k] = v
        with open (path, 'wb') as f:
            pickle.dump ({ "comment": comment, "vars": d }, f)
        for v in cls._loggedData.values ():
            v.clear ()
        
    @classmethod
    def hasDataToSave (cls): 
        for v in cls._loggedData.values ():
            if (len (v) > 0): return True
        return False

    @classmethod
    def clear (cls): 
        cls._loggedData.clear()
        
    @classmethod
    def logData (cls, module: nn.Module, desc: str, x: torch.Tensor, format: LoggedFormat, is_activation: bool, K: int = 0): 
        if (cls.enabled and (is_activation is None or is_activation == True and not cls.varConfig.skip_activations or is_activation == False and not cls.varConfig.skip_weights) and (K > 0 or not cls.varConfig.discretized_only)):
            moduleID = id (module)
            if not (moduleID, desc) in cls._loggedData:     # logged variable seen for the first time -> add to the list
                layer_name = f"{moduleID}"
                desc_prefix = ""
                if moduleID in cls._moduleLayerNames.keys ():
                    layer_name = cls._moduleLayerNames [moduleID]
                if (moduleID in cls._moduleParents.keys () and id (cls._moduleParents [moduleID]) in cls._moduleLayerNames.keys () and cls._moduleParents [moduleID].__class__.__name__ in cls.varConfig.groups): 
                    layer_name = cls._moduleLayerNames [id (cls._moduleParents [moduleID])]
                    desc_suffix = f'#{[ k for k,v in cls._moduleParents.items () if v == cls._moduleParents [moduleID] ].index (moduleID) + 1} '

                cls._loggedData [(moduleID, desc)] = _LoggedVariable (layer_name, desc_prefix + desc, is_activation = is_activation)
                cls._loggedData [(moduleID, desc)].size = list (x.size ())
                cls._loggedData [(moduleID, desc)].format = format
                cls._loggedData [(moduleID, desc)].K = K
                cls._loggedData [(moduleID, desc)].log (1 if cls.logFirstRun else cls.loggingFrequency, cls.transformData (x, format, K), K)
            else:
                if cls._loggedData [(moduleID, desc)].acceptsData (cls.loggingFrequency):
                    cls._loggedData [(moduleID, desc)].log (cls.loggingFrequency, cls.transformData (x, format, K), K)


    @classmethod
    def transformData (cls, input: torch.Tensor, form: LoggedFormat, K: int):   # from a tensor to histogram(s)

        if (form == LoggedFormat.RAW):
            return input.detach ().cpu ()

        if (form == LoggedFormat.HISTOGRAM):
            x = input.detach ().cpu ()
            if K > 0:
                binEdges = torch.from_numpy (np.linspace (-0.5, K - 0.5, num = K + 1, dtype = 'float32'))
                h, e = torch.histogram (x, binEdges)      # fixed binning for discretized data
            else:
                h, e = torch.histogram (x, 1000)          # adaptive binning
            return (h, e)

        if (form == LoggedFormat.PER_CHANNEL_HISTOGRAM):
            if (isinstance (input, torch.Tensor) and len (input.size ()) == 4):  # 4D tensors only, the second dimension is assumed to be the channel - true only for ACTIVATIONS
                x = input.detach ().cpu ()
                if K > 0:
                    binEdges = torch.from_numpy (np.linspace (-0.5, K - 0.5, num = K + 1, dtype = 'float32'))
                    globalH, globalE = torch.histogram (x, binEdges)         # fixed binning for discretized data
                else:
                    globalH, globalE = torch.histogram (x, 1000)             # determine bin edges that will be common for all channels
                nChannels = x.size (1)
                perChannelH : List [torch.return_types.histogram] = [None] * nChannels

                for ci in range (nChannels):
                    h, e = torch.histogram (x [:, ci, :, :], globalE)
                    perChannelH [ci] = h

                return (globalH, globalE, perChannelH)                        # return per-channel histograms and also the histogram of the whole tensor
            else:
                return None

        return None

    @classmethod
    def logPreForward (cls, layer: nn.Module, method, input: torch.Tensor, *args, **kwargs):    # called at the beginning of Method::dispatch
        if cls.enabled:
            vList = [ x for x in cls.varConfig if x.layer_class_name == layer.__class__.__name__ ]
            for v in vList:
                if v.condition is None or eval (v.condition) == True:
                    if (v.log_forward_input):
                        cls.logData (layer, v.description, input, v.logged_format, None if v.is_activation is None else eval (v.is_activation), 0 if v.discretization_K is None else eval (v.discretization_K))

    @classmethod
    def logPostForward (cls, layer: nn.Module, method, input: torch.Tensor, output: torch.Tensor, *args, **kwargs):     # called at the end of Method::dispatch
        if cls.enabled:
            vList = [ x for x in cls.varConfig if x.layer_class_name == layer.__class__.__name__ ]
            for v in vList:
                if v.condition is None or eval (v.condition) == True:
                    if (v.log_expression is not None):
                        cls.logData (layer, v.description, eval (v.log_expression), v.logged_format, False if v.is_activation is None else eval (v.is_activation), 0 if v.discretization_K is None else eval (v.discretization_K))
                    elif (v.log_forward_output):
                        cls.logData (layer, v.description, output, v.logged_format, None if v.is_activation is None else eval (v.is_activation), 0 if v.discretization_K is None else eval (v.discretization_K))


    _loggedData : Dict [Tuple [int, str], _LoggedVariable] = {}
    _nSaved : int = 0
    _dateTimeString = "" # "_" + time.strftime ("%Y%m%d_%H%M%S")   # goes to the pickle filenames
    _moduleLayerNames : Dict [int, str] = {}
    _moduleParents : Dict [int, nn.Module] = {}
# end class DataProfiler

class DataProfilerModule (nn.Module):   # not used anymore, a module that can be inserted into the net, which then logs data passing through
    instanceCounter : int = 0

    def __init__(self, is_activation: bool, desc: str, K:int = 0):
        super ().__init__()
        self.instanceID = DataProfilerModule.instanceCounter
        DataProfilerModule.instanceCounter = DataProfilerModule.instanceCounter + 1
        self.K = K
        self.is_activation = is_activation
        self.desc = desc + (" A" if is_activation else " W")
                
    def forward (self, x):
        if (DataProfiler.enabled and (self.is_activation and not DataProfiler.varConfig.skip_activations or not self.is_activation and not DataProfiler.varConfig.skip_weights)):
            assert isinstance(x, torch.Tensor)
            if self.is_activation:
                DataProfiler.logHistogram (self, self.desc, x.detach (), self.is_activation, self.K)
            else:
                DataProfiler.logData (self, self.desc, x.detach (), self.is_activation, self.K)
        return x





if __name__ == "__main__":    ### test
    lv = LoggedVarsConfiguration ()
    lv.setDefaults ()
    lv.save ('test.json')

    x = LoggedVarsConfiguration ()
    x.load ('test.json')
    print (x [0].logged_format == 'RAW')