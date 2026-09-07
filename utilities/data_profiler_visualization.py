# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
# logPath = "res/imagenette-160-ResNet18-v4-c26/W=2 A=2 data_seed=3/ST/n=U BS64 MD/o=Adam lr=0.01 wd=0/training_dp_20241105_195751"
# fileRangeFirst = 0
# fileRangeStep = 1
# fileRangeLast = None
# plotW = 800
# plotH = 600
# plotAsDensities = False
# nHistogramBins = 64

from argparse import ArgumentParser
argumentParser = ArgumentParser()
argumentParser.add_argument ("log_path_base", help = "path to the log files, without the incremental counter and the .pkl extension. Files matching (log_path_base + '*.pkl') will be processed.")
argumentParser.add_argument ("-first", "--file_range_first", type = int, default = 0, help = "Selection of logs from the sequence, visualized = log_files [first:last:step]. If first == -1 only the latest file will be loaded and visualized")
argumentParser.add_argument ("-last", "--file_range_last", type = int, default = None, help = "Selection of logs from the sequence, last file index. Default is None, indicating the latest file")
argumentParser.add_argument ("-step", "--file_range_step", type = int, default = 1, help = "Selection of logs from the sequence, index increment, default = 1")

argumentParser.add_argument ("-vh", "--visualize_histograms", action = "store_true", default = False, help = "Visualize histograms of tensors")
argumentParser.add_argument ("-vd", "--visualize_dead_channels", action = "store_true", default = False, help = "Visualize dead channels (majority of values are 0s or (K-1)s) in discretized activations")
argumentParser.add_argument ("-vdg", "--visualize_dead_channels_graymap", action = "store_true", default = False, help = "Visualize dead channels in discretized activations as an image where the percentage of 0s and (K-1)s is mapped to the shades of gray")

argumentParser.add_argument("-pd", "--plot_densities", action = "store_true", default = False, help = "Plot densities instead of histograms. Densities are directly comparable at the cost of losing visual resolution. Histograms may have different bin sizes (within one plot), therefore individual curves may be scaled differently")
argumentParser.add_argument("-pw", "--plot_width", type = int, default = 800, help = "Width of individual plots, in pixels, default 800")
argumentParser.add_argument("-ph", "--plot_height", type = int, default = 640, help = "Height of individual plots, in pixels, default 640")
argumentParser.add_argument("-nb", "--num_histogram_bins", type = int, default = 64, help = "Number of histogram bins, default 64")






import os, sys, glob
import PIL.Image
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils
from typing import Callable
from typing import List
import pickle
import gc
import matplotlib
import PIL
from itertools import chain

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import colormaps

sys.path.append ('.')
sys.path.append ('..')

import data_profiler 


def listFromHistograms (d):
# Combines histograms with different bin boundaries in a naive stupid way, by replicating count-times all the values.
# The resulting long list of values can be histogrammed again

    n = len (d)
    concatenated = []
    nElements = int (sum (sum (x ['data'][0]) for x in d))
    reductionFactor = max (10 ** (int (np.log10 (nElements)) - 4), 1)
    for i in range (n):                                 # over multiple records of one stored variable, eg. over batches
        h = d [i]['data'][0]                        
        binEdges = d [i]['data'][1]
        mids = ((binEdges [0:-1] + binEdges [1:])/2).tolist ()
        ts = [ [ val ] * int (count / reductionFactor) for val, count in zip (mids, h) ]
        concatenated.append (list (chain.from_iterable (ts)))
    return list (chain.from_iterable (concatenated)), reductionFactor



def genHistogramCollage ():
    collageImage = PIL.Image.new ('RGBA', (o.plot_width * maxCountPerLayer, o.plot_height * len (uniqueLayerNames)))
    cm = colormaps ['jet']

    for i in range (nVars):

        layer_name = last [i].layer_name
        desc = last [i].desc
        print (f'{i+1}/{nVars}: ' + layer_name + ' ' + desc + '                        ', end = '\r')

        row = uniqueLayerNames.index (layer_name)
        indices = [j for j, x in enumerate (last) if x.layer_name == layer_name ]
        col = indices.index (i)
        
        fig = plt.Figure (figsize = (o.plot_width/300, o.plot_height/300), dpi=300)
        canvas = FigureCanvas (fig)
        ax = fig.add_subplot(111)

        for j, d in enumerate (data):
            if len (d) > i and len (d [i]) > 0:

                if (d [i].format == data_profiler.LoggedFormat.RAW and d [i].K != 0):
                    h, e = torch.histogram (d [i][0]['data'], bins = torch.Tensor (range (d [i].K + 1)) - 0.5, density = o.plot_densities)
                    ax.bar ((e - 0.5).tolist ()[1:], h.tolist (), fill = False, edgecolor = cm (j / nSamples), linewidth = 0.5)

                elif (d [i].format == data_profiler.LoggedFormat.RAW):
                    h, e = torch.histogram (d [i][0]['data'], bins = o.num_histogram_bins, density = o.plot_densities)
                    ax.plot (e.tolist ()[1:], h.tolist (), color = cm (j / nSamples), linewidth = 0.5)

                elif ((d [i].format == data_profiler.LoggedFormat.HISTOGRAM or d [i].format == data_profiler.LoggedFormat.PER_CHANNEL_HISTOGRAM) and d [i].K != 0):
                    values, reductionFactor = listFromHistograms (d [i])
                    h, e = torch.histogram (torch.Tensor (values), bins = torch.Tensor (range (d [i].K + 1)) - 0.5, density = o.plot_densities)
                    ax.bar ((e - 0.5).tolist ()[1:], (h * reductionFactor).tolist (), fill = False, edgecolor = cm (j / nSamples), linewidth = 0.5)

                elif (d [i].format == data_profiler.LoggedFormat.HISTOGRAM or d [i].format == data_profiler.LoggedFormat.PER_CHANNEL_HISTOGRAM):
                    values, reductionFactor = listFromHistograms (d [i])
                    h, e = torch.histogram (torch.Tensor (values), bins = o.num_histogram_bins, density = o.plot_densities)
                    ax.plot (e.tolist ()[1:], (h * reductionFactor).tolist (), color = cm (j / nSamples), linewidth = 0.5)

                else:
                    ax.plot ([0], [0]) # to keep the line counter updated for the legend 
            else:
                ax.plot ([0], [0]) # to keep the line counter updated for the legend 

        if (row == 0 and col == 0): ax.legend (sampleNames, fontsize = 3)

        ax.set_title (layer_name + " " + desc + ", " + ("A " if last [i].is_activation == True else "W " if last [i].is_activation == False else "") + "x".join (str(x) for x in last [i].size) + (f", K = {d [i].K}" if d [i].K > 0 else "") , fontsize = 4)
        ax.tick_params(labelsize=4)
        ax.yaxis.get_offset_text().set_fontsize(4)
        
        fig.canvas.draw ()
        plotImage = PIL.Image.frombytes ('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
        plt.close (fig)
        
        collageImage.paste (plotImage, (o.plot_width * col, o.plot_height * row))

    print ("");
    return collageImage


def genDeadChannelsGraymap ():

    nEpochs = len (data)
    d = []
    for v in data:
        d.append ([ x for x in v if x.K > 0 and x.is_activation and len (x.size) == 4 and (x.format == data_profiler.LoggedFormat.RAW or x.format == data_profiler.LoggedFormat.PER_CHANNEL_HISTOGRAM)])   # only 4D discretized activations
    nVars = max ([ len (x) for x in d ])

    tensorSizes = [ x.size for x in d [-1] ]
    nChannels = [ x [1] for x in tensorSizes ]
    maxNChannels = max (nChannels) if len (nChannels) > 0 else 1

    cellWidth = 128//nEpochs
    cellHeight = 2
    borderWidth = 16
    collageImage = PIL.Image.new ('F', (cellWidth * nEpochs * nVars + borderWidth * (nVars - 1), cellHeight * maxNChannels * 2))

    for vi in range (nVars):

        layer_name = last [vi].layer_name
        desc = last [vi].desc
        print (f'{vi+1}/{nVars}: ' + layer_name + ' ' + desc + '                        ', end = '\r')

        img0 = np.zeros ((cellHeight * nChannels [vi], cellWidth * nEpochs), dtype = float)
        imgK = np.zeros ((cellHeight * nChannels [vi], cellWidth * nEpochs), dtype = float)
        total = tensorSizes [vi][0] * tensorSizes [vi][2] * tensorSizes [vi][3]
        K = d [-1][vi].K

        if (d [-1][vi].format == data_profiler.LoggedFormat.RAW):
            for ei in range (nEpochs):
                t = d [ei][vi][0]['data']
                for ci in range (nChannels [vi]):
                    x0 = float (torch.sum (t [:, ci, :, :] == 0)) / total           # percentage of zeroes
                    xK = float (torch.sum (t [:, ci, :, :] == K-1)) / total         # percentage of max vals (K-1)
                    img0 [ ci * cellHeight : (ci + 1) * cellHeight, ei * cellWidth : (ei + 1) * cellWidth ] = x0
                    imgK [ ci * cellHeight : (ci + 1) * cellHeight, ei * cellWidth : (ei + 1) * cellWidth ] = xK

        if (d [-1][vi].format == data_profiler.LoggedFormat.PER_CHANNEL_HISTOGRAM):
            for ei in range (nEpochs):
                t = d [ei][vi][0]['data']
                if (len (t [2]) == nChannels [vi]):     # check: number of stored histograms == number of channels
                    for ci in range (nChannels [vi]):
                        x0 = float (t [2][ci][0]) / total        # first bin of the histogram <- number of zeroes in the channel
                        xK = float (t [2][ci][-1]) / total        # last bin of the histogram <- number of maxvals in the channel
                        img0 [ ci * cellHeight : (ci + 1) * cellHeight, ei * cellWidth : (ei + 1) * cellWidth ] = x0
                        imgK [ ci * cellHeight : (ci + 1) * cellHeight, ei * cellWidth : (ei + 1) * cellWidth ] = xK

        collageImage.paste (PIL.Image.fromarray (img0 * 255), (cellWidth * nEpochs * vi + borderWidth * vi, (cellHeight * maxNChannels - img0.shape [0])//2))
        collageImage.paste (PIL.Image.fromarray (imgK * 255), (cellWidth * nEpochs * vi + borderWidth * vi, cellHeight * maxNChannels + (cellHeight * maxNChannels - imgK.shape [0])//2))

    print ("");
    return collageImage.convert ('L')


def genDeadChannelHistogramsCollage ():

    nEpochs = len (data)
    d = []
    for v in data:
        d.append ([ x for x in v if x.K > 0 and x.is_activation and len (x.size) == 4 and (x.format == data_profiler.LoggedFormat.RAW or x.format == data_profiler.LoggedFormat.PER_CHANNEL_HISTOGRAM)])   # only 4D discretized activations
    nVars = max ([ len (x) for x in d ])
    
    tensorSizes = [ x.size for x in d [-1] ]
    nChannels = [ x [1] for x in tensorSizes ]
    maxNChannels = max (nChannels)

    collageImage = PIL.Image.new ('RGBA', (o.plot_width * nVars, o.plot_height * 2))
    cm = colormaps ['jet']

    for vi in range (nVars):

        layer_name = d [-1][vi].layer_name
        desc = d[-1][vi].desc
        print (f'{vi+1}/{nVars}: ' + layer_name + ' ' + desc + '                                  ', end = '\r')

        fig = plt.Figure (figsize = (o.plot_width/300, 2*o.plot_height/300), dpi=300)
        canvas = FigureCanvas (fig)
        ax1 = fig.add_subplot (211)
        ax2 = fig.add_subplot (212)
        fig.tight_layout ()

        total = tensorSizes [vi][0] * tensorSizes [vi][2] * tensorSizes [vi][3]
        legends = []
        K = d [-1][vi].K

        for ei in [-1]: #range (nEpochs):
            t = d [ei][vi][0]['data']
            y0 = []
            yK = []
            yOther = []
            nAll0 = 0
            nAllK = 0
            n950 = 0
            n95K = 0

            if (d [ei][vi].format == data_profiler.LoggedFormat.RAW):
                for ci in range (nChannels [vi]):
                    n0 = float (torch.sum (t [:, ci, :, :] == 0))
                    nK = float (torch.sum (t [:, ci, :, :] == K - 1))
                    y0.append (n0 / total * 100)
                    yK.append (nK / total * 100)
                    if K > 2: yOther.append ((total - n0 - nK) / total * 100)
                
                    if n0 == total: nAll0 = nAll0 + 1
                    if nK == total: nAllK = nAllK + 1
                    if n0 >= total*0.95: n950 = n950 + 1
                    if nK >= total*0.95: n95K = n95K + 1

            if (d [ei][vi].format == data_profiler.LoggedFormat.PER_CHANNEL_HISTOGRAM):
                if (len (t [2]) == nChannels [vi]):     # check: number of stored histograms == number of channels
                    for ci in range (nChannels [vi]):
                        n0 = float (t [2][ci][0])
                        nK = float (t [2][ci][-1])
                        y0.append (n0 / total * 100)
                        yK.append (nK / total * 100)
                        if K > 2: yOther.append ((total - n0 - nK) / total * 100)

                        if (t [2][ci][0] == total): nAll0 = nAll0 + 1
                        if (t [2][ci][-1] == total): nAllK = nAllK + 1
                        if (t [2][ci][0] >= total*0.95): n950 = n950 + 1
                        if (t [2][ci][-1] >= total*0.95): n95K = n95K + 1
            
            legends.append (f'% of 0s, all 0s: {nAll0}x, 95% 0s: {n950}x')
            legends.append (f'% of {K-1}s, all {K-1}s: {nAllK}x, 95% {K-1}s: {n95K}x')
            if K > 2: legends.append ('% of other values')
            
            ax1.plot (range (nChannels [vi]), y0, 'o', linewidth = 0.5, markersize = 1)
            ax1.plot (range (nChannels [vi]), yK, 'o', linewidth = 0.5, markersize = 1)
            if K > 2: ax1.plot (range (nChannels [vi]), yOther, 'o', linewidth = 0.5, markersize = 1)

            ax2.plot (np.histogram (y0, range (101))[0], linewidth = 0.5, marker = 'o', markersize = 1)
            ax2.plot (np.histogram (yK, range (101))[0], linewidth = 0.5, marker = 'o', markersize = 1)
            if K > 2: ax2.plot (np.histogram (yOther, range (101))[0], linewidth = 0.5)

        ax1.set_title (layer_name + " " + desc + " " + "x".join (str(x) for x in d [-1][vi].size) + f", K = {K}", fontsize = 4)
        ax1.set_xlim (0, nChannels [vi])
        ax1.set_ylim (0, 100)
        ax1.set_xlabel ('Channel #', fontsize = 4)
        ax1.set_ylabel ('Percentage of 0s/Ks', fontsize = 4)
#        ax1.legend (legends, fontsize = 3)
        ax1.tick_params(labelsize=4)
        ax1.yaxis.get_offset_text().set_fontsize(4)

#        ax2.set_xlim (0, 100)
        ax2.set_xlabel ('Percentage of 0s/Ks', fontsize = 4)
        ax2.set_ylabel ('Histogram / channel count', fontsize = 4)
        ax2.legend (legends, fontsize = 3)
        ax2.tick_params(labelsize=4)
        ax2.yaxis.get_offset_text().set_fontsize(4)
        
        fig.canvas.draw ()
        plotImage = PIL.Image.frombytes ('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
        plt.close (fig)
        
        collageImage.paste (plotImage, (o.plot_width * vi, 0))

    print ("");
    return collageImage



# main

o = argumentParser.parse_args ()

logFiles = glob.glob (o.log_path_base + '*.pkl')
logFiles.sort ()
if (len (logFiles) == 0): print ("No datafiles found"); exit ()
logFiles = logFiles [o.file_range_first : o.file_range_last : o.file_range_step]
if (len (logFiles) == 0): print ("No datafiles selected"); exit ()

data = [None] * len (logFiles)
sampleNames = [str] * len (logFiles)
for i, fn in enumerate (logFiles):
    print (f'Loading file {i+1}/{len (logFiles)}', end = '\r')
    with open (fn, "rb") as f:
        data [i] = pickle.load (f)
        sampleNames [i] = data [i]['comment']
        data [i] = list (data [i]['vars'].values ())
print ("Data loaded              ")

last = data [-1]            # Logged variables can be added during the progress of training. The last file will contain records for all the stored variables, even if empty. So the last is used to pull names of variables from
nSamples = len (logFiles)
nVars = len (last)
assert (nVars == max ([ len (d) for d in data ]))

layerNames = [v.layer_name for v in last ]
uniqueLayerNames : List [str] = []
[ uniqueLayerNames.append (i) for i in layerNames if not uniqueLayerNames.count (i)] # order-preserving unique

perLayerCounts = [ layerNames.count (s) for s in uniqueLayerNames ]
maxCountPerLayer = max (perLayerCounts)

print ("Generating visualization ...")

if (o.visualize_histograms):
    collageImage = genHistogramCollage ()
    imgPath = o.log_path_base + ('_densities' if o.plot_densities else '') + '.png'
    print ("Saving image " + imgPath + " ...")
    collageImage.save (imgPath)

if (o.visualize_dead_channels_graymap):
    collageImage = genDeadChannelsGraymap ()
    imgPath = o.log_path_base + '_dead_channels_graymap.png'
    print ("Saving image " + imgPath + " ...")
    collageImage.save (imgPath)

if (o.visualize_dead_channels):
    collageImage = genDeadChannelHistogramsCollage ()
    imgPath = o.log_path_base + '_dead_channel_histograms.png'
    print ("Saving image " + imgPath + " ...")
    collageImage.save (imgPath)

print ("Done")
    