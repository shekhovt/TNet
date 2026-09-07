# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
class OpCounter:

    def __init__ (self):
        self.clear ();

    def clear (self):
        self.nLayers = 0;
        self.nMults = 0;
        self.nAdds = 0;
        self.nCmps = 0;
        self.nShifts = 0;
        self.nMemReads = 0;
        self.nMemWrites = 0;
        self.nFusedReads = 0;
        self.nFusedWrites = 0;
        self.nFusedMemReadsW = 0;
        self.nFusedMemReadsA = 0;

    def opConv2D (self, inSize, outSize, kernelSize):
        self.nLayers += 1;
        self.nMults += outSize [2] * outSize [3] * kernelSize [0] * kernelSize [1] * kernelSize [2] * kernelSize [3];
        self.nAdds +=  outSize [2] * outSize [3] * kernelSize [0] * kernelSize [1] * kernelSize [2] * kernelSize [3];
        self.nMemReads += inSize [1] * inSize [2] * inSize [3] + kernelSize [0] * kernelSize [1] * kernelSize [2] * kernelSize [3];
        self.nMemWrites += outSize [1] * outSize [2] * outSize [3];
        self.nFusedReads += inSize [1] * inSize [2] * inSize [3] + kernelSize [0] * kernelSize [1] * kernelSize [2] * kernelSize [3];
        self.nFusedWrites += outSize [1] * outSize [2] * outSize [3];
        self.nFusedReadsA += inSize [1] * inSize [2] * inSize [3]
        self.nFusedReadsW += kernelSize [0] * kernelSize [1] * kernelSize [2] * kernelSize [3];

    def opBatchNorm2D (self, inSize, outSize):
        self.nLayers += 1;
        self.nMults += inSize [1] * inSize [2] * inSize [3] * 2;
        self.nAdds +=  inSize [1] * inSize [2] * inSize [3] * 2;
        self.nMemReads += inSize [1] * inSize [2] * inSize [3] + 2 * inSize [1];
        self.nMemWrites += outSize [1] * outSize [2] * outSize [3];
        self.nFusedReads += 2 * inSize [1];
        self.nFusedReadsW += 2 * inSize [1];

    def opReLU (self, inSize, outSize):
        self.nLayers += 1;
        self.nCmps += inSize [1] * inSize [2] * inSize [3];
        self.nMemReads += inSize [1] * inSize [2] * inSize [3];
        self.nMemWrites += outSize [1] * outSize [2] * outSize [3];

    def opReLU6 (self, inSize, outSize):
        self.nLayers += 1;
        self.nCmps += inSize [1] * inSize [2] * inSize [3] * 2;
        self.nMemReads += inSize [1] * inSize [2] * inSize [3];
        self.nMemWrites += outSize [1] * outSize [2] * outSize [3];

    def opAdd (self, inSize, outSize):
        self.nLayers += 1;
        self.nAdds += inSize [1] * inSize [2] * inSize [3];
        self.nMemReads += inSize [1] * inSize [2] * inSize [3] * 2;
        self.nMemWrites += outSize [1] * outSize [2] * outSize [3];

    def opLinear (self, inSize, outSize):
        self.nLayers += 1;
        self.nMult += inSize [1] * outSize [1];
        self.nAdds += inSize [1] * outSize [1] + outSize [1];
        self.nMemReads += inSize [1] * outSize [1] + inSize [1];
        self.nMemWrites += outSize [1];
        self.nFusedReads += inSize [1] * outSize [1] + inSize [1];
        self.nFusedReadsW += inSize [1] * outSize [1] + inSize [1];

    def opMaxPool2D (self, inSize, outSize):
        self.nLayers += 1;
        self.nCmps += inSize [1] * inSize [2] * inSize [3];
        self.nMemReads += inSize [1] * inSize [2] * inSize [3];
        self.nMemWrites += outSize [1] * outSize [2] * outSize [3];
        self.nFusedWrites += outSize [1] * outSize [2] * outSize [3] - inSize [1] * inSize [2] * inSize [3];

    def opAvgPool2D (self, inSize, outSize):
        self.nLayers += 1;
        self.nAdds += inSize [1] * inSize [2] * inSize [3] * 2;
        self.nMults += outSize [1] * outSize [2] * outSize [3];
        self.nMemReads += inSize [1] * inSize [2] * inSize [3];
        self.nMemWrites += outSize [1] * outSize [2] * outSize [3];
        self.nFusedWrites += outSize [1] * outSize [2] * outSize [3] - inSize [1] * inSize [2] * inSize [3];

    def opDropout (self, inSize, outSize):
        self.nLayers += 1;
        self.nMemReads += inSize [1];
        self.nMemWrites += outSize [1];


    def addLayer (self, layerName, inputSize, outputSize, params):
#        print ('      ', layerName, inputSize, outputSize, params);
        foo = (self.nMults, self.nAdds, self.nCmps, self.nMemReads, self.nMemWrites, self.nFusedReads, self.nFusedWrites);
        if (layerName == 'Conv2d'):
            self.opConv2D (tuple (inputSize), tuple (outputSize), tuple (params [0]))
        elif (layerName == 'Identity'):
            pass
        elif (layerName == 'Flatten'):
            pass
        elif (layerName == 'Sequential'):
            pass
        elif (layerName == 'ReLU'):
            self.opReLU (tuple (inputSize), tuple (outputSize))
        elif (layerName == 'ReLU6'):
            self.opReLU6 (tuple (inputSize), tuple (outputSize))
        elif (layerName == 'BatchNorm2d'):
            self.opBatchNorm2D (tuple (inputSize), tuple (outputSize))
        elif (layerName == 'add_' or layerName == 'ResNetBasicLayer' or layerName == 'ResNetBottleNeckLayer' or layerName == 'MobileNetV2InvertedResidual'):
            self.opAdd (tuple (inputSize), tuple (outputSize))
        elif (layerName == 'Linear'):
            self.opLinear (tuple (inputSize), tuple (outputSize))
        elif (layerName == 'MaxPool2d'):
            self.opMaxPool2D (tuple (inputSize), tuple (outputSize))
        elif (layerName == 'AvgPool2d'):
            self.opAvgPool2D (tuple (inputSize), tuple (outputSize))
        elif (layerName == 'AdaptiveAvgPool2d'):
            self.opMaxPool2D (tuple (inputSize), tuple (outputSize))
        elif (layerName == 'Dropout'):
            self.opDropout (tuple (inputSize), tuple (outputSize))
        elif (layerName.startswith ('ResNet')):
            pass
        elif (layerName.startswith ('MobileNet')):
            pass
        else:
            print (f'Skipping unimplemented layer {layerName}')
        bar = (self.nMults, self.nAdds, self.nCmps, self.nMemReads, self.nMemWrites, self.nFusedReads, self.nFusedWrites);
#        print (f'        {[ bar [i] - foo [i] for i in range (len (foo)) ]}') # print increments


    def printStats (self):
        print (f'nLayers: {self.nLayers}');
        print (f'nMults: {self.nMults}, {self.nMults/1000/1000:.2f}M');
        print (f'nAdds: {self.nAdds}, {self.nAdds/1000/1000:.2f}M');
        print (f'nCmps: {self.nCmps}, {self.nCmps/1000/1000:.2f}M');
        print (f'nShifts: {self.nShifts}, {self.nShifts/1000/1000:.2f}M');
        print (f'nMemWrites: {self.nMemWrites}, {self.nMemWrites/1024/1024:.2f}M');
        print (f'nMemReads: {self.nMemReads}, {self.nMemReads/1024/1024:.2f}M');
        print (f'nFusedWrites: {self.nFusedWrites}, {self.nFusedWrites/1024/1024:.2f}M');
        print (f'nFusedReads: {self.nFusedReads}, {self.nFusedReads/1024/1024:.2f}M');
        print (f'   nFusedReadsW: {self.nFusedReadsW}, {self.nFusedReadsW/1024/1024:.2f}M');
        print (f'   nFusedReadsA: {self.nFusedReadsA}, {self.nFusedReadsA/1024/1024:.2f}M');

    def printNonzeroStats (self):
        if self.nMults > 0: print (f'nMults: {self.nMults}, {self.nMults/1024/1024:.2f}M');
        if self.nAdds > 0: print (f'nAdds: {self.nAdds}, {self.nAdds/1024/1024:.2f}M');
        if self.nCmps > 0: print (f'nCmps: {self.nCmps}, {self.nCmps/1024/1024:.2f}M');
        if self.nShifts > 0: print (f'nShifts: {self.nShifts}, {self.nShifts/1024/1024:.2f}M');
        if self.nMemWrites > 0: print (f'nMemWrites: {self.nMemWrites}, {self.nMemWrites/1024/1024:.2f}M');
        if self.nMemReads > 0: print (f'nMemReads: {self.nMemReads}, {self.nMemReads/1024/1024:.2f}M');
        if self.nFusedWrites > 0: print (f'nFusedWrites: {self.nFusedWrites}, {self.nFusedWrites/1024/1024:.2f}M');
        if self.nFusedReads > 0: print (f'nFusedReads: {self.nFusedReads}, {self.nFusedReads/1024/1024:.2f}M');

