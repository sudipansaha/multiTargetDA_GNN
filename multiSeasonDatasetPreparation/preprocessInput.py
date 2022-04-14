#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 20:05:36 2019

@author: sudipan
"""

import os
import numpy as np
import tifffile
import glob



def saturateSomePercentile(inputMap,percentileToSaturate):
    inputMapNormalized=(inputMap-np.amin(inputMap))/(np.percentile(inputMap,(100-percentileToSaturate))-np.amin(inputMap))
    inputMapNormalized[inputMapNormalized>1]=1
    return inputMapNormalized


def saturateSomePercentileForSentinel2(inputMap,percentileToSaturate):
    inputMapNormalized=(inputMap-np.amin(inputMap))/(np.percentile(inputMap,(100-percentileToSaturate))-np.amin(inputMap))
    inputMapNormalized[inputMapNormalized>1]=1
    return inputMapNormalized


def preprocessSentinel2DividedBy10000(inputMap):
    inputMapDividedBy10000 = inputMap/10000
    inputMapDividedBy10000[inputMapDividedBy10000>1]=1
    return inputMapDividedBy10000 




