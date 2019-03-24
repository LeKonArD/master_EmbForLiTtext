import pandas as pd
import numpy as np
from sklearn import preprocessing as prp



## from https://github.com/cligs/pyzeta
def calculate_scores(docprops1, docprops2, absolute1, absolute2,
                     logaddition, segmentlength):

    """
    This function implements several variants of Zeta by modifying some key parameters.
    Scores can be document proportions (binary features) or relative frequencies.
    Scores can be taken directly or subjected to a log-transformation (log2, log10)
    Scores can be subtracted from each other or divided by one another.
    The combination of document proportion, no transformation and subtraction is Burrows' Zeta.
    The combination of relative frequencies, no transformation, and division corresponds to
    the ratio of relative frequencies.
    """
    # Define logaddition and division-by-zero avoidance addition
    logaddition = logaddition
    divaddition = 0.00000000001
    # == Calculate subtraction variants ==
    # sd0 - Subtraction, docprops, untransformed a.k.a. "original Zeta"
    sd0 = docprops1 - docprops2
    print(len(docprops1))
    sd0 = pd.Series(sd0, name="sd0")
    print(len(sd0))
    # Prepare scaler to rescale variants to range of sd0 (original Zeta)
    scaler = prp.MinMaxScaler(feature_range=(min(sd0),max(sd0)))
    # sd2 - Subtraction, docprops, log2-transformed
    sd2 = np.log2(docprops1 + logaddition) - np.log2(docprops2 + logaddition)
    sd2 = pd.Series(sd2, name="sd2")
    sd2 = scaler.fit_transform(sd2.values.reshape(-1, 1))

    # Calculate Gries "deviation of proportions" (DP)
    segnum1 = len(absolute1.columns.values)
    segnum2 = len(absolute2.columns.values)
    seglens1 = [segmentlength] * segnum1
    seglens2 = [segmentlength] * segnum2
    crpsize1 = sum(seglens1)
    crpsize2 = sum(seglens2)
    #print("segments", segnum1, segnum2)
    totalfreqs1 = np.sum(absolute1, axis=1)
    totalfreqs2 = np.sum(absolute2, axis=1)
    #print("totalfreqs", totalfreqs1, totalfreqs2)
    expprops1 = np.array(seglens1) / crpsize1
    expprops2 = np.array(seglens2) / crpsize2
    #print("exprops", expprops1, expprops2)
    #print(absolute1.head())
    #print(totalfreqs1)
    obsprops1 = absolute1.div(totalfreqs1, axis=0)
    obsprops1 = obsprops1.fillna(expprops1[0]) # was: expprops1[0]
    obsprops2 = absolute2.div(totalfreqs2, axis=0)
    obsprops2 = obsprops2.fillna(expprops2[0]) # was: expprops2[0]
    devprops1 = (np.sum(abs(expprops1 - obsprops1), axis=1) /2 )
    devprops2 = (np.sum(abs(expprops2 - obsprops2), axis=1) /2 )
    #print(devprops1.head())
    #print(devprops2.head())

    # Calculate DP variants ("g" for Gries)
    sg0 = devprops1 - devprops2
    sg0 = pd.Series(sg0, name="sg0")
    sg0 = scaler.fit_transform(sg0.values.reshape(-1, 1))
    sg2 = np.log2(devprops1 + logaddition) - np.log2(devprops2 + logaddition)
    sg2 = pd.Series(sg2, name="sg2")
    sg2 = scaler.fit_transform(sg2.values.reshape(-1, 1))
    dg0 = (devprops1 + divaddition) / (devprops2 + divaddition)
    dg0 = pd.Series(dg0, name="dg0")
    dg0 = scaler.fit_transform(dg0.values.reshape(-1, 1))
    dg2 = np.log2(devprops1 + logaddition) / np.log2(devprops2 + logaddition)
    dg2 = pd.Series(dg2, name="dg2")
    dg2 = scaler.fit_transform(dg2.values.reshape(-1, 1))

    # Return all zeta variant scores
    return sd0, sd2.flatten(), sg0.flatten(), sg2.flatten(), dg0.flatten(), dg2.flatten(), devprops1, devprops2
