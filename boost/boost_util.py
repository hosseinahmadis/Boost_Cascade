import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def read_boost_data_hoss(args):
    print("start of reading data.........................................")
    
    fileFullPath_master = args.get("fileFullPath_master")
    frameIdx = args.get("frameIdx", 1)
    numSamplePerChirp = args.get("numSamplePerChirp", 256)
    numChirpPerLoop = args.get("numChirpPerLoop", 12)
    numRXPerDevice = args.get("numRXPerDevice", 4)
    number_of_lane = args.get("number_of_lane", 2)
    
    numLoops = 1

    data_IQ = readBinFile(fileFullPath_master, frameIdx, numSamplePerChirp, numChirpPerLoop, numLoops, numRXPerDevice, number_of_lane)
    
    print(f"finish of reading data......................................... frame number {frameIdx}")
    print(f'data_IQ={data_IQ.shape}')

   

    return data_IQ
#-----------------------------------------------------------------------------------------
def readBinFile(fileFullPath, frameIdx, numSamplePerChirp, numChirpPerLoop, numLoops, numRXPerDevice,lane=0):
    expected_num_samples_per_frame = numSamplePerChirp * numChirpPerLoop * numLoops * numRXPerDevice * 4
    dataSizeOneChirp = 4 * numSamplePerChirp * numRXPerDevice
    dataSizeOneFrame = dataSizeOneChirp * numChirpPerLoop*numLoops
    print(f'dataSizeOneFrame={dataSizeOneFrame}   expected_num_samples_per_frame={expected_num_samples_per_frame}')
  
    with open(fileFullPath, 'rb') as fp:
        fp.seek((frameIdx - 1) * dataSizeOneFrame )
        adc_data1 = np.fromfile(fp, dtype=np.uint16, count=dataSizeOneFrame // 2).astype(np.float32)
    print(f'shape of adc_data={adc_data1.shape}')
    frameData=adc_data1 - (adc_data1 >= 2**15) * 2**16
    if lane==2 or lane ==4 :
        frameData=dp_reshapeLVDS(frameData, lane)
    print(f'frameData shape is {frameData.shape}')    
    frameCplx = frameData[:, 0] + 1j * frameData[:, 1]
    
    frameComplex = np.zeros((numChirpPerLoop, numRXPerDevice, numSamplePerChirp), dtype=np.complex64)

    # Change interleave data to non-interleave
    temp = frameCplx.reshape(numChirpPerLoop, numSamplePerChirp * numRXPerDevice)
    for chirp in range(numChirpPerLoop):
       frameComplex[chirp, :, :] = temp[chirp, :].reshape(numRXPerDevice, numSamplePerChirp)
    
    
    
    return frameComplex

#-----------------------------------------------------------------------------------------

def dp_reshapeLVDS(rawData, numLanes):
    if numLanes == 4:
        # 4-lane configuration
        rawData8 = np.reshape(rawData, (8, len(rawData) // 8))
        rawDataI = np.reshape(rawData8[:4, :], -1)
        rawDataQ = np.reshape(rawData8[4:, :], -1)
    elif numLanes == 2:
        # 2-lane configuration
        rawData4 = np.reshape(rawData, (4, len(rawData) // 4))
        rawDataI = np.reshape(rawData4[:2, :], -1)
        rawDataQ = np.reshape(rawData4[2:, :], -1)
    else:
        raise ValueError("Unsupported number of lanes. Only 2 or 4 lanes are supported.")
    
    frameData = np.column_stack((rawDataI, rawDataQ))
    return frameData



#-----------------------------------------------------------------------------------------
