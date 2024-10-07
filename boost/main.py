import sys
import os
from boost_util import *

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from global_util import *
def test_boost():
   
   args_hoss = {
    "fileFullPath_master": "D:/Hossein/datasets/data_radar/Cascade1/master_0000_data.bin",
    "frameIdx": 1,
    "numSamplePerChirp": 256,
    "numChirpPerLoop": 12,
    "numRXPerDevice": 1,
    "number_of_lane": 2
   }

   data_iq_hoss=  read_boost_data_hoss(args_hoss) 
   
   print(f'shape of data_iq_hoss is {data_iq_hoss.shape}   and shape of data_iq_ti={1}\n') #print(f'\nshape of data_iq_hoss is {data_iq_hoss[0,0,0:10]} \n  and shape of data_iq_ti={data_iq_ti[0,0,0:10]}\n')   
   plot_adc_data_complex(data_iq_hoss[0,0,0:256],'boost')
   plot_fft(data_iq_hoss[0,0,0:256],'boost')

test_boost()

    