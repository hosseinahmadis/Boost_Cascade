import sys
import os
from cascade_util import *
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from global_util import *

def test_cascade():
   folder_path="D:/Hossein/datasets/data_radar/Cascade1"
   args_hoss = {
    "fileFullPath_master": os.path.join(folder_path,"master_0000_data.bin"),
    "fileFullPath_slave1": os.path.join(folder_path,"slave1_0000_data.bin"),
    "fileFullPath_slave2": os.path.join(folder_path,"slave2_0000_data.bin"),
    "fileFullPath_slave3": os.path.join(folder_path,"slave3_0000_data.bin"),
    "frameIdx": 1,
    "numSamplePerChirp": 256,
    "numChirpPerLoop": 12,
    "numRXPerDevice": 1,
    "number_of_lane": 2
   }

   data_iq_master,data_iq_slave1,data_iq_slave2,data_iq_slave3=  read_cascade_data_hoss(args_hoss) 
  
   plot_adc_data_complex(data_iq_master[0,0,0:256],'cascade_master')
   plot_adc_data_complex(data_iq_slave1[0,0,0:256],'cascade_slave1')


test_cascade()

    