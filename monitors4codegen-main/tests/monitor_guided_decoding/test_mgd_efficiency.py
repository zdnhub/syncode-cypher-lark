
from monitors4codegen.monitor_guided_decoding.infer import Infer
import pytest


def test_generation_length_vs_time():
    original_times = []
    mgd_times = []

    for k in [10, 20, 50, 100, 200]:
        original = Infer(max_new_tokens= k)
        mgd = Infer(max_new_tokens= k, mode = 'mgd')
        original_times.append(original.mean_generation)
        mgd_times.append(mgd.mean_generation)
    
    with open(original.times_path, 'w') as f:
        f.writelines(str(original_times))
    
    with open(mgd.times_path, 'w') as f:
        f.writelines(str(mgd_times))
    
    
    
