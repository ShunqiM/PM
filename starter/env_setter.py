import os

def set_visible_gpus(device):
    assert isinstance(device, str)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    print('CUDA_VISIBLE_DEVICES (updated):', os.environ["CUDA_VISIBLE_DEVICES"])