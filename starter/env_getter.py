import os

def get_env(name):
    assert name in os.environ, name

    value = os.environ.get(name)
    assert value is not None, name

    if name in ['SPLITS']:
        value = value.split(':')
    
    return value