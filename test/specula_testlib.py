
from specula import cp, np

def cpu_and_gpu(f):
    '''
    Decorator to run a test function first on GPU (if available)
    and the on CPU. If the GPU is not available, it will be
    skipped silently.
    '''
    def test_gpu(self):
        return f(self, target_device_idx=0, xp=cp)
    
    def test_cpu(self):
        return f(self, target_device_idx=-1, xp=np)
    
    def test_both(self):
        if cp is not None:
            test_gpu(self)
        test_cpu(self)
        
    return test_both