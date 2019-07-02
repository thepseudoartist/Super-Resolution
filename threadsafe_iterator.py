import threading

class _threadsafe_iter:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.iterator.next()

def threadsafe_generator(f):
    def g(*args, **kwargs):
        return _threadsafe_iter(f(*args, **kwargs))
    
    return g