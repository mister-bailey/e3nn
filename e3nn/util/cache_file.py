'''
Cache in files
'''
#import fcntl
import glob
import gzip
import os
import pickle
import sys
from random import randrange
from functools import lru_cache, wraps
from itertools import chain, count


class FileSystemMutex:
    '''
    Mutual exclusion of different **processes** using the file system
    '''

    def __init__(self, filename):
        self.handle = None
        self.filename = filename

    def acquire(self):
        '''
        Locks the mutex
        if it is already locked, it waits (blocking function)
        '''
        self.handle = open(self.filename, 'w')
        fcntl.lockf(self.handle, fcntl.LOCK_EX)
        self.handle.write("{}\n".format(os.getpid()))
        self.handle.flush()

    def release(self):
        '''
        Unlock the mutex
        '''
        if self.handle is None:
            raise RuntimeError()
        fcntl.lockf(self.handle, fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def cached_picklesjar(dirname, maxsize=128, open_jar=gzip.open,
                      load=pickle.load, save=pickle.dump, ext='pickle', temp_ext='temp'):
    '''
    Cache a function with a directory

    :param dirname: the directory path
    :param maxsize: maximum size of the RAM cache (there is no limit for the directory cache)
    '''

    def decorator(func):
        '''
        The actual decorator
        '''

        @lru_cache(maxsize=maxsize)
        @wraps(func)
        def wrapper(*args, **kwargs):
            '''
            The wrapper of the function
            '''
            try:
                os.makedirs(dirname)
            except FileExistsError:
                pass
            except PermissionError:
                return func(*args, **kwargs)

            if not os.access(dirname, os.W_OK):
                return func(*args, **kwargs)

            #mutexfile = os.path.join(dirname, "mutex")

            key = (args, frozenset(kwargs.items()), func.__defaults__)

            #with FileSystemMutex(mutexfile):
            # why do we try all the files? could we just rely on the naming convention?
            for file in glob.glob(os.path.join(dirname, "*.{}".format(ext))):
                with open_jar(file, "rb") as file:
                    loadedkey = load(file)
                    if key == loadedkey:
                        return load(file)

            sys.stdout.flush()
            result = func(*args, **kwargs)
            sys.stdout.flush()

            #with FileSystemMutex(mutexfile):
            name = " ".join(map(str, args))
            if kwargs:
                name += " " + " ".join(
                    "{}={}".format(key, value)
                    for key, value in sorted(kwargs.items())
                )
            #for postfix in chain([''], ('_{}'.format(i) for i in count())):
            #    file = os.path.join(dirname, "{}{}.{}".format(name, postfix, ext))
            #    if not os.path.isfile(file):
            #        break
            file = os.path.join(dirname, "{}{}.{}".format(name, ext))  
                
            temp_name = '{:X}'.format(randrange(16 ** 8))
            temp_file = os.path.join(dirname, "{}.{}".format(temp_name, temp_ext))

            try:
                with open_jar(temp_file, "wb") as temp_file:
                    save(key, temp_file)
                    save(result, temp_file)
                os.replace(temp_file, file)
            except PermissionError:
                pass

            return result

        return wrapper

    return decorator
