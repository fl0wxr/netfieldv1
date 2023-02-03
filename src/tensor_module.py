from functools import partial


class cpu:

    def __init__(self):

        import numpy as np
        self.tm = np

        self.np = np

        self.as_strided = partial(self.tm.lib.stride_tricks.as_strided, writeable=False)
        self.insert = self.tm.insert

        pass

    def memory_deall(self):

        pass

class gpu:

    def __init__(self):

        import cupy as cp
        self.tm = cp

        import numpy as np
        self.np = np

        self.as_strided = partial(self.tm.lib.stride_tricks.as_strided)
        self.insert = self.insert_
        self.append = self.append_

        self.mempool = cp.cuda.MemoryPool()
        cp.cuda.set_allocator(self.mempool.malloc)
        self.pinned_mempool = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(self.pinned_mempool.malloc)

    def insert_(self, arr, obj, values, axis=None):
        '''
        Inputs:
            <arr>: cupy ndarray, array where the insertions will occur.
            <obj>: cupy ndarray, integer index.
            <values>: float or int, this is the value which will be inserted on each corresponding <arr>'s position.
            <axis>: int, the associated axis across which the insertions will happen.
        '''

        return self.tm.array( self.np.insert( self.tm.asnumpy(arr), self.tm.asnumpy(obj), values, axis ) )

    def append_(self, arr, values, axis=None):

        return self.tm.array( self.np.append( self.tm.asnumpy(arr), self.tm.asnumpy(values), axis ) )

    def memory_deall(self):

        self.mempool.free_all_blocks()
        self.pinned_mempool.free_all_blocks()
