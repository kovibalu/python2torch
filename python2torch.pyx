import ctypes

import cython

import numpy as np
cimport numpy as np

ctypes.CDLL("libluajit.so", mode=ctypes.RTLD_GLOBAL)

cdef extern from "<lua.h>":
    cdef struct lua_State

    cdef int LUA_MULTRET

    cdef lua_State * luaL_newstate()
    cdef void luaL_openlibs(lua_State *)
    cdef const char * lua_tostring(lua_State *, int)
    cdef int lua_pcall(lua_State *, int, int, int)
    cdef void lua_close(lua_State *)
    cdef void lua_getglobal(lua_State *, const char*)
    cdef void lua_pushstring(lua_State *, const char*)
    cdef int lua_gettop(lua_State *)
    cdef void lua_pop(lua_State *, int)
    cdef void lua_pushinteger(lua_State *, ptrdiff_t)
    cdef void lua_pushnumber(lua_State *, double)
    cdef void lua_rawset(lua_State *, int)
    cdef void lua_newtable(lua_State *)
    cdef int lua_isuserdata(lua_State *, int)
    cdef int lua_istable(lua_State *, int)
    cdef int lua_isstring(lua_State *, int)
    cdef int lua_isnumber(lua_State *, int)
    cdef int lua_isfunction(lua_State *, int)
    cdef int lua_isnil(lua_State *, int)
    cdef int lua_next(lua_State *, int)
    cdef void lua_pushnil(lua_State *)
    cdef double lua_tonumber(lua_State *, int)
    cdef void lua_pushvalue(lua_State *, int)
    cdef void lua_pushglobaltable(lua_State *)

cdef extern from "<lauxlib.h>":
    cdef int luaL_loadfile(lua_State *, const char*)

cdef extern from "<luaT.h>":
    cdef void luaT_pushudata(lua_State *, void*, const char*)
    cdef void * luaT_checkudata(lua_State *, int, const char*)
    cdef const char * luaT_typename(lua_State *, int)

cdef extern from "<TH/TH.h>":
    cdef struct THStorage
    cdef struct THTensor
    cdef struct THLongStorage
    cdef struct THFloatTensor

    cdef THStorage * THFloatStorage_newWithData(float *, long)
    cdef THTensor * THFloatTensor_newWithStorage(THStorage *, long,
                                                 THLongStorage*,
                                                 THLongStorage*)

    cdef THStorage * THDoubleStorage_newWithData(double *, long)
    cdef THTensor * THDoubleTensor_newWithStorage(THStorage *, long,
                                                  THLongStorage*,
                                                  THLongStorage*)

    cdef THStorage * THByteStorage_newWithData(char *, long)
    cdef THTensor * THByteTensor_newWithStorage(THStorage *, long,
                                                THLongStorage*, THLongStorage*)

    cdef THStorage * THLongStorage_newWithData(long *, long)
    cdef THTensor * THLongTensor_newWithStorage(THStorage *, long,
                                                THLongStorage*, THLongStorage*)

    cdef THStorage * THLongStorage_newWithSize(long)
    cdef long * THLongStorage_data(const THStorage *)

    cdef float * THFloatTensor_data(const THTensor *)
    cdef int THFloatTensor_nDimension(const THTensor *)
    cdef long THFloatTensor_size(const THTensor *, int)
    cdef long THFloatTensor_stride(const THTensor *, int)

    cdef double * THDoubleTensor_data(const THTensor *)
    cdef int THDoubleTensor_nDimension(const THTensor *)
    cdef long THDoubleTensor_size(const THTensor *, int)
    cdef long THDoubleTensor_stride(const THTensor *, int)

    cdef long * THLongTensor_data(const THTensor *)
    cdef int THLongTensor_nDimension(const THTensor *)
    cdef long THLongTensor_size(const THTensor *, int)
    cdef long THLongTensor_stride(const THTensor *, int)

    cdef char * THByteTensor_data(const THTensor *)
    cdef int THByteTensor_nDimension(const THTensor *)
    cdef long THByteTensor_size(const THTensor *, int)
    cdef long THByteTensor_stride(const THTensor *, int)

cdef class PyTorchExtension(object):
    cdef lua_State * L

    def __cinit__(self, str script_name not None):
        self.L = luaL_newstate()
        luaL_openlibs(self.L)

        lua_getglobal(self.L, 'require')
        lua_pushstring(self.L, 'torch')
        if lua_pcall(self.L, 1, 0, 0):
            raise Exception('require torch failed')

        if luaL_loadfile(self.L, script_name) or lua_pcall(self.L, 0, 0, 0):
            raise Exception(lua_tostring(self.L, -1))

        lua_pushglobaltable(self.L)
        lua_pushnil(self.L)

        self.function_names = []

        while lua_next(self.L, -2):
            name = lua_tostring(self.L, -2)
            if lua_isfunction(self.L, -1):
                self.function_names.append(name)
            lua_pop(self.L, 1)

        lua_pop(self.L, 1)

    @staticmethod
    cdef THTensor * PyToTHFloat(np.ndarray[float] py, THLongStorage * size,
                                THLongStorage * stride):
        storage = THFloatStorage_newWithData(& py[0], py.size)
        tensor = THFloatTensor_newWithStorage(storage, 0, size, stride)
        return tensor

    @staticmethod
    cdef THTensor * PyToTHDouble(np.ndarray[double] py, THLongStorage * size,
                                 THLongStorage * stride):
        storage = THDoubleStorage_newWithData(& py[0], py.size)
        tensor = THDoubleTensor_newWithStorage(storage, 0, size, stride)
        return tensor

    @staticmethod
    cdef THTensor * PyToTHByte(np.ndarray[char] py, THLongStorage * size,
                               THLongStorage * stride):
        storage = THByteStorage_newWithData(& py[0], py.size)
        tensor = THByteTensor_newWithStorage(storage, 0, size, stride)
        return tensor

    @staticmethod
    cdef THTensor * PyToTHLong(np.ndarray[long] py, THLongStorage * size,
                               THLongStorage * stride):
        storage = THLongStorage_newWithData(& py[0], py.size)
        tensor = THLongTensor_newWithStorage(storage, 0, size, stride)
        return tensor

    @staticmethod
    cdef THLongStorage * PyArraySize(np.ndarray array):
        storage = THLongStorage_newWithSize(array.ndim)
        data = THLongStorage_data(storage)
        for i in xrange(array.ndim):
            data[i] = array.shape[i]
        return < THLongStorage * >storage

    @staticmethod
    cdef THLongStorage * PyArrayStride(np.ndarray array):
        storage = THLongStorage_newWithSize(array.ndim)
        data = THLongStorage_data(storage)
        for i in xrange(array.ndim):
            data[i] = array.strides[i] / array.itemsize
        return < THLongStorage * >storage

    @staticmethod
    cdef THToPyFloat(THTensor * output_tensor):
        num_dims = THFloatTensor_nDimension(output_tensor)
        shape = []
        strides = []
        for i in xrange(num_dims):
            shape.append(THFloatTensor_size(output_tensor, i))
            strides.append(THFloatTensor_stride(output_tensor, i))
        data = THFloatTensor_data(output_tensor)

        total_size = 1
        for size in shape:
            total_size *= size

        array = np.empty((total_size,), dtype=np.float32)

        for i in xrange(total_size):
            array[i] = data[i]

        return array.reshape(shape)

    @staticmethod
    cdef THToPyDouble(THTensor * output_tensor):
        num_dims = THDoubleTensor_nDimension(output_tensor)
        shape = []
        strides = []
        for i in xrange(num_dims):
            shape.append(THDoubleTensor_size(output_tensor, i))
            strides.append(THDoubleTensor_stride(output_tensor, i))
        data = THDoubleTensor_data(output_tensor)

        total_size = 1
        for size in shape:
            total_size *= size

        array = np.empty((total_size,), dtype=np.float64)

        for i in xrange(total_size):
            array[i] = data[i]

        return array.reshape(shape)

    @staticmethod
    cdef THToPyLong(THTensor * output_tensor):
        num_dims = THLongTensor_nDimension(output_tensor)
        shape = []
        strides = []
        for i in xrange(num_dims):
            shape.append(THLongTensor_size(output_tensor, i))
            strides.append(THLongTensor_stride(output_tensor, i))
        data = THLongTensor_data(output_tensor)

        total_size = 1
        for size in shape:
            total_size *= size

        array = np.empty((total_size,), dtype=np.long)

        for i in xrange(total_size):
            array[i] = data[i]

        return array.reshape(shape)

    @staticmethod
    cdef THToPyByte(THTensor * output_tensor):
        num_dims = THByteTensor_nDimension(output_tensor)
        shape = []
        strides = []
        for i in xrange(num_dims):
            shape.append(THByteTensor_size(output_tensor, i))
            strides.append(THByteTensor_stride(output_tensor, i))
        data = THByteTensor_data(output_tensor)

        total_size = 1
        for size in shape:
            total_size *= size

        array = np.empty((total_size,), dtype=np.int8)

        for i in xrange(total_size):
            array[i] = data[i]

        return array.reshape(shape)

    cdef PyToLuaTensor(self, np.ndarray arg):
        cdef THTensor * torch_tensor
        cdef THLongStorage * size_tensor
        cdef THLongStorage * stride_tensor

        shape = arg.shape
        arg = np.require(arg, requirements='C')
        size_tensor = PyTorchExtension.PyArraySize(arg)
        stride_tensor = PyTorchExtension.PyArrayStride(arg)
        type_string = None
        if arg.dtype == np.float32:
            torch_tensor = PyTorchExtension.PyToTHFloat(
                arg.ravel(), size_tensor, stride_tensor)
            type_string = 'torch.FloatTensor'
        elif arg.dtype == np.float64:
            torch_tensor = PyTorchExtension.PyToTHDouble(
                arg.ravel(), size_tensor, stride_tensor)
            type_string = 'torch.DoubleTensor'
        elif arg.dtype == np.byte:
            torch_tensor = PyTorchExtension.PyToTHByte(
                arg.ravel(), size_tensor, stride_tensor)
            type_string = 'torch.ByteTensor'
        elif arg.dtype == np.long:
            torch_tensor = PyTorchExtension.PyToTHLong(
                arg.ravel(), size_tensor, stride_tensor)
            type_string = 'torch.LongTensor'
        else:
            raise Exception('unknown dtype')
        luaT_pushudata(self.L, torch_tensor, type_string)

    cdef PyToLuaString(self, str arg):
        lua_pushstring(self.L, arg)

    cdef PyToLuaFloat(self, float arg):
        lua_pushnumber(self.L, arg)

    cdef PyToLuaInteger(self, int arg):
        lua_pushinteger(self.L, arg)

    cdef PyToLuaList(self, args):
        lua_newtable(self.L)

        for index, arg in enumerate(args):
            lua_pushnumber(self.L, index + 1)
            self.PyToLua(arg)
            lua_rawset(self.L, -3)

    cdef PyToLua(self, pytype):
        if type(pytype) is str:
            self.PyToLuaString(pytype)
        elif type(pytype) is np.ndarray:
            self.PyToLuaTensor(pytype)
        elif type(pytype) is int:
            self.PyToLuaInteger(pytype)
        elif type(pytype) is float:
            self.PyToLuaFloat(pytype)
        elif type(pytype) in (list, tuple):
            self.PyToLuaList(pytype)
        else:
            raise Exception('unknown type {0}'.format(str(type(pytype))))

    cdef LuaToPy(self):
        if lua_isuserdata(self.L, -1):
            typename = str(luaT_typename(self.L, -1))
            output_tensor = <THTensor * >luaT_checkudata(self.L, -1, typename)
            if typename == 'torch.DoubleTensor':
                output_py = PyTorchExtension.THToPyDouble(output_tensor)
            elif typename == 'torch.FloatTensor':
                output_py = PyTorchExtension.THToPyFloat(output_tensor)
            elif typename == 'torch.LongTensor':
                output_py = PyTorchExtension.THToPyLong(output_tensor)
            elif typename == 'torch.ByteTensor':
                output_py = PyTorchExtension.THToPyByte(output_tensor)
            else:
                raise Exception('unknown type {0}'.format(typename))
        elif lua_istable(self.L, -1):
            output_py = {}
            lua_pushnil(self.L)
            while lua_next(self.L, -2):
                lua_pushvalue(self.L, -2)
                key = self.LuaToPy()
                value = self.LuaToPy()
                output_py[key] = value
        elif lua_isnil(self.L, -1):
            output_py = None
        elif lua_isnumber(self.L, -1):
            output_py = lua_tonumber(self.L, -1)
        elif lua_isstring(self.L, -1):
            output_py = lua_tostring(self.L, -1)
        else:
            raise Exception('unknown type')

        lua_pop(self.L, 1)
        return output_py

    def __call__(self, str function_name not None, args not None):
        before_call = lua_gettop(self.L)

        lua_getglobal(self.L, function_name)

        for arg in args:
            self.PyToLua(arg)

        if lua_pcall(self.L, len(args), LUA_MULTRET, 0):
            raise Exception(lua_tostring(self.L, -1))

        after_call = lua_gettop(self.L)

        num_retval = after_call - before_call

        outputs = []
        for i in xrange(num_retval):
            outputs.append(self.LuaToPy())
        outputs.reverse()

        return outputs

    def __dealloc__(self):
        lua_close(self.L)


class PyTorch(PyTorchExtension):

    def __init__(self, *args, **kwargs):
        super(PyTorch, self).__init__(*args, **kwargs)

        def wrapped_call_closure(name):
            def wrapped_call(*args):
                return self(name, args)
            return wrapped_call

        for name in self.function_names:
            setattr(self, name, wrapped_call_closure(name))
