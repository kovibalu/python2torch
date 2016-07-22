# python2torch
Call torch functions from python.

Did you train some super awesome nn model in Torch7?  Now do you want to deploy it as a web service with a python web server or use it as a subroutine in a PySpark script for big data analytics?  Using Lua in C is pretty easy, and to make it even easier to convert between Lua and Python types, here's a short cython script to get you started.

# Installation
```
git clone https://github.com/kmatzen/python2torch.git
cd python2torch
```
Now you have to modify `include_dirs` and `library_dirs` in setup.py so they point to your torch distribution. Then run:
```
python setup.py build
sudo python setup.py install
```
You are all set!

# How to Use
Let's start with an example.  Say you have this following lua script which defines the functions you wish to call.

mah_stuff.lua
```
function Hello()
  print "Is it me you're looking for?"
end

function DoIt(anInt, aFloat, aString, anArray, aTensor)
  print(anInt)
  print(aFloat)
  print(aString)
  for k, v in ipairs(anArray) do
    print(k, v)
  end
  print(aTensor:type(), aTensor:size())
  
  return torch.linspace(1, 10, 10)
end
```

Now let's call this stuff from python.  First, set up the torch object.
```
import python2torch

torch = python2torch.PyTorch('mah_stuff.lua')
```

The global functions from the lua state will be used to populate the torch object.  For example:
```
>>> print(dir(torch))
['DoIt', 'Hello', '__call__', '__class__', '__delattr__', '__dict__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__pyx_vtable__', '__qualname__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'assert', 'collectgarbage', 'dofile', 'error', 'function_names', 'gcinfo', 'getfenv', 'getmetatable', 'include', 'ipairs', 'load', 'loadfile', 'loadstring', 'module', 'newproxy', 'next', 'pairs', 'pcall', 'print', 'rawequal', 'rawget', 'rawlen', 'rawset', 'require', 'select', 'setfenv', 'setmetatable', 'tonumber', 'tostring', 'type', 'unpack', 'xpcall']
```
You can see that the functions we defined, ```DoIt``` and ```Hello``` have been added.

Let's call one.
```
>>> output = torch.Hello()
Is it me you're looking for?
>>> print output
[]
```
Great!  It printed our message.  In this implementation, all return values are returned as a python list.  When there are zero return values, the result is the empty list.

Let's try calling a function that takes some arguments and produces a return value.
```
>>> import numpy
>>> output = torch.DoIt(1, 3.14159, 'boomplesnoot', [ 'easy', 'as', 123 ], numpy.random.randn(10))
1
3.1415901184082
boomplesnoot
1	easy
2	as
3	123
torch.DoubleTensor
 10
[torch.LongStorage of size 1]

>>> print output
[array([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.])]
```

And that's it!  What's missing?  While writing this README I found that I forgot to convert the following pytypes to their lua equivalent:
```
None
dict
```
Other way around should be fine.

# Some gotchas
- I treat python tuples and lists more or less the same.
- This will break if the returned torch tensors are not contiguously allocated.  A fix will come shortly.
- If a lua table is returned and that table was created using contiguous integer indices, I do not automatically convert that to a list.  It will become a dict.  I didn't want to get into the business of auto-converting 1-based indices to 0-based indices.
- I've tried this using several torch packages and I haven't had anything clearly break, but I did have to add a hack to get the torch libs to import properly.  Probably not robust.  Most notable hack was calling ```ctypes.CDLL("libluajit.so", mode=ctypes.RTLD_GLOBAL)```.
