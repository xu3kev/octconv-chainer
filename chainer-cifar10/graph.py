import chainer.computational_graph as c
from importlib import import_module
import numpy as np
from octconv import Conv_BN_ACT
import os
model_file="models/oct_resnet.py"
model_name="OctResNet"
ext = os.path.splitext(model_file)[1]
mod_path = '.'.join(os.path.split(model_file)).replace(ext, '')
mod = import_module(mod_path)
net = getattr(mod, model_name)(10, alpha=0.25)
x = np.random.normal(0,1,(1,3,32,32)).astype(np.float32)

vs = net(x)
g = c.build_computational_graph([vs])

#vs = Conv_BN_ACT(None, 64, 3, 1, 1, True, alpha_out=0.25)(x)
#vs = [vs[0], vs[1]]
#g = c.build_computational_graph(vs)

with open('comp_graph', 'w') as o:
    o.write(g.dump())
