import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L

def get_num_channel(channels, alpha):
    if channels is None or alpha is None:
        return None, None
    l_channels = int(alpha * channels)
    h_channels = channels - l_channels
    return (h_channels, l_channels)

def oct_relu(x):
    if type(x) is tuple:
        return F.relu(x[0]), F.relu(x[1])
    else:
        return F.relu(x)

def oct_add(x1, x2):
    if type(x1) is tuple:
        if type(x2) is tuple:
            return (x1[0]+x2[0], x1[1]+x2[1])
        else:
            return (x1[0]+x2, x1[1])
    else:
        if type(x2) is tuple:
            return (x1+x2[0], x2[1])
        else:
            return x1+x2

def oct_function(f):
    def f2(x):
        ret = (f(x[0]), f(x[1]))
        if ret[1] is None:
            return ret[0]
        else:
            return ret
    return f2




class OctConv(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0
                 ,nobias=True, initialW=None, initial_bias=None, dilate=1, groups=1, alpha_in = None, alpha_out = 0.5):
        """Octave Convolution

        Noted that initialW and initial_bias should only accept initializer
        """
        super(OctConv, self).__init__()
        self.ksize = ksize
        self.pad = pad
        self.dilate = dilate
        self.out_channels = out_channels
        self.groups = int(groups)

        #self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        #self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        assert (alpha_in is None or 0 <= alpha_in <= 1) and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1."

        self.alpha_in, self.alpha_out = alpha_in, alpha_out

        h_in_channels, l_in_channels = get_num_channel(in_channels, alpha_in)
        self.h_out_channels, self.l_out_channels = get_num_channel(out_channels, alpha_out)
	
        with self.init_scope():
            self.conv_l2l = None if l_in_channels == 0 or self.l_out_channels == 0 else \
                            L.Convolution2D(l_in_channels, self.l_out_channels,
                                      ksize, 1, pad, nobias=nobias, initialW=initialW, initial_bias=initial_bias, dilate=dilate, groups=groups)
            self.conv_l2h = None if l_in_channels == 0 or self.h_out_channels == 0 else \
                            L.Convolution2D(l_in_channels, self.h_out_channels,
                                      ksize, 1, pad, nobias=nobias, initialW=initialW, initial_bias=initial_bias, dilate=dilate, groups=groups)
            self.conv_h2l = None if h_in_channels == 0 or self.l_out_channels == 0 else \
                            L.Convolution2D(h_in_channels, self.l_out_channels,
                                      ksize, 1, pad, nobias=nobias, initialW=initialW, initial_bias=initial_bias, dilate=dilate, groups=groups)
            self.conv_h2h = None if h_in_channels == 0 or self.h_out_channels == 0 else \
                            L.Convolution2D(h_in_channels, self.h_out_channels,
                                      ksize, 1, pad, nobias=nobias, initialW=initialW, initial_bias=initial_bias, dilate=dilate, groups=groups)

    def forward(self, x):
        x_h, x_l = x if type(x) is tuple else (x, None)

        #TODO Here average_pooling_2d should use cover_all mode. However, currently chainer does not support this feature.
        # Hence, only following sizes are support.
        if self.l_out_channels > 0:
            if self.stride == 1:
                assert x_h.shape[2]%2==0 and x_h.shape[3]%2==0
            elif self.stride == 2:
                assert x_h.shape[2]//2%2==0 and x_h.shape[3]//2%2==0

        if x_h is not None:
            x_h = F.average_pooling_2d(x_h, 2) if self.stride == 2 else x_h
            x_h2h = self.conv_h2h(x_h)
            x_h2l = self.conv_h2l(F.average_pooling_2d(x_h, 2)) if self.l_out_channels > 0 else None

        if x_l is not None:
            x_l2h = self.conv_l2h(x_l)
            _, _, H, W = x_h2h.shape
            x_l2h = F.unpooling_2d(x_l2h, 2, outsize=(H, W)) if self.stride == 1 else x_l2h
            x_l2l = F.average_pooling_2d(x_l, 2) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.l_out_channels > 0 else None 
            x_h = x_l2h + x_h2h
            x_l = x_h2l + x_l2l if self.l_out_channels > 0 else None
            h_out, l_out = x_h, x_l
        else:
            h_out, l_out = x_h2h, x_h2l

        return h_out if l_out is None else (h_out, l_out)

        

class Conv_BN(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, dilate=1, groups=1,
            nobias=True, initialW=None, initial_bias=None, bn_kwargs={},
            alpha_in = None, alpha_out = 0.5):
        super(Conv_BN, self).__init__()
        h_out_channels, l_out_channels = get_num_channel(out_channels, alpha_out)

        if 'comm' in bn_kwargs:
            norm_layer = L.MultiNodeBatchNormalization
        else:
            norm_layer = L.BatchNormalization

        with self.init_scope():
            self.conv = OctConv(in_channels, out_channels, ksize, stride, pad, nobias=nobias, initialW=initialW, initial_bias=initial_bias,
                                   groups=groups, dilate=dilate, alpha_in=alpha_in, alpha_out=alpha_out)
            self.bn_h = None if h_out_channels == 0 else norm_layer(h_out_channels, **bn_kwargs)
            self.bn_l = None if l_out_channels == 0 else norm_layer(l_out_channels, **bn_kwargs)

    def forward(self, x):
        x = self.conv(x)
        x_h, x_l = x if type(x) is tuple else (x, None)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None
        return (x_h, x_l) if x_l is not None else x_h


class Conv_BN_ACT(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, dilate=1, groups=1,
            nobias=True, initialW=None, initial_bias=None, activ=F.relu, bn_kwargs={},
            alpha_in = None, alpha_out = 0.5):
        super(Conv_BN_ACT, self).__init__()
        h_out_channels, l_out_channels = get_num_channel(out_channels, alpha_out)
        assert h_out_channels > 0

        if 'comm' in bn_kwargs:
            norm_layer = L.MultiNodeBatchNormalization
        else:
            norm_layer = L.BatchNormalization

        with self.init_scope():
            self.conv = OctConv(in_channels, out_channels, ksize, stride, pad, nobias=nobias, initialW=initialW, initial_bias=initial_bias,
                                   groups=groups, dilate=dilate, alpha_in=alpha_in, alpha_out=alpha_out)
            self.bn_h = None if h_out_channels == 0 else norm_layer(h_out_channels, **bn_kwargs)
            self.bn_l = None if l_out_channels == 0 else norm_layer(l_out_channels, **bn_kwargs)
            self.act = activ

    def forward(self, x):
        x = self.conv(x)
        x_h, x_l = x if type(x) is tuple else (x, None)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None

        if self.act is not None:
            x_h = self.act(x_h)
            if x_l is not None:
                x_l = self.act(x_l)

        return (x_h, x_l) if x_l is not None else x_h

if __name__ == "__main__":
    np.random.seed(0)
    oc = Conv_BN(None, 3, ksize=3, alpha_in=None,alpha_out=0.5, stride=1, pad=1)
    oc1 = Conv_BN(None, 10, ksize=7, alpha_in=None, alpha_out=0.8, stride=2, pad=3)
    oc2 = Conv_BN(None, 1, ksize=3, alpha_in=None, alpha_out=0, stride=1, pad=1)

    x = np.random.randn(4, 3, 32, 32).astype(np.float32)

    out = oc2(oc1(oc(x)))

    print(x)
    print(out)
    print(out.shape)

