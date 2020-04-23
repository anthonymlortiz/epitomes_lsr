import torch as T

class EpitomeModel(T.nn.Module):

    def __init__(self, size, channels=3, layers=1):
        super(EpitomeModel, self).__init__()
        
        self.size = size
        self.channels = channels
        self.layers = layers
        
        self.mean = T.nn.Parameter(T.empty((layers,channels,size,size)).uniform_()/10.+0.5)
        self.prior = T.nn.Parameter(T.zeros((layers,size,size)))
        self.ivar = T.nn.Parameter(1./T.empty((layers,channels,size,size)).uniform_().clamp(min=0.1,max=0.1))
    
    def forward(self, inputs):
        
        w = inputs.shape[-1]
        
        mean_pad = T.nn.functional.pad(self.mean, (w//2,)*4, 'circular')
        var_pad = T.nn.functional.pad(self.ivar, (w//2,)*4, 'circular')
        
        r = T.nn.functional.conv2d(input=mean_pad * var_pad, weight=inputs)
        a = T.nn.functional.avg_pool2d(mean_pad**2 * var_pad, (w,w), stride=1).sum(1)*(w**2)
        z = T.nn.functional.conv2d(input=var_pad, weight=inputs**2)

        lv = T.nn.functional.avg_pool2d(T.log(var_pad), (w,w), stride=1).sum(1)*(w**2)

        p = -(a.unsqueeze(1)-2*r+z) + lv.unsqueeze(1)
        
        return p/2 + self.prior.view(-1,self.size*self.size).log_softmax(1).view(self.layers,-1,self.size,self.size)

        