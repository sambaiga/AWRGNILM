import torch
import torch.nn as nn
import torch.nn.functional as F

class ReccurrentBlock(torch.nn.Module):
    """[summary]
    
    Arguments:
        torch {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    def __init__(self, eps=10, delta=10):
        super(ReccurrentBlock, self).__init__()
        self.eps         = torch.nn.Parameter(torch.randn(1), requires_grad = True)
        self.delta       = torch.nn.Parameter(torch.randn(1), requires_grad = True)
        torch.nn.init.constant_(self.eps, eps)
        torch.nn.init.constant_(self.delta, delta)
        
    def forward(self, dist):
        dist = torch.floor(dist*self.eps)
        dist[dist>self.delta]=self.delta
        return dist

class MLPLayer(nn.Module):
    def __init__(self, in_size, hidden_arch=[128, 512, 1024], output_size=1, activation=nn.ReLU(True),
                 batch_norm=True):
        
        super(MLPLayer, self).__init__()
        self.in_size = in_size
        self.output_size = output_size
        layer_sizes = [in_size] + [x for x in hidden_arch]
        self.layers = []
        
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)            
            if batch_norm and i!=0:# if used as discriminator, then there is no batch norm in the first layer
                bn = nn.BatchNorm1d(layer_sizes[i+1])
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)
            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)
        if output_size is not None:
            layer = nn.Linear(layer_sizes[-1], output_size)
            self.layers.append(layer)
            self.add_module("layer"+str(len(self.layers)), layer)
       
       
        self.init_weights()
        self.mlp_network =  nn.Sequential(*self.layers)
        
    def forward(self, z):
        return self.mlp_network(z)
        
    def init_weights(self):
        for layer in self.layers:
            try:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight.data)
                    nn.init.constant_(layer.bias.data, 0)
            except: pass
                



class Conv2D(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.2, width=50):
        super(Conv2D, self).__init__()
        self.in_size=in_size
        if width==30:
            self.hidden_out = 64
        if width==50:
            self.hidden_out = 576 
        if width==60:
            self.hidden_out = 1024 
        if width==80:
            self.hidden_out = 3136
        if width==100:
            self.hidden_out = 5184
        

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, 16, 5, 2),
            nn.ReLU()
            
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 2),
            nn.ReLU()
            
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 2),
            nn.ReLU()
            
        )
        self.fc_out=nn.Sequential(
            nn.Linear(self.hidden_out, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, out_size)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x= self.fc_out(x)
        return x

class Conv2DAdaptiveRecurrence(nn.Module):
    def __init__(self, in_size=1, out_size=12, dropout=0.2, eps=10, delta=10, width=50):
        super(Conv2DAdaptiveRecurrence, self).__init__()
        
        self.in_size  = in_size
        self.rec_block = ReccurrentBlock(eps=eps, delta=delta)
        self.con_layer = Conv2D(in_size, out_size, dropout, width)
        
    def forward(self, x):
        
        x  = self.rec_block(x)
        prediction = self.con_layer(x)
        return prediction
    
    

class MLPAdaptiveRecurrence(nn.Module):
    def __init__(self, in_size, out_size=12, dropout=0.2, eps=10, delta=10, width=50, image_type="vi"):
        super().__init__()
        
        self.in_size  = in_size
        self.image_type = image_type
        if image_type!="vi":
            self.rec_block = ReccurrentBlock(eps=eps, delta=delta)
        self.mlp = MLPLayer(in_size=width*width*in_size, hidden_arch=[128, 256, 512, 1024], output_size=out_size)
        
    def forward(self, x):
        if self.image_type!="vi":
            x  = self.rec_block(x)
        prediction = self.mlp(x.reshape(x.size(0), -1))
        return prediction    