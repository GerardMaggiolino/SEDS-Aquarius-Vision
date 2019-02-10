import torch 
import torch.nn as nn

class LanderCNN(nn.Module): 
    '''
    CNN architecture for terrain classification task. 

    Loosely based off MobileNet architecture, uses depthwise separable 
    convolutional layers. Striding used over pooling for dimensionality
    reduction. 

    Expected input of batches of 1x250x250 image. Output for ordinal
    classification, multi-dimension sigmoid output. 

    Filters of 5, 3, 3, ..., 7 Avg pool. 
    '''

    def __init__(self): 
        super().__init__()

        # Conv activation
        act = nn.ReLU()
        # Pad for proper striding 
        pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        
        # Define conv layers and register under ModuleList
        self.conv_layers = nn.ModuleList([
            # First normal conv layer 251 -> 124
            pad,
            nn.Conv2d(1, 8, 5, stride=2, bias=False),
            nn.BatchNorm2d(8),
            act,
            # First depthwise separable convolution 125 -> 62
            pad,
            nn.Conv2d(8, 8, 3, stride=2, bias=False, groups=8),
            nn.BatchNorm2d(8),
            act,
            nn.Conv2d(8, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            act,
            # Second depthwise separable convolution 63 -> 31
            pad, 
            nn.Conv2d(16, 16, 3, stride=2, bias=False, groups=16),
            nn.BatchNorm2d(16),
            act,
            nn.Conv2d(16, 32, 1, bias=False),
            nn.BatchNorm2d(32), 
            act,
            # Third depthwise separable convolution, 31 -> 15, 
            nn.Conv2d(32, 32, 3, stride=2, bias=False, groups=32),
            nn.BatchNorm2d(32), 
            act, 
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            act, 
            # Fourth depthwise separable convolution, 15 -> 7,
            nn.Conv2d(64, 64, 3, stride=2, bias=False, groups=64), 
            nn.BatchNorm2d(64),
            act, 
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            act
        ])

        # Define fc and output layers
        self.pool = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(128, 4)
        self.activation = nn.Sigmoid()
        
        # Weight initialization
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
            if isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
        nn.init.xavier_normal_(self.fc1.weight)


    def forward(self, batch): 
        ''' 
        Expected (BATCH_SIZEx1x250x250) input. Output is sigmoid logits.
        '''
        for layer in self.conv_layers: 
            batch = layer(batch)
        batch = self.pool(batch)
        batch = batch.view(-1, self.num_flat_features(batch))
        return self.fc1(batch)


    def num_flat_features(self, inputs):
        size = inputs.size()[1:]
        num_features = 1

        for s in size:
            num_features *= s
  
        return num_features
        

    def num_parameters(self): 
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

