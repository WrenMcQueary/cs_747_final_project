import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import pdb

NUM_CLASSES = 104


#
# CLASSES
#
class Classifier(nn.Module):
    def __init__(self, n_conv_layers: int, n_fc_layers: int, dilation=int):
        """Defines layers of our neural net"""
        super(Classifier, self).__init__()
        
        self.n_conv_layers = n_conv_layers
        self.n_fc_layers = n_fc_layers
        self.dilation = dilation
        
        # Add convolutional layers
        input_channels_by_layer = get_input_channels_by_layer_count(self.n_conv_layers)
        for ii in range(self.n_conv_layers):
            # Calculate this_input_channels
            this_input_channels = input_channels_by_layer[ii]
            # Calculate this_output_channels
            if ii < self.n_conv_layers - 1:
                this_output_channels = input_channels_by_layer[ii+1]
            else:
                this_output_channels = 16
            # Calculate this_kernel_size
            if ii == 0:
                this_kernel_size = 5
            else:
                this_kernel_size = 3
            # Calculate this_dilation
            this_dilation = self.dilation
            # Add layer
            exec(f"self.conv{ii+1} = nn.Conv2d({this_input_channels}, {this_output_channels}, {this_kernel_size}, dilation={this_dilation})")
        
        # Define maxpooling
        # Calculate this_kernel_size
        this_kernel_size = 2
        # Calculate this_stride
        this_stride = 2
        # Define
        exec(f"self.pool = nn.MaxPool2d({this_kernel_size}, {this_stride})")
            
        # Add fully connected layers
        n_convs_and_dilation_to_initial_fc_layer_dims = {
            (1, 1): 61504,
            (2, 1): 14400,
            (2, 3): 10816,
            (3, 1): 3136,
            (3, 3): 1600,
            (4, 1): 576,
            (4, 2): 256,
            (4, 3): 64,
            (5, 1): 64,
        }
        initial_fc_layer_dims = n_convs_and_dilation_to_initial_fc_layer_dims[(self.n_conv_layers, self.dilation)]
        consec_layer_dim_ratio = (initial_fc_layer_dims/NUM_CLASSES) ** (1/self.n_fc_layers)    # Ratio of consecutive layers' dimensions, roughly
        for ii in range(self.n_fc_layers):
            # Calculate this_input_dimensions
            this_input_dimensions = int(initial_fc_layer_dims / (consec_layer_dim_ratio ** ii))
            # Calculate this_output_dimensions
            this_output_dimensions = int(initial_fc_layer_dims / (consec_layer_dim_ratio ** (ii+1)))
            # Hard code to ensure we end up with NUM_CLASSES layers
            if ii == (self.n_fc_layers - 1):
                this_output_dimensions = NUM_CLASSES
            # Add layer
            exec(f"self.fc{ii+1} = nn.Linear({this_input_dimensions}, {this_output_dimensions})")
        
        #self.conv1 = nn.Conv2d(3, 64, 5)
        #self.conv2 = nn.Conv2d(64, 32, 3)
        #self.conv3 = nn.Conv2d(32, 16, 3)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.fc1 = nn.Linear(3136, 480)
        #self.fc2 = nn.Linear(480, 240)
        #self.fc3 = nn.Linear(240, NUM_CLASSES)

    def forward(self, x):
        """Defines how signal passes between the layers of our neural net.
        Process batch_size images at a time.
        """
        # Convolutional and pooling passes
        if self.n_conv_layers > 10:
            raise NotImplementedError("Wren's code can't yet handle more than 10 convolutional layers")
        if self.n_conv_layers >= 1:
            x = self.pool(F.relu(self.conv1(x)))
        if self.n_conv_layers >= 2:
            x = self.pool(F.relu(self.conv2(x)))
        if self.n_conv_layers >= 3:
            x = self.pool(F.relu(self.conv3(x)))
        if self.n_conv_layers >= 4:
            x = self.pool(F.relu(self.conv4(x)))
        if self.n_conv_layers >= 5:
            x = self.pool(F.relu(self.conv5(x)))
        if self.n_conv_layers >= 6:
            x = self.pool(F.relu(self.conv6(x)))
        if self.n_conv_layers >= 7:
            x = self.pool(F.relu(self.conv7(x)))
        if self.n_conv_layers >= 8:
            x = self.pool(F.relu(self.conv8(x)))
        if self.n_conv_layers >= 9:
            x = self.pool(F.relu(self.conv9(x)))
        if self.n_conv_layers >= 10:
            x = self.pool(F.relu(self.conv10(x)))
        
        # Run view
        x = x.view(x.size()[0], x.size()[1] * x.size()[2] * x.size()[3])
        
        # Fully connected passes
        if self.n_fc_layers > 10:
            raise NotImplementedError("Wren's code can't yet handle more than 10 fully connected layers")
        if self.n_fc_layers > 1:
            x = F.relu(self.fc1(x))
        if self.n_fc_layers > 2:
            x = F.relu(self.fc2(x))
        if self.n_fc_layers > 3:
            x = F.relu(self.fc3(x))
        if self.n_fc_layers > 4:
            x = F.relu(self.fc4(x))
        if self.n_fc_layers > 5:
            x = F.relu(self.fc5(x))
        if self.n_fc_layers > 6:
            x = F.relu(self.fc6(x))
        if self.n_fc_layers > 7:
            x = F.relu(self.fc7(x))
        if self.n_fc_layers > 8:
            x = F.relu(self.fc8(x))
        if self.n_fc_layers > 9:
            x = F.relu(self.fc9(x))
        #---
        if self.n_fc_layers == 1:
            x = self.fc1(x)
        if self.n_fc_layers == 2:
            x = self.fc2(x)
        if self.n_fc_layers == 3:
            x = self.fc3(x)
        if self.n_fc_layers == 4:
            x = self.fc4(x)
        if self.n_fc_layers == 5:
            x = self.fc5(x)
        if self.n_fc_layers == 6:
            x = self.fc6(x)
        if self.n_fc_layers == 7:
            x = self.fc7(x)
        if self.n_fc_layers == 8:
            x = self.fc8(x)
        if self.n_fc_layers == 9:
            x = self.fc9(x)
        if self.n_fc_layers == 10:
            x = self.fc10(x)
        
        return x
        
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = self.pool(F.relu(self.conv3(x)))
        #x = x.view(x.size()[0], 16 * 14 * 14)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        #return x

        
#
# HELPER FUNCTIONS
#
def get_input_channels_by_layer_count(layers):
    """Return the number of input channels that every CONVOLUTIONAL LAYER should have, given the number of layers.
    :param layers: the number of convolutional layers
    """
    input_channels_by_layer = []
    for ii in range(layers):
        if ii == layers-1:
            input_channels_by_layer = [3] + input_channels_by_layer
        else:
            input_channels_by_layer = [32 * 2**ii] + input_channels_by_layer
    return input_channels_by_layer
