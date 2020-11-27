# Importing libraries
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


# Building the model architecture for the face detection model
class Net(nn.Module):

    def __init__( self ):
        super( Net, self ).__init__()

        # Defining layers of the CNN
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)

        # max pooling layer
        self.pool = nn.MaxPool2d(2,2)

        # Linear layer
        self.fc1 = nn.Linear(256 * 3 * 3, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        #self.fc3 = nn.Linear(1024, 500)
        self.predict = nn.Linear(2048, 30)

        
        # Batch Norm conv
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5_bn = nn.BatchNorm2d(256)
            
        # Batch Norm linear
        self.fc1_bn = nn.BatchNorm1d(4096)
        self.fc2_bn = nn.BatchNorm1d(2048)
        #self.fc3_bn = nn.BatchNorm1d(500)

        # Droupout Layer
        self.dropout = nn.Dropout(0.5)

    def forward( self, x ):
        ## Define forward behavior
        x = self.pool(F.leaky_relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.leaky_relu(self.conv4_bn(self.conv4(x))))
        x = self.pool(F.leaky_relu(self.conv5_bn(self.conv5(x))))
            
        # flatten image input
        x = x.view(-1, 256 * 3 * 3 )
            
        # add 1st hidden layer, with relu activation function
        x = F.leaky_relu( self.fc1_bn(self.fc1(x)) )
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = F.leaky_relu( self.fc2_bn(self.fc2(x)) )
        x = self.dropout(x)
        # add 3rd hidden layer
        x = self.predict(x)
        return x