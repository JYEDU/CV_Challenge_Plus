from torch import nn
import torchvision
import torch
import numpy as np

class MSCN(nn.Module):

    def __init__ (self, kernel_size=3, n_channels=64, scaling_factor=2):
        super(MSCN, self).__init__()

        # filter calculation
        # layer = 7
        # max_filter = 196
        # min_filter = 32


        # ACTIVE FUNC & DROPOUT

        self.drop = torch.nn.Dropout(p=0.8)
        self.prelu = torch.nn.PReLU()

        # FEATURE EXTRACTION LEVEL

        self.conv3_1 = torch.nn.Conv2d(1,196,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.conv3_2 = torch.nn.Conv2d(196,147,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.conv3_3 = torch.nn.Conv2d(147,117,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.conv3_4 = torch.nn.Conv2d(117,92,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.conv3_5 = torch.nn.Conv2d(92,70,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.conv3_6 = torch.nn.Conv2d(70,50,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.conv3_7 = torch.nn.Conv2d(50,32,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        
        self.conv1_1 = torch.nn.Conv2d(1,196,1,stride=1,padding=0,padding_mode="replicate",bias=True)
        self.conv1_2 = torch.nn.Conv2d(196,147,1,stride=1,padding=0,padding_mode="replicate",bias=True)
        self.conv1_3 = torch.nn.Conv2d(147,117,1,stride=1,padding=0,padding_mode="replicate",bias=True)
        self.conv1_4 = torch.nn.Conv2d(117,92,1,stride=1,padding=0,padding_mode="replicate",bias=True)
        self.conv1_5 = torch.nn.Conv2d(92,70,1,stride=1,padding=0,padding_mode="replicate",bias=True)
        self.conv1_6 = torch.nn.Conv2d(70,50,1,stride=1,padding=0,padding_mode="replicate",bias=True)
        self.conv1_7 = torch.nn.Conv2d(50,32,1,stride=1,padding=0,padding_mode="replicate",bias=True)        
 
        torch.nn.init.kaiming_normal_(self.conv3_1.weight)
        torch.nn.init.kaiming_normal_(self.conv3_2.weight)
        torch.nn.init.kaiming_normal_(self.conv3_3.weight)
        torch.nn.init.kaiming_normal_(self.conv3_4.weight)
        torch.nn.init.kaiming_normal_(self.conv3_5.weight)
        torch.nn.init.kaiming_normal_(self.conv3_6.weight)
        torch.nn.init.kaiming_normal_(self.conv3_7.weight)

        torch.nn.init.kaiming_normal_(self.conv1_1.weight)
        torch.nn.init.kaiming_normal_(self.conv1_2.weight)
        torch.nn.init.kaiming_normal_(self.conv1_3.weight)
        torch.nn.init.kaiming_normal_(self.conv1_4.weight)
        torch.nn.init.kaiming_normal_(self.conv1_5.weight)
        torch.nn.init.kaiming_normal_(self.conv1_6.weight)
        torch.nn.init.kaiming_normal_(self.conv1_7.weight)

        # RECONSTRUCTION NETWORK LEVEL

        self.A1 = torch.nn.Conv2d(1408,32,1,stride=1,bias=True)
        self.B1 = torch.nn.Conv2d(1408,32,3,stride=1,padding=1,padding_mode="replicate",bias=True)

        torch.nn.init.kaiming_normal_(self.A1.weight)
        torch.nn.init.kaiming_normal_(self.B1.weight)


        # Upsampled layer
        self.upconv = torch.nn.Conv2d(64,2*2*96,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.pixelshufflerlayer = torch.nn.PixelShuffle(2)

        torch.nn.init.kaiming_normal_(self.upconv.weight)

        self.reconv = torch.nn.Conv2d(96,1,3,stride=1,padding=1,padding_mode="replicate",bias=False)

        torch.nn.init.kaiming_normal_(self.reconv.weight)

    def forward(self, lr):

        #Feature Update
        output1 = self.drop(self.prelu(self.conv3_1(lr)))
        s3_1 = output1
        output2 = self.drop(self.prelu(self.conv1_1(lr)))
        s1_1 = output2        
        output = torch.cat([s3_1,s1_1], 1)
        s1 = output
                
        output1 = self.drop(self.prelu(self.conv3_2(output1)))
        s3_2 = output1
        output2 = self.drop(self.prelu(self.conv1_2(output2)))
        s1_2 = output2    
        output = torch.cat([s3_2,s1_2], 1)   
        s2 = output
                
        output1 = self.drop(self.prelu(self.conv3_3(output1)))
        s3_3 = output1
        output2 = self.drop(self.prelu(self.conv1_3(output2)))
        s1_3 = output2        
        output = torch.cat([s3_3,s1_3], 1)   
        s3 = output
                        
        output1 = self.drop(self.prelu(self.conv3_4(output1)))
        s3_4 = output1
        output2 = self.drop(self.prelu(self.conv1_4(output2)))
        s1_4 = output2        
        output = torch.cat([s3_4,s1_4], 1)   
        s4 = output
                
        output1 = self.drop(self.prelu(self.conv3_5(output1)))
        s3_5 = output1
        output2 = self.drop(self.prelu(self.conv1_5(output2)))
        s1_5 = output2        
        output = torch.cat([s3_5,s1_5], 1)   
        s5 = output
        
        output1 = self.drop(self.prelu(self.conv3_6(output1)))
        s3_6 = output1
        output2 = self.drop(self.prelu(self.conv1_6(output2)))
        s1_6 = output2   
        output = torch.cat([s3_6,s1_6], 1)   
        s6 = output
                
        output1 = self.drop(self.prelu(self.conv3_7(output1)))
        s3_7 = output1
        output2 = self.drop(self.prelu(self.conv1_7(output2)))
        s1_7 = output2        
        output = torch.cat([s3_7,s1_7], 1)   
        s7 = output        

        output = torch.cat([s1,s2,s3,s4,s5,s6,s7], dim = 1)
        

        # Reconstruction Update

        a1_out = self.drop(self.prelu(self.A1(output)))
        b1_out = self.drop(self.prelu(self.B1(output)))
        
        output = torch.cat([a1_out, b1_out], dim = 1)

        
        # transposed

        up_out = self.pixelshufflerlayer(self.upconv(output))
        re_out = self.reconv(up_out)
        
        return re_out

