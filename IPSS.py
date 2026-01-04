import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import os
import math
import cv2
from einops import rearrange
from torchvision import transforms as transforms,models
from torchvision.ops.deform_conv import DeformConv2d
from torch.utils.data import DataLoader
from erp_dataset import MyDataset
import warnings

warnings.filterwarnings("ignore", message="To copy construct from a tensor.*")
# layer_dic = torch.load('/mnt/10T/wkc/fix_offset/layerdic_8k.pt')
# offset_dic = torch.load('/mnt/10T/wkc/fix_offset/offsetdic_8k.pt')

# def creat_off_set(layer,position=None,patch_u=None,patch_v=None,layer_dic=layer_dic,offset_dic=offset_dic,device=None):
#     offset_global =offset_dic[layer_dic[layer]]
#     u,v = layer_dic[layer][0]
#     if position[0] >= 0.94 or position[1] >= 0.89:
#         position[0] = 0.92
#         position[1] = 0.83
        
#     u_star = int(math.ceil(position[0]*u))
#     v_star = int(math.ceil(position[1]*v))
#     offset_local=offset_global[:,:,u_star:u_star+patch_u,v_star:v_star+patch_v]
#     offset_local = torch.tensor(offset_local).clone().detach().to(device=device)
#     return offset_local

class EquiMDConv(nn.Module):
    def __init__(self,input_channel,out_put_channel,layer):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = out_put_channel
        self.conv = DeformConv2d(in_channels=self.input_channel, out_channels=self.output_channel, kernel_size=3, stride=1, padding=1)
        # 用于生成 offset 的卷积层
        self.conv_offset = nn.Conv2d(self.input_channel, 18, kernel_size=3, stride=1, padding=1)
        init_offset = torch.Tensor(np.zeros([18, self.input_channel, 3, 3]))
        self.conv_offset.weight = torch.nn.Parameter(init_offset)  
        # 用于生成 mask 的卷积层
        self.conv_mask = nn.Conv2d(self.input_channel, 9, kernel_size=3, stride=1, padding=1)
        init_mask = torch.Tensor(np.zeros([9, self.input_channel, 3, 3])+np.array([0.5]))
        self.conv_mask.weight = torch.nn.Parameter(init_mask)  
        self.layer = layer
 
    def forward(self, x, position):
        # offset_fix = creat_off_set(self.layer,position,x.shape[2],x.shape[3],layer_dic,offset_dic,x.device)
        #learnable offset
        offset = self.conv_offset(x)
        # offset = offset+offset_fix    
        mask = torch.sigmoid(self.conv_mask(x))
        out = self.conv(input=x, offset=offset, mask=mask)
        return out


class EMConvBlock(nn.Module):
    def __init__(self,input_channel,output_channel,layer,stride=2):
        super(EMConvBlock,self).__init__()
        self.conv1 = EquiMDConv(input_channel,output_channel,layer)
        self.conv = nn.Conv2d(input_channel,output_channel,kernel_size=1)
        self.use_1x1 = nn.Conv2d(input_channel,output_channel,kernel_size=1,stride = stride)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3,padding=1,stride=stride)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x, position):

        out_line = self.use_1x1(x)                  
        x1 = self.conv1(x,position) 
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.max_pool(x1)          
        out = x1 + out_line           
        out = self.sigmoid(out)
        
        return out

class BackBone_conv(nn.Module):
    def __init__(self):
        super(BackBone_conv,self).__init__()
        
        #layer1
        self.layer1_block1 = EMConvBlock(3,64,0)
        self.layer1_block2 = EMConvBlock(64,64,0,1)
        #layer2
        self.layer2_block1 = EMConvBlock(64,128,1)
        self.layer2_block2 = EMConvBlock(128,128,1,1)
        #layer3
        self.layer3_block1 = EMConvBlock(128,256,2)
        self.layer3_block2 = EMConvBlock(256,256,2,1)
    
    def forward(self,x,position):
        
        out = self.layer1_block1(x,position)
        out = self.layer1_block2(out,position)
        out = self.layer2_block1(out,position)
        out = self.layer2_block2(out,position)
        out = self.layer3_block1(out,position)
        out = self.layer3_block2(out,position)
        
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class PatchAttention(nn.Module):
    def __init__(self, in_dim):
        super(PatchAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.position_encoding = PositionalEncoding(d_model=in_dim)      
 
    def INF(self, B, H, W, device):
        inf_tensor = torch.tensor(float("inf"), device=device)
        return -torch.diag(inf_tensor.repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
    
    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        encoded_x = self.position_encoding(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        proj_query = self.query_conv(x)          # torch.Size([2, 320, 14, 14])
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)        # torch.Size([28, 14, 320])
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)        # torch.Size([28, 14, 320])
        proj_key = self.key_conv(x)              # torch.Size([2, 320, 14, 14])
        encoded_x = self.key_conv(encoded_x)
        encode_H = encoded_x.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        encode_W = encoded_x.permute(0,2,1,3).contiguous().view(m_batchsize*width,-1,width)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)      # torch.Size([28, 320, 14])
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)      # torch.Size([28, 320, 14])
        proj_value = self.value_conv(x)          # torch.Size([2, 2560, 14, 14])
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)  # torch.Size([28, 2560, 14])
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)  # torch.Size([28, 2560, 14])
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width, x.device)).view(m_batchsize,width,height,height).permute(0,2,1,3)   # torch.Size([2, 14, 14, 14])
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)                                                            # torch.Size([2, 14, 14, 14])
        encode_H = (torch.bmm(proj_query_H, encode_H )+self.INF(m_batchsize, height, width, x.device)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        encode_W = torch.bmm(proj_query_W, encode_W).view(m_batchsize,height,width,width) 
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))      # torch.Size([2, 14, 14, 28])
        concate_encode = self.softmax(torch.cat([encode_H, encode_W], 3))
        concate = concate + concate_encode
 
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)   # torch.Size([28, 14, 14])
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)          # torch.Size([28, 14, 14])
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)   # torch.Size([2, 2560, 14, 14])
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)   # torch.Size([2, 2560, 14, 14])

        return self.gamma*(out_H + out_W) + x


class EfficientChannelAttention(nn.Module):           
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(y)
        out = torch.multiply(x, out)
        return out

    
class MultiHead_SelfAttention(nn.Module):
    def __init__(self,input_dim,num_heads):
        super().__init__()
        self.num_heads=num_heads
        self.head_dim=input_dim//num_heads
        assert input_dim%num_heads==0 

        self.query=nn.Linear(input_dim,input_dim)
        self.key=nn.Linear(input_dim,input_dim)
        self.value=nn.Linear(input_dim,input_dim)

        self.output_linear=nn.Linear(input_dim,input_dim)

    def forward(self, x):
        batch_size,seq_len,input_dim=x.size()
        query=self.query(x).view(batch_size,seq_len,self.num_heads,self.head_dim)

        key=self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value=self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        query=query.transpose(1,2)
        key=key.transpose(1,2)
        value=value.transpose(1,2)
        attention_scores=torch.matmul(query,key.transpose(-2,-1))/torch.sqrt(torch.tensor(self.head_dim,dtype=torch.float))

        attention_weights=torch.softmax(attention_scores,dim=-1)
        attention=torch.matmul(attention_weights,value)
        attention=attention.transpose(1,2).contiguous().view(batch_size,seq_len,input_dim)
        output=self.output_linear(attention)

        return output

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
 
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class DAI(nn.Module):
    def __init__(self, in_dim):
        super(DAI,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.gamma = nn.Parameter(torch.zeros(1)) 
        self.addnorm = AddNorm(128, 0.2)
        self.ma = MultiHead_SelfAttention(128,8)
        self.eca = EfficientChannelAttention(128)
        
    def forward(self, x, y):
        B, C, H, W = x.size()
        dis = rearrange(x,'b c h w -> b (h w) c')    # torch.Size([2, 784, 128])
        ref = rearrange(y,'b c h w -> b (h w) c')    # torch.Size([2, 784, 128])
        dis_ma = self.ma(dis)
        ref_ma = self.ma(ref)
        dis_ad = self.addnorm(dis, dis_ma)
        ref_ad = self.addnorm(ref, ref_ma)
        dis = rearrange(dis_ad, 'b (h w) c -> b c h w', h=28, w=28)
        ref = rearrange(ref_ad, 'b (h w) c -> b c h w', h=28, w=28)
        diff = dis - ref
        
        k_x = self.key_conv(x).view(B, -1, W*H) 
        q_y = self.query_conv(diff).view(B, -1, W*H).permute(0,2,1)
        energy_x = torch.bmm(q_y,k_x)
        attention_x = self.softmax(energy_x)
        v_x = self.value_conv(x).view(B, -1, W*H)
        out_x = torch.bmm(v_x,attention_x.permute(0,2,1))
        out_x = out_x.view(B, C, H, W)
        out_x = self.gamma*out_x + x
        
        k_y = self.key_conv(y).view(B, -1, W*H) 
        q_x = self.query_conv(diff).view(B, -1, W*H).permute(0,2,1)
        energy_y = torch.bmm(q_x,k_y)
        attention_y = self.softmax(energy_y)
        v_out = self.value_conv(x).view(B, -1, W*H)
        out_y = torch.bmm(v_out,attention_y.permute(0,2,1))
        out_y = out_y.view(B, C, H, W)
        out_y = self.gamma*out_y + y
        
        out = torch.cat((out_x, out_y),dim=1)
        out = self.conv(out)
        out = self.eca(out)
        
        return out

class IPSS(nn.Module):
    def __init__(self):
        super(IPSS,self).__init__()
        self.backbone_conv = BackBone_conv()
        self.backbone = nn.Sequential(*list(models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1).children())[:-3])
        self.equi_md_conv = EquiMDConv(256, 256, 2)
        self.ppa = PatchAttention(in_dim=128)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten() 
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.dai = DAI(128)
        self.conv = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels=5120, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64, 1)
    
    
    def forward_vector(self, x, y, position):
        B, N, C, H, W = x.shape
        feature_dis = torch.tensor([]).to(x.device)
        feature_ref = torch.tensor([]).to(x.device)
        for i in range(N):
            input_x = x[:,i,:,:,:]
            input_y = y[:,i,:,:,:]
            input_position = position[:,i,:]
            input_position = input_position[0]
            cnn_x = self.backbone_conv(input_x, input_position)                # torch.Size([4, 256, 28, 28])
            cnn_y = self.backbone_conv(input_y, input_position)

            dis = self.backbone(input_x)
            ref = self.backbone(input_y)
         
            out_x = self.upsample(self.upsample(self.conv(dis)))
            out_y = self.upsample(self.upsample(self.conv(ref)))

            out_x = self.equi_md_conv(out_x, input_position)
            out_y = self.equi_md_conv(out_y, input_position)
            dis = torch.cat((cnn_x, out_x),dim=1)                              # torch.Size([4, 512, 28, 28])
            ref = torch.cat((cnn_y, out_y),dim=1)                              # torch.Size([4, 512, 28, 28])

            feature_dis = torch.cat((dis, feature_dis),dim=1)
            feature_ref = torch.cat((ref, feature_ref),dim=1)

        return feature_dis, feature_ref
    
    def forward(self, x, y, position): 
        dis, ref = self.forward_vector(x, y, position)  
        dis = self.conv1(dis)          # torch.Size([1, 128, 28, 28])
        ref = self.conv1(ref)
        feat = self.dai(dis, ref)
        feat = self.ppa(self.ppa(feat))
        feat = self.gap(feat)
        feat = self.flatten(feat)
        out_put = self.fc1(feat)
        out_put = self.fc2(out_put)
        score = torch.squeeze(out_put,dim=-1)

        return score
    
if __name__ == "__main__":
        device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")             
        net = IPSS().to(device=device)
        test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
        
        train_dataset = MyDataset('/mnt/10T/lzy/csvfiles/bd_test.csv',transform=test_transform,seed=80)
        train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=False,
    )
        for (img1,img2,mos),position in train_loader:
            img1 = img1.to(device=device)
            img2 = img2.to(device=device)
            position = position.to(device=device)
            out_put = net(img1,img2,position)
            print("out:",out_put)
            print("out:",out_put.shape)
            break