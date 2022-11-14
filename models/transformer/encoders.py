from torch.nn import functional as F
from torch.nn.modules.activation import LeakyReLU
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention, MultiHeadGeometryAttention
from models.transformer.grid_aug import BoxRelationalEmbedding


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # nn.Sequential一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，
        # 同时以神经网络模块为元素的有序字典也可以作为传入参数
        # nn.Conv2d 二维卷积 先实例化再使用 在Pytorch的nn模块中，它是不需要你手动定义网络层的权重和偏置的
        self.conv1 = nn.Sequential(  # input shape (1,28,28)
            nn.Conv2d(in_channels=64,  # input height 必须手动提供 输入张量的channels数
                      out_channels=16,  # n_filter 必须手动提供 输出张量的channels数
                      kernel_size=5,  # filter size 必须手动提供 卷积核的大小
                      # 如果左右两个数不同，比如3x5的卷积核，那么写作kernel_size = (3, 5)，注意需要写一个tuple，而不能写一个列表（list）
                      stride=1,  # filter step 卷积核在图像窗口上每次平移的间隔，即所谓的步长
                      padding=2  # con2d出来的图片大小不变 Pytorch与Tensorflow在卷积层实现上最大的差别就在于padding上
                      ),  # output shape (16,28,28) 输出图像尺寸计算公式是唯一的 # O = （I - K + 2P）/ S +1
            nn.ReLU(),  # 分段线性函数，把所有的负值都变为0，而正值不变，即单侧抑制
            nn.MaxPool2d(kernel_size=1)
            # 2x2采样，28/2=14，output shape (16,14,14) maxpooling有局部不变性而且可以提取显著特征的同时降低模型的参数，从而降低模型的过拟合
        )
        self.conv2 = nn.Sequential(nn.Conv2d(16, 8, 5, 1, 2),  # output shape (32,7,7)
                                   nn.ReLU(),
                                   nn.MaxPool2d(1))
        # 因上述几层网络处理后的output为[32,7,7]的tensor，展开即为7*7*32的一维向量，接上一层全连接层，最终output_size应为10，即识别出来的数字总类别数
        # 在二维图像处理的任务中，全连接层的输入与输出一般都设置为二维张量，形状通常为[batch_size, size]
        # self.out = nn.Linear(32 * 7 * 7, 10)  # 全连接层 7*7*32, num_classes

    def forward(self, x):
        x = self.conv1(x)  # 卷一次
        x = self.conv2(x)  # 卷两次
        # x = x.view(x.size(0), -1)  # flat (batch_size, 32*7*7)
        # 将前面多维度的tensor展平成一维 x.size(0)指batchsize的值
        # view()函数的功能根reshape类似，用来转换size大小
        # output = self.out(x)  # fc out全连接层 分类器
        return x


class SR(nn.Module):
    def __init__(self, N, d_model=512):
        super(SR, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(N*d_model, N*d_model),
            nn.LeakyReLU(),
            nn.Linear(N*d_model, d_model),
            nn.LeakyReLU()
        )

    def forward(self, x, layers, relative_geometry_weights, attention_mask = None, attention_weights = None, pos = None):
        out = x
        outs = []
        for l in layers:
            out = l(out, out, out,relative_geometry_weights, attention_mask, attention_weights, pos=pos)
            outs.append(out)
        outs = self.MLP(torch.cat(outs, -1))
        out = 0.2 * outs + out
        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadGeometryAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                                attention_module=attention_module,
                                                attention_module_kwargs=attention_module_kwargs)
        self.mhatt1 = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None, pos=None):
        if pos is not None:
            q = queries + pos
            k = keys + pos
            att = self.mhatt(q, k, values, relative_geometry_weights, attention_mask, attention_weights)
        else:
            q = queries
            k = keys
            att = self.mhatt1(q, k, values, attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module, # ScaledDotProductAttention
                                                  attention_module_kwargs=attention_module_kwargs) # {'m': args.m}
                                     for _ in range(N)])
        self.SR = SR(N, d_model)
        self.padding_idx = padding_idx

        # self.WGs = nn.ModuleList([nn.Sequential(nn.Linear(64, 1, bias=True), nn.LeakyReLU()) for _ in range(h)])

        # self.MLP = nn.Sequential(
        #     nn.Linear(N * d_model, N * d_model),
        #     nn.LeakyReLU(),
        #     nn.Linear(N * d_model, d_model),
        #     nn.LeakyReLU()
        # )
        self.cnn = CNN()

    def forward(self, input, attention_weights=None, pos=None):

        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        # grid geometry embedding
        # relative_geometry_embeddings = BoxRelationalEmbedding(input)
        # flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
        # box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        # box_size_per_head.insert(1, 1)
        # relative_geometry_weights_per_head = [layer(flatten_relative_geometry_embeddings).view(box_size_per_head) for
        #                                       layer in self.WGs]
        # relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        # relative_geometry_weights = F.relu(relative_geometry_weights)


        relative_geometry_embeddings = BoxRelationalEmbedding(input)
        relative_geometry_embeddings = relative_geometry_embeddings.permute(0, 3, 2, 1)
        # flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
        # box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        # box_size_per_head.insert(1, 1)
        # relative_geometry_weights_per_head = [layer(flatten_relative_geometry_embeddings).view(box_size_per_head) for
        #                                       layer in self.WGs]
        # relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        # relative_geometry_weights = F.relu(relative_geometry_weights)
        relative_geometry_weights = self.cnn(relative_geometry_embeddings)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        out = self.SR(input, self.layers, relative_geometry_weights, attention_mask, attention_weights, pos=pos)

        return out, attention_mask


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_weights=None, pos=None):
        mask = (torch.sum(input, dim=-1) == 0).unsqueeze(-1)
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = out.masked_fill(mask, 0)
        return super(TransformerEncoder, self).forward(out, attention_weights=attention_weights, pos=pos)
