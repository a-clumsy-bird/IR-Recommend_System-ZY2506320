import numpy as np
import torch

# 1x1卷积实现的FFN
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # 恢复为(N, Length, C)
        return outputs


class RecModel(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(RecModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        # 添加固定位置编码的维度参数
        # self.hidden_units = args.hidden_units
        # self.maxlen = args.maxlen
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # 注意力模块的层归一化
        self.attention_layers = torch.nn.ModuleList()  # 多头注意力层
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            # 层归一化
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            # 多头注意力
            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)
            # 前馈网络的层归一化
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            # 前馈网络
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
    
    def create_fixed_position_encoding(self, seq_len, d_model):
        """
        创建Transformer原始的正弦位置编码
        seq_len: 序列长度
        d_model: 隐藏层维度
        """
        position = torch.arange(seq_len).unsqueeze(1).float()  # [seq_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(torch.log(torch.tensor(10000.0)) / d_model))  # [d_model/2]
        
        pos_encoding = torch.zeros(seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用sin
        pos_encoding[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用cos
        
        return pos_encoding  # [seq_len, d_model]

    # 将输入序列转换为特征表示
    def log2feats(self, log_seqs): 
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5  # 缩放嵌入向量

        # # 创建固定位置编码（只在第一次运行时创建并缓存）
        # if not hasattr(self, 'fixed_pos_encoding'):
        #     # 创建最大长度的位置编码并缓存
        #     self.fixed_pos_encoding = self.create_fixed_position_encoding(
        #         self.maxlen, self.hidden_units
        #     ).to(self.dev)  # [maxlen, hidden_units]
        
        # # 获取当前序列的实际长度
        # seq_len = log_seqs.shape[1]
        
        # # 取前seq_len个位置编码
        # pos_encoding = self.fixed_pos_encoding[:seq_len, :]  # [seq_len, hidden_units]
        
        # # 扩展维度以匹配批次大小
        # pos_encoding = pos_encoding.unsqueeze(0)  # [1, seq_len, hidden_units]
        # pos_encoding = pos_encoding.expand(log_seqs.shape[0], -1, -1)  # [batch, seq_len, hidden_units]
        
        # # 添加位置编码
        # seqs += pos_encoding

        # 创建位置编码矩阵，形状为(B,L)
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= (log_seqs != 0)  # 将padding位置的位置编码置零

        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # 因果注意力掩码矩阵
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))  # 下三角矩阵

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)  # 转置为(L,B,C)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x,
                                                attn_mask=attention_mask)
                seqs = seqs + mha_outputs # 残差连接
                seqs = torch.transpose(seqs, 0, 1) # (B,L,C)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
                                                attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):       
        log_feats = self.log2feats(log_seqs) # 获取序列的特征表示

        #  正样本和负样本的嵌入
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        # 通过点积计算相似度
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): 
        log_feats = self.log2feats(log_seqs) # 获取序列的特征表示
        final_feat = log_feats[:, -1, :] # 最后一个输出作为用户表示
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        
        # 通过矩阵乘法计算用户表示和候选物品相似度
        # item_embs: (batch_size, num_candidates, hidden_units)
        # final_feat: (batch_size, hidden_units) -> unsqueeze后变为 (batch_size, hidden_units, 1)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1) # (B,N)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits 
