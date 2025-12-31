import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
import json


def contrastive_loss(z_emb: torch.Tensor,
                            l_emb: torch.Tensor,
                            tau: float = 0.1) -> torch.Tensor:
    """
    z_emb: (N, D)  视觉端（slot）embedding，所有 batch+所有对象展平后的结果
    l_emb: (N, D)  文本端（caption）embedding，对齐到每个 slot
    tau:   温度，CTRL-O 用 0.1

    返回：标量 loss
    """
    assert z_emb.shape == l_emb.shape
    N, D = z_emb.shape

    # 相似度矩阵 (N, N)，行：视觉，列：文本
    logits = z_emb @ l_emb.t() / tau   # (N, N)

    # 正样本就是对角线：第 i 个 slot 对第 i 个 caption
    target = torch.arange(N, device=z_emb.device)

    loss = F.cross_entropy(logits, target)
    return loss

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout, dim=1):
        if not (self.training and dropout):
            return x
        return x.new_empty(x.shape[:dim] + (1,) + x.shape[dim+1:]).bernoulli_(1 - dropout) / (1 - dropout) * x

# class TextEncoder(nn.Module):
#     """
#         极简版文本编码器：
#         - 输入：token ids (B, L) 和长度 x_len (B,)
#         - 做 embedding 后，对非 PAD 的位置做平均池化
#         - 输出：
#             ret    : (B, D) 句子级 embedding（给对比学习用）
#             output : (B, L, D) 每个 token 的 embedding（如果你想做别的用）
#             attns  : None（这里不做视觉 attention）
#     """
#     def __init__(self, vocab, embedding_dim=512, pad_token_id=0, dropout=0.0):
#         super().__init__()
#         self.vocab = vocab               # word -> id 字典
#         self.idx2word = {i: w for w, i in vocab.items()}
#         self.vocab_size = len(vocab)

#         self.embedding_dim = embedding_dim
#         self.pad_token_id = pad_token_id

#         self.embedding = nn.Embedding(
#             num_embeddings=self.vocab_size,
#             embedding_dim=self.embedding_dim,
#             padding_idx=self.pad_token_id,
#         )

#         self.dropout_o = dropout
#         self.dropout = LockedDropout()
#         self.output_dropout = nn.Dropout(self.dropout_o)

#         self.hidden_dim = self.embedding_dim

#         self.text_encoder = "embedding"
#         self.embedding_type = None

#     def forward(self, x, x_len, image_features=None, image_feature_map=None):
#         """
#         x      : LongTensor (B, L)  句子 token ids（已对齐、pad 好）
#         x_len  : LongTensor (B,)    每句真实长度（不含 pad）
#         返回:
#             ret    : (B, D) 句向量（平均池化）
#             output : (B, L, D) 每个 token 的 embedding
#             attns  : None
#         """
#         # (B, L, D)
#         attns = None
#         embedding = self.embedding(x)
        
#         if self.text_encoder == "embedding":
#             raw_output = embedding
#             if self.embedding_type == "flat":
#                 ret = torch.sum(raw_output, dim=1) / x_len.unsqueeze(1)
        
#         output = self.lockdrop(raw_output, self.dropout_o)

#         if self.embedding_type == "flat":
#             ret = self.output_dropout(ret)
#         elif self.embedding_type == "spatial":
#             ret = output

#         return ret, output, attns

#     @property
#     def regressional(self):
#         # 保留接口，按原代码逻辑：只有 lstm 才是 regressional，这里返回 False 就行
#         return False
        
class SimpleConvBackbone(nn.Module):
    """
    CNN backbone
    output: feature map (B, C, h, w)。
    change: DINO、ViT 
    """
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        return self.encoder(x)   # (B, C, h, w)

class SlotTextModel(nn.Module):
    """
    input: slot_feat (B,D_slot), caption tokens (B,L)
    output: loss + embeddings
    """
    def __init__(
            self, 
            textencoder: nn.Module,
            slot_dim:int,
            embed_dim:int =256,
            tau: float =0.1,
        ):
        super().__init__()
        self.text_encoder = textencoder
        self.tau = tau

        self.vision_proj = nn.Linear(slot_dim, embed_dim)
        self.text_proj = nn.Linear(getattr(textencoder, "embed_dim", embed_dim), embed_dim)

    @staticmethod
    def _l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)
    
    def forward(self, slot_feat, captions):
        B = slot_feat.size(0)

        z = self.vision_proj(slot_feat)  # (B, D)
        z = self._l2norm(z)

        t = self.text_encoder(captions)  # (B, D)
        t = self._l2norm(t)

        logits = (z @ t.t()) / self.tau           # (B, B)
        targets = torch.arange(B, device=logits.device)

        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.t(), targets)
        loss = 0.5 * (loss_i2t + loss_t2i)

        return loss, z, t

        
        
    
# class MultimodalModel(nn.Module):
#     """
#     intput:
#       images:   List[Tensor(B,3,H,W)]
#       masks:    List[Tensor(B,K_i,H,W)]
#       captions: List[List[str]]

#     输出：
#       loss_cc:  InfoNCE
#       z_all:    (N, D) vision embedding
#       l_all:    (N, D) text embedding
#     """
#     def __init__(self,
#                  vision_backbone: nn.Module = None,
#                  text_encoder: nn.Module = None,
#                  embed_dim: int = 256,
#                  tau: float = 0.1):
#         super().__init__()
#         self.nlp = spacy.load("en_core_web_sm")
#         with open("/home/wz3008/slot-attn/vocab.json") as f:
#             self.vocab = json.load(f)

        
#         if not vision_backbone:
#             self.vision_backbone = SimpleConvBackbone(feature_dim=embed_dim)
#         else:
#             self.vision_backbone = vision_backbone

#         self.tau = tau
    
#         self.text_encoder = text_encoder

#         self.vision_proj = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(embed_dim, embed_dim),
#         )
#         self.text_proj = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(embed_dim, embed_dim),
#         )

#     def tokenize(self, captions):
#         """
#         captions: List[str]
#         token_ids: List[List[int]], len=N, each is the list of caption's token id
#         """
#         max_seq_len = 25

#         if isinstance(captions, str):
#             captions = [captions]
        
#         all_tokens = []
#         token_lengths = []

#         for cap in captions:
#             doc = self.nlp(cap)
#             word_tokens = [token.text for token in doc]

#             if len(word_tokens) > max_seq_len - 2:
#                 word_tokens = word_tokens[:max_seq_len - 2]
            
#             token_lenght = len(word_tokens) + 2

#             tokens = [self.vocab["<sos>"]] + [self.vocab.get(token, self.vocab["<unk>"]) for token in word_tokens] + [self.vocab["<eos>"]] + [self.vocab["<pad>"]] * (max_seq_len - len(word_tokens) - 2)
#             all_tokens.append(tokens)
#             token_lengths.append(token_lenght)
#         return tokens, token_lengths

#     def forward(self, images, masks, captions):
#         """
#         images:   List[Tensor(3,H,W)]
#         gt_masks: List[Tensor(K_i,H,W)]
#         captions: List[List[str]]
#         """
#         device = next(self.parameters()).device

#         if isinstance(images, list):
#             imgs = torch.stack([img.to(device) for img in images], dim=0)  # (B,3,H,W)
#         B = imgs.shape[0]

#         feats = self.vision_backbone(imgs)   # (B, C, h, w)
#         B, C, h, w = feats.shape

#         # 3. 用 mask 对 feature map 做加权平均，得到每个 object 的视觉向量
#         all_slot_feats = []   # List[Tensor(K_i, C)]
#         all_captions_flat = []  # List[str]，和 slot 对齐

#         for b in range(B):
#             feat_b = feats[b]                   # (C, h, w)
#             mask_b = masks[b].to(device)        # (K_i, H, W)
#             K_i, H, W = mask_b.shape

#             # 将 mask resize 到 feature map 的空间大小 (h, w)
#             mask_resized = F.interpolate(
#                 mask_b.unsqueeze(1).float(),    # (K_i,1,H,W)
#                 size=(h, w),
#                 mode="nearest"
#             ).squeeze(1)                         # (K_i,h,w)

#             # 展平空间维度
#             feat_flat = feat_b.view(C, -1)      # (C, h*w)
#             mask_flat = mask_resized.view(K_i, -1)  # (K_i, h*w)

#             # 防止除 0
#             eps = 1e-6
#             weights = mask_flat / (mask_flat.sum(dim=1, keepdim=True) + eps)  # (K_i, h*w)
#             # (K_i, C) = (K_i,h*w) @ (h*w,C)
#             slot_feats = weights @ feat_flat.t()      # (K_i, C)

#             all_slot_feats.append(slot_feats)
#             all_captions_flat.extend(captions[b])     # 将这帧的 K_i 个 caption 接到总 list 里

#         # 拼成 (N, C)
#         z_vis = torch.cat(all_slot_feats, dim=0)   # (N, C)

#         # 4. 文本编码：把所有 object caption 拼成一个大 list，一次性编码
#         l_txt = self.text_encoder(all_captions_flat)  # (N, D)

#         # 5. 投影到公共空间
#         z_emb = self.vision_proj(z_vis)  # (N, D)
#         l_emb = self.text_proj(l_txt)    # (N, D)

#         # 6. CTRL-O 风格 InfoNCE 对比损失
#         loss_cc = contrastive_loss(z_emb, l_emb, tau=self.tau)

#         return loss_cc, z_emb, l_emb

class TextEncoder(nn.Module):
    """
    - Python hash: token to [0, vocab_size)
    - nn.Embedding + GRU encoder
    - change to  CLIP / LLaMA encoder。
    """
    """
    - tokenizer:
    -vocab: 
    -special token:
    -padding mask:
    -nlp model: need to match nlp model's input and output
    """
    def __init__(self, embed_dim: int = 256, vocab_size: int = 10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.nlp = spacy.load("en_core_web_sm")
        with open("/home/wz3008/slot-attn/vocab.json") as f:
            self.vocab = json.load(f)

    def tokenize(self, captions):
        """
        captions: List[str]
        token_ids: List[List[int]], len=N, each is the list of caption's token id
        """
        max_seq_len = 25
        all_tokens = []
        token_lengths = []
        # import pdb; pdb.set_trace()
        # if isinstance(captions, str):
        #     captions = [captions]

        for cap in captions:
            doc = self.nlp(cap)
            word_tokens = [token.text for token in doc]

            if len(word_tokens) > max_seq_len - 2:
                word_tokens = word_tokens[:max_seq_len - 2]
            
            token_lenght = len(word_tokens) + 2

            tokens = [self.vocab["<sos>"]] + [self.vocab.get(token, self.vocab["<unk>"]) for token in word_tokens] + [self.vocab["<eos>"]] + [self.vocab["<pad>"]] * (max_seq_len - len(word_tokens) - 2)
            all_tokens.append(tokens)
            token_lengths.append(token_lenght)
        return all_tokens, token_lengths

    def forward(self, captions)-> torch.Tensor:
        """
        captions: List[str]
        sent_emb: (N, D): Tensor
        """
        token_ids, token_lengths = self.tokenize(captions)  
        max_len = max(len(ids) for ids in token_ids)

        # padding
        padded = []
        lengths = []
        for ids in token_ids:
            lengths.append(len(ids))
            if len(ids) < max_len:
                ids = ids + [0] * (max_len - len(ids))
            padded.append(ids)

        padded = torch.tensor(padded, dtype=torch.long, device=self.embedding.weight.device)  # (N, L)
        emb = self.embedding(padded)  # (N, L, D)

        # GRU
        out, h_n = self.gru(emb)      # h_n: (1, N, D)
        sent_emb = h_n.squeeze(0)     # (N, D)
        return sent_emb