import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
import json
from typing import List

from torch.nn.utils.rnn import pack_padded_sequence
import timm

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
            Mmax: int = 10,
            # scene_pool: str = "attn" # {"attn","mean","concat"}
        ):
        super().__init__()
        self.text_encoder = textencoder
        self.tau = tau
        self.Mmax = Mmax
        # self.scene_pool = scene_pool

        self.vision_proj = nn.Linear(slot_dim, embed_dim)
        self.text_proj = nn.Linear(getattr(textencoder, "embed_dim", embed_dim), embed_dim)

        # for concatenate linear pooling
        self.scene_proj = nn.Linear(Mmax * embed_dim, embed_dim)
        self.pad_text = nn.Parameter(torch.zeros(embed_dim))
        # # for attentive pooling over slots (scene-level)
        # self.scene_attn_q = nn.Parameter(torch.randn(embed_dim))
        # self.scene_attn_k = nn.Linear(embed_dim, embed_dim, bias=False)

        # # for concat pooling (needs fixed K at runtime; we lazy-init on first use)
        # self._scene_concat_proj = None  # nn.Linear(K*D, D)
        
        self.scene_pma_q = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        ############################ Early fusion ##############################
        self.max_text_len = 50
        self.ef_cls = nn.Parameter(torch.randn(1, 1, embed_dim))      # CLS token
        self.ef_type_emb = nn.Embedding(3, embed_dim)                 # 0:vision, 1:text, 2:cls
        self.ef_pos_emb = nn.Parameter(torch.randn(1, 1 + Mmax + self.max_text_len, embed_dim))
        
        ef_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=4 * embed_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        
        self.ef_xf = nn.TransformerEncoder(ef_layer, num_layers=2)
        ########################################################################

    @staticmethod
    def _l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)
    
    def infonce_loss(self, z, t):
        B = z.shape[0]
        tau = getattr(self, "tau", 0.07)
        logits = (z @ t.t()) / tau
        labels = torch.arange(B, device=logits.device)
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_i2t + loss_t2i)
        
        loss_batch = F.cross_entropy(logits, labels, reduction="none")
        return loss_batch
    
    def attentive_pooling(self, z_slots):
        q = z_slots.mean(dim=1, keepdim=True)                 # (B,1,D)
        attn = (z_slots * q).sum(dim=-1)                      # (B,M)
        alpha = F.softmax(attn, dim=1).unsqueeze(-1)          # (B,M,1)
        z_scene = (z_slots * alpha).sum(dim=1)                # (B,D)
        return z_scene
    
    def PMA(self,z_slots):  #learnable pooling by multihead attention
        B = z_slots.shape[0]
        q = self.scene_pma_q.expand(B, -1, -1)                       # (B,1,D)
        scores = torch.matmul(q, z_slots.transpose(1, 2))            # (B,1,M)
        scores = scores / (z_slots.shape[-1] ** 0.5)
        alpha = F.softmax(scores, dim=-1)                            # (B,1,M)
        z_scene = torch.matmul(alpha, z_slots).squeeze(1)            # (B,D)
        return z_scene
    
    def early_fuse(self, z_slots, tok_ids, tok_lens):
        """
        z_slots: (B,M,D)   after vision_proj + l2norm
        tok_ids: (B,1,L)
        tok_lens:(B,1)
        return:  z_fuse (B,D)  # CLS pooling
        """
        B, M, D = z_slots.shape
        L = tok_ids.shape[-1]
        device = z_slots.device
        
        t_words = self.text_encoder.embedding(tok_ids.squeeze(1)) #B,1,L -> B,L,D
        
        # ---- build multimodal token sequence: [CLS, V(=slots), T(=words)] ----
        cls = self.ef_cls.expand(B, 1, D)                            # (B,1,D)
        x = torch.cat([cls, z_slots, t_words], dim=1)                # (B, 1+M+L, D)
        
        # ---- type embedding ----
        type_ids = torch.cat([
            torch.full((B, 1), 2, device=device, dtype=torch.long),  # CLS
            torch.full((B, M), 0, device=device, dtype=torch.long),  # vision
            torch.full((B, L), 1, device=device, dtype=torch.long),  # text
        ], dim=1) 
        
        x = x + self.ef_type_emb(type_ids)                          # (B, 1+M+L, D)
        x = x + self.ef_pos_emb[:, : (1 + M + L), :].to(device)     # (B, 1+M+L, D)
        
        # key padding mask: mask text pads only
        text_len = tok_lens.squeeze(1).to(device)                   # (B,)
        text_pos = torch.arange(L, device=device).unsqueeze(0)      # (1,25)
        text_mask = text_pos >= text_len.unsqueeze(1)               # (B,25)
    
        key_padding_mask = torch.cat([
            torch.zeros((B,1+M), device=device, dtype=torch.bool),
            text_mask
        ], dim=1)                                                   # (B,1+M+25)
    
        y = self.ef_xf(x, src_key_padding_mask=key_padding_mask)    # (B,1+M+25,D)
        z_fuse = self._l2norm(y[:, 0])                              # CLS pooling -> (B,D)
        return z_fuse
        
    def forward(self, slot_feat, tok_ids, tok_lens):
        """
        slot_feat: (B,M,Dslot)
        tok_ids:   (B,M,L) / (B,1,L)
        tok_lens:  (B,M)   / (B,1) 
        """
        B, M, _ = slot_feat.shape
        z_slots = self._l2norm(self.vision_proj(slot_feat))   # B M D
        
        # text encode once
        if tok_ids.shape[1] != 1:
            tok_ids_flat = tok_ids.view(B*M, -1).to(slot_feat.device)
            tok_lens_flat = tok_lens.view(B*M).to(slot_feat.device)
            t_flat = self.text_encoder.forward_ids(tok_ids_flat, tok_lens_flat)  # (B*M,Dtxt)
            t_slots = self._l2norm(self.text_proj(t_flat)).view(B, M, -1) 
            
            # scene loss
            valid = (tok_lens > 1).float().unsqueeze(-1)       # (B,M,1)
            
            z_scene = (z_slots * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)  # (B,D)
            z_scene = self._l2norm(z_scene)
            t_scene = (t_slots * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)  # (B,D)
            t_scene = self._l2norm(t_scene)
        else:
            tok_ids_flat = tok_ids.view(B*1, -1).to(slot_feat.device)
            tok_lens_flat = tok_lens.view(B*1).to(slot_feat.device)
            t_slots = self.text_encoder.forward_ids(tok_ids_flat,tok_lens_flat)  #B,Dtxt
            t_slots = self._l2norm(self.text_proj(t_slots)) #B,256
            
            t_scene = t_slots
            z_scene = z_slots.sum(dim=1)
            #z_scene = self.attentive_pooling(z_slots)
            #z_scene = self.PMA(z_slots)
            #z_scene = self.early_fuse(z_slots, tok_ids, tok_lens)
            
        # attribute loss
        #sim_attr = (z_slots * t_slots).sum(dim=-1)
        #attribute_loss = ((1.0 - sim_attr).pow(2)).mean()
        #attribute_loss = torch.tensor(0).to(slot_feat.device)
        
        ##################################
        # Huber
        #sim_scene = (z_scene * t_scene).sum(dim=-1)       # (B,)
        #scene_loss = (1.0 - sim_scene).pow(2).mean()
        
        scene_loss = self.infonce_loss(z_scene, t_scene)
        scene_loss = scene_loss.mean()
        
        attribute_loss = torch.zeros_like(scene_loss).to(scene_loss.device)
        
        loss = attribute_loss + scene_loss

        return attribute_loss, scene_loss, loss
        
    def evaluate(self, slot_feat, tok_ids, tok_lens):
        """
        slot_feat: (B,M,Dslot)
        tok_ids:   (B,M,L)
        tok_lens:  (B,M)
        output:
        attribute_loss: 4
        scene_loss: 4
        loss: 4
        """
        #import pdb; pdb.set_trace()
        B, M, _ = slot_feat.shape
        z_slots = self._l2norm(self.vision_proj(slot_feat))
        
        # text encode once
        if tok_ids.shape[1] != 1:
            tok_ids_flat = tok_ids.view(B*M, -1).to(slot_feat.device)
            tok_lens_flat = tok_lens.view(B*M).to(slot_feat.device)
            t_flat = self.text_encoder.forward_ids(tok_ids_flat, tok_lens_flat)  # (B*M,Dtxt)
            t_slots = self._l2norm(self.text_proj(t_flat)).view(B, M, -1) 
            
            # scene loss
            valid = (tok_lens > 1).float().unsqueeze(-1)       # (B,M,1)
            
            z_scene = (z_slots * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)  # (B,D)
            z_scene = self._l2norm(z_scene)
            t_scene = (t_slots * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)  # (B,D)
            t_scene = self._l2norm(t_scene)
        else:
            tok_ids_flat = tok_ids.view(B*1, -1).to(slot_feat.device)
            tok_lens_flat = tok_lens.view(B*1).to(slot_feat.device)
            t_slots = self.text_encoder.forward_ids(tok_ids_flat,tok_lens_flat)  #B,Dtxt
            t_slots = self._l2norm(self.text_proj(t_slots)) 
            
            t_scene = t_slots
            z_scene = z_slots.sum(dim=1)
            #z_scene = self.attentive_pooling(z_slots)
            #z_scene = self.PMA(z_slots)
            #z_scene = self.early_fuse(z_slots, tok_ids, tok_lens)
        
        # attribute loss
        # sim_attr = (z_slots * t_slots).sum(dim=-1)  # 4,10
        # attribute_loss = ((1.0 - sim_attr).pow(2)).mean(1)  # 4,10
        #attribute_loss = torch.zeros(4,10).to(slot_feat.device)
        
        ##############################
        # Huber
        #sim_scene = (z_scene * t_scene).sum(dim=-1)       # (B,)
        #scene_loss = (1.0 - sim_scene).pow(2)   
        
        scene_loss = self.infonce_loss(z_scene, t_scene)
        
        attribute_loss = torch.zeros_like(scene_loss).to(scene_loss.device)
        
        loss = attribute_loss + scene_loss

        return attribute_loss, scene_loss, loss
        
    
        
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
        with open("/scratch/wz3008/new_SlotAttn/slot_attn_new/vocab.json") as f:
            self.vocab = json.load(f)
            
        self.pad_id = 1
    
    #def embedding(self, token_ids):
    #    """
    #    token_ids: (B,L) long
    #    return:    (B,L,D) float
    #    """
    #    return self.embedding(token_ids)
    
    def forward_ids(self, token_ids: torch.Tensor, lengths: torch.Tensor)-> torch.Tensor:
        """
        token_ids: (N, L)  long
        lengths:   (N,)    long
        return: sent_emb (N, D)
        """
        lengths = lengths.clamp(min=1).cpu()
        emb = self.embedding(token_ids)  # (N,L,D)
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)        # (1,N,D)
        return h_n.squeeze(0)            # (N,D)

    def tokenize(self, captions):
        """
        captions: List[str]
        token_ids: List[List[int]], len=N, each is the list of caption's token id
        """
        max_seq_len = 25
        all_tokens = []
        token_lengths = []
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
        
class VitTextModel(nn.Module):
    """
    input: slot_feat (B,D_slot), caption tokens (B,L)
    output: loss + embeddings
    """
    def __init__(
            self, 
            args,
            textencoder: nn.Module,
            embed_dim:int =256,
            tau: float =0.1,
            Mmax: int = 10,
            vit_name: str = "vit_base_patch16_224",  # ViT-B/14
            # scene_pool: str = "attn" # {"attn","mean","concat"}
        ):
        super().__init__()
        self.text_encoder = textencoder
        self.tau = tau
        self.Mmax = Mmax

        self.vit = timm.create_model(vit_name, pretrained=False, num_classes=0)
        vit_dim = getattr(self.vit, "num_features", None) or getattr(self.vit, "embed_dim", None)
        if args.vit_checkpoint:
            ckpt_path = args.vit_checkpoint 
            ckpt = torch.load(ckpt_path, map_location="cpu") 
            state = ckpt["student"] 
            self.vit.load_state_dict(state, strict=False) 
        vit_dim = getattr(self.vit, "num_features", None) or getattr(self.vit, "embed_dim", None)
        self.vision_proj = nn.Linear(vit_dim, embed_dim)
        
        self.text_proj = nn.Linear(getattr(textencoder, "embed_dim", embed_dim), embed_dim)
        
        self.mask_pooling = args.mask_pool

    @staticmethod
    def _l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)
    
    def forward(self, images: torch.Tensor, tok_ids: torch.Tensor, tok_lens: torch.Tensor, masks: torch.Tensor):
        """
        images:  (B,3,H,W)
        tok_ids: (B,M,L)
        tok_lens:(B,M)
        masks: B 1 H W
        return: scalar loss
        """
        if not self.mask_pool:
            z_img = self._l2norm(self.vision_proj(self.vit(images))) #(B,D)
        else:
            tok = self.vit.forward_features(images) #B,197,768
            tok = tok[:,1:,:]
            B,N,C = tok.shape
            S = int(N ** 0.5)
            feat_map = tok.transporse(1,2).reshape(B,C,S,S)
            masks_f = F.interpolate(
                masks.float().view(B*masks.shape[1], 1, masks.shape[-2], masks.shape[-1]),
                size = (S, S),
                mode = "nearest"
            ).view(B, masks.shape[1], 1, S, S) #B M 1 S S
            masks_f = masks_f.flatten(3) # (B,M,1,196)
            feat_map_flat = feat_map.flatten(2).unsqueeze(1) # (B,1,768,196)
            pooled = (masks_f * feat_map_flat).sum(dim=-1) / masks_f.sum(dim=-1).clamp_min(1e-6) #B,M,768
            pooled_mean = pooled.mean(dim=1)
            z_img = self._l2norm(self.vision_proj(pooled))
        
        B,M,_ = tok_ids.shape
        tok_ids_flat = tok_ids.view(B*M, -1)
        tok_lens_flat = tok_lens.view(B*M)
        t_flat = self.text_encoder.forward_ids(tok_ids_flat, tok_lens_flat)  # (B*M,Dtxt)
        t_slots = self._l2norm(self.text_proj(t_flat)).view(B, M, -1) 

        valid = (tok_lens > 1).float().unsqueeze(-1)       # (B,M,1)
        t_scene = (t_slots * valid).sum(dim=1) / (valid.sum(dim=1).clamp(min=1.0))  # (B,D)
        t_scene = self._l2norm(t_scene) # (B,D)

        # sim = (z_img * t_scene).sum(dim=-1)
        #loss_b = (1.0 - sim).pow(2)                         # (B,)
        #loss = loss_b.mean()
        tau = getattr(self, "tau", 0.07)
        logits = (z_img @ t_scene.t()) / tau
        labels = torch.arange(B, device=logits.device)
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_i2t + loss_t2i)
        
        loss_b = F.cross_entropy(logits, labels, reduction="none")
        return loss, loss_b
    
    def evaluate(self, images: torch.Tensor, tok_ids: torch.Tensor, tok_lens: torch.Tensor, masks: torch.Tensor):
        """
        images:  (B,3,H,W)
        tok_ids: (B,M,L)
        tok_lens:(B,M)
        masks: B 1 H W
        return: scalar loss
        """
        if not self.mask_pool:
            z_img = self._l2norm(self.vision_proj(self.vit(images))) #(B,D)
        else:
            tok = self.vit.forward_features(images) #B,197,768
            tok = tok[:,1:,:]
            B,N,C = tok.shape
            S = int(N ** 0.5)
            feat_map = tok.transporse(1,2).reshape(B,C,S,S)
            masks_f = F.interpolate(
                masks.float().view(B*masks.shape[1], 1, masks.shape[-2], masks.shape[-1]),
                size = (S, S),
                mode = "nearest"
            ).view(B, masks.shape[1], 1, S, S) #B M 1 S S
            masks_f = masks_f.flatten(3) # (B,M,1,196)
            feat_map_flat = feat_map.flatten(2).unsqueeze(1) # (B,1,768,196)
            pooled = (masks_f * feat_map_flat).sum(dim=-1) / masks_f.sum(dim=-1).clamp_min(1e-6) #B,M,768
            pooled_mean = pooled.mean(dim=1)
            z_img = self._l2norm(self.vision_proj(pooled))
        
        B,M,_ = tok_ids.shape
        tok_ids_flat = tok_ids.view(B*M, -1)
        tok_lens_flat = tok_lens.view(B*M)
        t_flat = self.text_encoder.forward_ids(tok_ids_flat, tok_lens_flat)  # (B*M,Dtxt)
        t_slots = self._l2norm(self.text_proj(t_flat)).view(B, M, -1) 

        valid = (tok_lens > 1).float().unsqueeze(-1)       # (B,M,1)
        t_scene = (t_slots * valid).sum(dim=1) / (valid.sum(dim=1).clamp(min=1.0))  # (B,D)
        t_scene = self._l2norm(t_scene) # (B,D)

        sim = (z_img * t_scene).sum(dim=-1)
        loss_b = (1.0 - sim).pow(2)                         # (B,)
        return loss_b
