import os
import json
import math
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

from model import TextEncoder, SlotTextModel

import spacy
from tqdm import *

#def collate_slot_caption(batch):
#    # batch: List[Dict]
#    out = {}
#    out["video_idx"] = torch.tensor([b["video_idx"] for b in batch], dtype=torch.long)
#    out["frame_idx"] = torch.tensor([b["frame_idx"] for b in batch], dtype=torch.long)
#    out["slot_feat"] = torch.stack([b["slot_feat"] for b in batch], dim=0)  # (B,M,D)

#    out["captions"] = [b["captions"] for b in batch]  # List[List[str]]

#    if "neg_captions" in batch[0]:
#        out["neg_captions"] = [b["neg_captions"] for b in batch]  # List[List[List[str]]]
#    return out
# def bytes_to_gb(x: int) -> float:
#     return x / (1024**3)
    
class SlotCaptionDataset(Dataset):
    def __init__(self, args, index_json: str, root_dir: str, split: str, train_percent: float = 0.9):
        self.root_dir = Path(root_dir)
        self.index_path = Path(index_json)
        self.split = split
        self.train_percent = train_percent
        groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        # groups = {
        #     (0, 0): [rec1, rec2, rec3, ...],
        #     (0, 1): [rec9, rec10, ...],
        #     (0, 2): [rec17, ...],
        # }
        with open("/scratch/wz3008/new_SlotAttn/slot_attn_new/vocab.json") as f:
            self.vocab = json.load(f)
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["ner", "parser", "lemmatizer", "tagger", "attribute_ruler"]
        )
        self.sos = self.vocab["<sos>"]
        self.eos = self.vocab["<eos>"]
        self.pad = self.vocab["<pad>"]
        self.unk = self.vocab["<unk>"]
        self.max_seq_len = 50
        
        # small cache to avoid repeated spaCy on same captions (very helpful)
        self._tok_cache: Dict[str, Tuple[List[int], int]] = {}
        
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (int(rec["video_idx"]), int(rec["frame_idx"]))
                groups.setdefault(key,[]).append(rec)
                # self.records.append(json.loads(line))

        self.frame_keys = sorted(groups.keys())

        n_train = int(len(self.frame_keys)*self.train_percent)
        n_val = int(len(self.frame_keys)*0.9)
        if args.dev:
            n_train = int(len(self.frame_keys)*0.005)
            n_val = int(len(self.frame_keys)*0.995)

        # n_train = int(len(self.records) * 0.8)
        if split == "train":
            self.frame_keys = self.frame_keys[:n_train]
            # self.records = self.records[:n_train]
            # print(f"Loading training data from {self.index_path}")
        elif split == "val":
            self.frame_keys = self.frame_keys[n_val:]
            # self.records = self.records[n_train:]
            # print(f"Loading validation data from {self.index_path}")

        self.groups = groups

        self.Mmax = 10

    def preload_all_feats(self):
        all_paths = []
        for k in self.frame_keys:
            for r in self.groups[k]:
                all_paths.append(str(self.root_dir / r["slot_feat_path"]))

        self._feat_cache = {}
        for p in tqdm(all_paths, desc="preload feats"):
            self._feat_cache[p] = torch.load(p, map_location="cpu", weights_only=True).float().view(-1)
        print(f"load {len(self._feat_cache)} slot feats to cache")
        

    def __len__(self) -> int:
        return len(self.frame_keys)
    
    def _tokenize_one(self, cap:str) -> Tuple[List[int], int]:
        if cap in self._tok_cache:
            return self._tok_cache[cap]
        
        doc = self.nlp(cap)
        words = [t.text for t in doc]
        words = words[: self.max_seq_len - 2]
        
        ids = [self.sos] + [self.vocab.get(w, self.unk) for w in words] + [self.eos]
        length = len(ids)
        
        if len(ids) < self.max_seq_len:
            ids = ids + [self.pad] * (self.max_seq_len - len(ids))
        else:
            ids = ids[: self.max_seq_len]
            length = min(length, self.max_seq_len)
        
        self._tok_cache[cap] = (ids, length)
        return ids, length
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.frame_keys[idx]
        recs = self.groups[key]
        # recs = sorted(recs,key=lambda r:int(r.get("instance_idx", 0)))
        
        feats: List[torch.Tensor] = []
        captions: List[str] = []
        
        tok_ids_list: List[List[int]] = []
        tok_len_list: List[int] = []
        
        for r in recs:
            # p = self.root_dir / r["slot_feat_path"]
            p = str(self.root_dir / r["slot_feat_path"])
            if hasattr(self, "_feat_cache"):
                feat = self._feat_cache[p]
            else:
                feat = torch.load(p, map_location="cpu", weights_only=True).float().view(-1)
            
            # feat = torch.load(p, map_location="cpu", weights_only=True).float().view(-1)  # (D,)
            feats.append(feat)
            
            ids, ln = self._tokenize_one(r["caption"])
            tok_ids_list.append(ids)
            tok_len_list.append(ln)
            
            captions.append(r["caption"])
        
        whole_caption = "There is "
        for i, x in enumerate(captions):
            if i == 0:
                whole_caption += f" {x}"
            else:
                whole_caption += f" and {x}"
        whole_caption += "."
        caption_id, caption_len = self._tokenize_one(whole_caption)
        caption_id = torch.tensor([caption_id], dtype=torch.long)  # (1, L)
        caption_len = torch.tensor([caption_len], dtype=torch.long) # (1,)
        
        slot_dim = feats[0].shape[-1]
        
        while len(feats) < self.Mmax:
            feats.append(torch.zeros(slot_dim))
            tok_ids_list.append([self.pad] * self.max_seq_len)
            tok_len_list.append(1)
            # captions.append("<pad>")

        slot_feat = torch.stack(feats, dim=0)
        tok_ids = torch.tensor(tok_ids_list, dtype=torch.long)  # (M, L)
        tok_lens = torch.tensor(tok_len_list, dtype=torch.long) # (M,)
            
        out = {
            "video_idx": key[0],
            "frame_idx": key[1],
            "slot_feat": slot_feat,
            "tok_ids": tok_ids,
            "tok_lens": tok_lens,
            "caption_id": caption_id,
            "caption_len": caption_len
        }
        
        if self.split == "val":
            import random
            rnd = random.Random(42+idx)
            
            # neg_captions: List[List[str]] = []
            neg_tok_ids = []
            neg_tok_lens = []
            # neg_captions = []
            
            for _ in range(3):
                j = rnd.randrange(len(self.frame_keys))
                if len(self.frame_keys) > 1 and j == idx:
                    j = (j + 1) % len(self.frame_keys)
                other_key = self.frame_keys[j]
                
                recs2 = self.groups[other_key]
                
                # ids2, lens2 = [], []
                captions2 = []
                for r in recs2:
                    #ids, ln = self._tokenize_one(r["caption"])
                    #ids2.append(ids)
                    #lens2.append(ln)
                    captions2.append(r["caption"])
                    
                #r2 = recs2[rnd.randrange(len(recs2))]
                
                #while len(ids2) < self.Mmax:
                #    ids2.append([self.pad] * self.max_seq_len)
                #    lens2.append(1)
                
                #neg_tok_ids.append(torch.tensor(ids2, dtype=torch.long))   # (M,L)
                #neg_tok_lens.append(torch.tensor(lens2, dtype=torch.long)) # (M,)
                
                whole_caption = "There is "
                for i,x in enumerate(captions2):
                    if i == 0:
                        whole_caption += f" {x}"
                    else:
                        whole_caption += f" and {x}"
                whole_caption += "."
                caption_id, caption_len = self._tokenize_one(whole_caption)
        
                neg_tok_ids.append(torch.tensor([caption_id], dtype=torch.long))
                neg_tok_lens.append(torch.tensor([caption_len], dtype=torch.long))
            
            out["neg_tok_ids"] = neg_tok_ids       # List[Tensor(M,L)] len=3
            out["neg_tok_lens"] = neg_tok_lens     # List[Tensor(M,)]  len=3
        return out
            
            

@torch.no_grad()
def evaluate(model:nn.Module, loader:DataLoader, device: torch.device, args) -> Dict[str, float]:
    model.eval()
    total_loss, total_n = 0.0, 0
    total_attr_loss = 0.0
    total_scene_loss = 0.0

    correct, correct_attr, correct_scene = 0,0,0
    
    for batch in loader:
        slot_feats = batch["slot_feat"].to(device)  #1,M,D
        # captions = batch["captions"]
        # captions = [list(x) for x in zip(*captions)]#List[List[str]] (1,M,)
        # captions = captions[0]
        # neg_captions = batch["neg_captions"] #List[List[str]] (3,M,)
        # for i in range(3):
        #     neg_captions[i] = [list(x) for x in zip(*neg_captions[i])][0]
        
        #tok_ids = batch["tok_ids"].to(device)
        #tok_lens = batch["tok_lens"].to(device)
        tok_ids = batch["caption_id"].to(device)
        tok_lens = batch["caption_len"].to(device)
        
        
        neg_tok_ids = [t.to(device) for t in batch["neg_tok_ids"]]
        neg_tok_lens = [t.to(device) for t in batch["neg_tok_lens"]]
        
        #loss, z, t = model(slot_feats, captions)
        #import pdb; pdb.set_trace()
        attribute_loss, scene_loss, loss = model.evaluate(
            slot_feats.repeat(4,1,1), 
            torch.stack([tok_ids] + neg_tok_ids).squeeze(1),
            torch.stack([tok_lens] + neg_tok_lens).squeeze(1)
        )
        
        total_loss += loss.mean().item()
        total_attr_loss += attribute_loss.mean().item()
        total_scene_loss += scene_loss.mean().item()
        
        pred = loss.argmin().item()
        pred_attr = attribute_loss.argmin().item()
        pred_scene = scene_loss.argmin().item()
        
        correct += int(pred == 0)
        correct_attr += int(pred_attr == 0)
        correct_scene += int(pred_scene == 0)
        
        total_n += 1
        
        
        #B = slot_feats.shape[0]
        #total_loss += loss.item() * B
        #total_n += B

        #logits = z @ t.t()
        #pred = logits.argmax(dim=1)
        #target = torch.arange(B, device=device)
        #correct += (pred == target).sum().item()
        
        if args.dev:
            break
    
    return {
        "loss": total_loss / max(total_n, 1),
        "loss_attr": total_attr_loss / max(total_n, 1),
        "loss_scene": total_scene_loss / max(total_n, 1),
        "acc_i2t_top1": correct / max(total_n, 1),
        "acc_i2t_top1_attr": correct_attr / max(total_n, 1),
        "acc_i2t_top1_scene": correct_scene / max(total_n, 1),
    }

def train(args):
    print("slot experiment")
    print(f"=============== Start {args.train_percent} percent train ===========")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device: ", device)
    
    root_dir = args.data_root        

    with open(args.vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)


    train_set = SlotCaptionDataset(
        args,
        index_json=os.path.join(root_dir, "index_movi_a_train.json"),
        root_dir=root_dir, 
        split = "train", 
        train_percent = args.train_percent
    )
    val_set = SlotCaptionDataset(
        args,
        index_json=os.path.join(root_dir, "index_movi_a_train.json"),
        root_dir=root_dir, 
        split = "val"
    )

    # MODIFICATION
    print(f"len train set = {len(train_set)}")
    print(f"len val set = {len(val_set)}")
    
    train_set.preload_all_feats()
    val_set.preload_all_feats()
    
    d_slot = train_set[0]["slot_feat"].shape[-1]

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle =True,
        num_workers = args.num_workers,
        pin_memory =True,
        # TODO: collate_fn = make_collate(tokenizer),
        drop_last =True,
        #collate_fn=collate_slot_caption,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle =False,
        num_workers = args.num_workers,
        pin_memory =True,
        drop_last =False,
        #collate_fn=collate_slot_caption,
        # persistent_workers=True, #
        # prefetch_factor=4,
    )

    textencoder =  TextEncoder()
    model = SlotTextModel(
        textencoder=textencoder,
        slot_dim=d_slot,
        embed_dim=256,
        tau=0.1,
    ).to(device)

    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    logdir = Path(args.logdir) / (time.strftime("%Y%m%d_%H%M%S") + "_" + str(args.train_percent))
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    global_step = 0
    best_val = math.inf
    
    print("eval with scratch model")
    
    if val_loader is not None:
            metrics = evaluate(model, val_loader, device, args)
            
            writer.add_scalar("val/loss", metrics["loss"], epoch)
            #writer.add_scalar("val/loss_attr", metrics["loss_attr"], epoch)
            #writer.add_scalar("val/loss_scene", metrics["loss_scene"], epoch)
            writer.add_scalar("val/acc_i2t_top1", metrics["acc_i2t_top1"], epoch)
            #writer.add_scalar("val/acc_i2t_top1_attr", metrics["acc_i2t_top1_attr"], epoch)
            #writer.add_scalar("val/acc_i2t_top1_scene", metrics["acc_i2t_top1_scene"], epoch)
            
            print(
                f"loss {metrics['loss']:.4f} "
                f"loss_attr {metrics['loss_attr']:.4f} "
                f"loss_scene {metrics['loss_scene']:.4f} "
                f"acc_i2t_top1 {metrics['acc_i2t_top1']:.4f} "
                f"acc_i2t_top1_attr {metrics['acc_i2t_top1_attr']:.4f} "
                f"acc_i2t_top1_scene {metrics['acc_i2t_top1_scene']:.4f}"
            )
    if args.eval:
        args.epochs = 0

    for epoch in range(args.epochs):
        if not args.eval:
            model.train()
            t0 = time.time()
            
            for it, batch in enumerate(train_loader):
                slot_feat = batch["slot_feat"].to(device)
                # captions = batch["captions"]
                # captions = [list(x) for x in zip(*captions)]
                #tok_ids   = batch["tok_ids"].to(device)        #B,10,25
                #tok_lens  = batch["tok_lens"].to(device)       #B,10
                tok_ids = batch["caption_id"].to(device)    #B,1,25
                tok_lens = batch["caption_len"].to(device)  #B,1
                
                
                attribute_loss, scene_loss, loss = model(slot_feat, tok_ids, tok_lens)
    
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
    
                if global_step % args.log_every == 0:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    print(f"epoch {epoch:02d} iter {it:05d} step {global_step:07d} loss {loss.item():.4f}")
                global_step += 1
                
                if args.dev:
                    break
            
            writer.add_scalar("train/epoch_time_sec", time.time() - t0, epoch)
            print("train/epoch_time_sec", time.time() - t0, epoch)
        
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, args)
            
            writer.add_scalar("val/loss", metrics["loss"], epoch)
            writer.add_scalar("val/loss_attr", metrics["loss_attr"], epoch)
            writer.add_scalar("val/loss_scene", metrics["loss_scene"], epoch)
            writer.add_scalar("val/acc_i2t_top1", metrics["acc_i2t_top1"], epoch)
            writer.add_scalar("val/acc_i2t_top1_attr", metrics["acc_i2t_top1_attr"], epoch)
            writer.add_scalar("val/acc_i2t_top1_scene", metrics["acc_i2t_top1_scene"], epoch)
            
            print(
                f"[val] epoch {epoch:02d} "
                f"loss {metrics['loss']:.4f} "
                f"loss_attr {metrics['loss_attr']:.4f} "
                f"loss_scene {metrics['loss_scene']:.4f} "
                f"acc_i2t_top1 {metrics['acc_i2t_top1']:.4f} "
                f"acc_i2t_top1_attr {metrics['acc_i2t_top1_attr']:.4f} "
                f"acc_i2t_top1_scene {metrics['acc_i2t_top1_scene']:.4f}"
            )


            if metrics["loss"] < best_val:
                best_val = metrics["loss"]
                save_path = logdir / "best_model.pth"
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": vars(args),
                    "d_slot": d_slot,
                }
                torch.save(ckpt, save_path)
                print(f"[ckpt] saved best.pt (val loss {best_val:.4f})")
        
        if (epoch + 1) % args.save_every == 0:
            save_path = logdir / f"model_epoch{epoch:02d}.pth"
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args),
                "d_slot": d_slot,
            }
            torch.save(ckpt, save_path)
            print(f"[ckpt] saved model_epoch{epoch:02d}.pth")
        
        if args.dev:
            break
    writer.close()
    print("Training completed.")



if __name__=="__main__":
    # dataset = SlotCaptionDataset(
    #     index_json="/home/wz3008/slot-attn/output/index_movi_a_train.json",
    #     root_dir="/home/wz3008/slot-attn/output"
    # )
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default ="/home/wz3008/slot-attn/output")
    p.add_argument("--vocab-path", type=str, default="/home/wz3008/slot-attn/vocab.json")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--logdir", type=str, default="./logs/")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--train_percent", type=float, default=0.9)
    
    p.add_argument("--dev", action="store_true")
    p.add_argument("--eval", action="store_true")
    

    args = p.parse_args()
    train(args)

    
