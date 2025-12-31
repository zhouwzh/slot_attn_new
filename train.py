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

class SlotCaptionDataset(Dataset):
    def __init__(self, index_json: str, root_dir: str, split: str):
        self.root_dir = Path(root_dir)
        self.index_path = Path(index_json)
        self.records: List[Dict[str, Any]] = []

        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))
        n_train = int(len(self.records) * 0.8)
        if split == "train":
            self.records = self.records[:n_train]
            # print(f"Loading training data from {self.index_path}")
        elif split == "val":
            self.records = self.records[n_train:]
            # print(f"Loading validation data from {self.index_path}")

    def __len__(self) -> int:
        return len(self.records)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        slot_feat_path = self.root_dir / rec["slot_feat_path"]
        caption = rec["caption"]
    
        return {
            "slot_feat": torch.load(slot_feat_path, map_location="cpu").float().view(-1),
            "caption": caption
        }

@torch.no_grad()
def evaluate(model:nn.Module, loader:DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss, total_n = 0.0, 0

    correct = 0
    for batch in loader:
        slot_feats = batch["slot_feat"].to(device)
        captions = batch["caption"]
        loss, z, t = model(slot_feats, captions)
        B = slot_feats.shape[0]
        total_loss += loss.item() * B
        total_n += B

        logits = z @ t.t()
        pred = logits.argmax(dim=1)
        target = torch.arange(B, device=device)
        correct += (pred == target).sum().item()
    
    return {
        "loss": total_loss / max(total_n, 1),
        "acc_i2t_top1": correct / max(total_n, 1),
    }

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    root_dir = args.data_root        

    with open(args.vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    train_set = SlotCaptionDataset(
        index_json=os.path.join(root_dir, "index_movi_a_train.json"),
        root_dir=root_dir, split = "train")
    val_set = SlotCaptionDataset(
        index_json=os.path.join(root_dir, "index_movi_a_train.json"),
        root_dir=root_dir, split = "val")
    
    d_slot = train_set[0]["slot_feat"].shape[0]

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle =True,
        num_workers = args.num_workers,
        pin_memory =True,
        # TODO: collate_fn = make_collate(tokenizer),
        drop_last =True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle =False,
        num_workers = args.num_workers,
        pin_memory =True,
        drop_last =False
    )

    textencoder =  TextEncoder()
    model = SlotTextModel(
        textencoder=textencoder,
        slot_dim=d_slot,
        embed_dim=256,
        tau=0.1,
    ).to(device)

    
    opt = opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    logdir = Path(args.logdir) / time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    global_step = 0
    best_val = math.inf

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()

        for it, batch in enumerate(train_loader):
            slot_feat = batch["slot_feat"].to(device)
            captions = batch["caption"]
            
            loss, _,_ = model(slot_feat, captions)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            if global_step % args.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                print(f"epoch {epoch:02d} iter {it:05d} step {global_step:07d} loss {loss.item():.4f}")
            global_step += 1
        
        writer.add_scalar("train/epoch_time_sec", time.time() - t0, epoch)
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device)
            writer.add_scalar("val/loss", metrics["loss"], epoch)
            writer.add_scalar("val/acc_i2t_top1", metrics["acc_i2t_top1"], epoch)
            # print(f"[val] epoch {epoch:02d} loss {metrics['loss']:.4f} acc_i2t_top1 {metrics['acc_i2t_top1']:.4f}")

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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--logdir", type=str, default="./logs/")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5)
    

    args = p.parse_args()
    train(args)

    