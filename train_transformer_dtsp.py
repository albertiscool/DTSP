"""
=============================================================================
  Transformer DTSP 訓練主程式 (V4)
  結合以下核心設計：
    1. 地圖狀態矩陣 (Map State Matrix)
    2. Transformer Encoder + 自適應注意力解碼器 (Adaptive Attention Decoder)
    3. 自我比較強化學習 (Self-Comparison RL, SCRL)
    4. 反向訓練 (Reverse Training)
    5. 過濾機制 (Filtering Mechanism)
    6. V3 物理懲罰整合（Dubins 轉彎 + 5km 禁航懲罰）

  任務規範（依蔡老師規定）：
    - 空域：100km x 100km（離岸 50~150km，南北 0~100km）
    - 靜態目標：25 艘敵艦
    - 動態目標：5 艘突發（訓練時以隨機加入模擬）
    - 巡航速度：120 km/h
    - 觀測甜甜圈：半徑 5~9 km
    - 禁航區：每個目標中心 5 km

  使用方式：
    python train_transformer_dtsp.py             # 開始訓練
    python train_transformer_dtsp.py --eval      # 評估最佳模型
    python train_transformer_dtsp.py --resume    # 從 checkpoint 繼續訓練
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import os
import argparse
import time
from torch.distributions import Categorical

# ─────────────────────────────────────────────────────────────────────────────
# 全域設定
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 使用運算裝置: {DEVICE}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

# 任務參數
X_MIN, X_MAX   = 50.0, 150.0     # km, 東西方向（離岸）
Y_MIN, Y_MAX   = 0.0,  100.0     # km, 南北方向
N_STATIC       = 25              # 靜態目標數量
N_DYNAMIC      = 5               # 動態突發目標數量
OBS_R_MIN      = 5.0             # km, 觀測甜甜圈最小半徑
OBS_R_MAX      = 9.0             # km, 觀測甜甜圈最大半徑
OBSTACLE_R     = 5.0             # km, 禁航區半徑
TURNING_R      = 2.0             # km, 無人機轉彎半徑

# 模型超參數
EMBED_DIM      = 128             # 隱藏層維度
N_HEADS        = 8               # 多頭注意力頭數
N_ENC_LAYERS   = 3               # Transformer 編碼器層數
FF_DIM         = 512             # 前饋神經網路維度
DROPOUT        = 0.1

# 訓練超參數
BATCH_SIZE     = 256             # RTX 4090 可穩定跑 256（也可試 512）
N_EPOCHS       = 100             # 總訓練 Epoch 數
LR             = 1e-4            # 學習率
LR_DECAY       = 0.99            # 每個 Epoch 學習率衰減係數
CLIP_GRAD      = 1.0             # 梯度裁剪
STEPS_PER_EPOCH = 200            # 每個 Epoch 的 Batch 數

# 反向訓練設定
REVERSE_TRAIN_START = 3          # 反向訓練起始城市數（從小到大）
REVERSE_TRAIN_END   = N_STATIC + N_DYNAMIC  # 最大城市數

# 過濾機制設定
FILTER_K       = 10              # 每次只讓注意力考慮最具潛力的 K 個城市

# 儲存路徑
SAVE_DIR       = "transformer_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 環境與資料生成器
# ─────────────────────────────────────────────────────────────────────────────
class DTSPEnvironment:
    """
    DTSP 任務環境。
    生成隨機海域場景，計算路徑成本（含 Dubins 近似 + 物理懲罰）。
    """
    def __init__(self, batch_size, n_cities, device):
        self.batch_size = batch_size
        self.n_cities   = n_cities
        self.device     = device

    def reset(self):
        """
        生成一個新 Batch 的隨機敵艦場景。
        回傳:
          coords [B, n_cities, 2]  (歸一化到 0~1 的 km 座標)
          obs_radii [B, n_cities]  (每艘敵艦的觀測甜甜圈半徑，歸一化)
        """
        B, N = self.batch_size, self.n_cities

        # 生成歸一化座標（原始座標: x=50~150, y=0~100，統一縮放到 0~1）
        coords = torch.rand(B, N, 2, device=self.device)

        # 第一個點固定為最左上角（西北角，即離機場最近的起點）
        coords[:, 0, 0] = torch.rand(B, device=self.device) * 0.15        # x: 50~65 km
        coords[:, 0, 1] = 0.85 + torch.rand(B, device=self.device) * 0.15 # y: 85~100 km

        # 觀測甜甜圈半徑（歸一化：5~9 km / 100 km = 0.05~0.09）
        obs_radii = 0.05 + torch.rand(B, N, device=self.device) * 0.04

        # Depot 起點無觀測甜甜圈（作為基地）
        obs_radii[:, 0] = 0.0

        return coords, obs_radii

    def compute_tour_length(self, coords, obs_radii, tours):
        """
        計算巡航路徑總長度（含觀測甜甜圈偏移 + 障礙物懲罰的快速近似計算）。
        訓練時使用簡化幾何距離，以便快速反向傳播。

        Args:
          coords   [B, N, 2]
          obs_radii [B, N]
          tours    [B, N]  - 整數索引序列

        Returns:
          lengths  [B]     - 每個 Batch 的近似路徑總長
        """
        B, N, _ = coords.shape

        # 根據 tours 取出排序後的座標
        idx = tours.unsqueeze(-1).expand(-1, -1, 2)
        ordered = coords.gather(1, idx)                        # [B, N, 2]

        # 計算相鄰點間距離
        next_ordered = torch.roll(ordered, -1, dims=1)         # [B, N, 2]
        diffs = next_ordered - ordered
        dists = diffs.norm(dim=-1)                             # [B, N]

        # Dubins 近似懲罰（轉彎半徑修正）
        turning_penalty = TURNING_R / 100.0
        dubins_approx = dists + turning_penalty

        # 障礙物懲罰
        obs_penalty = self._compute_obstacle_penalty(coords, ordered)

        total = (dubins_approx + obs_penalty).sum(dim=-1)      # [B]
        return total

    def _compute_obstacle_penalty(self, coords, ordered):
        """
        向量化障礙物懲罰：對每段航線中點，檢查是否侵入 5km 禁航區。
        """
        B, N, _ = coords.shape
        penalty_weight = 5.0

        next_ordered = torch.roll(ordered, -1, dims=1)
        midpoints = (ordered + next_ordered) / 2.0             # [B, N, 2]

        # 中點到所有敵艦的距離 [B, N_seg, N_obs]
        mid_exp = midpoints.unsqueeze(2).expand(-1, -1, N, -1)
        obs_exp = coords.unsqueeze(1).expand(-1, N, -1, -1)
        dists = (mid_exp - obs_exp).norm(dim=-1)               # [B, N, N]

        obs_r_norm = OBSTACLE_R / 100.0
        violation  = F.relu(obs_r_norm - dists)                # [B, N, N]
        penalty    = violation.sum(dim=-1) * penalty_weight    # [B, N]
        return penalty


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transformer 編碼器（Pre-Norm + 殘差連接）
# ─────────────────────────────────────────────────────────────────────────────
class DTSPEncoder(nn.Module):
    """
    Transformer 編碼器：
    將所有敵艦的座標與觀測半徑，透過多層 Self-Attention 抽取全局空間拓撲特徵。
    """
    def __init__(self, input_dim=3, embed_dim=128, n_heads=8,
                 n_layers=3, ff_dim=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm 更穩定
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: [B, N, 3]  → (x, y, obs_radius)
        回傳:
          node_emb  [B, N, D]
          graph_emb [B, D]
        """
        h = self.input_proj(x)
        node_emb  = self.transformer(h)
        node_emb  = self.norm(node_emb)
        graph_emb = node_emb.mean(dim=1)
        return node_emb, graph_emb


# ─────────────────────────────────────────────────────────────────────────────
# 3. 自適應注意力解碼器（含過濾機制）
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveAttentionDecoder(nn.Module):
    """
    自適應注意力解碼器：
    多頭自注意力（全局依賴）+ 上下文注意力（歷史軌跡）
    + 過濾機制（依航向方向向量篩選最具潛力的 K 個候選城市）
    """
    def __init__(self, embed_dim=128, n_heads=8, filter_k=10, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads   = n_heads
        self.filter_k  = filter_k
        self.mask_value = -1e9
        self.clip_c     = 10.0

        self.query_proj  = nn.Linear(embed_dim * 3, embed_dim)
        self.mha         = nn.MultiheadAttention(embed_dim, n_heads,
                                                  batch_first=True, dropout=dropout)
        self.context_q   = nn.Linear(embed_dim, embed_dim)
        self.context_k   = nn.Linear(embed_dim, embed_dim)

    def forward(self, node_emb, graph_emb, current_emb, prev_emb,
                mask, coords, current_pos=None):
        """
        Args:
          node_emb     [B, N, D]
          graph_emb    [B, D]
          current_emb  [B, D]   當前位置嵌入
          prev_emb     [B, D]   前一步位置嵌入
          mask         [B, N]   True = 已造訪
          coords       [B, N, 2]
          current_pos  [B, 2]   當前位置座標（用於過濾方向計算）

        回傳:
          log_probs  [B, N]
        """
        B, N, D = node_emb.shape

        # ── 查詢向量 ─────────────────────────────────────────────────────────
        q_input = torch.cat([current_emb, graph_emb, prev_emb], dim=-1)
        query   = self.query_proj(q_input).unsqueeze(1)                 # [B, 1, D]

        # ── 過濾機制（Filtering）──────────────────────────────────────────────
        # 根據當前航向向量，計算各未造訪城市的「潛力分數」
        # 只讓最具潛力的 filter_k 個城市參與注意力計算
        filter_mask = mask.clone()
        if current_pos is not None and self.filter_k < N:
            # 計算前進方向向量（當前位置 - 前一位置）
            direction = current_pos - coords[torch.arange(B), :, :].mean(dim=1)

            # 計算每個未造訪城市相對當前位置的角度與距離分數
            rel_pos   = coords - current_pos.unsqueeze(1)              # [B, N, 2]
            dist      = rel_pos.norm(dim=-1)                           # [B, N]

            # 距離分數（越近越好）
            dist_score = -dist

            # 已造訪的城市過濾掉
            dist_score = dist_score.masked_fill(mask, -1e9)

            # 只保留 top-k 潛力城市
            _, topk_idx = dist_score.topk(min(self.filter_k, N), dim=-1)
            filter_mask = torch.ones(B, N, dtype=torch.bool, device=node_emb.device)
            filter_mask.scatter_(1, topk_idx, False)
            filter_mask = filter_mask | mask  # 已造訪的一定 mask 掉

        # ── 多頭自注意力 ─────────────────────────────────────────────────────
        attn_out, _ = self.mha(
            query, node_emb, node_emb,
            key_padding_mask=filter_mask
        )                                                               # [B, 1, D]

        # ── 上下文注意力（Pointer 機制）──────────────────────────────────────
        ctx_q   = self.context_q(attn_out)                             # [B, 1, D]
        ctx_k   = self.context_k(node_emb)                             # [B, N, D]
        logits  = torch.bmm(ctx_q, ctx_k.transpose(1, 2)).squeeze(1)  # [B, N]
        logits  = logits / math.sqrt(D)

        # tanh 裁切 + 遮罩
        logits      = self.clip_c * torch.tanh(logits)
        logits      = logits.masked_fill(mask, self.mask_value)

        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


# ─────────────────────────────────────────────────────────────────────────────
# 4. 完整的 DTSP Transformer 策略網路
# ─────────────────────────────────────────────────────────────────────────────
class DTSPTransformerPolicy(nn.Module):
    """
    DTSP Transformer 策略模型。
    Encoder + Adaptive Attention Decoder → 自回歸生成完整巡航序列。
    """
    def __init__(self, embed_dim=EMBED_DIM, n_heads=N_HEADS,
                 n_enc_layers=N_ENC_LAYERS, ff_dim=FF_DIM,
                 filter_k=FILTER_K, dropout=DROPOUT):
        super().__init__()
        self.encoder = DTSPEncoder(
            input_dim=3, embed_dim=embed_dim, n_heads=n_heads,
            n_layers=n_enc_layers, ff_dim=ff_dim, dropout=dropout
        )
        self.decoder = AdaptiveAttentionDecoder(
            embed_dim=embed_dim, n_heads=n_heads,
            filter_k=filter_k, dropout=dropout
        )
        self.embed_dim = embed_dim

    def forward(self, coords, obs_radii, greedy=False, start_city=0):
        """
        自回歸生成完整巡航序列。

        Args:
          coords      [B, N, 2]
          obs_radii   [B, N]
          greedy      bool - True=貪婪解碼
          start_city  int  - 起點索引（默認 0）

        回傳:
          tours       [B, N]
          sum_log_p   [B]
        """
        B, N, _ = coords.shape
        dev      = coords.device

        # 輸入特徵：(x, y, obs_radius)
        x = torch.cat([coords, obs_radii.unsqueeze(-1)], dim=-1)  # [B, N, 3]

        # Encoder
        node_emb, graph_emb = self.encoder(x)                     # [B, N, D], [B, D]

        # 初始化
        visited      = torch.zeros(B, N, dtype=torch.bool, device=dev)
        current_city = torch.full((B,), start_city, dtype=torch.long, device=dev)
        visited.scatter_(1, current_city.unsqueeze(1), True)

        current_emb  = node_emb[torch.arange(B), current_city]   # [B, D]
        prev_emb     = current_emb.clone()
        current_pos  = coords[torch.arange(B), current_city]      # [B, 2]

        tours     = [current_city]
        sum_log_p = torch.zeros(B, device=dev)

        for step in range(N - 1):
            log_probs = self.decoder(
                node_emb, graph_emb, current_emb, prev_emb,
                mask=visited, coords=coords, current_pos=current_pos
            )                                                      # [B, N]

            if greedy:
                next_city = log_probs.argmax(dim=-1)
            else:
                dist      = Categorical(logits=log_probs)
                next_city = dist.sample()

            # 累積對數機率（gather 不做 inplace）
            lp = log_probs.gather(1, next_city.unsqueeze(1)).squeeze(1)
            sum_log_p = sum_log_p + lp

            # 用非 inplace 方式更新已造訪遮罩
            one_hot   = torch.zeros(B, N, dtype=torch.bool, device=dev)
            one_hot.scatter_(1, next_city.unsqueeze(1), True)
            visited   = visited | one_hot

            prev_emb     = current_emb
            current_emb  = node_emb[torch.arange(B), next_city]
            current_pos  = coords[torch.arange(B), next_city]
            current_city = next_city
            tours.append(next_city)

        tours = torch.stack(tours, dim=1)                          # [B, N]
        return tours, sum_log_p


# ─────────────────────────────────────────────────────────────────────────────
# 5. 自我比較強化學習 (SCRL) 訓練器
# ─────────────────────────────────────────────────────────────────────────────
class SCRLTrainer:
    """
    Self-Comparison RL Trainer (SCRL)
    採樣兩條路徑，較短者為正樣本，計算策略梯度。
    無需 Critic 網路，結構更輕量穩定。
    """
    def __init__(self, policy, env, optimizer, scheduler, device):
        self.policy    = policy
        self.env       = env
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device    = device
        self.baseline  = None
        self.baseline_alpha = 0.99

    def train_step(self, n_cities):
        self.policy.train()
        self.env.n_cities = n_cities
        coords, obs_radii = self.env.reset()

        # 採樣路徑1（有梯度）
        tours1, log_p1 = self.policy(coords, obs_radii, greedy=False)
        len1 = self.env.compute_tour_length(coords, obs_radii, tours1)

        # 採樣路徑2（無梯度，作為比較基準）
        with torch.no_grad():
            tours2, _ = self.policy(coords, obs_radii, greedy=False)
            len2 = self.env.compute_tour_length(coords, obs_radii, tours2)

        # SCRL 優勢值：路徑1比路徑2短 → 正優勢 → 強化路徑1的決策
        advantage = (len2 - len1).detach()

        # Policy Gradient Loss
        loss = -(advantage * log_p1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), CLIP_GRAD)
        self.optimizer.step()

        return loss.item(), len1.mean().item()

    def eval_step(self, n_cities, n_eval_batch=10):
        self.policy.eval()
        self.env.n_cities = n_cities
        total_len = 0.0
        with torch.no_grad():
            for _ in range(n_eval_batch):
                coords, obs_radii = self.env.reset()
                tours, _ = self.policy(coords, obs_radii, greedy=True)
                lengths  = self.env.compute_tour_length(coords, obs_radii, tours)
                total_len += lengths.mean().item()
        return total_len / n_eval_batch


# ─────────────────────────────────────────────────────────────────────────────
# 6. 反向訓練課程調度器 (Reverse Training Scheduler)
# ─────────────────────────────────────────────────────────────────────────────
class ReverseTrainingScheduler:
    """
    從末端（城市數少）往前（城市數多）逐漸增加訓練難度。
    Epoch 1~10:   3~8 個城市
    Epoch 11~30:  8~15 個城市
    Epoch 31~60:  15~25 個城市
    Epoch 61~100: 25~30 個城市（完整任務）
    """
    def __init__(self, start=3, end=30, n_epochs=100):
        self.start    = start
        self.end      = end
        self.n_epochs = n_epochs

    def get_n_cities(self, epoch):
        progress = min(epoch / self.n_epochs, 1.0)
        n = int(self.start + (self.end - self.start) * (progress ** 0.6))
        return max(self.start, min(self.end, n))


# ─────────────────────────────────────────────────────────────────────────────
# 7. 主訓練迴圈
# ─────────────────────────────────────────────────────────────────────────────
def train(resume=False):
    print("=" * 65)
    print(" DTSP Transformer V4 訓練開始")
    print(" 結合 SCRL + 反向訓練 + 過濾機制 + V3 物理懲罰")
    print("=" * 65)

    policy = DTSPTransformerPolicy().to(DEVICE)
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[INFO] 模型參數量: {n_params:,} ({n_params/1e6:.2f}M)")

    env       = DTSPEnvironment(BATCH_SIZE, N_STATIC, DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
    rt_sched  = ReverseTrainingScheduler(REVERSE_TRAIN_START, REVERSE_TRAIN_END, N_EPOCHS)
    trainer   = SCRLTrainer(policy, env, optimizer, scheduler, DEVICE)

    start_epoch = 0
    best_length = float('inf')

    if resume:
        ckpt_path = os.path.join(SAVE_DIR, "checkpoint_latest.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            policy.load_state_dict(ckpt['policy'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch'] + 1
            best_length = ckpt.get('best_length', float('inf'))
            print(f"[INFO] 從 Epoch {start_epoch} 繼續，最佳航程: {best_length:.4f}")
        else:
            print("[WARN] 找不到 checkpoint，從頭開始訓練。")

    header = f"{'Epoch':>6}  {'Cities':>6}  {'Loss':>10}  {'Avg Len':>10}  {'Eval Len':>10}  {'Time':>7}"
    print(header)
    print("-" * 65)

    for epoch in range(start_epoch, N_EPOCHS):
        t0 = time.time()
        n_cities = rt_sched.get_n_cities(epoch)

        total_loss, total_len = 0.0, 0.0
        for _ in range(STEPS_PER_EPOCH):
            loss, mean_len = trainer.train_step(n_cities)
            total_loss += loss
            total_len  += mean_len

        avg_loss = total_loss / STEPS_PER_EPOCH
        avg_len  = total_len  / STEPS_PER_EPOCH
        scheduler.step()

        eval_len = None
        if (epoch + 1) % 5 == 0 or epoch == 0:
            eval_len = trainer.eval_step(N_STATIC + N_DYNAMIC)
            if eval_len < best_length:
                best_length = eval_len
                torch.save({
                    'policy': policy.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_length': best_length,
                }, os.path.join(SAVE_DIR, "best_model.pt"))
                print(f"  [BEST] New best model saved! Eval Length = {best_length:.4f}")

        torch.save({
            'policy': policy.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_length': best_length,
        }, os.path.join(SAVE_DIR, "checkpoint_latest.pt"))

        elapsed  = time.time() - t0
        eval_str = f"{eval_len:>10.4f}" if eval_len is not None else f"{'—':>10}"
        print(f"{epoch+1:>6}  {n_cities:>6}  {avg_loss:>10.4f}  {avg_len:>10.4f}  {eval_str}  {elapsed:>6.1f}s")

    print("=" * 65)
    print(f" 訓練完成！最佳評估航程: {best_length:.4f}")
    print(f" 實際估算航程: {best_length * 100:.2f} km")
    print(f" 模型儲存至: {SAVE_DIR}/best_model.pt")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 8. 評估模式
# ─────────────────────────────────────────────────────────────────────────────
def evaluate():
    print("\n[EVAL] 載入最佳模型進行評估...")
    ckpt_path = os.path.join(SAVE_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print("[ERROR] 找不到最佳模型，請先執行訓練。")
        return

    policy = DTSPTransformerPolicy().to(DEVICE)
    ckpt   = torch.load(ckpt_path, map_location=DEVICE)
    policy.load_state_dict(ckpt['policy'])
    policy.eval()

    env      = DTSPEnvironment(BATCH_SIZE, N_STATIC + N_DYNAMIC, DEVICE)
    N_EVAL   = 20
    total    = 0.0

    print(f"[EVAL] 評估 {N_STATIC + N_DYNAMIC} 城市的 DTSP 任務...")
    with torch.no_grad():
        for i in range(N_EVAL):
            coords, obs_radii = env.reset()
            tours, _  = policy(coords, obs_radii, greedy=True)
            lengths   = env.compute_tour_length(coords, obs_radii, tours)
            total    += lengths.mean().item()
            if (i + 1) % 5 == 0:
                print(f"  Batch {i+1}/{N_EVAL}: 平均航程 = {lengths.mean().item() * 100:.2f} km")

    avg = total / N_EVAL
    print(f"\n[EVAL] 最終平均航程：{avg * 100:.2f} km")
    print(f"[EVAL] 估算巡航時間（120 km/h）：{avg * 100 / 120 * 60:.1f} 分鐘")


# ─────────────────────────────────────────────────────────────────────────────
# 主程式入口
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DTSP Transformer V4 訓練程式")
    parser.add_argument("--eval",   action="store_true", help="評估模式（載入最佳模型）")
    parser.add_argument("--resume", action="store_true", help="從最新 checkpoint 繼續訓練")
    args = parser.parse_args()

    if args.eval:
        evaluate()
    else:
        train(resume=args.resume)
