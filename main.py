import os
import re
import time
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_FILE = "mini_bitnet.pt"

# ==========================================
# 1. Wikipedia記事取得関数（新規学習時のみ使用）
# ==========================================
def fetch_en_wikipedia_articles(num_articles=50):
    print(f"英語Wikipediaから {num_articles} 件の記事を新規取得中...")
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "MiniBitNetExperiment/1.0 (educational)"}
    texts = []

    while len(texts) < num_articles:
        params = {
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": 10,
            "prop": "extracts",
            "explaintext": True,
        }
        try:
            res = requests.get(url, params=params, headers=headers, timeout=10).json()
            pages = res.get("query", {}).get("pages", {})
            for page in pages.values():
                body = page.get("extract", "")
                body = re.sub(r"==+.*?==+", "", body)
                body = re.sub(r"\n+", "\n", body).strip()
                if len(body) > 500:
                    texts.append(body)
                    if len(texts) >= num_articles:
                        break
            time.sleep(0.2)
        except Exception as e:
            print(f"通信リトライ中: {e}")
            time.sleep(1)

    return "\n\n".join(texts)

# ==========================================
# 2. チェックポイント判定（新規 vs 継続学習）
# ==========================================
checkpoint = None
is_resume = False

if os.path.exists(MODEL_FILE):
    try:
        try:
            loaded = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
        except TypeError:
            loaded = torch.load(MODEL_FILE, map_location="cpu")

        if isinstance(loaded, dict) and "raw_text" in loaded and "model_state_dict" in loaded:
            checkpoint = loaded
            is_resume = True
            print(f"【継続モード】{MODEL_FILE} 内のデータと重みを検出しました！")
        else:
            print(f"【新規モード】{MODEL_FILE} に学習データが含まれていないため、新しく学習を開始します。")
    except Exception as e:
        print(f"【新規モード】{MODEL_FILE} の読み込みに失敗したため、新しく学習を開始します ({e})")
else:
    print(f"【新規モード】{MODEL_FILE} が見つからないため、新しく学習を開始します。")

if is_resume:
    raw_text = checkpoint["raw_text"]
    chars = checkpoint["chars"]
    start_step = checkpoint.get("total_steps", 0)
else:
    raw_text = fetch_en_wikipedia_articles(num_articles=50)
    chars = sorted(list(set(raw_text)))
    start_step = 0

print(f"学習テキスト総文字数: {len(raw_text):,} 文字 | 累積ステップ数: {start_step}")

# ==========================================
# 3. トークナイザとデータセット準備
# ==========================================
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(raw_text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 128
batch_size = 16

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x, y

# ==========================================
# 4. 1.58bit LLM (BitNet b1.58) モデル定義
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

class BitLinear(nn.Linear):
    def forward(self, x):
        scale_x = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_q = x + ((x * scale_x).round().clamp(-128, 127) / scale_x - x).detach()

        scale_w = self.weight.abs().mean().clamp(min=1e-5)
        w_ternary = (self.weight / scale_w).round().clamp(-1, 1)
        w_q = self.weight + (w_ternary * scale_w - self.weight).detach()

        return F.linear(x_q, w_q, self.bias)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = BitLinear(d_model, 3 * d_model, bias=False)
        self.proj = BitLinear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        x = self.norm(x)
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.w1 = BitLinear(d_model, 4 * d_model, bias=False)
        self.w2 = BitLinear(4 * d_model, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(self.norm(x))))

class BitBlock(nn.Module):
    def __init__(self, d_model, n_head, block_size):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_head, block_size)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x

class MiniBitNetLM(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_head=6, n_layer=6, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([BitBlock(d_model, n_head, block_size) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=150):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / 0.8
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================
# 5. モデル初期化と重み復元
# ==========================================
model = MiniBitNetLM(vocab_size, d_model=192, n_head=6, n_layer=6, block_size=block_size)

if is_resume:
    state_dict_fp32 = {k: v.float() for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict_fp32)
    print("前回の重みを復元しました！")

# ==========================================
# 6. 学習ループの実行
# ==========================================
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
add_iters = 400  # 1回の実行で追加学習するステップ数

print(f"\n--- 学習開始（ステップ {start_step} → {start_step + add_iters}） ---")
start_time = time.time()
for step in range(add_iters):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 50 == 0 or step == add_iters - 1:
        current_step = start_step + step + 1
        print(f"Step {current_step:4d} | Loss: {loss.item():.4f} | 経過時間: {time.time() - start_time:.1f}秒")

# ==========================================
# 7. データと重みを1ファイルに軽量保存（GitHub対応）
# ==========================================
save_payload = {
    "model_state_dict": {k: v.detach().cpu().half() for k, v in model.state_dict().items()},
    "raw_text": raw_text,
    "chars": chars,
    "total_steps": start_step + add_iters,
}
torch.save(save_payload, MODEL_FILE)
file_size_mb = os.path.getsize(MODEL_FILE) / (1024 * 1024)
print(f"\n保存完了！{MODEL_FILE} ({file_size_mb:.2f} MB) にデータと重みを保存しました。")

# ==========================================
# 8. テスト文章生成と重み分布の確認
# ==========================================
print("\n--- 生成テスト ---")
context = torch.tensor([[stoi.get("T", 0)]], dtype=torch.long)
generated = decode(model.generate(context, max_new_tokens=200)[0].tolist())
print(generated)

sample_w = model.blocks[0].attn.qkv.weight
w_ternary = (sample_w / sample_w.abs().mean()).round().clamp(-1, 1)
print("\n--- 量子化された重みの分布（第1層） ---")
print(f"-1の数: {(w_ternary == -1).sum().item()}, 0の数: {(w_ternary == 0).sum().item()}, +1の数: {(w_ternary == 1).sum().item()}")
