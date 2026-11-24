import re
import string
import time
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

# MyBinderのメモリ膨張・CPU競合を防ぐ設定
torch.set_num_threads(1)
device = "cpu"
torch.manual_seed(42)

# ==========================================
# 1. 英語Wikipedia APIから記事データを取得
# ==========================================
def fetch_en_wikipedia_articles(num_articles=50):
    url = "https://en.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "Mini1BitLLMStudy/1.0 (https://mybinder.org/)"
    }
    # 基本の英語文字（ASCII印字可能文字＋改行）のみ許可して語彙を約95種に固定
    allowed_chars = set(string.printable)
    texts = []
    print(f"英語Wikipediaから {num_articles} 件の記事を取得中...")

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
        res = requests.get(url, params=params, headers=headers, timeout=15).json()
        pages = res.get("query", {}).get("pages", {})

        for page in pages.values():
            body = page.get("extract", "")
            body = re.sub(r"==+.*?==+", "", body)
            body = re.sub(r"\n+", "\n", body).strip()
            # 特殊記号や他言語文字を除外
            body = "".join(ch for ch in body if ch in allowed_chars)
            if len(body) > 400:
                texts.append(body)
                if len(texts) >= num_articles:
                    break
        time.sleep(0.5)

    full_text = "\n\n".join(texts)
    print(f"取得完了: 合計 {len(full_text):,} 文字")
    return full_text

raw_text = fetch_en_wikipedia_articles(num_articles=50)

# 文字レベルトークナイザー（語彙数 約95種類）
chars = sorted(list(set(raw_text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(raw_text), dtype=torch.long)
print(f"語彙数（文字種）: {vocab_size} 種類")

# ==========================================
# 2. 1bitLLM（BitNet b1.58）のモデル定義
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.norm = RMSNorm(in_features)

    def forward(self, x):
        x = self.norm(x)
        # 入力を8ビット整数(-128〜127)にスケーリング量子化 + STE
        x_scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_quant = (x * x_scale).round().clamp(-128, 127) / x_scale
        x_ste = x + (x_quant - x).detach()

        # 重みを {-1, 0, +1} の1.58ビットに量子化 + STE
        w = self.weight
        w_scale = w.abs().mean().clamp(min=1e-5)
        w_quant = (w / w_scale).round().clamp(-1, 1) * w_scale
        w_ste = w + (w_quant - w).detach()

        return F.linear(x_ste, w_ste, self.bias)

class BitCausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = BitLinear(d_model, 3 * d_model)
        self.proj = BitLinear(d_model, d_model)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class BitBlock(nn.Module):
    def __init__(self, d_model, n_head, block_size):
        super().__init__()
        self.attn = BitCausalSelfAttention(d_model, n_head, block_size)
        self.ffn = nn.Sequential(
            BitLinear(d_model, 4 * d_model),
            nn.GELU(),
            BitLinear(4 * d_model, d_model),
        )

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
        self.blocks = nn.Sequential(*[BitBlock(d_model, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================
# 3. 学習ループの実行
# ==========================================
block_size = 128
batch_size = 16
max_iters = 800

# 語彙が少ない分、層を6層・次元を192に増やして賢さを底上げ
model = MiniBitNetLM(vocab_size, d_model=192, n_head=6, n_layer=6, block_size=block_size).to(device)
total_params = sum(p.numel() for p in model.parameters())
bit_params = sum(p.numel() for m in model.modules() if isinstance(m, BitLinear) for p in m.parameters())
print(f"総パラメータ数: {total_params:,} (うち1.58bit層: {bit_params/total_params*100:.1f}%)")

optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3)

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y

print("学習スタート...")
start_time = time.time()
for step in range(max_iters):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 100 == 0 or step == max_iters - 1:
        elapsed = time.time() - start_time
        print(f"Step {step:3d}/{max_iters} | Loss: {loss.item():.4f} | 経過時間: {elapsed:.1f}秒")

# ==========================================
# 4. 英語テキスト生成テストと重み分布の確認
# ==========================================
prompt = "The "
start_ids = torch.tensor([encode(prompt)], dtype=torch.long)

generated = model.generate(start_ids, max_new_tokens=200)
print("\n--- 生成された英文 ---")
print(decode(generated[0].tolist()))

sample_w = model.blocks[0].attn.qkv.weight
w_ternary = (sample_w / sample_w.abs().mean()).round().clamp(-1, 1)
print("\n--- 量子化された重みの分布（第1層） ---")
print(f"-1の数: {(w_ternary == -1).sum().item()}, 0の数: {(w_ternary == 0).sum().item()}, +1の数: {(w_ternary == 1).sum().item()}")
