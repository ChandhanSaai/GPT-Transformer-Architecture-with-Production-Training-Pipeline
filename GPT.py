import torch
import torch.nn as nn
import torch.functional as F
import tiktoken
import time
import math

class GPT(nn.Module):
  def __init__(self, vocab_size, model_dim, context_length, num_transformer, num_heads):
    super().__init__()

    self.embedding = nn.Embedding(vocab_size, model_dim)
    self.position_embedding = nn.Embedding(context_length, model_dim)
    self.transformers = nn.Sequential()
    for transf in range(num_transformer):
      self.transformers.append(Transformer(model_dim, num_heads))
    self.ln_3 = nn.LayerNorm(model_dim)
    self.linear_3 = nn.Linear(model_dim, vocab_size, bias=False)
    self.linear_3.weight = self.embedding.weight
    
    self.apply(self._init_weights)
    

  def forward(self, context):
    embedding = self.embedding(context)
    context_len = context.shape[1]
    position = torch.arange(context_len, device=context.device)
    embedding = embedding + self.position_embedding(position)

    return self.linear_3(self.ln_3(self.transformers(embedding)))

  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
      if isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

class Transformer(nn.Module):
  def __init__(self, model_dim : int, num_heads : int):
    super().__init__()

    self.mhsf = multihead_attention(model_dim, num_heads)
    self.snn = simple_neural_network(model_dim)
    self.ln_1 = nn.LayerNorm(model_dim)
    self.ln_2 = nn.LayerNorm(model_dim)

  def forward(self, embedding):
    embedding = embedding + self.mhsf(self.ln_1(embedding))
    embedding = embedding + self.snn(self.ln_2(embedding))
    return embedding

class simple_neural_network(nn.Module):
  def __init__(self, model_dim : int):
    super().__init__()

    self.liner_1 = nn.Linear(model_dim, model_dim)
    self.gelu = nn.GELU(approximate='tanh')
    self.liner_2 = nn.Linear(model_dim, model_dim)
    self.drop = nn.Dropout(0.2)

  def forward(self, x):
    return self.drop(self.liner_2(self.gelu(self.liner_1(x))))

class multihead_attention(nn.Module):
  def __init__(self, model_dim : int, num_heads : int):
    super().__init__()

    self.heads = nn.ModuleList()
    for head in range(num_heads):
      self.heads.append(singlehead_attention(model_dim, model_dim // num_heads))

    self.compute = nn.Linear(model_dim, model_dim)
    self.dropout = nn.Dropout(0.2)

  def forward(self, embedding):
    head_outputs = []
    for head in self.heads:
      head_outputs.append(head(embedding))
    concat_output = torch.cat(head_outputs, dim=-1)
    return self.dropout(self.compute(concat_output))

class singlehead_attention(nn.Module):
    def __init__(self, model_dim: int, head_size: int):
        super().__init__()
        self.q = nn.Linear(model_dim, head_size, bias=False)
        self.k = nn.Linear(model_dim, head_size, bias=False)
        self.v = nn.Linear(model_dim, head_size, bias=False)

    def forward(self, embedding):
        q = self.q(embedding)
        k = self.k(embedding)
        v = self.v(embedding)

        # scores = q @ torch.transpose(k, 1, 2)
        # B, K, D = scores.shape
        # score = scores / (D ** 0.5)

        # lower_triangular_mask = torch.tril(torch.ones(K, K, device=embedding.device))
        # score = score.masked_fill(lower_triangular_mask == 0, float('-inf'))

        # score = nn.functional.softmax(score, dim=-1)


        score = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal = True)

        return score
    
# ----------------------------------------------------------------------------------------------------

def Generate(model,new_chars, context, context_length):
  tokenizer = tiktoken.get_encoding("gpt2")
  gen = []
  for i in range(new_chars):
    if len(context.T) > context_length:
      context = context[:, -context_length:]
    value = model(context)
    value = value[:, -1, :]
    value = nn.functional.softmax(value, dim=-1)
    value = torch.multinomial(value, 1)
    context = torch.cat((context, value), dim = -1)
    # gen.append(int_to_char[value.item()])
    # gen.append(value.item())
    gen.append(tokenizer.decode([value.item()]))
    # print(gen)
  return ''.join(gen)
  # return gen

# ---------------------------------------------------------------------------------------------

def general(model, context, targets):
    value = model(context)
    
    loss = None
    if targets is not None:
        loss = torch.nn.functional.cross_entropy(value.view(-1, value.size(-1)), targets.view(-1))
    
    
    # print(value.shape)
    # value = value[:,-1,:]
    # value = nn.functional.softmax(value, dim=-1)
    # value = torch.multinomial(value, 1)
    return value, loss

# ------------------------------------------------------------------------------------------------

import re

def clean_text(text: str) -> str:
    # 1) Remove any character that isn’t a letter or whitespace
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # 2) Collapse all runs of whitespace (spaces, tabs, newlines) to a single space
    text = re.sub(r'\s+', ' ', text)
    # 3) Strip leading/trailing spaces
    return text.strip()

#   ------------------------------------------------------------------------------------------
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = 50304
model_dim = 768
context_length = 1024
num_transformer = 12
num_heads = 12


model = GPT(vocab_size, model_dim, context_length, num_transformer, num_heads)
model.to("cuda")
model = torch.compile(model)
# -----------------------------------------------------------------------------------------------------------

class data_loader:
  def __init__(self, b, t):
    self.B = b
    self.T = t
    with open('/content/drive/MyDrive/Colab Notebooks/a_room_with_a_view.txt', 'r', encoding='utf-8') as f:
        self.text = f.read()
    self.tokens = tokenizer.encode(self.text)
    
    self.sub_batch = 0
  def dataloader(self):
    b, t = self.B, self.T
    buf = torch.tensor(self.tokens[self.sub_batch : self.sub_batch + (b*t + 1)]).to("cuda")
    x = buf[:-1].view(b, t)
    y = buf[1:].view(b, t)
    self.sub_batch += (b*t)
    if self.sub_batch + (b*t + 1)> len(self.tokens):
      self.sub_batch = 0
    return x, y

# ---------------------------------------------------------------------------------------------------------
# with open('/content/drive/MyDrive/Colab Notebooks/a_room_with_a_view.txt', 'r', encoding='utf-8') as f:
#     text = f.read()

# # text = clean_text(text)

# text = text[:]

# tokens = tokenizer.encode(text)
# print(len(tokens))

total_batch_size = 524288

B, T = 4, 1024

data = data_loader(B, T)

# buf = torch.tensor(tokens[:B*T + 1]).to("cuda")
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)

torch.set_float32_matmul_precision('high')

# ------------------------------------------------------------------------------------------------
# context = tokenizer.encode("I am god")
# context_tensor = torch.tensor([context], dtype=torch.long).to("cuda")
# # tex, loss = general(model, x, y)
# # print(tokenizer.decode([tex.item()]))
# # print(loss)
# generated_text = Generate(model, 100, context_tensor, context_length)
# print(generated_text)

# ---------------------------------------------------------------------------------------------------

max_lr = 6e-4
min_lr = max_lr * 0.1
warm_up = 1
max_steps = 10

def get_lr(step):
    if step < warm_up:
        return min_lr * (step + 1) / warm_up
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warm_up) / (max_steps - warm_up)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# --------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------

optimizer = torch.optim.AdamW(model.parameters())

epoch = 10
for i in range(epoch):
    total_loss = 0
    x_1 = time.time()
    optimizer.zero_grad()

    for _ in range(total_batch_size // (B*T)):
      x, y = data.dataloader()
      with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
          tex, loss = general(model, x, y)
      loss = loss / (total_batch_size // (B*T))   
      total_loss += loss.detach()      
      loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = get_lr(i)
    optimizer.step()
    torch.cuda.synchronize()
    x_2 = time.time()
    print(f"loss : {total_loss.item()}, epoch = {i}, time = {((x_2 - x_1) * 1000):.2f}ms, token/sec = {((B*T*(total_batch_size // (B*T)))/(x_2 - x_1)):.2f}")
    
    
    
context = tokenizer.encode("I am gpt")
context_tensor = torch.tensor([context], dtype=torch.long).to("cuda")

generated_text = Generate(model, 100, context_tensor, context_length)
print(generated_text)
