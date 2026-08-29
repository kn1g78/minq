# minq

[English README](README.md)

从零开发的 LLaMA 架构族(LLaMA / Qwen2 / Qwen3 风格)Transformer CPU 推理引擎,使用 Rust 实现,不依赖任何现成深度学习框架。所有组件，包括张量、块量化内核、Transformer 前向传播、KV cache 增量解码、采样——均基于 `rayon` 手工实现,热路径使用 AVX2+FMA 内核。

## 架构

```
                 ┌─────────────────────────── minq ───────────────────────────┐
                 │                                                              │
  prompt ──────► │  tokenizer ──► engine ──────────────────────► detokenized    │
  (text)         │  (BPE wrap)    (prefill + decode loop)        text           │
                 │                   │ 1 fwd pass/prompt  │ 1 fwd pass/token    │
                 │                   ▼                    ▼                     │
                 │                 model  ◄──── KV cache (per layer K/V)        │
                 │   embed → [RMSNorm → QKV(+bias) → RoPE → GQA attention       │
                 │            → O proj → +residual → RMSNorm → SwiGLU           │
                 │            → +residual] × N → RMSNorm → lm_head → logits     │
                 │                   │                    │                     │
                 │           tensor (f32 kernels)   quantize (Q8_0 / Q4_0       │
                 │           rayon matmul/matvec    fused block matvec)         │
                 │                   ▲                    ▲                     │
                 │            format: .minq ◄── quantize CLI (from              │
                 │            header+JSON+tensors   safetensors export)         │
                 └──────────────────────────────────────────────────────────────┘
```

## 模块

| 模块 | 职责 |
|------|------|
| `tensor.rs` | 稠密 f32 `Tensor`(Vec + shape + strides),rayon 并行 `matmul`/`matvec`,`rmsnorm`、`softmax`、逐元素算子;AVX2+FMA `dot` 运行时分发 |
| `quantize.rs` | 研究核心:Q8_0 / Q4_0 块量化(32 权重/块),量化/反量化,以及融合反量化-乘加的 `matvec` 热路径(标量 + AVX2 内核) |
| `model.rs` | Transformer 本体:RMSNorm、RoPE、带 KV cache 的 MHA/GQA 注意力、SwiGLU MLP、可选的 Qwen3 逐头 q/k-norm;支持从 safetensors(F32/F16/BF16)与 `.minq` 加载权重 |
| `format.rs` | `.minq` 文件格式:8 字节 magic + config JSON + 带类型的张量记录 |
| `tokenizer.rs` | `tokenizers` crate 封装(HF `tokenizer.json`) |
| `sampler.rs` | greedy / temperature / top-k / top-p 采样,种子可复现 |
| `engine.rs` | 生成循环:prefill + 增量 decode,流式回调(增量 detokenize,无乱码) |
| `ppl.rs` | 困惑度评估:不重叠上下文窗口,log-sum-exp NLL(f64),逐位置流式读取 logits |
| `main.rs` | CLI:`run`、`quantize`、`bench`、`eval-ppl` |

## 构建

任意较新的 stable Rust 工具链(>= 1.85)即可:

```bash
cargo build --release
```


## 使用

```bash
cargo build --release

# 生成文本(可直接读 HuggingFace 的 f32/fp16 权重,或 .minq 文件)
./target/release/minq run \
  --model models/qwen2-0.5b \
  --tokenizer models/qwen2-0.5b/tokenizer.json \
  --prompt "The capital of France is" \
  --max-tokens 64 --temperature 0.7 --top-p 0.9

# 导出量化模型(读取目录下的 config.json + *.safetensors)
./target/release/minq quantize \
  --input models/qwen2-0.5b \
  --output models/qwen2-0.5b/qwen2-0.5b-q8_0.minq --dtype q8_0

# 基准测试 prefill / decode 吞吐
./target/release/minq bench --model models/qwen2-0.5b/qwen2-0.5b-q8_0.minq \
  --prompt-tokens 128 --gen-tokens 64 --threads 8

# 文本困惑度(量化质量指标)
./target/release/minq eval-ppl --model models/qwen2-0.5b/qwen2-0.5b-q8_0.minq \
  --tokenizer models/qwen2-0.5b/tokenizer.json \
  --input wikitext2-test.txt --context 512 --max-tokens 2000
```

`--model` 接受 `.minq` 文件、单个 `.safetensors` 文件,或包含 `config.json` 与一个或多个 `*.safetensors` 分片的目录。改名前导出的旧文件(扩展名 `.minfer`,magic `MINFER01`)二进制布局完全相同,加载器仍然接受;新导出统一使用 magic `MINQ0001`。

模型权重**不属于**本仓库。放在 `models/` 目录下,每个模型一个子目录,例如 `models/qwen3-4b-base/` 内存放从 HuggingFace 或 ModelScope 下载的 `config.json`、`tokenizer.json`、`*.safetensors`,以及你从中导出的 `.minq` 文件。

## 量化格式

两种格式都把每行权重按 32 个值分块,每块共享一个 f32 scale(GGML 风格),因此反量化只是逐元素一次乘法,可以融合进 matvec 累加循环:

- **Q8_0** — 36 字节/块:`d = max|x| / 127`,`x ≈ q · d`,`q ∈ [-127, 127]` i8。体积为 f32 的 1/3.56,逐元素误差 ≤ `max|x| / 254`。
- **Q4_0** — 20 字节/块:`d = max|x| / 8`,`x ≈ (code − 8) · d`,4-bit 码两两打包进一字节(低 nibble = 元素 `2i`,高 nibble = `2i+1`)。体积为 f32 的 1/6.4,逐元素误差 ≤ `max|x| / 16`。

激活始终保持 f32——只有权重存储被量化。这是纯权重量化(W4/W8, A32),正是 CPU 上内存带宽受限的逐 token 解码所关心的区间。

## SIMD 内核

解码热路径(`QuantizedTensor::matvec` 以及注意力与稠密权重使用的 f32 `dot`)配有手写 `std::arch` AVX2+FMA 内核:

- **Q8_0**:32 个 i8 码按四组 8-lane 符号扩展为 f32,用 FMA 累加;块 scale 每块只乘一次。
- **Q4_0**:nibble 用 `and`/`srli`+`and` 解包,用 `unpacklo/hi_epi8` 交错回顺序布局(因此 `x` 侧无需 shuffle),减 8 居中后走同一条 FMA 路径。
- **分发**在运行时进行:`is_x86_feature_detected!("avx2")` + `"fma"`(结果缓存于 `OnceLock`);标量内核保留为可移植回退,所有 `unsafe` SIMD 代码都包裹在带 SAFETY 注释的安全公共 API 之后。无 AVX2 的机器行为不变。

实测(i5-12450HX,12 线程,Qwen2-0.5B,`minq bench --prompt-tokens 128 --gen-tokens 64`):

| 模型 | 阶段 | 标量 | AVX2+FMA | 加速比 |
|------|--------|--------|----------|--------|
| Q8_0 | prefill | 15.5 tok/s | 22.3 tok/s | 1.44x |
| Q8_0 | decode | 15.2 tok/s | 20.2 tok/s | 1.33x |
| Q4_0 | prefill | 13.4 tok/s | 23.4 tok/s | 1.75x |
| Q4_0 | decode | 13.5 tok/s | 21.3 tok/s | 1.58x |

注意 SIMD 构建**翻转了 Q4_0 与 Q8_0 的快慢顺序**:计算不再是瓶颈后,Q4_0 更小的内存占用开始占优——这正是纯权重量化瞄准的带宽受限区间。

## 架构支持

- **LLaMA / Qwen2**:RMSNorm、RoPE、MHA/GQA、SwiGLU;可选 Q/K/V 偏置。
- **Qwen3**:Q、K 上的逐头 RMSNorm(`q_norm` / `k_norm`,head_dim 维,在 RoPE 之前作用),`head_dim` 显式给出、与 `hidden_size / n_heads` 解耦,无 QKV 偏置。是否启用 q/k-norm 完全由对应张量是否存在驱动,因此 Qwen2 checkpoint 行为不变。tied 与非 tied 的 LM head 均已处理。

## 设计决策与取舍

- **无框架、小表面。** 整个引擎约 3300 行。rayon 并行按行 matvec 就是全部"运行时";没有算子图、没有自动微分、没有设备抽象。
- **融合反量化 matvec,而不是先反量化再 matmul。** 解码逐 token 进行,是纯内存带宽受限:量化的收益来自少读 4–6 倍权重字节。物化 f32 权重会把收益还回去,所以 `QuantizedTensor::matvec` 在块内累加 `Σ qᵢ·xᵢ`,每块只乘一次 scale。
- **纯权重量化。** 量化激活(W8A8 等)在 CPU、batch=1 下收益很小却损失真实精度,有意不做。
- **嵌入表保持 f32**(它是查表而非乘法),RMSNorm 增益和 Q/K/V 偏置同理。其余所有 2-D 且块对齐的张量都会被量化。
- **两条独立前向路径。** `forward`(KV cache 增量)与 `forward_full`(一次性因果扫描)并存,让测试套件能够**证明** cache 精确,而不只是看起来对。
- **自研 `.minq` 格式而非 GGUF。** 可以 hexdump 检查的教学格式:magic、JSON 配置、长度前缀张量记录。没有元数据沼泽,没有对齐填充规则。
- **已知局限(如实列出):** 权重未 mmap(读入内存);无 flash-attention(解码是 matvec 受限,影响很小);单序列(无 batching);SIMD 仅覆盖 AVX2(尚无 AVX-512 / NEON 内核)。

## 测试

`cargo test` 使用程序生成的随机小模型(无需下载),覆盖:

- Q8_0 往返相对误差有界(< 1%),且 Q4_0 严格差于 Q8_0(单调性 sanity check)
- 融合量化 matvec ≡ 先反量化再 matvec
- RMSNorm 对手算参考值
- RoPE:范数保持与相对位置性质 `⟨RoPE(q,m), RoPE(k,n)⟩ = f(m − n)`
- **KV cache 精确性**:增量解码 logits ≡ 一次性全前向 logits(含 prefill+decode 分段)
- GQA:`n_kv_heads = 1` ≡ 显式复制 K/V 头的 MHA
- 随机 2 层模型(dim 64, vocab 128)greedy 生成的端到端确定性
- 采样器定律:`top_k = 1` ≡ greedy,`temperature = 0` ≡ greedy,种子可复现
- `.minq` 文件 roundtrip、坏 magic 拒绝、旧 magic `MINFER01` 兼容
- AVX2 内核 ≡ 标量内核(相对误差 < 1e-5)
- Qwen3:逐头 q/k-norm 下的 KV cache 精确性、q/k-norm 确实生效、`head_dim` 与 `hidden_size / n_heads` 解耦

## 模型验证

在 RTX 4050 6GB 笔记本(CPU 推理,12 线程)上对 Qwen2-0.5B 官方权重(fp16 safetensors,HuggingFace)的端到端结果:

| 精度 | 体积 | prefill | decode | greedy 输出 vs fp16 |
|---|---|---|---|---|
| f32(fp16 读入) | 1976 MB | 9.2 tok/s | 6.8 tok/s | — |
| Q8_0 | 947 MB | 15.6 tok/s | 15.7 tok/s(2.3×) | **逐 token 一致**(中英文 prompt) |
| Q4_0 | 768 MB | 14.2 tok/s | 13.7 tok/s(2.0×) | 发散:重复循环、事实漂移 |

注意此处 Q4_0 *慢于* Q8_0——12 线程下 nibble 解包开销吃掉了带宽收益。手写 AVX2 内核(见上文 *SIMD 内核*)消除了该开销,并把快慢顺序翻转为 Q4_0 占优。

规模验证:**Qwen3-4B-Base**(ModelScope,同机;fp16→f32 基线约需 16GB 内存,超出本机,故直接对量化档比较):

| 精度 | 体积 | prefill | decode | 生成质量(greedy,中英文) |
|---|---|---|---|---|
| Q8_0 | 5644 MB(2.85× 压缩) | 3.9 tok/s | 4.1 tok/s | 连贯、事实正确 |
| Q4_0 | 3827 MB(4.20× 压缩) | 5.4 tok/s | **5.2 tok/s** | 连贯——未出现 0.5B 上的重复循环 |

两个结论:AVX2 内核就位后,Q4_0 在 4B 上速度超过 Q8_0;0.5B 上可见的 4-bit 质量损失在 4B 上消失——**小模型对量化更脆弱,质量-速度权衡是规模相关的**。

内存墙验证:**Qwen3-8B-Base**(ModelScope,同机 16GB 内存):f32 展开 32.8 GB,Q8_0 11.0 GB **无法加载**(内存不足),Q4_0 7.2 GB 正常运行:

| 精度 | 体积 | prefill | decode | 生成质量(greedy,中英文) |
|---|---|---|---|---|
| Q8_0 | 11005 MB(2.98× 压缩) | —(内存不足) | —(内存不足) | — |
| Q4_0 | 7221 MB(4.54× 压缩) | 4.8 tok/s | **4.6 tok/s** | 连贯、事实正确 |

对端侧部署而言,在**不采用 mmap** 的引擎上,4 bit 不是 8B 的优化项,而是入场券(采用 mmap 按需调页的引擎如 llama.cpp 可以移动这堵墙——这正是本引擎把 mmap 列为下一步的直接动机)。另一观察:Q4_0 解码从 4B 到 8B 几乎没掉速(5.2 → 4.6 tok/s)——8B 的 matvec 行数更多,12 线程并行利用率更高,有效带宽从约 20 GB/s 提升到约 33 GB/s,每 token 时间随规模次线性增长。

定量质量评估:WikiText-2 测试集前 10,000 token(context 512,统一口径,引擎内置 `eval-ppl`):

| 模型 | 精度 | PPL | 相对基线 |
|---|---|---|---|
| Qwen2-0.5B | f32 | 19.15 | — |
| Qwen2-0.5B | Q8_0 | 19.14 | **−0.04%(噪声内)** |
| Qwen2-0.5B | Q4_0 | 21.76 | +13.7% |
| Qwen3-4B | Q8_0 | 11.01 | — |
| Qwen3-4B | Q4_0 | 12.01 | +9.1% |

Q8_0 在 0.5B 上与 fp16 的 PPL 差异在噪声内;Q4_0 的相对质量损失随规模收窄(0.5B +13.7% → 4B +9.1%);且 4B Q4_0 的绝对 PPL 远优于 0.5B fp16——端侧内存预算内,"更大的模型 + 更激进的量化"优于"小模型 + 高保真",独立复现了 k-bit 推理缩放律文献(Dettmers & Zettlemoyer, 2023)的核心结论。
