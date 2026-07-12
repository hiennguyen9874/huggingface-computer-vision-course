# Unit 13 — Các hướng kiến trúc mới ngoài Transformer truyền thống trong Computer Vision

## 1. Bối cảnh chung

Transformer, đặc biệt là Vision Transformer — ViT, đã trở thành nền tảng quan trọng trong computer vision. Tuy nhiên, self-attention có một nhược điểm lớn:

\[
\text{Cost} = O(L^2)
\]

với \(L\) là số token.

Trong ảnh, số token tăng rất nhanh khi:

- ảnh có độ phân giải cao hơn,
- patch size nhỏ hơn,
- xử lý video,
- xử lý ảnh y tế, ảnh vệ tinh, ảnh panorama.

Ví dụ ảnh \(224 \times 224\), patch \(16 \times 16\):

\[
L = 14 \times 14 = 196
\]

Nhưng ảnh \(1024 \times 1024\), patch \(8 \times 8\):

\[
L = 128 \times 128 = 16384
\]

Self-attention khi đó phải tính ma trận attention kích thước:

\[
16384 \times 16384
\]

rất tốn bộ nhớ và tính toán.

Unit 13 giới thiệu một số hướng thay thế hoặc cải tiến:

| Kiến trúc | Ý tưởng chính |
|---|---|
| **Hyena** | Thay attention bằng long convolution + gating |
| **I-JEPA** | Self-supervised learning bằng dự đoán trong embedding space |
| **RetNet / RMT / ViR** | Thay attention bằng retention mechanism |
| **Hiera** | Giữ Transformer đơn giản, dùng pretraining tốt thay vì thêm module phức tạp |

---

# 2. Hyena

## 2.1. Hyena là gì?

**Hyena** là một loại operator được đề xuất để thay thế cơ chế attention.

Mục tiêu chính:

- giữ khả năng nhìn toàn cục như attention,
- giảm chi phí tính toán từ \(O(L^2)\),
- xử lý chuỗi dài hiệu quả hơn,
- áp dụng được cho language, DNA, ảnh, video, dữ liệu N-dimensional.

Hyena được xây dựng bằng cách kết hợp:

1. **Long convolution**
2. **Implicit parametrization**
3. **Element-wise gating**
4. **Recursive mixing giữa nhiều projection**

---

## 2.2. Vì sao cần thay attention?

Self-attention có hai đặc tính quan trọng:

### 1. Global context

Mỗi token có thể tương tác với mọi token khác.

Trong ảnh, điều này nghĩa là một patch ở góc trái có thể tương tác với patch ở góc phải.

### 2. Data-dependent operation

Attention score phụ thuộc vào input thông qua:

\[
Q = XW_q
\]

\[
K = XW_k
\]

\[
V = XW_v
\]

Sau đó:

\[
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
\]

Tức là cách trộn thông tin thay đổi tùy theo nội dung ảnh.

Hyena cố gắng mô phỏng hai đặc tính này mà không dùng attention chuẩn.

---

## 2.3. Long convolution

Convolution thông thường dùng kernel nhỏ, ví dụ \(3 \times 3\), \(5 \times 5\).

Long convolution dùng kernel có độ dài bằng toàn bộ input.

Nếu input sequence có độ dài \(L\), kernel cũng có độ dài xấp xỉ \(L\).

Điều này tạo ra **global receptive field**, tương tự attention.

Ví dụ đơn giản với sequence 1D:

```python
import torch
import torch.nn.functional as F

x = torch.randn(1, 1, 1024)       # batch=1, channel=1, sequence length=1024
kernel = torch.randn(1, 1, 1024)  # long convolution kernel

y = F.conv1d(x, kernel, padding=1023)
```

Trong thực tế, long convolution không được triển khai ngây thơ như trên vì rất tốn kém. Hyena thường dùng FFT để tăng tốc.

---

## 2.4. Vì sao dùng FFT?

Convolution trực tiếp có chi phí:

\[
O(L^2)
\]

Nhưng convolution có thể được tính qua Fourier Transform:

\[
\text{Conv}(x,h) = \mathcal{F}^{-1}(\mathcal{F}(x) \cdot \mathcal{F}(h))
\]

Chi phí giảm còn:

\[
O(L \log L)
\]

Đây là một lý do quan trọng khiến Hyena hiệu quả với chuỗi dài.

Pseudo-code:

```python
def fft_convolution(x, h):
    # x: input sequence
    # h: convolution kernel
    X = torch.fft.rfft(x)
    H = torch.fft.rfft(h)
    Y = X * H
    y = torch.fft.irfft(Y, n=x.shape[-1])
    return y
```

---

## 2.5. Implicit convolution

Nếu long convolution kernel có độ dài \(L\), học trực tiếp kernel sẽ tốn nhiều tham số và khó scale.

Hyena không học trực tiếp từng phần tử của kernel.

Thay vào đó, nó học một hàm nhỏ:

\[
h = \gamma_\theta(t)
\]

Trong đó:

- \(t\) là vị trí/token index,
- \(\gamma_\theta\) là một neural network nhỏ,
- output là giá trị kernel tại vị trí đó.

Tức là thay vì học:

```python
h = torch.nn.Parameter(torch.randn(L))
```

Hyena học:

```python
h_t = gamma(position_t)
```

Ví dụ minh họa:

```python
import torch
import torch.nn as nn

class ImplicitFilter(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, L):
        positions = torch.linspace(0, 1, L).unsqueeze(-1)
        h = self.net(positions).squeeze(-1)
        return h
```

Ý tưởng giống như thay vì lưu từng điểm của đường thẳng, ta chỉ học \(a\) và \(b\) trong:

\[
y = ax + b
\]

rồi sinh giá trị khi cần.

---

## 2.6. Gating trong Hyena

Attention phụ thuộc vào dữ liệu vì attention score được sinh từ input.

Hyena tạo tính data-dependent bằng **element-wise gating**.

Gating thường có dạng:

\[
y = a \odot b
\]

Trong đó \(\odot\) là nhân từng phần tử.

Ví dụ:

```python
gate = torch.sigmoid(W_g(x))
value = W_v(x)

y = gate * value
```

Gate quyết định thông tin nào đi tiếp, thông tin nào bị chặn.

---

## 2.7. Hyena operator hoạt động như thế nào?

Hyena thường tạo nhiều projection từ input, tương tự attention tạo \(q, k, v\).

Với Hyena bậc 2, có thể hiểu đơn giản:

1. Từ input \(u\), tạo các projection:

\[
x_1, x_2, x_3
\]

2. Áp dụng long convolution lên một projection.

3. Trộn bằng gating.

4. Lặp lại theo dạng recursive.

Một dạng pseudo-code đơn giản:

```python
def hyena_operator(u):
    x1, x2, x3 = project(u)  # giống q, k, v

    h1 = implicit_filter(length=u.shape[1])
    h2 = implicit_filter(length=u.shape[1])

    y = fft_convolution(x3, h1)
    y = x2 * y              # gating

    y = fft_convolution(y, h2)
    y = x1 * y              # gating

    return y
```

Điểm quan trọng:

- Không dùng softmax.
- Không tính \(QK^T\).
- Không tạo ma trận attention \(L \times L\).
- Dùng convolution toàn cục và gating để thay thế.

---

## 2.8. Hyena trong computer vision

Hyena có thể thay attention trong các backbone như:

- ViT,
- Swin,
- DeiT.

Với ảnh, token sequence đến từ image patches.

Hyena đặc biệt hữu ích khi số patch lớn:

- ảnh độ phân giải cao,
- patch nhỏ,
- ảnh y tế,
- remote sensing,
- video.

Lợi ích chính:

- giảm GPU memory,
- xử lý sequence dài tốt hơn,
- vẫn giữ khả năng modeling toàn cục.

---

## 2.9. Điểm cần nhớ về Hyena

Cần nắm các ý sau:

1. Hyena là attention replacement.
2. Nó dùng **long convolution** để có global context.
3. Nó dùng **gating** để tạo data dependency.
4. Nó dùng **implicit filter** để không phải học trực tiếp kernel dài.
5. FFT giúp giảm chi phí xuống khoảng:

\[
O(L \log L)
\]

6. Hyena phù hợp với dữ liệu có sequence rất dài.

---

# 3. I-JEPA

## 3.1. I-JEPA là gì?

**I-JEPA** là viết tắt của:

**Image-based Joint-Embedding Predictive Architecture**

Đây là một phương pháp **self-supervised learning** do Meta AI giới thiệu.

Mục tiêu:

- học representation tốt cho ảnh,
- không cần label,
- không cần hand-crafted augmentation mạnh,
- không reconstruct pixel trực tiếp như MAE,
- dự đoán trong embedding space.

---

## 3.2. Hai hướng self-supervised learning phổ biến

### 3.2.1. Invariance-based methods

Các phương pháp này tạo nhiều view khác nhau từ cùng một ảnh bằng augmentation:

- crop,
- rotate,
- color jitter,
- blur,
- scale.

Sau đó ép model tạo embedding giống nhau cho các view đó.

Ví dụ:

\[
f(x_1) \approx f(x_2)
\]

với \(x_1, x_2\) là hai biến thể của cùng ảnh.

Ưu điểm:

- học semantic representation tốt,
- hiệu quả cho classification.

Nhược điểm:

- phụ thuộc nhiều vào augmentation do con người thiết kế,
- augmentation có thể gây bias,
- không phải task nào cũng cần invariant với crop/rotate/color.

Ví dụ object counting hoặc depth estimation có thể bị ảnh hưởng xấu nếu augmentation làm mất thông tin không gian.

---

### 3.2.2. Generative methods

Các phương pháp này thường mask một phần ảnh rồi yêu cầu model tái tạo phần bị che.

Ví dụ MAE:

- chia ảnh thành patch,
- mask 75% patch,
- encode phần còn lại,
- decode để reconstruct pixel của patch bị mask.

Ưu điểm:

- không cần label,
- dễ mở rộng sang nhiều modality.

Nhược điểm:

- reconstruct pixel có thể khiến model tập trung vào chi tiết thấp cấp,
- tốn compute,
- cần nhiều dữ liệu để học tốt.

---

## 3.3. Joint-Embedding Architecture

Joint-Embedding Architecture dùng hai network để đưa các input khác nhau vào cùng một embedding space.

Ví dụ:

```text
view 1 -> encoder A -> embedding z1
view 2 -> encoder B -> embedding z2
```

Mục tiêu:

\[
z_1 \approx z_2
\]

Vấn đề lớn:

### Representation collapse

Model có thể output cùng một vector cho mọi input.

Ví dụ:

\[
f(x) = c
\]

với mọi ảnh \(x\).

Khi đó loss có thể thấp nhưng representation vô dụng.

---

## 3.4. I-JEPA khác gì?

I-JEPA kết hợp ý tưởng từ generative learning và joint embedding, nhưng có khác biệt quan trọng:

### Không reconstruct pixel

Thay vì dự đoán pixel bị che, I-JEPA dự đoán **embedding** của vùng bị che.

Đây gọi là:

**abstract prediction**

Nghĩa là model học dự đoán ở mức representation/semantic, không phải màu từng pixel.

---

## 3.5. Kiến trúc I-JEPA

I-JEPA có ba thành phần chính.

### 1. Context Encoder

Nhận vùng context của ảnh.

Ví dụ một số patch không bị mask.

Output:

\[
z_x = f_\theta(x_{\text{context}})
\]

Context encoder thường là ViT.

---

### 2. Target Encoder

Nhận ảnh hoặc vùng target, sinh target representation.

Output:

\[
z_y = f_{\bar{\theta}}(x_{\text{target}})
\]

Target encoder thường là một bản cập nhật chậm của context encoder, ví dụ bằng EMA trong nhiều framework self-supervised.

---

### 3. Predictor

Nhận:

- embedding của context,
- mask token,
- thông tin vị trí vùng cần dự đoán.

Sau đó dự đoán embedding target:

\[
\hat{z}_y = g_\phi(z_x, \text{mask tokens})
\]

Loss có dạng:

\[
\mathcal{L} = || \hat{z}_y - z_y ||^2
\]

---

## 3.6. Multi-block masking

I-JEPA không mask patch nhỏ ngẫu nhiên đơn lẻ.

Nó mask các block lớn.

Lý do:

- ép model hiểu ngữ cảnh rộng,
- học semantic structure,
- tránh chỉ dựa vào texture/local pattern.

Ví dụ nếu che cả phần thân của con chó, model phải hiểu từ đầu, chân, background rằng vùng bị che thuộc về con chó.

---

## 3.7. Pseudo-code đơn giản cho I-JEPA

```python
def ijepa_step(image):
    context_blocks, target_blocks = sample_blocks(image)

    context_repr = context_encoder(context_blocks)

    with torch.no_grad():
        target_repr = target_encoder(target_blocks)

    pred_repr = predictor(
        context_repr,
        target_positions=target_blocks.positions
    )

    loss = ((pred_repr - target_repr) ** 2).mean()

    return loss
```

Điểm quan trọng là loss không nằm trên pixel:

```python
# Không phải:
loss = mse(predicted_pixels, target_pixels)

# Mà là:
loss = mse(predicted_embedding, target_embedding)
```

---

## 3.8. Vì sao I-JEPA quan trọng?

I-JEPA có các lợi thế:

1. Không cần hand-crafted augmentation mạnh.
2. Không reconstruct pixel nên học semantic tốt hơn generative method truyền thống.
3. Hiệu quả hơn MAE trên nhiều benchmark.
4. Tốt trên cả semantic task và low-level task.
5. Pretraining khá hiệu quả, theo tài liệu là dưới 1200 GPU hours trên ImageNet.

---

## 3.9. Điểm cần nhớ về I-JEPA

1. I-JEPA là self-supervised learning cho ảnh.
2. Nó dự đoán embedding của vùng bị mask, không dự đoán pixel.
3. Có ba module: context encoder, target encoder, predictor.
4. Dùng multi-block masking để học semantic tốt.
5. Giảm phụ thuộc vào augmentation thủ công.

---

# 4. Retention trong Vision

## 4.1. RetNet là gì?

**Retentive Network — RetNet** là kiến trúc được đề xuất như một hướng thay thế Transformer trong language modeling.

Mục tiêu giải quyết ba vấn đề:

1. Train song song hiệu quả như Transformer.
2. Inference rẻ như RNN.
3. Performance tốt khi scale.

Thành phần chính là:

**Multi-Scale Retention — MSR**

---

## 4.2. Retention mechanism có gì đặc biệt?

RetNet có ba dạng tính toán.

### 1. Parallel representation

Dùng khi training.

Tương tự self-attention, có thể tận dụng GPU để xử lý song song.

### 2. Recurrent representation

Dùng khi inference autoregressive.

Ưu điểm:

- memory \(O(1)\),
- không cần KV cache lớn như Transformer,
- latency thấp hơn.

### 3. Chunkwise recurrent representation

Dùng cho sequence dài.

Ý tưởng:

- trong mỗi chunk: xử lý song song,
- giữa các chunk: xử lý recurrent.

Cách này cân bằng giữa tốc độ training và tiết kiệm memory.

---

## 4.3. Từ RetNet sang thị giác máy tính

Trong vision, hai hướng nổi bật được nhắc đến là:

1. **RMT — Retentive Networks Meet Vision Transformers**
2. **ViR — Vision Retention Networks**

---

# 4.4. RMT

## RMT là gì?

RMT là vision backbone lấy cảm hứng từ RetNet.

Mục tiêu:

- cải thiện ViT,
- thêm spatial prior rõ ràng,
- giảm complexity,
- xử lý ảnh hiệu quả hơn.

RMT dùng cơ chế:

**Manhattan Self-Attention — MaSA**

---

## 4.5. Manhattan distance trong ảnh

Với hai token ảnh ở vị trí:

\[
(i_1, j_1)
\]

và

\[
(i_2, j_2)
\]

Manhattan distance là:

\[
d = |i_1 - i_2| + |j_1 - j_2|
\]

Nó đo khoảng cách theo lưới 2D.

Ví dụ:

```python
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
```

Trong ảnh, token càng xa thì attention nên bị decay mạnh hơn.

---

## 4.6. Manhattan Self-Attention — MaSA

MaSA thêm spatial decay matrix vào attention.

Ý tưởng:

- token gần nhau có ảnh hưởng mạnh hơn,
- token xa nhau vẫn có thể tương tác nhưng bị giảm trọng số,
- giữ global receptive field,
- thêm inductive bias phù hợp với ảnh.

Attention thông thường:

\[
A = QK^T
\]

MaSA thêm decay theo Manhattan distance:

\[
A_{ij} = Q_iK_j^T + D_{ij}
\]

hoặc có thể hiểu là attention score bị điều chỉnh bởi distance bias.

Trong đó \(D_{ij}\) phụ thuộc vào khoảng cách Manhattan giữa token \(i\) và token \(j\).

---

## 4.7. Decomposed Manhattan Self-Attention — MaSAD

MaSAD phân rã attention theo hai trục:

1. chiều ngang,
2. chiều dọc.

Thay vì tính toàn bộ attention 2D rất lớn, nó xử lý theo axis.

Lợi ích:

- giảm complexity,
- giữ được spatial prior,
- vẫn có receptive field toàn cục,
- scale tốt hơn với ảnh lớn.

Đây là kỹ thuật tương tự các ý tưởng axial attention, nhưng được thiết kế với Manhattan decay.

---

## 4.8. RMT khác RetNet gốc thế nào?

RetNet gốc:

- training dùng parallel,
- inference dùng recurrent.

RMT:

- dùng MaSA cho cả training và inference.

Theo nội dung tài liệu, tác giả so sánh MaSA với các representation khác của RetNet và cho thấy MaSA có throughput tốt và accuracy cao.

---

# 4.9. ViR — Vision Retention Networks

ViR là một vision backbone khác lấy cảm hứng từ RetNet.

Kiến trúc tổng thể khá giống ViT, nhưng thay:

```text
Multi-Head Attention
```

bằng:

```text
Multi-Head Retention
```

ViR có thể chuyển giữa các mode:

- parallel,
- recurrent,
- chunkwise.

Điểm đáng chú ý:

- không dùng gating function,
- scale tốt với ảnh độ phân giải lớn,
- throughput và memory tốt hơn khi resolution tăng.

Một khác biệt nhỏ với ViT:

- positional embedding được cộng vào patch embedding trước,
- sau đó mới append `[class]` token.

---

## 4.10. Điểm cần nhớ về Retention trong Vision

1. RetNet hướng tới thay self-attention bằng retention.
2. Retention có thể vừa train song song vừa inference recurrent.
3. RMT dùng Manhattan Self-Attention để thêm spatial decay.
4. MaSAD phân rã attention theo trục ngang/dọc để giảm cost.
5. ViR thay MHA bằng Multi-Head Retention.
6. Các mô hình retention hữu ích khi xử lý ảnh lớn hoặc sequence dài.

---

# 5. Hiera

## 5.1. Hiera là gì?

**Hiera — Hierarchical Vision Transformer** là một kiến trúc vision transformer phân cấp.

Điểm đặc biệt:

> Hiera đạt accuracy cao mà không cần nhiều thành phần chuyên biệt/phức tạp thường thấy trong các vision transformer hiện đại.

Thông điệp chính của Hiera:

> Nếu pretraining task đủ mạnh, model có thể học spatial reasoning mà không cần nhồi thêm quá nhiều inductive bias vào kiến trúc.

---

## 5.2. CNN, ViT và hierarchical representation

CNN phù hợp với ảnh vì có inductive bias mạnh:

- local receptive field,
- translation equivariance,
- hierarchical features.

CNN thường hoạt động theo nhiều stage:

| Stage | Spatial resolution | Channels | Feature |
|---|---:|---:|---|
| Early | cao | ít | edge, texture |
| Middle | trung bình | vừa | part, shape |
| Late | thấp | nhiều | object, semantic |

ViT nguyên bản thì đơn giản hơn:

- chia ảnh thành patch,
- flatten thành token sequence,
- xử lý bằng transformer block,
- resolution token thường giữ cố định.

ViT mạnh nhưng thiếu vision inductive bias.

Do đó nhiều mô hình cố thêm:

- convolution,
- relative positional embedding,
- window attention,
- pooling phức tạp,
- multi-scale module.

Nhưng các thành phần này làm model:

- chậm hơn,
- lớn hơn,
- khó scale hơn,
- khó triển khai hơn.

---

## 5.3. Hiera bắt đầu từ MViTv2

Hiera được đơn giản hóa từ **MViTv2**.

MViTv2 là hierarchical vision transformer với nhiều stage.

Nó học multi-scale representation bằng cách:

- early stage: spatial resolution cao, channel thấp,
- deeper stage: spatial resolution thấp, channel cao.

Giống CNN nhưng dùng transformer.

---

## 5.4. Ý tưởng chính của Hiera

Tác giả Hiera cho rằng:

> Nhiều thành phần vision-specific trong các hierarchical ViT là không thật sự cần thiết nếu model được pretrain bằng MAE tốt.

Do đó Hiera loại bỏ nhiều thành phần phức tạp.

---

## 5.5. Những thay đổi chính từ MViTv2 sang Hiera

### 1. Thay relative positional embedding bằng absolute positional embedding

MViTv2 dùng relative positional embedding trong attention.

Hiera quay lại dùng absolute positional embedding đơn giản hơn.

Lý do:

- relative position embedding làm model phức tạp hơn,
- khi dùng MAE pretraining, relative embedding không còn quá cần thiết,
- bỏ nó giúp model nhanh hơn và accuracy vẫn tốt.

---

### 2. Loại bỏ convolution

Nhiều hierarchical ViT thêm convolution để có spatial inductive bias.

Hiera loại bỏ convolution vì tin rằng MAE pretraining có thể giúp model tự học spatial structure.

Ban đầu thay convolution bằng max pooling.

Nhưng một số max pooling stride 1 gần như chỉ làm biến đổi cục bộ không cần thiết.

Sau đó Hiera loại bỏ các pooling stride 1 này, giúp:

- tốc độ tăng,
- accuracy gần như giữ lại.

Theo nội dung tài liệu, thay đổi này giúp tăng tốc đáng kể:

- khoảng 22% cho ảnh,
- khoảng 27% cho video.

---

## 5.6. MAE trong Hiera

MAE — Masked Autoencoder là pretraining task quan trọng.

Quy trình:

1. Chia ảnh thành patch.
2. Mask phần lớn patch, thường khoảng 75%.
3. Encoder chỉ nhìn các patch còn lại.
4. Decoder cố reconstruct patch bị thiếu.

Ví dụ trực quan:

```text
Input image patches:
[A, B, C, D, E, F, G, H]

Mask 75%:
[A, _, _, _, E, _, _, _]

Encoder sees:
[A, E]

Decoder reconstructs:
[B, C, D, F, G, H]
```

MAE buộc model học cấu trúc không gian và ngữ nghĩa ảnh.

Hiera dùng MAE như một pretext task mạnh, nhờ đó kiến trúc có thể đơn giản hơn.

---

## 5.7. Vì sao Hiera quan trọng?

Hiera cho thấy một quan điểm quan trọng:

> Không phải lúc nào thêm module phức tạp vào architecture cũng là cách tốt nhất. Đôi khi training objective tốt có thể thay thế một phần architectural bias.

Điều này quan trọng vì:

- model đơn giản dễ scale hơn,
- nhanh hơn,
- dễ triển khai hơn,
- ít lỗi hơn,
- dễ bảo trì hơn.

---

## 5.8. Điểm cần nhớ về Hiera

1. Hiera là hierarchical ViT đơn giản hóa.
2. Nó xuất phát từ MViTv2.
3. Nó loại bỏ relative positional embedding.
4. Nó loại bỏ convolution không cần thiết.
5. Nó dựa mạnh vào MAE pretraining.
6. Bài học chính: pretraining task tốt có thể giảm nhu cầu thêm kiến trúc phức tạp.

---

# 6. So sánh nhanh các phương pháp

| Phương pháp | Vấn đề giải quyết | Kỹ thuật chính | Điểm mạnh |
|---|---|---|---|
| Hyena | Attention \(O(L^2)\) quá đắt | Long convolution + gating | Sequence dài, memory tốt |
| I-JEPA | Self-supervised representation | Predict embedding thay vì pixel | Semantic representation tốt |
| RetNet/RMT/ViR | Attention tốn cache/compute | Retention, spatial decay | Train song song, inference/memory tốt |
| Hiera | Vision Transformer quá phức tạp | Đơn giản hóa + MAE pretraining | Nhanh, chính xác, dễ scale |

---

# 7. Các khái niệm kỹ thuật quan trọng cần nắm

## 7.1. Global receptive field

Một token có thể nhận thông tin từ toàn bộ input.

Attention có global receptive field tự nhiên.

Hyena đạt được điều này bằng long convolution.

RMT/MaSA đạt được bằng attention có spatial decay.

---

## 7.2. Data dependency

Operation thay đổi theo input.

Attention có data dependency qua \(Q,K,V\).

Hyena tạo data dependency bằng gating.

---

## 7.3. Inductive bias

Là giả định được đưa vào kiến trúc model.

CNN có inductive bias mạnh cho ảnh.

ViT có ít inductive bias hơn.

Hiera cho thấy có thể dùng pretraining task mạnh để giảm nhu cầu thêm inductive bias bằng tay.

---

## 7.4. Abstract prediction

Dự đoán ở không gian embedding thay vì pixel.

Đây là điểm cốt lõi của I-JEPA.

```text
Generative SSL:
masked image -> reconstruct pixels

I-JEPA:
context -> predict target representation
```

---

## 7.5. Spatial decay

Trong vision, token gần nhau thường liên quan mạnh hơn token xa nhau.

RMT dùng Manhattan distance để encode bias này.

---

# 8. Nếu cần nhớ ngắn gọn

- **Hyena**: thay attention bằng long convolution + gating, tốt cho sequence dài.
- **I-JEPA**: self-supervised, dự đoán embedding của vùng bị che thay vì reconstruct pixel.
- **Retention/RMT/ViR**: dùng retention hoặc attention có decay theo không gian để giảm chi phí và scale tốt hơn.
- **Hiera**: đơn giản hóa hierarchical ViT, dùng MAE pretraining mạnh thay vì thêm nhiều module phức tạp.