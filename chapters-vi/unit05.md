# Unit 5 — Generative Models trong Computer Vision

Unit này nói về các mô hình sinh ảnh, tức là các mô hình không chỉ phân loại hay phát hiện đối tượng, mà có thể **tạo ra ảnh mới**, **biến đổi ảnh**, **phục hồi ảnh**, hoặc **sinh ảnh từ text prompt**.

Các nhóm chính được đề cập:

1. Generative Models vs Discriminative Models  
2. GANs  
3. VAEs  
4. StyleGAN  
5. Diffusion Models  
6. Stable Diffusion  
7. DreamBooth, LoRA, ControlNet  
8. CycleGAN  
9. Đánh giá mô hình sinh ảnh  
10. Vấn đề đạo đức, privacy, bias

---

# 1. Generative Models là gì?

Trong machine learning có thể chia mô hình thành 2 nhóm lớn:

## Discriminative Models

Mô hình phân biệt học ranh giới giữa các class.

Ví dụ:

- Image classification
- Object detection
- Semantic segmentation
- Regression

Mục tiêu thường là học:

```text
p(y | x)
```

Tức là: với ảnh `x`, dự đoán label `y`.

Ví dụ: ảnh này là mèo hay chó?

---

## Generative Models

Mô hình sinh học phân phối dữ liệu.

Mục tiêu gần hơn với:

```text
p(x)
```

hoặc:

```text
p(x | condition)
```

Tức là học xem dữ liệu thật có phân phối như thế nào, sau đó sinh ra dữ liệu mới giống phân phối đó.

Ví dụ:

- Sinh ảnh từ noise
- Sinh ảnh từ text
- Biến ảnh ngựa thành ảnh ngựa vằn
- Inpainting vùng bị thiếu trong ảnh
- Super-resolution ảnh độ phân giải thấp

---

# 2. Các task sinh ảnh chính trong unit

Unit này tập trung vào các bài toán:

## Noise-to-image

Input là vector noise ngẫu nhiên, output là ảnh.

Ví dụ:

```text
z ~ N(0, I) -> Generator -> image
```

Thường gặp trong GAN, DCGAN, StyleGAN.

---

## Text-to-image

Input là text prompt, output là ảnh.

Ví dụ:

```text
"A photo of an astronaut riding a horse on Mars"
```

Mô hình sinh ra ảnh tương ứng.

Thường dùng Diffusion Models, Stable Diffusion.

---

## Image-to-image

Input là ảnh, output là ảnh đã biến đổi.

Ví dụ:

- Sketch -> photo
- Summer -> winter
- Horse -> zebra
- Low-res -> high-res
- Ảnh gốc + prompt -> ảnh chỉnh sửa

Thường dùng CycleGAN, diffusion image-to-image, ControlNet.

---

# 3. Đánh giá mô hình sinh ảnh

Đánh giá generative models rất khó vì thường không có ground truth duy nhất.

Ví dụ prompt:

```text
"A cat wearing sunglasses"
```

Có vô số ảnh hợp lệ. Vì vậy không thể chỉ dùng accuracy như classification.

---

## 3.1 FID — Fréchet Inception Distance

FID là metric phổ biến nhất.

Ý tưởng:

1. Lấy feature của ảnh thật bằng Inception-v3.
2. Lấy feature của ảnh sinh bằng Inception-v3.
3. Xem 2 phân phối feature này cách nhau bao xa.
4. Khoảng cách càng thấp thì ảnh sinh càng giống ảnh thật.

```text
FID thấp hơn -> tốt hơn
```

FID dùng Fréchet distance giữa 2 phân phối Gaussian trong feature space.

Điểm cần nhớ:

- FID đo chất lượng tổng quát của tập ảnh sinh.
- Không hoàn hảo.
- Phụ thuộc vào feature của Inception-v3 pretrained trên ImageNet.
- Có thể bị ảnh hưởng bởi domain khác ImageNet.

---

## 3.2 PSNR

PSNR gần với mean squared error.

Hay dùng trong image restoration, super-resolution.

```text
PSNR cao hơn -> tốt hơn
```

Giá trị thường gặp:

```text
25–34: ổn
>34: rất tốt
```

Nhược điểm: không luôn tương quan tốt với cảm nhận thị giác của con người.

---

## 3.3 SSIM

SSIM đo độ tương đồng cấu trúc giữa 2 ảnh.

Range:

```text
0 đến 1
1 là giống hoàn toàn
```

SSIM xét 3 thành phần:

- Luminance — độ sáng
- Contrast — độ tương phản
- Structure — cấu trúc

Hay dùng khi có ảnh ground truth.

---

## 3.4 Inception Score

Dùng Inception-v3 để đánh giá ảnh sinh.

```text
IS cao hơn -> tốt hơn
```

Ý tưởng:

- Ảnh sinh nên rõ class.
- Tập ảnh sinh nên đa dạng class.

Hiện nay ít được ưu tiên hơn FID.

---

## 3.5 CLIP Score

Dùng cho text-to-image.

Ý tưởng:

- Encode text bằng CLIP text encoder.
- Encode image bằng CLIP image encoder.
- Tính cosine similarity.

```text
CLIP Score cao hơn -> ảnh khớp prompt tốt hơn
```

Range thường được nói là:

```text
0–100
```

Quan trọng: CLIP Score đo độ khớp text-image, không đảm bảo ảnh đẹp hay không có artifact.

---

# 4. GAN — Generative Adversarial Networks

GAN được Ian Goodfellow giới thiệu năm 2014.

GAN gồm 2 mạng chính:

1. Generator
2. Discriminator

---

## 4.1 Generator

Generator nhận noise vector `z` và sinh ảnh giả.

```text
z -> Generator -> fake image
```

Mục tiêu: tạo ảnh giả đủ thật để đánh lừa Discriminator.

---

## 4.2 Discriminator

Discriminator nhận ảnh và dự đoán ảnh đó là thật hay giả.

```text
image -> Discriminator -> real/fake probability
```

Mục tiêu: phân biệt ảnh thật từ dataset và ảnh giả từ Generator.

---

## 4.3 Trực giác

GAN giống một trò chơi giữa:

- Generator: kẻ làm tranh giả
- Discriminator: chuyên gia phát hiện tranh giả

Theo thời gian:

- Generator ngày càng tạo ảnh thật hơn.
- Discriminator ngày càng giỏi phát hiện ảnh giả hơn.

---

## 4.4 Objective function của GAN

Hàm min-max cơ bản:

```text
min_G max_D L(D, G)
= E_{x ~ p_real(x)} [log D(x)]
+ E_{x ~ p_g(x)} [log(1 - D(x))]
```

Trong đó:

- `D(x)` là xác suất Discriminator cho rằng ảnh `x` là thật.
- `p_real(x)` là phân phối ảnh thật.
- `p_g(x)` là phân phối ảnh sinh bởi Generator.

Discriminator muốn maximize:

```text
log D(real) + log(1 - D(fake))
```

Generator muốn minimize:

```text
log(1 - D(fake))
```

Hoặc trong thực tế thường dùng non-saturating loss:

```text
maximize log D(fake)
```

để gradient mạnh hơn.

---

## 4.5 Vấn đề kỹ thuật của GAN

GAN có thể tạo ảnh rất sắc nét, nhưng khó train.

Các vấn đề thường gặp:

### Training instability

Do Generator và Discriminator cùng thay đổi, loss không ổn định như supervised learning bình thường.

---

### Mode collapse

Generator chỉ sinh một số kiểu ảnh lặp lại, không đa dạng.

Ví dụ dataset có mèo và chó, nhưng Generator chỉ sinh mèo.

Lý do: Generator tìm được vài mẫu dễ đánh lừa Discriminator và cứ lặp lại.

---

### Khó kiểm soát output

Vanilla GAN nhận noise vector, nhưng từng chiều trong latent vector thường bị entangled.

Ví dụ:

- Một chiều thay đổi vừa làm đổi giới tính, vừa đổi kính, vừa đổi pose.
- Khó yêu cầu “nữ đeo kính” nếu latent space không tách bạch feature.

Đây là lý do StyleGAN ra đời.

---

# 5. VAE — Variational Autoencoder

Trước khi hiểu VAE, cần hiểu Autoencoder.

---

## 5.1 Autoencoder

Autoencoder gồm:

1. Encoder
2. Decoder

```text
x -> Encoder -> z -> Decoder -> x'
```

Trong đó:

- `x`: ảnh gốc
- `z`: latent representation
- `x'`: ảnh tái tạo

Mục tiêu:

```text
x ≈ x'
```

Loss thường là reconstruction loss:

```text
L = 1/n * Σ (x_i - fθ(gφ(x_i)))²
```

Trong đó:

- `gφ`: encoder
- `fθ`: decoder

Autoencoder thường dùng cho:

- Compression
- Denoising
- Feature learning
- Dimensionality reduction

---

## 5.2 Hạn chế của Autoencoder thường

Autoencoder thường encode ảnh thành một vector cố định.

```text
x -> z
```

Nhưng để sinh ảnh mới, ta cần latent space có cấu trúc tốt, liên tục, dễ sample.

Vanilla Autoencoder không đảm bảo điều đó.

---

## 5.3 VAE khác Autoencoder như thế nào?

VAE không encode ảnh thành một vector cố định, mà encode thành một phân phối xác suất.

Thường là Gaussian:

```text
qφ(z | x) = N(μ, σ²)
```

Encoder output:

```text
μ, log σ²
```

Sau đó sample:

```text
z ~ N(μ, σ²)
```

Decoder dùng `z` để tái tạo hoặc sinh ảnh:

```text
pθ(x | z)
```

---

## 5.4 Reparameterization trick

Không thể backprop trực tiếp qua thao tác random sampling thông thường.

VAE dùng trick:

```text
z = μ + σ * ε
ε ~ N(0, I)
```

Nhờ đó gradient đi qua `μ` và `σ`.

Pseudo-code:

```python
import torch

mu = encoder_mu(x)
logvar = encoder_logvar(x)

std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)

z = mu + std * eps
x_recon = decoder(z)
```

---

## 5.5 Loss của VAE

VAE loss gồm 2 phần:

```text
VAE Loss = Reconstruction Loss + KL Divergence
```

### Reconstruction loss

Đo ảnh tái tạo giống ảnh gốc không.

```text
x vs x'
```

Có thể dùng:

- MSE
- BCE
- Perceptual loss tùy bài toán

---

### KL divergence

Ép phân phối latent học được gần với prior chuẩn, thường là:

```text
p(z) = N(0, I)
```

KL term:

```text
KL(qφ(z|x) || p(z))
```

Mục tiêu:

- Latent space có cấu trúc.
- Có thể sample từ `N(0, I)` để sinh ảnh mới.
- Các điểm gần nhau trong latent space tạo ra ảnh tương tự nhau.

---

## 5.6 Trade-off trong VAE

Nếu KL quá mạnh:

- Latent chứa ít thông tin.
- Ảnh tái tạo/sinh có thể mờ.

Nếu reconstruction quá mạnh:

- Model tái tạo tốt training image.
- Nhưng latent space kém regularized.
- Sinh ảnh mới có thể không tốt.

Vì vậy VAE phải cân bằng giữa:

```text
chất lượng tái tạo
```

và

```text
khả năng sinh ảnh mới
```

---

## 5.7 GAN vs VAE

| Tiêu chí | GAN | VAE |
|---|---|---|
| Chất lượng ảnh | Thường sắc nét hơn | Thường mờ hơn |
| Training | Khó, dễ mất ổn định | Ổn định hơn |
| Latent space | Có thể khó kiểm soát | Có cấu trúc xác suất rõ hơn |
| Ứng dụng | Image generation, super-resolution, image translation | Denoising, anomaly detection, latent representation |
| Vấn đề | Mode collapse, unstable | Blurry output |

---

# 6. StyleGAN

StyleGAN là biến thể GAN nổi tiếng của NVIDIA, đặc biệt mạnh trong sinh ảnh khuôn mặt.

Điểm chính: StyleGAN chủ yếu thay đổi **Generator**, còn Discriminator gần như giữ tinh thần như GAN.

---

# 6.1 Vanilla GAN thiếu gì?

Vanilla GAN thiếu khả năng kiểm soát rõ ràng.

Ví dụ:

- `z1` sinh nam đeo kính
- `z2` sinh nữ không đeo kính

Nếu muốn nữ đeo kính thì không có cách trực tiếp và ổn định.

Lý do: latent features bị entangled.

---

# 6.2 StyleGAN giải quyết bằng gì?

StyleGAN đưa vào các thành phần:

1. Mapping Network
2. AdaIN
3. Noise injection
4. Progressive/synthesis architecture

---

## 6.3 Mapping Network

Thay vì đưa noise vector `z` trực tiếp vào Generator:

```text
z -> Generator
```

StyleGAN dùng MLP nhiều layer để map:

```text
z -> Mapping Network -> w
```

Trong StyleGAN1, mapping network thường là 8-layer MLP.

Ý nghĩa:

- `z` nằm trong latent space ban đầu.
- `w` nằm trong intermediate latent space.
- `w` thường disentangled hơn.

Nếu latent disentangled tốt, thay đổi một chiều có thể tương ứng một thuộc tính cụ thể:

```text
smile
age
pose
glasses
hair style
```

---

## 6.4 Synthesis Network

Generator trong StyleGAN gọi là Synthesis Network.

Thay vì đưa latent chỉ vào đầu mạng, StyleGAN đưa style information vào nhiều layer.

Các layer thấp điều khiển đặc trưng high-level:

- Pose
- Face shape
- General hairstyle
- Eyeglasses

Các layer cao hơn, resolution lớn hơn, điều khiển chi tiết nhỏ:

- Skin pores
- Hair placement
- Fine texture
- Eye details

---

## 6.5 AdaIN — Adaptive Instance Normalization

AdaIN điều chỉnh feature map theo style.

Instance normalization chuẩn hóa feature map:

```text
x_norm = (x - μ(x)) / σ(x)
```

AdaIN thêm scale và bias từ style vector:

```text
AdaIN(x, y) = y_s * ((x - μ(x)) / σ(x)) + y_b
```

Trong đó:

- `x`: feature map
- `y`: style derived from latent `w`
- `y_s`: scale
- `y_b`: bias

Ý nghĩa: mỗi layer được điều khiển bởi style khác nhau.

---

## 6.6 Noise Injection

StyleGAN thêm noise map vào từng block của synthesis network.

Mục tiêu: kiểm soát stochastic details.

Ví dụ:

- Vị trí sợi tóc
- Lỗ chân lông
- Texture nhỏ
- Chi tiết da

Những chi tiết này không nên cố định hoàn toàn theo latent style, mà cần randomness cục bộ.

---

# 6.7 StyleGAN2

StyleGAN1 có artifact.

Hai vấn đề lớn:

## Blob-like artifact

Một số vùng ảnh xuất hiện dạng “blob” kỳ lạ.

Nguyên nhân được cho là liên quan đến normalization trong StyleGAN1.

StyleGAN2 thay đổi cách modulation/demodulation để giảm artifact.

---

## Location preference artifact

Progressive growing có thể làm model phụ thuộc không tốt vào vị trí tuyệt đối.

Khi interpolate latent, ảnh có thể biến đổi không tự nhiên.

StyleGAN2 bỏ progressive growing, dùng skip generator và residual discriminator.

---

# 6.8 StyleGAN3

StyleGAN2 vẫn có vấn đề aliasing.

Hiện tượng: texture bị “dính” vào tọa độ pixel khi animation/interpolation.

Ví dụ khi mặt quay, texture nhỏ không di chuyển tự nhiên theo object mà có vẻ cố định theo canvas.

StyleGAN3 xử lý vấn đề aliasing và giúp animation/interpolation tự nhiên hơn.

---

# 6.9 Ứng dụng StyleGAN

- Sinh khuôn mặt realistic
- Image editing
- Inpainting
- Style transfer
- Synthetic data
- Privacy-preserving data
- Anonymization
- Fashion design
- Virtual environment
- 3D-aware generation như StyleNeRF

---

# 7. Diffusion Models

Diffusion Models là nhóm mô hình sinh ảnh hiện đại, là nền tảng của Stable Diffusion, Imagen, Latent Diffusion Models.

---

## 7.1 Ý tưởng chính

Diffusion gồm 2 quá trình:

1. Forward diffusion
2. Reverse diffusion

---

## 7.2 Forward diffusion

Thêm noise Gaussian vào ảnh thật từng bước.

```text
x0 -> x1 -> x2 -> ... -> xT
```

Trong đó:

- `x0`: ảnh thật
- `xT`: gần như pure Gaussian noise

Ở mỗi timestep `t`, thêm một lượng noise nhỏ.

Sau nhiều bước, ảnh mất toàn bộ thông tin nhìn được.

---

## 7.3 Reverse diffusion

Train neural network học cách khử noise từng bước.

```text
xT -> xT-1 -> ... -> x0
```

Khi inference:

1. Bắt đầu từ noise ngẫu nhiên.
2. Denoise từng bước.
3. Tạo ảnh mới.

---

## 7.4 Vì sao diffusion ổn định hơn GAN?

GAN dùng adversarial training nên dễ unstable, mode collapse.

Diffusion thường train theo objective dạng denoising/likelihood-based, ổn định hơn và đa dạng hơn.

Tuy nhiên diffusion có nhược điểm lớn:

```text
Inference chậm hơn GAN
```

vì phải denoise nhiều bước.

---

# 8. Các nhánh chính của diffusion models

Unit đề cập 3 framework lớn.

---

## 8.1 DDPM — Denoising Diffusion Probabilistic Models

DDPM dùng latent variables để mô hình hóa phân phối xác suất.

Có thể xem như một dạng đặc biệt của VAE:

- Forward diffusion giống encoding.
- Reverse diffusion giống decoding.

DDPM học reverse process để sinh ảnh từ noise.

---

## 8.2 NCSN — Noise Conditioned Score Networks

NCSN học score function:

```text
∇x log p(x)
```

Tức gradient của log density.

Model được train với nhiều noise levels khác nhau.

Ý tưởng: biết hướng nào trong không gian dữ liệu làm tăng probability density, từ đó dần move từ noise về data manifold.

---

## 8.3 SDE — Stochastic Differential Equations

SDE mô hình hóa diffusion bằng phương trình vi phân ngẫu nhiên.

Đây là cách tổng quát hơn, bao phủ được cả DDPM và NCSN.

Ưu điểm:

- Lý thuyết mạnh.
- Có thể dẫn đến sampling strategy hiệu quả hơn.

---

# 9. Ứng dụng của diffusion models

Diffusion dùng cho rất nhiều task:

- Text-to-image
- Image generation
- Super-resolution
- Inpainting
- Image editing
- Image-to-image translation
- Background/attribute editing
- Segmentation representation
- Classification representation
- Anomaly detection

---

# 10. Hạn chế của diffusion models

## 10.1 Inference chậm

Vì phải chạy nhiều denoising steps.

Ví dụ DDPM gốc có thể dùng `T = 1000` bước.

Các phương pháp mới như Latent Consistency Models cố gắng giảm số bước inference.

---

## 10.2 Khó render text trong ảnh

Một số mô hình text-to-image dùng CLIP embeddings.

CLIP tốt cho semantic alignment, nhưng không encode spelling chi tiết tốt.

Vì vậy diffusion model thường gặp lỗi khi sinh chữ trong ảnh:

- Chữ méo
- Sai spelling
- Không đọc được

---

# 11. Stable Diffusion

Stable Diffusion là mô hình generative AI sinh ảnh photorealistic từ text hoặc image prompt.

Ra mắt năm 2022, dựa trên Latent Diffusion Models.

---

# 11.1 Thành phần chính của Stable Diffusion

Stable Diffusion gồm 3 thành phần cực kỳ quan trọng:

1. VAE
2. U-Net denoiser
3. CLIP text encoder

---

## 11.2 VAE trong Stable Diffusion

Ảnh pixel-space rất lớn, xử lý trực tiếp tốn compute.

Self-attention đặc biệt đắt vì complexity tăng theo bình phương số token/pixel.

Ví dụ:

- Ảnh 64x64 có 4096 pixel.
- Ảnh 128x128 có số pixel gấp 4.
- Attention có thể tốn xấp xỉ 16 lần compute/memory.

Stable Diffusion giải quyết bằng cách không diffusion trực tiếp trên pixel image, mà trên latent representation.

```text
image -> VAE Encoder -> latent
latent -> diffusion process
latent -> VAE Decoder -> image
```

VAE nén ảnh thành latent nhỏ hơn nhưng vẫn giữ thông tin quan trọng.

Đây là lý do gọi là Latent Diffusion.

---

## 11.3 CLIP text encoder

Prompt được tokenizer rồi đưa qua CLIP text encoder.

Với SD 1.x:

```text
embedding dimension: 768
```

Với SD 2.x:

```text
embedding dimension: 1024
```

Prompt thường được pad/truncate về:

```text
77 tokens
```

Conditioning tensor có dạng gần như:

```text
77 x 1024
```

hoặc tùy version.

---

## 11.4 U-Net denoiser

U-Net nhận noisy latent tại timestep `t` và dự đoán noise hoặc denoised latent.

Input chính:

- Noisy latent
- Timestep
- Text conditioning

Output:

- Noise prediction hoặc denoising signal

---

## 11.5 Cross-attention

Cross-attention là cách U-Net dùng text prompt.

Ở nhiều layer trong U-Net, mỗi spatial location trong latent có thể “attend” tới token trong prompt.

Ví dụ prompt:

```text
"a red car on a snowy mountain"
```

Một số vùng ảnh attend tới:

- `red`
- `car`
- `snowy`
- `mountain`

Nhờ đó model đưa thông tin text vào quá trình denoise.

---

# 12. Dùng Stable Diffusion với Diffusers

Cài thư viện:

```bash
pip install diffusers
```

---

## 12.1 Text-to-image

```python
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

generator = torch.Generator(device="cuda").manual_seed(31)

image = pipeline(
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    generator=generator,
).images[0]

image.save("astronaut_jungle.png")
```

Điểm kỹ thuật:

- `torch_dtype=torch.float16` giảm memory.
- `.to("cuda")` chạy trên GPU.
- `manual_seed` giúp reproducibility.

---

## 12.2 Image-to-image

Image-to-image nhận ảnh gốc và prompt.

```python
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder",
    torch_dtype=torch.float16,
    use_safetensors=True,
)

pipeline.enable_model_cpu_offload()

init_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
)

prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

image = pipeline(prompt, image=init_image).images[0]

grid = make_image_grid([init_image, image], rows=1, cols=2)
grid.save("cat_wizard.png")
```

Ý nghĩa:

- Ảnh đầu vào giữ một phần structure.
- Prompt điều khiển style/content mới.

---

## 12.3 Inpainting

Inpainting cần:

1. Ảnh gốc
2. Mask image
3. Prompt
4. Optional negative prompt

```python
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint",
    torch_dtype=torch.float16,
)

pipeline.enable_model_cpu_offload()

init_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
)

mask_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"
)

prompt = "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
negative_prompt = "bad anatomy, deformed, ugly, disfigured"

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    mask_image=mask_image,
).images[0]

grid = make_image_grid([init_image, mask_image, image], rows=1, cols=3)
grid.save("inpainting_result.png")
```

Negative prompt dùng để nói model tránh gì.

Ví dụ:

```text
bad anatomy, blurry, ugly, distorted
```

---

# 13. Kiểm soát Diffusion Models

Diffusion model mạnh nhưng đôi khi khó sinh đúng subject/style mong muốn.

Unit nói về 3 kỹ thuật:

1. DreamBooth
2. LoRA
3. ControlNet

---

# 13.1 DreamBooth

DreamBooth là kỹ thuật fine-tune diffusion model bằng vài ảnh của một subject hoặc style cụ thể.

Ví dụ có 4 ảnh của một con chó, DreamBooth học token đặc biệt đại diện cho con chó đó.

Prompt sau fine-tune:

```text
"a photo of sks dog in a park"
```

Trong đó `sks dog` là identifier gắn với subject.

Ưu điểm:

- Cá nhân hóa model.
- Sinh subject cụ thể trong nhiều bối cảnh.

Nhược điểm:

- Fine-tune toàn bộ diffusion model tốn compute.
- Dễ overfit nếu dữ liệu ít hoặc train quá lâu.

---

# 13.2 LoRA — Low-Rank Adaptation

LoRA là kỹ thuật fine-tune hiệu quả.

Thay vì update toàn bộ weight matrix `W`, LoRA học update dạng low-rank:

```text
ΔW = A B
```

Trong đó:

- `A` và `B` là ma trận rank thấp.
- Model gốc được freeze.
- Chỉ train LoRA parameters.

Ưu điểm:

- File nhỏ hơn, thường khoảng vài chục đến vài trăm MB.
- Train nhanh hơn.
- Dễ chia sẻ.
- Có thể load/unload vào model gốc.

---

## Load LoRA trong Diffusers

```python
from diffusers import StableDiffusionXLPipeline
import torch

model = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionXLPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
).to("cuda")

# Load từ file local
pipe.load_lora_weights("lora_weights.safetensors")

# Hoặc load từ Hugging Face repo
pipe.load_lora_weights("ostris/crayon_style_lora_sdxl")

# Fuse LoRA vào model
pipe.fuse_lora(lora_scale=0.8)

image = pipe("a castle drawn with crayon style").images[0]
image.save("crayon_castle.png")
```

Gỡ LoRA:

```python
pipe.unfuse_lora()
```

`lora_scale` quyết định mức ảnh hưởng của LoRA.

```text
0.0: không dùng LoRA
1.0: dùng hoàn toàn
0.7–1.0: thường là vùng nên thử
```

---

# 13.3 ControlNet

ControlNet giúp điều khiển diffusion bằng một ảnh điều kiện.

Ảnh điều kiện có thể chứa:

- Edge map
- Pose skeleton
- Depth map
- Segmentation map
- Normal map
- Scribble
- Canny edges

Ví dụ:

```text
Input: edge image of a bird
Prompt: "a colorful bird"
Output: ảnh chim có shape theo edge input nhưng màu sắc/style theo prompt
```

ControlNet hữu ích vì diffusion thường thiếu consistency về bố cục.

Với ControlNet, ta ép model giữ structure cụ thể.

---

# 14. CycleGAN

CycleGAN dùng cho unpaired image-to-image translation.

---

## 14.1 Paired vs Unpaired translation

### Paired

Có cặp ảnh tương ứng:

```text
photo -> sketch
```

Mỗi ảnh photo có đúng sketch tương ứng.

---

### Unpaired

Chỉ có 2 tập ảnh riêng biệt:

```text
Set X: horse images
Set Y: zebra images
```

Không có cặp horse-zebra tương ứng.

Mục tiêu: học chuyển domain.

```text
horse -> zebra
zebra -> horse
```

---

# 14.2 Kiến trúc CycleGAN

CycleGAN có 2 Generator và 2 Discriminator.

Giả sử 2 domain là `X` và `Y`.

Generator:

```text
G: X -> Y
F: Y -> X
```

Discriminator:

```text
D_Y: phân biệt real Y và fake Y
D_X: phân biệt real X và fake X
```

---

## 14.3 Adversarial loss

`G` cố tạo ảnh `G(x)` giống domain `Y`.

`D_Y` cố phân biệt:

```text
real y
fake G(x)
```

Tương tự với `F` và `D_X`.

---

## 14.4 Cycle consistency loss

Đây là ý tưởng cốt lõi.

Nếu chuyển từ `X` sang `Y`, rồi chuyển ngược về `X`, ta nên thu lại ảnh ban đầu.

```text
x -> G(x) -> F(G(x)) ≈ x
```

Tương tự:

```text
y -> F(y) -> G(F(y)) ≈ y
```

Loss:

```text
L_cycle = ||F(G(x)) - x|| + ||G(F(y)) - y||
```

Thường dùng L1 loss.

Ý nghĩa:

- Giữ content gốc.
- Chỉ đổi style/domain.
- Tránh generator biến ảnh thành bất kỳ ảnh nào thuộc target domain.

---

## 14.5 Identity loss

Identity loss là loss tùy chọn để giữ màu hoặc đặc trưng khi ảnh đã thuộc target domain.

Ví dụ đưa ảnh ngựa vào generator `zebra -> horse`, vì ảnh đã là horse, output nên gần như không đổi.

```text
F(x) ≈ x nếu x thuộc domain X
G(y) ≈ y nếu y thuộc domain Y
```

Loss:

```text
L_identity = ||F(x) - x|| + ||G(y) - y||
```

Hữu ích khi cần bảo toàn màu sắc.

---

## 14.6 Least Squares GAN loss

CycleGAN thường dùng least-squares adversarial loss thay vì BCE loss.

Mục tiêu:

- Giảm vanishing gradient.
- Giảm mode collapse.
- Training ổn định hơn.

Discriminator không chỉ phân loại 0/1 bằng BCE, mà minimize khoảng cách bình phương tới label thật/giả.

---

## 14.7 PatchGAN Discriminator

CycleGAN dùng PatchGAN discriminator.

Thay vì đánh giá toàn bộ ảnh là real/fake, PatchGAN đánh giá từng patch.

Ưu điểm:

- Tập trung vào texture cục bộ.
- Tạo chi tiết local realistic hơn.
- Ít tham số hơn full-image discriminator.

---

## 14.8 Generator architecture

Generator trong CycleGAN lấy cảm hứng từ:

- U-Net
- DCGAN
- Residual blocks

Cấu trúc thường gồm:

```text
downsampling -> residual blocks -> upsampling
```

Residual connections giúp học identity mapping và hỗ trợ transform sâu hơn.

---

# 15. So sánh nhanh GAN, VAE, Diffusion

| Tiêu chí | GAN | VAE | Diffusion |
|---|---|---|---|
| Sinh ảnh sắc nét | Rất tốt | Trung bình | Rất tốt |
| Training stability | Khó | Dễ hơn | Ổn định hơn GAN |
| Inference speed | Nhanh | Nhanh | Chậm hơn |
| Diversity | Có thể mode collapse | Tốt | Tốt |
| Latent structure | Không đảm bảo | Tốt | Tùy kiến trúc |
| Control bằng text | Không mặc định | Không mặc định | Rất mạnh |
| Ứng dụng nổi bật | Face generation, editing | Denoising, anomaly detection | Text-to-image, inpainting, editing |

---

# 16. Vấn đề đạo đức, privacy, bias

Generative vision models mạnh nhưng có nhiều rủi ro xã hội.

---

## 16.1 Deepfake và mất niềm tin truyền thông

AI có thể tạo ảnh/video giả rất thật.

Rủi ro:

- Misinformation
- Fake news
- Manipulated political content
- Làm giảm niềm tin vào media thật

---

## 16.2 Quấy rối và bôi nhọ cá nhân

AI image editing có thể bị dùng để:

- Tạo ảnh giả của cá nhân
- Defamation
- Harassment
- Non-consensual synthetic media

---

## 16.3 Beauty standards phi thực tế

Công cụ chỉnh sửa AI có thể làm lan rộng tiêu chuẩn sắc đẹp không thực tế.

Ảnh hưởng:

- Body image
- Self-esteem
- Áp lực xã hội

---

## 16.4 Bias

Mô hình học từ dữ liệu internet có thể chứa bias.

Hệ quả:

- Sinh ảnh thiên lệch giới tính/nghề nghiệp.
- Thiếu đại diện cho một số nhóm.
- Khuếch đại stereotype.

---

## 16.5 Bản quyền và quyền nghệ sĩ

Các công ty train text-to-image model thường dùng dữ liệu scrape từ internet.

Vấn đề:

- Tác phẩm nghệ sĩ bị dùng không xin phép.
- Không credit.
- Không bồi thường.
- Có tranh chấp pháp lý.

Một hướng phản kháng là image poisoning:

- Nghệ sĩ thêm nhiễu rất nhỏ, mắt người khó thấy.
- Nếu ảnh bị scrape để train, nhiễu có thể làm giảm chất lượng học của model.

---

## 16.6 Các hướng xử lý hiện tại

### Transparency và labeling

Gắn nhãn ảnh/video do AI tạo hoặc chỉnh sửa.

---

### Fact-checking và verification

Dùng công cụ kiểm chứng media.

---

### Legal framework

Xây dựng luật về:

- Deepfake
- Consent
- Copyright
- Misuse accountability

---

### Detection models

Phát triển model phát hiện ảnh AI-generated.

Nhưng đây là cuộc đua cat-and-mouse:

```text
Generator tốt hơn -> detector phải tốt hơn -> generator lại né detector
```

---

# 17. Các điểm kỹ thuật cần nắm chắc

Nếu học Unit 5 để đi làm hoặc nghiên cứu, nên nắm rõ các điểm sau:

## Generative vs discriminative

- Discriminative học ranh giới.
- Generative học phân phối dữ liệu.

---

## GAN

Cần hiểu:

- Generator
- Discriminator
- Min-max objective
- Mode collapse
- Training instability
- Vì sao GAN sinh ảnh sắc nét
- Vì sao khó kiểm soát latent space

---

## VAE

Cần hiểu:

- Encoder/decoder
- Latent distribution
- `μ`, `σ`
- Reparameterization trick
- Reconstruction loss
- KL divergence
- Trade-off giữa reconstruction và generation

---

## StyleGAN

Cần hiểu:

- Mapping network `z -> w`
- Disentanglement
- AdaIN
- Noise injection
- Layer-wise style control
- StyleGAN1 artifact
- StyleGAN2 demodulation/progressive growing issue
- StyleGAN3 aliasing

---

## Diffusion

Cần hiểu:

- Forward process
- Reverse process
- Gaussian noise
- Denoising objective
- Vì sao training ổn định
- Vì sao inference chậm
- DDPM/NCSN/SDE khác nhau ở mức ý tưởng

---

## Stable Diffusion

Cần hiểu:

- Latent diffusion
- Vai trò VAE
- Vai trò U-Net
- Vai trò CLIP text encoder
- Cross-attention
- Text-to-image
- Image-to-image
- Inpainting
- Negative prompt

---

## Control techniques

Cần hiểu:

- DreamBooth: personalize subject bằng vài ảnh.
- LoRA: low-rank fine-tuning, nhẹ và dễ chia sẻ.
- ControlNet: dùng ảnh điều kiện để kiểm soát structure.

---

## CycleGAN

Cần hiểu:

- Unpaired image-to-image translation
- 2 generators, 2 discriminators
- Cycle consistency loss
- Identity loss
- PatchGAN
- Least-squares adversarial loss

---

# 18. Tóm tắt ngắn gọn

Unit 5 giới thiệu nền tảng của các mô hình sinh ảnh hiện đại.

- **GAN** sinh ảnh sắc nét nhưng khó train và dễ mode collapse.
- **VAE** ổn định, có latent space xác suất, nhưng ảnh thường mờ hơn.
- **StyleGAN** cải tiến GAN để kiểm soát style tốt hơn và sinh ảnh photorealistic.
- **Diffusion Models** học cách khử noise từng bước, hiện là nền tảng của text-to-image hiện đại.
- **Stable Diffusion** dùng VAE để nén ảnh vào latent space, U-Net để denoise, CLIP để hiểu text.
- **DreamBooth/LoRA/ControlNet** giúp cá nhân hóa và kiểm soát diffusion model.
- **CycleGAN** giải quyết image-to-image translation khi không có paired dataset.
- **Ethics** là phần bắt buộc phải hiểu vì generative models có thể gây deepfake, bias, vi phạm privacy và copyright.