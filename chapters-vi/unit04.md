# Unit 4 — Multimodal Text and Vision Models

Unit này nói về **mô hình đa phương thức**, đặc biệt là các mô hình kết hợp **thị giác + ngôn ngữ** như CLIP, BLIP, OWL-ViT, VQA, image captioning, text-to-image generation, v.v.

Trọng tâm chính:

1. Vì sao cần multimodal learning.
2. Vision-Language Models hoạt động như thế nào.
3. Các task phổ biến giữa ảnh và văn bản.
4. CLIP và contrastive learning.
5. BLIP cho sinh văn bản từ ảnh.
6. OWL-ViT cho open-vocabulary object detection.
7. Transfer learning cho mô hình multimodal.

---

# 1. Multimodality là gì?

**Modality** là một dạng dữ liệu hoặc một kênh thông tin.

Ví dụ:

| Modality | Dữ liệu |
|---|---|
| Vision | ảnh, video |
| Text | câu, đoạn văn, caption |
| Audio | giọng nói, âm thanh |
| Sensor | LiDAR, IMU, EEG, depth, thermal |

Con người ra quyết định bằng nhiều giác quan cùng lúc. Ví dụ, chỉ nghe tiếng động trong đêm thì khó biết chuyện gì xảy ra, nhưng khi bật đèn và nhìn thấy cửa sổ đang mở, ta hiểu được nguyên nhân. Đây là ý tưởng cốt lõi của multimodality: **nhiều nguồn thông tin bổ sung cho nhau để tạo ra hiểu biết đầy đủ hơn**.

Trong AI, mô hình unimodal chỉ xử lý một loại dữ liệu, ví dụ:

- CNN/ViT cho ảnh.
- BERT/GPT cho text.
- Wav2Vec cho audio.

Mô hình multimodal xử lý nhiều loại dữ liệu cùng lúc, ví dụ:

- Ảnh + câu hỏi → câu trả lời.
- Ảnh → caption.
- Text prompt → ảnh.
- Text query → tìm ảnh liên quan.
- Ảnh + text query → bounding box của vật thể.

---

# 2. Multimodal dataset

Một dataset multimodal chứa nhiều modality đi kèm nhau.

Ví dụ:

## Vision + Text

- **MS COCO**: ảnh + nhiều caption.
- **VQA dataset**: ảnh + câu hỏi + câu trả lời.
- **LAION-5B**: hàng tỷ cặp ảnh + text từ web.
- **Conceptual Captions**: ảnh + caption từ internet.

## Vision + Audio

- VGG-Sound.
- RAVDESS.
- AVID.

## Vision + Audio + Text

- RECOLA.
- IEMOCAP.

Điểm quan trọng: dữ liệu web thường rất lớn nhưng **nhiễu**. Ví dụ alt-text của ảnh có thể không mô tả đúng nội dung ảnh. Vì vậy các mô hình như BLIP có thêm cơ chế lọc và sinh caption để cải thiện chất lượng dữ liệu.

---

# 3. Kiến trúc tổng quát của multimodal model

Một mô hình Vision-Language thường gồm:

```text
Image ──> Vision Encoder ──┐
                            ├──> Fusion / Alignment module ──> Output
Text  ──> Text Encoder ────┘
```

Các thành phần chính:

## 3.1 Vision encoder

Dùng để biến ảnh thành vector embedding.

Có thể là:

- CNN.
- ResNet.
- Vision Transformer, ViT.
- Swin Transformer.

## 3.2 Text encoder

Dùng để biến văn bản thành embedding.

Có thể là:

- BERT-like encoder.
- Transformer encoder.
- CLIP text encoder.
- LLM embedding layer.

## 3.3 Fusion module

Kết hợp thông tin từ ảnh và text.

Có các chiến lược:

### Early fusion

Gộp thông tin từ đầu.

Ví dụ: chia ảnh thành patch, coi mỗi patch như token, rồi đưa chung với token text vào Transformer.

### Late fusion

Xử lý từng modality riêng, sau đó mới so sánh hoặc kết hợp embedding.

Ví dụ điển hình: CLIP.

### Hybrid fusion

Kết hợp cả hai cách.

---

# 4. Vision-Language Models, VLM

**Vision-Language Model** là mô hình hiểu đồng thời ảnh và ngôn ngữ.

Các VLM thường được pre-train trên lượng lớn cặp ảnh-văn bản, sau đó dùng cho downstream tasks như:

- Zero-shot classification.
- Image-text retrieval.
- Visual Question Answering.
- Image captioning.
- Object detection theo text query.
- Text-to-image generation.

---

# 5. Các mục tiêu huấn luyện VLM

Có 3 nhóm objective chính:

## 5.1 Contrastive objective

Dùng để học không gian embedding chung giữa ảnh và text.

Ý tưởng:

- Ảnh và caption đúng phải gần nhau.
- Ảnh và caption sai phải xa nhau.

Đây là cách CLIP học.

Ví dụ batch có `N` cặp ảnh-text:

```text
(image_1, text_1)
(image_2, text_2)
...
(image_N, text_N)
```

Mô hình tính similarity giữa mọi ảnh và mọi text, tạo ma trận `N x N`.

```text
          text_1  text_2  text_3
image_1    đúng     sai     sai
image_2    sai      đúng    sai
image_3    sai      sai     đúng
```

Mục tiêu là làm similarity trên đường chéo chính cao, các ô còn lại thấp.

## 5.2 Generative objective

Mô hình học cách sinh dữ liệu.

Ví dụ:

- Ảnh → caption.
- Ảnh + câu hỏi → câu trả lời.
- Text → ảnh.
- Document image → markdown/text.

Các model như BLIP, Donut, Nougat dùng hướng này.

## 5.3 Alignment objective

Căn chỉnh ảnh và text ở cấp độ global hoặc local.

Ví dụ:

- Global: toàn ảnh khớp với toàn caption.
- Local: vùng ảnh “red apple” khớp với cụm từ “red apple”.

Task như visual grounding cần alignment local rất tốt.

---

# 6. Contrastive learning và contrastive loss

Contrastive learning là kỹ thuật học representation sao cho:

- Mẫu giống nhau nằm gần nhau trong embedding space.
- Mẫu khác nhau nằm xa nhau.

Công thức trong tài liệu:

\[
L = \mathbb{1}[y_i = y_j]||x_i - x_j||^2 + \mathbb{1}[y_i \neq y_j]max(0, \epsilon - ||x_i - x_j||^2)
\]

Hiểu đơn giản:

- Nếu hai mẫu cùng loại: giảm khoảng cách giữa chúng.
- Nếu hai mẫu khác loại: tăng khoảng cách giữa chúng, ít nhất tới một margin `epsilon`.

Trong CLIP, thay vì chỉ so từng cặp, mô hình so toàn bộ ảnh và text trong batch bằng cosine similarity, rồi dùng symmetric cross-entropy loss.

---

# 7. CLIP

CLIP là viết tắt của **Contrastive Language-Image Pre-training**.

CLIP gồm hai encoder độc lập:

```text
Image ──> Image Encoder ──> Image Embedding
Text  ──> Text Encoder  ──> Text Embedding
```

Sau đó tính cosine similarity giữa image embedding và text embedding.

## 7.1 CLIP học như thế nào?

CLIP được train trên nhiều cặp ảnh-caption từ internet.

Với một batch gồm nhiều ảnh và caption tương ứng:

- Caption đúng của ảnh phải có similarity cao.
- Caption không đúng phải có similarity thấp.
- Chiều ngược lại cũng vậy: text phải gần ảnh đúng và xa ảnh sai.

Điểm mạnh của CLIP là nó không chỉ học class cố định như “cat”, “dog”, “car”, mà học từ ngôn ngữ tự nhiên. Vì vậy có thể dùng cho **zero-shot classification**.

## 7.2 Zero-shot image classification với CLIP

Ví dụ phân loại ảnh là mèo hay chó:

```python
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=image,
    return_tensors="pt",
    padding=True,
)

outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

print(probs)
```

CLIP sẽ trả xác suất ảnh phù hợp với từng prompt.

Kỹ thuật quan trọng: label nên được viết thành prompt tự nhiên, ví dụ:

```python
"a photo of a cat"
```

thường tốt hơn chỉ dùng:

```python
"cat"
```

## 7.3 Ứng dụng của CLIP

- Zero-shot image classification.
- Image-text retrieval.
- Text-to-image search.
- Image similarity search.
- Conditioning cho diffusion models.
- Là backbone cho các model khác như OWL-ViT.

## 7.4 Hạn chế của CLIP

CLIP mạnh nhưng không phải toàn năng:

- Không nhất thiết tốt hơn model fine-tune chuyên biệt.
- Nhạy với prompt.
- Có bias từ dữ liệu internet.
- Khó với reasoning phức tạp, quan hệ không gian, đếm vật thể, compositional reasoning.
- Generalization vẫn có giới hạn với domain quá khác dữ liệu train.

---

# 8. Image-text retrieval

Task này có hai chiều:

## Text-to-image retrieval

Input:

```text
"a black cat sitting on a sofa"
```

Output:

```text
Danh sách ảnh liên quan nhất
```

## Image-to-text retrieval

Input:

```text
Một ảnh
```

Output:

```text
Caption hoặc đoạn text phù hợp nhất
```

CLIP rất phù hợp vì ảnh và text được map vào cùng embedding space.

Ví dụ:

```python
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open(requests.get(
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    stream=True
).raw)

texts = ["a photo of a cat", "a photo of a dog"]

inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True
)

outputs = model(**inputs)
scores = outputs.logits_per_image.softmax(dim=1)

print(scores)
```

Với hệ thống search thật, thường làm như sau:

```text
1. Encode toàn bộ ảnh trong database thành vector.
2. Lưu vector vào vector database.
3. Khi user nhập text query, encode query thành vector.
4. Tìm ảnh có cosine similarity cao nhất.
```

---

# 9. Visual Question Answering, VQA

VQA là task:

```text
Input: ảnh + câu hỏi
Output: câu trả lời
```

Ví dụ:

```text
Image: ảnh hai con mèo
Question: "How many cats are there?"
Answer: "2"
```

Có hai dạng:

## Multiple-choice VQA

Model chọn một đáp án trong danh sách có sẵn.

## Open-ended VQA

Model sinh câu trả lời tự do.

Nhiều VQA model xem task này như classification trên tập answer phổ biến.

Ví dụ dùng BLIP-VQA:

```python
from PIL import Image
from transformers import pipeline

vqa_pipeline = pipeline(
    "visual-question-answering",
    model="Salesforce/blip-vqa-capfilt-large"
)

image = Image.open("elephant.jpeg")
question = "Is there an elephant?"

result = vqa_pipeline(image, question, top_k=1)
print(result)
```

Các model phổ biến:

- BLIP-VQA.
- ViLT.
- DePlot cho chart/plot QA.

---

# 10. Visual reasoning

Visual reasoning yêu cầu mô hình không chỉ nhận diện vật thể, mà còn suy luận quan hệ.

Ví dụ:

- “Is the cube left of the sphere?”
- “Are there more red objects than blue objects?”
- “Is the person holding an umbrella?”

Khó hơn VQA thường vì cần hiểu:

- Quan hệ không gian.
- So sánh số lượng.
- Quan hệ nguyên nhân-ngữ cảnh.
- Logic giữa text và image.

Dataset như **CLEVR** và **Winoground** được dùng để kiểm tra khả năng reasoning/compositionality.

---

# 11. Document VQA

DocVQA là task hỏi đáp trên ảnh tài liệu.

Input:

```text
Document image + question
```

Output:

```text
Answer extracted hoặc generated từ tài liệu
```

Khó hơn VQA bình thường vì model phải hiểu:

- Text trong ảnh.
- Layout.
- Bảng biểu.
- Form.
- Quan hệ vị trí giữa các trường thông tin.

Ví dụ:

```text
Question: "What is the purchase amount?"
Answer: "20,000$"
```

Các model phổ biến:

## 11.1 LayoutLM

Dùng OCR + layout information.

Nó không chỉ đọc text mà còn biết vị trí của text trên trang.

```python
from transformers import pipeline
from PIL import Image

pipe = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa"
)

image = Image.open("your-document.png")
question = "What is the purchase amount?"

print(pipe(image=image, question=question))
```

## 11.2 Donut

Donut là **OCR-free Document Understanding Transformer**.

Nó xử lý trực tiếp ảnh tài liệu, không cần OCR pipeline riêng.

Kiến trúc:

```text
Document Image ──> Vision Encoder ──> Text Decoder ──> Answer/Text
```

Ưu điểm:

- End-to-end.
- Tránh lỗi OCR trung gian.
- Tốt cho form, receipt, document QA.

## 11.3 Nougat

Nougat chuyên đọc paper/PDF khoa học và chuyển thành markup/markdown.

Đặc biệt hữu ích với:

- Công thức toán.
- Bảng.
- Cấu trúc paper.
- Scanned scientific documents.

---

# 12. Image captioning

Image captioning là task:

```text
Input: ảnh
Output: caption mô tả ảnh
```

Ví dụ:

```text
Ảnh: cầu thủ nhảy bắt bóng
Caption: "a soccer player jumping to catch the ball"
```

Mô hình thường gồm:

```text
Image Encoder ──> Visual Embedding ──> Text Decoder ──> Caption
```

Các model phổ biến:

## 12.1 ViT-GPT2

Kết hợp:

- ViT làm image encoder.
- GPT-2 làm text decoder.

```python
from transformers import pipeline

image_to_text = pipeline(
    "image-to-text",
    model="nlpconnect/vit-gpt2-image-captioning"
)

result = image_to_text(
    "https://ankur3107.github.io/assets/images/image-captioning-example.png"
)

print(result)
```

## 12.2 BLIP Image Captioning

BLIP mạnh hơn vì được pre-train trên vision-language tasks và có cơ chế lọc dữ liệu.

```python
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)

img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)

caption = processor.decode(out[0], skip_special_tokens=True)
print(caption)
```

---

# 13. BLIP

BLIP là **Bootstrapping Language-Image Pre-training**.

CLIP chủ yếu học alignment ảnh-text. BLIP mở rộng sang các tác vụ **sinh văn bản**, như:

- Image captioning.
- VQA.
- Image-text retrieval.
- Text generation grounded on image.

## 13.1 Vấn đề BLIP giải quyết

Dữ liệu web rất lớn nhưng noisy.

Ví dụ alt-text của ảnh có thể là:

```text
"IMG_2022 holiday sale click here"
```

không mô tả đúng nội dung ảnh.

BLIP dùng cơ chế **CapFilt**, gồm:

```text
Captioner: sinh caption tốt hơn cho ảnh
Filter: lọc cặp ảnh-text nhiễu
```

Cách này cho thấy chất lượng dữ liệu quan trọng hơn chỉ tăng kích thước dataset.

## 13.2 Kiến trúc BLIP

BLIP gồm:

### Vision Transformer

Encode ảnh thành visual embedding.

### Unimodal Text Encoder

Giống BERT, dùng để encode text riêng.

### Image-Grounded Text Encoder

Dùng cross-attention để text nhìn vào image embedding.

### Image-Grounded Text Decoder

Sinh text theo kiểu autoregressive, dùng cho captioning hoặc VQA.

## 13.3 BLIP-2

BLIP-2 tiếp tục phát triển BLIP, thường kết nối vision encoder với language model lớn.

Ví dụ VQA với BLIP-2:

```python
from PIL import Image
import requests
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16
).to(device)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "Question: How many remotes are there? Answer:"

inputs = processor(
    images=image,
    text=prompt,
    return_tensors="pt"
).to(device, torch.float16)

outputs = model.generate(**inputs)

answer = processor.tokenizer.batch_decode(
    outputs,
    skip_special_tokens=True
)

print(answer)
```

---

# 14. Visual grounding

Visual grounding là task:

```text
Input: ảnh + text query
Output: vùng ảnh tương ứng, thường là bounding box hoặc mask
```

Ví dụ:

```text
Query: "the red apple in the bowl"
Output: bounding box quanh quả táo đỏ
```

Nó yêu cầu model hiểu cả:

- Ngôn ngữ.
- Vị trí vật thể.
- Quan hệ giữa các đối tượng.
- Mapping giữa cụm từ và vùng ảnh.

---

# 15. OWL-ViT

OWL-ViT là model cho **open-vocabulary object detection**.

Object detection truyền thống như YOLO thường chỉ phát hiện các class đã được train, ví dụ:

```text
person, car, dog, cat, bicycle...
```

OWL-ViT có thể phát hiện object bằng text query tự do, ví dụ:

```text
"cat tail"
"remote control"
"red backpack"
"open laptop"
```

Ngay cả khi class đó không nằm trong tập label detector truyền thống.

## 15.1 OWL-ViT hoạt động thế nào?

Nó bắt đầu giống CLIP:

```text
Image Encoder + Text Encoder + Contrastive Learning
```

Sau đó fine-tune cho object detection.

Khác với CLIP chỉ tạo một embedding cho toàn ảnh, OWL-ViT tạo embedding cho từng image token/patch.

```text
Image patches ──> token embeddings ──> object embeddings
                                    ├── classification với text query
                                    └── box prediction bằng MLP
```

## 15.2 Dùng OWL-ViT

```python
import requests
from PIL import Image, ImageDraw
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

texts = [["a photo of a cat", "a photo of a dog", "remote control", "cat tail"]]

inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt"
)

outputs = model(**inputs)

target_sizes = torch.Tensor([image.size[::-1]])

results = processor.post_process_object_detection(
    outputs=outputs,
    target_sizes=target_sizes,
    threshold=0.1
)

boxes = results[0]["boxes"]
scores = results[0]["scores"]
labels = results[0]["labels"]

draw = ImageDraw.Draw(image)

for box, score, label in zip(boxes, scores, labels):
    box = [round(x, 2) for x in box.tolist()]
    print(
        f"Detected {texts[0][label]} "
        f"with confidence {round(score.item(), 3)} "
        f"at location {box}"
    )
    draw.rectangle(box, outline="red")

image
```

Điểm quan trọng:

- `texts` là list các query.
- Model trả về box, score, label.
- `post_process_object_detection` scale box về kích thước ảnh gốc.
- Threshold thấp giúp thấy nhiều detection hơn nhưng dễ tăng false positive.

---

# 16. Text-to-image generation

Task:

```text
Input: text prompt
Output: image
```

Có hai nhóm mô hình chính.

## 16.1 Autoregressive models

Xem ảnh như chuỗi image tokens.

Quy trình:

```text
Text prompt ──> Encoder
Image tokenizer ──> image tokens
Decoder sinh từng image token một
```

Tương tự language model sinh từng token text.

Ưu điểm:

- Có thể kiểm soát theo chuỗi.
- Tự nhiên với framework Transformer.

Nhược điểm:

- Chậm.
- Khó với ảnh độ phân giải cao.
- Chuỗi image token dài.

## 16.2 Diffusion models

Stable Diffusion thuộc nhóm này.

Ý tưởng:

```text
Noise ──> denoise dần dần dưới điều kiện text prompt ──> image
```

Stable Diffusion dùng **latent diffusion**:

- Không denoise trực tiếp trong pixel space.
- Denoise trong latent space để tiết kiệm bộ nhớ và compute.

Thành phần chính:

```text
Text prompt ──> CLIP text encoder ──┐
                                    ├── UNet denoising in latent space ──> VAE decoder ──> Image
Noise latent ───────────────────────┘
```

Ví dụ dùng Diffusers:

```bash
pip install diffusers transformers accelerate safetensors invisible_watermark
```

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

pipe.to("cuda")

prompt = "An astronaut riding a unicorn"

image = pipe(prompt=prompt).images[0]
image.save("output.png")
```

---

# 17. Các chiến lược xây dựng VLM

Tài liệu nêu một số hướng chính.

## 17.1 Biến ảnh thành token như text token

Ảnh được chia thành patches, mỗi patch như một token.

```text
[image_patch_1, image_patch_2, ..., text_token_1, text_token_2]
```

Sau đó đưa vào Transformer chung.

Ví dụ:

- VisualBERT.
- SimVLM.

## 17.2 Image embedding làm prefix cho language model

Giữ LLM đóng băng, chỉ học cách map ảnh vào embedding space tương thích với LLM.

Ví dụ:

- Frozen.
- ClipCap.

Ý tưởng:

```text
Image ──> learned prefix embeddings ──> Frozen LM ──> generated text
```

Ưu điểm:

- Ít train hơn.
- Tận dụng LLM mạnh.

## 17.3 Cross-attention để fuse ảnh vào language model

Text decoder hoặc LM layer dùng cross-attention nhìn vào visual embeddings.

Ví dụ:

- VisualGPT.
- BLIP-like architectures.

## 17.4 Không train thêm

Một số phương pháp dùng guided decoding hoặc kết hợp model có sẵn mà không fine-tune.

Ví dụ: MAGiC.

---

# 18. Transfer learning trong multimodal models

Có 3 cách dùng model cho task mới:

## 18.1 Zero-shot / few-shot learning

Dùng model pretrained lớn mà không fine-tune hoặc chỉ dùng rất ít ví dụ.

Ví dụ:

- CLIP zero-shot classification.
- OWL-ViT open-vocabulary detection.
- BLIP-2 VQA với prompt.

Phù hợp khi:

- Không có label.
- Có rất ít dữ liệu.
- Cần prototype nhanh.

## 18.2 Train from scratch

Khởi tạo trọng số ngẫu nhiên rồi train từ đầu.

Chỉ nên dùng khi:

- Không có pretrained model phù hợp.
- Domain quá khác biệt.
- Có rất nhiều dữ liệu và compute.

Nhược điểm:

- Tốn dữ liệu.
- Tốn GPU.
- Dễ underperform nếu data không đủ.

## 18.3 Transfer learning / fine-tuning

Dùng pretrained weights làm khởi tạo rồi fine-tune trên task riêng.

Ưu điểm:

- Cần ít data hơn.
- Hội tụ nhanh hơn.
- Tận dụng knowledge đã học.

Rủi ro:

### Domain shift

Dữ liệu mới khác dữ liệu gốc quá nhiều.

Ví dụ model train ảnh internet nhưng fine-tune trên ảnh y tế hoặc ảnh vệ tinh.

### Catastrophic forgetting

Khi fine-tune, model có thể quên kiến thức cũ.

Cách giảm:

- Learning rate nhỏ.
- Freeze một phần backbone.
- LoRA/adapters.
- Mix dữ liệu cũ và mới nếu cần.
- Early stopping.

---

# 19. Các task fine-tuning phổ biến

| Task | Model ví dụ |
|---|---|
| Fine-tune CLIP | `openai/clip-vit-base-patch32` |
| VQA | `dandelin/vilt-b32-mlm` |
| Image captioning | `microsoft/git-base` |
| Open-set object detection | YOLO-World |
| Multimodal assistant | LLaVA |

---

# 20. Các điểm kỹ thuật quan trọng cần nhớ

## 20.1 Embedding space chung là nền tảng của nhiều VLM

CLIP-style models học cách đưa ảnh và text vào cùng không gian vector.

Điều này cho phép:

- So sánh ảnh và text bằng cosine similarity.
- Search đa phương thức.
- Zero-shot classification.
- Retrieval.

## 20.2 Prompt ảnh hưởng lớn tới kết quả

Với CLIP, prompt:

```text
"a photo of a cat"
```

thường tốt hơn:

```text
"cat"
```

Với domain đặc thù, prompt engineering rất quan trọng.

Ví dụ medical image:

```text
"a chest x-ray showing pneumonia"
```

có thể khác nhiều với:

```text
"pneumonia"
```

## 20.3 Không phải thêm modality lúc nào cũng tốt

Thêm modality có thể:

- Tăng thông tin.
- Nhưng cũng tăng noise.
- Một modality có thể dominate modality khác.
- Có thể thiếu dữ liệu ở một modality.
- Quan hệ giữa modality có thể phức tạp.

## 20.4 Dữ liệu web lớn nhưng noisy

Model như BLIP nhấn mạnh:

```text
data quality > raw data quantity
```

Cơ chế CapFilt là ví dụ tốt: sinh caption + lọc caption để cải thiện pretraining.

## 20.5 Zero-shot tốt nhưng không thay thế fine-tuning

CLIP/OWL-ViT mạnh khi không có dữ liệu label, nhưng model fine-tuned vẫn thường tốt hơn trên task/domain cụ thể.

## 20.6 Evaluation phải kiểm tra reasoning thật

Dataset dễ như COCO có thể không đủ để chứng minh model “hiểu”.

Các benchmark khó hơn:

- CLEVR.
- Hateful Memes.
- Winoground.
- NoCaps.

Chúng kiểm tra:

- Generalization.
- Compositionality.
- Reasoning.
- Out-of-domain concepts.

---

# 21. Khi nào nên dùng model nào?

## Cần phân loại ảnh zero-shot

Dùng CLIP.

```text
Image + list prompt labels → probabilities
```

## Cần tìm ảnh bằng text

Dùng CLIP/OpenCLIP + vector database.

```text
Text query → embedding → nearest image embeddings
```

## Cần hỏi đáp trên ảnh

Dùng BLIP/BLIP-2/ViLT.

```text
Image + question → answer
```

## Cần caption ảnh

Dùng BLIP image captioning, GIT, ViT-GPT2.

```text
Image → caption
```

## Cần hỏi đáp trên tài liệu scan

Dùng LayoutLM, Donut, Nougat.

```text
Document image + question → answer
```

## Cần detect object bằng mô tả text

Dùng OWL-ViT hoặc Grounding DINO.

```text
Image + "red backpack" → bounding box
```

## Cần sinh ảnh từ text

Dùng Stable Diffusion / SDXL qua Diffusers.

```text
Prompt → generated image
```

---

# 22. Tóm tắt ngắn gọn

Unit 4 giới thiệu cách AI kết hợp ảnh và văn bản. Ý tưởng cốt lõi là học representation chung hoặc học cách fuse nhiều modality để giải quyết các task mà unimodal model làm kém.

Các model cần nhớ:

- **CLIP**: image-text alignment, contrastive learning, zero-shot classification, retrieval.
- **BLIP**: vision-language generation, captioning, VQA, CapFilt để xử lý dữ liệu noisy.
- **BLIP-2**: kết nối vision model với language model lớn.
- **OWL-ViT**: open-vocabulary object detection bằng text query.
- **LayoutLM/Donut/Nougat**: document understanding.
- **Stable Diffusion**: text-to-image generation bằng latent diffusion.

Các khái niệm cần nắm chắc:

- Modality.
- Multimodal fusion.
- Vision encoder / text encoder.
- Shared embedding space.
- Contrastive loss.
- Zero-shot learning.
- Fine-tuning / transfer learning.
- Domain shift.
- Catastrophic forgetting.
- Open-vocabulary detection.
- Visual grounding.
- Image-text retrieval.