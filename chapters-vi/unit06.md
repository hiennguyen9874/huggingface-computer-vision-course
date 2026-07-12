# Unit 6 — Các tác vụ Computer Vision cơ bản

Unit này nhằm giúp hiểu rõ các bài toán nền tảng trong thị giác máy tính:

1. **Image Classification**: phân loại ảnh.
2. **Object Detection**: phát hiện và định vị vật thể.
3. **Image Segmentation**: phân vùng ảnh theo pixel.

Mỗi task khác nhau ở mức độ chi tiết của đầu ra:

| Task | Đầu vào | Đầu ra |
|---|---|---|
| Image Classification | Một ảnh | Một hoặc nhiều nhãn cho toàn ảnh |
| Object Detection | Một ảnh | Bounding box + class label + confidence |
| Image Segmentation | Một ảnh | Mask theo từng pixel |

---

# 1. Image Classification

## Mục tiêu

Image Classification là bài toán xác định ảnh thuộc lớp nào.

Ví dụ:

- Ảnh chứa chó → `dog`
- Ảnh X-quang → `pneumonia` hoặc `normal`
- Ảnh sản phẩm → `shoe`, `shirt`, `bag`

Đây là task đơn giản nhất trong ba task vì model chỉ cần trả lời:

> “Ảnh này là gì?”

Nó không cần biết vật thể nằm ở đâu trong ảnh.

## Đầu ra kỹ thuật

Model thường trả về vector xác suất:

```text
cat: 0.02
dog: 0.93
horse: 0.05
```

Class có xác suất cao nhất là dự đoán cuối cùng.

## Code ví dụ với Hugging Face

```python
from transformers import pipeline
from PIL import Image

classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224"
)

image = Image.open("image.jpg").convert("RGB")

result = classifier(image)

print(result)
```

Ví dụ output:

```python
[
    {"label": "golden retriever", "score": 0.91},
    {"label": "Labrador retriever", "score": 0.05}
]
```

## Điểm kỹ thuật cần nắm

Image classification thường dùng:

- CNN: ResNet, EfficientNet, ConvNeXt.
- Vision Transformer: ViT, DeiT, Swin Transformer.

Các metric phổ biến:

- Accuracy.
- Top-1 Accuracy.
- Top-5 Accuracy.
- Precision / Recall / F1 nếu dữ liệu mất cân bằng.

---

# 2. Object Detection

## Mục tiêu

Object Detection là bài toán vừa:

1. **Nhận diện vật thể là gì**.
2. **Xác định vị trí vật thể trong ảnh**.

Nói cách khác, object detection kết hợp:

- **Classification**: vật thể thuộc class nào.
- **Localization**: vật thể nằm ở đâu.

Ví dụ với ảnh đường phố, model có thể trả về:

```text
car: box = [50, 80, 200, 160], score = 0.98
person: box = [300, 70, 350, 220], score = 0.94
traffic light: box = [420, 30, 450, 90], score = 0.87
```

## Bounding Box

Object detection thường biểu diễn vị trí vật thể bằng **bounding box**.

Có nhiều format:

### Format 1: `[x_min, y_min, x_max, y_max]`

```text
x_min, y_min: góc trên trái
x_max, y_max: góc dưới phải
```

### Format 2: `[x_center, y_center, width, height]`

Thường gặp trong YOLO.

Ví dụ:

```python
box = {
    "xmin": 50,
    "ymin": 80,
    "xmax": 200,
    "ymax": 160
}
```

## Object Detection khác Classification thế nào?

| Classification | Object Detection |
|---|---|
| Chỉ biết ảnh có gì | Biết ảnh có gì và ở đâu |
| Một nhãn cho toàn ảnh | Nhiều object trong cùng ảnh |
| Output đơn giản | Output gồm box, label, score |
| Không xử lý vị trí | Cần localization chính xác |

Ví dụ ảnh có 3 quả táo và 2 quả cam:

- Classification: `fruit`
- Object Detection: phát hiện từng quả táo/cam với box riêng.

---

# 3. Ứng dụng của Object Detection

Object detection được dùng nhiều trong thực tế:

## Xe tự lái

Phát hiện:

- Người đi bộ.
- Xe khác.
- Đèn giao thông.
- Biển báo.
- Làn đường.

## Giám sát an ninh

Phát hiện:

- Người xâm nhập.
- Hành vi bất thường.
- Vật thể bị bỏ quên.

## Y tế

Phát hiện:

- Khối u.
- Vùng tổn thương.
- Dị vật.
- Bất thường trong ảnh X-ray, MRI, CT.

## Sản xuất công nghiệp

Phát hiện:

- Lỗi bề mặt.
- Sản phẩm sai hình dạng.
- Thiếu linh kiện.

## AR / VR

Phát hiện vật thể ngoài đời thực để chồng thông tin ảo lên đúng vị trí.

---

# 4. Code Object Detection với Transformers

Ví dụ trong nội dung dùng model `facebook/detr-resnet-50`.

DETR là viết tắt của **DEtection TRansformer**.

```python
from transformers import pipeline
from PIL import Image

pipe = pipeline(
    "object-detection",
    model="facebook/detr-resnet-50"
)

image = Image.open("path/to/your/image.jpg").convert("RGB")

bounding_boxes = pipe(image)

print(bounding_boxes)
```

Output mẫu:

```python
[
    {
        "score": 0.998,
        "label": "person",
        "box": {
            "xmin": 100,
            "ymin": 50,
            "xmax": 250,
            "ymax": 400
        }
    },
    {
        "score": 0.965,
        "label": "dog",
        "box": {
            "xmin": 300,
            "ymin": 120,
            "xmax": 500,
            "ymax": 380
        }
    }
]
```

## Ý nghĩa các trường

```python
"score"
```

Độ tự tin của model.

```python
"label"
```

Class dự đoán.

```python
"box"
```

Bounding box của object.

---

# 5. Đánh giá Object Detection

Object Detection khó đánh giá hơn Classification vì phải đánh giá cả:

1. Class đúng không.
2. Box có khớp với ground truth không.
3. Có bỏ sót object không.
4. Có detect nhầm không.

Hai metric quan trọng nhất:

---

## 5.1 Intersection over Union — IoU

IoU đo độ chồng lấp giữa bounding box dự đoán và bounding box thật.

Công thức:

```text
IoU = Area of Intersection / Area of Union
```

Trong đó:

- Intersection: phần giao nhau giữa predicted box và ground-truth box.
- Union: tổng vùng bao phủ bởi cả hai box.

Ví dụ:

```text
IoU = 1.0  → box trùng hoàn toàn
IoU = 0.5  → box chồng lấp vừa đủ
IoU = 0.0  → không chồng lấp
```

Thông thường, một detection được coi là đúng nếu:

```text
IoU >= 0.5
```

Hoặc trong benchmark nghiêm ngặt hơn:

```text
IoU từ 0.5 đến 0.95
```

## Code tính IoU cho bounding box

```python
def compute_iou(box_a, box_b):
    """
    box format: [xmin, ymin, xmax, ymax]
    """

    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = box_a_area + box_b_area - intersection_area

    return intersection_area / union_area
```

---

## 5.2 Mean Average Precision — mAP

mAP là metric phổ biến nhất cho object detection.

Nó kết hợp:

- **Precision**: trong các object model phát hiện, bao nhiêu cái là đúng?
- **Recall**: trong các object thật, model tìm được bao nhiêu cái?

```text
Precision = True Positives / (True Positives + False Positives)

Recall = True Positives / (True Positives + False Negatives)
```

Trong object detection:

- True Positive: detect đúng class và IoU vượt threshold.
- False Positive: detect sai hoặc box không đủ khớp.
- False Negative: object thật nhưng model bỏ sót.

mAP được tính bằng cách lấy trung bình Average Precision trên nhiều class và thường trên nhiều ngưỡng IoU.

Ví dụ COCO thường dùng:

```text
mAP@[0.5:0.95]
```

Tức là tính AP ở các threshold IoU:

```text
0.50, 0.55, 0.60, ..., 0.95
```

rồi lấy trung bình.

## Cần nhớ

- IoU đo chất lượng box.
- Precision/Recall đo chất lượng phát hiện.
- mAP tổng hợp cả localization và classification.

---

# 6. Image Segmentation

## Mục tiêu

Image Segmentation là bài toán chia ảnh thành các vùng có ý nghĩa.

Điểm cốt lõi:

> Segmentation có thể xem như classification cho từng pixel.

Thay vì hỏi:

> “Ảnh này là gì?”

Segmentation hỏi:

> “Mỗi pixel trong ảnh thuộc về class nào hoặc instance nào?”

Ví dụ trong ảnh đường phố:

- Pixel thuộc đường.
- Pixel thuộc vỉa hè.
- Pixel thuộc xe.
- Pixel thuộc người.
- Pixel thuộc cây.
- Pixel thuộc bầu trời.

## Đầu ra

Output là một hoặc nhiều **mask**.

Mask là ma trận cùng kích thước với ảnh, trong đó mỗi pixel chứa nhãn.

Ví dụ ảnh kích thước `512 x 512`, segmentation mask cũng thường là `512 x 512`.

```text
0 = background
1 = person
2 = car
3 = road
4 = sky
```

---

# 7. Các loại Image Segmentation

Nội dung unit nhấn mạnh ba loại segmentation chính:

1. Semantic Segmentation.
2. Instance Segmentation.
3. Panoptic Segmentation.

---

## 7.1 Semantic Segmentation

Semantic Segmentation gán class cho từng pixel.

Ví dụ:

- Tất cả pixel thuộc mèo → `cat`
- Tất cả pixel thuộc chó → `dog`
- Tất cả pixel thuộc nền → `background`

Điểm quan trọng:

> Semantic segmentation không phân biệt các object khác nhau trong cùng class.

Nếu ảnh có hai con mèo, cả hai đều được gán là `cat`, không có `cat_1`, `cat_2`.

Ví dụ output:

```text
pixel vùng mèo thứ nhất  → cat
pixel vùng mèo thứ hai   → cat
pixel vùng nền           → background
```

Phù hợp cho:

- Phân vùng đường, trời, cây, nhà.
- Ảnh vệ tinh.
- Ảnh y tế.
- Phân tích đất nông nghiệp.

---

## 7.2 Instance Segmentation

Instance Segmentation không chỉ biết class của từng pixel mà còn phân biệt từng object riêng biệt.

Nếu ảnh có hai con mèo:

```text
mèo thứ nhất → cat instance 1
mèo thứ hai  → cat instance 2
```

Nó kết hợp ý tưởng của:

- Object Detection: phân biệt từng object.
- Segmentation: tạo mask chính xác theo pixel.

Output thường là:

```python
[
    {
        "label": "cat",
        "score": 0.98,
        "mask": mask_for_cat_1
    },
    {
        "label": "cat",
        "score": 0.96,
        "mask": mask_for_cat_2
    }
]
```

Phù hợp cho:

- Đếm số lượng vật thể.
- Robot grasping.
- Phân tích từng tế bào trong ảnh y tế.
- Tách từng người trong ảnh.

---

## 7.3 Panoptic Segmentation

Panoptic Segmentation kết hợp Semantic và Instance Segmentation.

Nó trả lời cả hai câu hỏi:

1. Pixel này thuộc class nào?
2. Pixel này thuộc instance nào?

Ví dụ:

```text
road        → semantic class, không cần instance
sky         → semantic class, không cần instance
person #1   → instance riêng
person #2   → instance riêng
car #1      → instance riêng
```

Panoptic segmentation thường phân biệt:

## Stuff

Các vùng không đếm được hoặc không có instance rõ ràng:

- Road.
- Sky.
- Grass.
- Wall.
- Water.

## Things

Các vật thể đếm được:

- Person.
- Car.
- Dog.
- Chair.
- Bicycle.

Panoptic Segmentation hữu ích khi cần hiểu toàn bộ scene một cách đầy đủ.

---

# 8. So sánh ba loại Segmentation

| Loại | Phân loại từng pixel | Phân biệt từng object | Ví dụ |
|---|---:|---:|---|
| Semantic Segmentation | Có | Không | Tất cả mèo là `cat` |
| Instance Segmentation | Có | Có | `cat #1`, `cat #2` |
| Panoptic Segmentation | Có | Có, với object đếm được | `road`, `sky`, `person #1` |

---

# 9. Kiến trúc model cho Segmentation

## U-Net

U-Net là kiến trúc kinh điển cho segmentation, đặc biệt trong ảnh y tế.

Nó có hai pha chính:

### Downsampling path

Nén ảnh lại để học đặc trưng cấp cao.

Ví dụ:

```text
512x512 → 256x256 → 128x128 → 64x64
```

Mục tiêu:

- Học context.
- Hiểu object lớn.
- Trích xuất feature trừu tượng.

### Upsampling path

Phóng feature map trở lại kích thước ban đầu.

Ví dụ:

```text
64x64 → 128x128 → 256x256 → 512x512
```

Mục tiêu:

- Tạo mask chi tiết.
- Dự đoán class cho từng pixel.

U-Net thường dùng **skip connections** để nối feature ở encoder sang decoder, giúp giữ lại chi tiết không gian.

---

## Vision Transformer-based Segmentation

Computer Vision hiện đại đang chuyển mạnh từ CNN sang Transformer.

Một ví dụ quan trọng là:

# Segment Anything Model — SAM

SAM được Meta AI Research giới thiệu năm 2023.

Đặc điểm chính:

- Dựa trên Vision Transformer.
- Là model segmentation dạng promptable.
- Có khả năng zero-shot trên ảnh mới.
- Được train trên dataset rất lớn: hơn 1 tỷ mask trên 11 triệu ảnh.

## Promptable segmentation là gì?

Người dùng có thể đưa prompt để chỉ định vùng cần segment.

Prompt có thể là:

- Điểm.
- Bounding box.
- Mask thô.
- Một số hệ thống mở rộng có thể dùng text prompt.

Ý tưởng:

> Thay vì model tự segment mọi thứ theo class cố định, người dùng chỉ cho model biết muốn tách vùng nào.

## Zero-shot transfer

SAM có thể xử lý ảnh hoặc domain mới mà không cần fine-tune trực tiếp trên dataset đó.

---

# 10. Code dùng SAM với Transformers

Ví dụ từ nội dung:

```python
from transformers import pipeline
from PIL import Image

pipe = pipeline(
    "mask-generation",
    model="facebook/sam-vit-base",
    device=0
)

raw_image = Image.open("path/to/image").convert("RGB")

masks = pipe(raw_image)

print(masks)
```

Lưu ý:

- `device=0` dùng GPU đầu tiên.
- Nếu chạy CPU, có thể bỏ `device=0`.

```python
pipe = pipeline(
    "mask-generation",
    model="facebook/sam-vit-base"
)
```

Output thường chứa nhiều mask ứng viên.

---

# 11. Đánh giá Segmentation Model

Segmentation thường là supervised learning.

Dataset gồm:

```text
image + ground-truth mask
```

Các metric phổ biến:

1. IoU / Jaccard Index.
2. Pixel Accuracy.
3. Dice Coefficient.

---

## 11.1 IoU cho Segmentation

IoU đo độ chồng lấp giữa predicted mask và ground-truth mask.

Công thức:

```text
IoU = Intersection / Union
```

Ví dụ với binary mask:

```python
import numpy as np

def mask_iou(pred_mask, true_mask):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union
```

## Vì sao IoU quan trọng?

IoU ít bị ảnh hưởng hơn bởi class imbalance so với accuracy.

Ví dụ ảnh y tế có 99% background và 1% vùng bệnh. Model đoán toàn background có thể đạt accuracy rất cao, nhưng IoU vùng bệnh sẽ rất thấp.

---

## 11.2 Pixel Accuracy

Pixel Accuracy đo tỷ lệ pixel được phân loại đúng.

Công thức:

```text
Pixel Accuracy = Correct Pixels / Total Pixels
```

Code:

```python
def pixel_accuracy(pred_mask, true_mask):
    return (pred_mask == true_mask).sum() / true_mask.size
```

Nhược điểm lớn:

> Rất nhạy với mất cân bằng class.

Ví dụ:

- 95% pixel là background.
- Model đoán tất cả là background.
- Pixel accuracy có thể đạt 95%.
- Nhưng model hoàn toàn vô dụng với object cần phát hiện.

Vì vậy, không nên chỉ dùng Pixel Accuracy.

---

## 11.3 Dice Coefficient

Dice đo phần trăm overlap giữa prediction và ground truth.

Công thức:

```text
Dice = 2 * Intersection / (Predicted Area + Ground Truth Area)
```

Code:

```python
def dice_coefficient(pred_mask, true_mask):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    total = pred_mask.sum() + true_mask.sum()

    if total == 0:
        return 1.0

    return 2 * intersection / total
```

Dice thường được dùng nhiều trong ảnh y tế vì nhạy với vùng nhỏ.

---

# 12. IoU vs Dice

Hai metric này liên quan chặt chẽ.

| Metric | Công thức | Đặc điểm |
|---|---|---|
| IoU | `intersection / union` | Nghiêm khắc hơn |
| Dice | `2 * intersection / (pred + true)` | Nhạy với overlap, hay dùng trong medical imaging |

Nếu vùng dự đoán hơi lệch, Dice thường nhìn “dễ chịu” hơn IoU.

---

# 13. Dataset và Ground Truth

Với Object Detection, ground truth thường gồm:

```python
{
    "image": image,
    "annotations": [
        {
            "label": "person",
            "bbox": [xmin, ymin, xmax, ymax]
        },
        {
            "label": "car",
            "bbox": [xmin, ymin, xmax, ymax]
        }
    ]
}
```

Với Segmentation, ground truth gồm:

```python
{
    "image": image,
    "mask": mask
}
```

Hoặc với instance segmentation:

```python
{
    "image": image,
    "instances": [
        {
            "label": "person",
            "mask": mask_1
        },
        {
            "label": "person",
            "mask": mask_2
        }
    ]
}
```

---

# 14. Những điểm kỹ thuật quan trọng cần nhớ

## Object Detection

Cần nắm:

- Object detection = classification + localization.
- Output gồm `label`, `score`, `box`.
- Bounding box thường ở dạng `[xmin, ymin, xmax, ymax]` hoặc `[x_center, y_center, width, height]`.
- IoU dùng để đo box có khớp không.
- mAP là metric tổng hợp quan trọng nhất.
- DETR là object detection model dựa trên Transformer.

## Segmentation

Cần nắm:

- Segmentation là classification ở mức pixel.
- Output là mask.
- Semantic segmentation không phân biệt instance.
- Instance segmentation phân biệt từng object.
- Panoptic segmentation kết hợp semantic + instance.
- U-Net là kiến trúc kinh điển encoder-decoder.
- SAM là model segmentation hiện đại dựa trên ViT, có prompt và zero-shot tốt.
- IoU, Pixel Accuracy, Dice là metric chính.
- Pixel Accuracy có thể gây hiểu nhầm nếu class imbalance.

---

# 15. Tóm tắt ngắn gọn

```text
Image Classification:
    Ảnh này là gì?

Object Detection:
    Trong ảnh có gì và nằm ở đâu?

Semantic Segmentation:
    Mỗi pixel thuộc class nào?

Instance Segmentation:
    Mỗi pixel thuộc object instance nào?

Panoptic Segmentation:
    Mỗi pixel thuộc class nào và instance nào nếu có?
```

Nếu mới học, thứ tự nên nắm là:

1. Image Classification.
2. Object Detection.
3. Semantic Segmentation.
4. Instance Segmentation.
5. Panoptic Segmentation.
6. Metrics: IoU, mAP, Dice.