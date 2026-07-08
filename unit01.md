# Unit 1 — Tổng quan Computer Vision

## 1. Computer Vision là gì?

**Computer Vision — thị giác máy tính** là lĩnh vực nghiên cứu cách làm cho máy tính “nhìn”, xử lý, phân tích và hiểu dữ liệu hình ảnh hoặc video.

Nói kỹ hơn, Computer Vision bao gồm các phương pháp để:

1. **Thu nhận dữ liệu thị giác**  
   Ví dụ: ảnh từ camera, ảnh y tế, ảnh vệ tinh, video, ảnh hồng ngoại.

2. **Xử lý dữ liệu ảnh**  
   Ví dụ: resize, lọc nhiễu, tăng tương phản, chuẩn hóa pixel.

3. **Phân tích ảnh**  
   Ví dụ: tìm cạnh, phát hiện vùng, trích xuất đặc trưng.

4. **Hiểu nội dung ảnh**  
   Ví dụ: nhận biết vật thể, phân đoạn ảnh, mô tả ảnh bằng văn bản.

5. **Ra quyết định dựa trên ảnh**  
   Ví dụ: xe tự lái phanh khi thấy người đi bộ.

---

## 2. Vì sao Computer Vision khó?

Con người nhìn và hiểu cảnh vật rất tự nhiên, nhưng với máy tính thì cực kỳ khó.

Ví dụ: “nhận ra một quả bóng”.

Nếu lập trình bằng luật cứng:

```text
Nếu vật thể tròn thì là quả bóng
```

Luật này sai ngay vì:

- Không phải quả bóng nào cũng tròn hoàn hảo, ví dụ bóng rugby.
- Không phải vật tròn nào cũng là bóng, ví dụ kẹo, bong bóng, hành tinh.
- Ngữ cảnh ảnh hưởng đến nhận thức: quả bóng trong sân thể thao khác với tảng đá tròn trong hang động.

Vấn đề chính là hình ảnh chứa rất nhiều biến thiên:

- Góc nhìn khác nhau.
- Ánh sáng khác nhau.
- Vật thể bị che khuất.
- Kích thước thay đổi.
- Nền phức tạp.
- Nhiễu từ camera.
- Chất lượng ảnh thấp.
- Ngữ cảnh thay đổi.

Vì vậy, Computer Vision thường cần mô hình có khả năng học từ dữ liệu thay vì chỉ dùng luật thủ công.

---

# 3. Computer Vision và Deep Learning

Trước Deep Learning, pipeline thị giác máy tính thường là:

```text
Ảnh gốc
→ Tiền xử lý ảnh
→ Trích xuất đặc trưng thủ công
→ Mô hình Machine Learning cổ điển
→ Dự đoán
```

Ví dụ:

```text
Ảnh khuôn mặt
→ Tìm cạnh, texture, histogram màu
→ SVM / Random Forest
→ Nhận diện người
```

Nhược điểm:

- Phụ thuộc nhiều vào kiến thức chuyên gia.
- Cần thiết kế đặc trưng thủ công.
- Khó tổng quát với dữ liệu phức tạp.

Với Deep Learning, pipeline thường là:

```text
Ảnh gốc
→ Neural Network
→ Mô hình tự học đặc trưng
→ Dự đoán
```

Đặc biệt, **CNN — Convolutional Neural Network** giúp mô hình tự học các đặc trưng như:

- Cạnh.
- Góc.
- Texture.
- Hình dạng.
- Bộ phận vật thể.
- Toàn bộ vật thể.

Điểm quan trọng: Deep Learning không loại bỏ hoàn toàn tiền xử lý ảnh. Trong thực tế vẫn thường cần:

- Resize ảnh.
- Normalize pixel.
- Data augmentation.
- Lọc nhiễu nếu dữ liệu quá xấu.
- Chuyển đổi định dạng hoặc channel.

---

# 4. Các mức độ hiểu ảnh

Nội dung phân biệt ba cấp độ xử lý ảnh.

## 4.1. Low-level vision

Đây là các thao tác cơ bản trên ảnh.

Input là ảnh, output cũng là ảnh.

Ví dụ:

- Tăng độ sáng.
- Tăng tương phản.
- Làm mờ ảnh.
- Làm nét ảnh.
- Lọc nhiễu.
- Phát hiện cạnh.

Ví dụ với OpenCV:

```python
import cv2

img = cv2.imread("image.jpg")

# Làm mờ Gaussian
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Tìm cạnh Canny
edges = cv2.Canny(img, 100, 200)
```

---

## 4.2. Mid-level vision

Ở mức này, hệ thống bắt đầu rút ra thông tin có cấu trúc từ ảnh.

Ví dụ:

- Segmentation: tách vùng ảnh.
- Object classification: phân loại vật thể.
- Feature extraction: trích xuất đặc trưng.
- Object description: mô tả vùng/vật thể.

Output không nhất thiết là ảnh, mà có thể là:

```text
Ảnh → Danh sách vùng vật thể
Ảnh → Vector đặc trưng
Ảnh → Nhãn lớp
```

---

## 4.3. High-level vision

Đây là mức gần với nhận thức của con người.

Ví dụ:

- Hiểu toàn cảnh.
- Nhận diện hành động.
- Mô tả ảnh bằng ngôn ngữ tự nhiên.
- Dự đoán tình huống.
- Tái dựng cảnh 3D.
- Điều hướng robot.

Ví dụ:

```text
Ảnh → "Một con mèo đang nằm trên ghế sofa"
```

---

# 5. Các task phổ biến trong Computer Vision

Một số tác vụ quan trọng:

## 5.1. Image Classification

Phân loại toàn bộ ảnh vào một nhãn.

Ví dụ:

```text
Ảnh → "cat"
Ảnh → "dog"
Ảnh X-ray → "normal" hoặc "pneumonia"
```

---

## 5.2. Object Detection

Phát hiện vật thể và vị trí của chúng trong ảnh.

Output thường gồm:

```text
class label + bounding box + confidence score
```

Ví dụ:

```text
person: [x_min, y_min, x_max, y_max], score=0.97
car: [x_min, y_min, x_max, y_max], score=0.91
```

---

## 5.3. Segmentation

Phân đoạn ảnh theo pixel.

Có hai loại quan trọng:

### Semantic Segmentation

Gán mỗi pixel vào một lớp.

Ví dụ:

```text
pixel thuộc road
pixel thuộc sky
pixel thuộc car
pixel thuộc person
```

Không phân biệt các instance khác nhau cùng lớp.

### Instance Segmentation

Không chỉ biết pixel thuộc lớp nào, mà còn biết thuộc instance nào.

Ví dụ:

```text
person_1
person_2
person_3
```

---

## 5.4. Tracking

Theo dõi vật thể qua nhiều frame video.

Ví dụ:

```text
Frame 1: ball ở vị trí A
Frame 2: ball ở vị trí B
Frame 3: ball ở vị trí C
```

Ứng dụng:

- Thể thao.
- Xe tự lái.
- Giám sát.
- Robotics.

---

## 5.5. Image Captioning

Sinh mô tả văn bản từ ảnh.

Ví dụ:

```text
Ảnh → "A dog is playing with a ball in the park."
```

---

## 5.6. Image Generation

Sinh ảnh mới từ text hoặc điều kiện đầu vào.

Ví dụ:

```text
Prompt: "a futuristic city at night"
→ Ảnh thành phố tương lai
```

---

# 6. Ảnh là gì về mặt kỹ thuật?

Một ảnh có thể được xem là một hàm:

\[
F(x, y)
\]

Trong đó:

- `x`, `y`: tọa độ không gian.
- `F(x, y)`: giá trị cường độ sáng tại điểm đó.

Với ảnh grayscale:

```text
F(x, y) = intensity
```

Ví dụ pixel 8-bit:

```text
0   = đen
255 = trắng
```

Với ảnh màu RGB:

\[
F(x, y) = [R, G, B]
\]

Mỗi pixel có 3 giá trị:

```text
R: red channel
G: green channel
B: blue channel
```

Ví dụ:

```python
import cv2

img = cv2.imread("image.jpg")

print(img.shape)
```

Output có thể là:

```text
( height, width, channels )
( 720, 1280, 3 )
```

Lưu ý: OpenCV đọc ảnh theo thứ tự **BGR**, không phải RGB.

```python
img_bgr = cv2.imread("image.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
```

---

# 7. Pixel, voxel, channel, mask

## Pixel

Pixel là phần tử nhỏ nhất của ảnh 2D.

```text
pixel = picture element
```

Một ảnh grayscale 5x5 có thể là ma trận:

```text
[
 [  0,  10,  20,  30,  40],
 [ 50,  60,  70,  80,  90],
 [100, 110, 120, 130, 140],
 [150, 160, 170, 180, 190],
 [200, 210, 220, 230, 255]
]
```

---

## Voxel

Voxel là phần tử nhỏ nhất của ảnh 3D.

```text
voxel = volume element
```

Dùng trong:

- MRI.
- CT scan.
- Ảnh hiển vi 3D.
- Dữ liệu thể tích.

Ảnh 3D có thể được biểu diễn:

\[
F(x, y, z)
\]

---

## Channel

Channel là một lớp thông tin của ảnh.

Ví dụ RGB có 3 channel:

```text
Red
Green
Blue
```

Ảnh y tế hoặc vệ tinh có thể có nhiều channel hơn:

```text
visible light
infrared
x-ray
near-infrared
thermal
```

---

## Mask / Binary Image

Mask là ảnh nhị phân, thường dùng để biểu diễn vùng quan tâm.

Ví dụ:

```text
1 = foreground
0 = background
```

Ví dụ mask segmentation:

```python
import numpy as np

mask = np.array([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
])
```

---

# 8. Ảnh, video, tabular data khác nhau thế nào?

## Ảnh

Ảnh là dữ liệu không gian tại một thời điểm.

```text
F(x, y)
```

Thường biểu diễn bằng ma trận 2D hoặc tensor 3D:

```text
height × width × channels
```

---

## Video

Video là chuỗi ảnh theo thời gian.

```text
F(x, y, t)
```

Tensor video thường có dạng:

```text
frames × height × width × channels
```

Video khó hơn ảnh vì có thêm chiều thời gian:

- Chuyển động.
- Tracking.
- Temporal consistency.
- Frame rate.
- Optical flow.

---

## Tabular data

Tabular data có dạng hàng/cột:

```text
sample × features
```

Khác biệt lớn:

- Với tabular data, feature thường đã rõ ràng.
- Với ảnh, feature phải được trích xuất từ pixel.

Ví dụ:

```text
Tabular:
age, income, city, label

Image:
pixel values → cần học ra edges, textures, objects
```

---

# 9. Image acquisition — ảnh số được tạo ra như thế nào?

Quy trình tạo ảnh số:

```text
Nguồn năng lượng / ánh sáng
→ Tương tác với vật thể
→ Cảm biến thu nhận tín hiệu
→ Chuyển thành điện áp analog
→ Sampling
→ Quantization
→ Ảnh số
```

## 9.1. Sampling

Sampling là rời rạc hóa tọa độ không gian.

Tức là lấy mẫu không gian liên tục thành lưới pixel.

```text
Cảnh thật liên tục → ma trận pixel
```

Số lượng pixel càng cao thì spatial resolution càng cao.

---

## 9.2. Quantization

Quantization là rời rạc hóa giá trị cường độ.

Ví dụ ảnh 8-bit:

```text
2^8 = 256 mức
0 đến 255
```

Ảnh 16-bit:

```text
2^16 = 65536 mức
0 đến 65535
```

Quan trọng trong ảnh y tế, ảnh khoa học, ảnh vệ tinh.

---

## 9.3. Spatial Resolution

Spatial resolution là khả năng phân biệt chi tiết nhỏ trong không gian.

Ví dụ:

```text
Ảnh 20MP thường có nhiều chi tiết hơn ảnh 8MP
```

Nhưng độ phân giải cao không phải lúc nào cũng tốt hơn.

Vì:

- File lớn hơn.
- Training chậm hơn.
- Cần nhiều RAM/GPU hơn.
- Có thể chứa nhiều nhiễu hơn.
- Deployment khó hơn trên thiết bị yếu.

---

## 9.4. Intensity Resolution

Intensity resolution là khả năng phân biệt thay đổi nhỏ về cường độ sáng.

Ví dụ:

```text
8-bit  = 256 mức sáng
12-bit = 4096 mức sáng
16-bit = 65536 mức sáng
```

Ảnh medical imaging thường cần bit-depth cao hơn ảnh thông thường.

---

## 9.5. Dynamic Range

Dynamic range là tỉ lệ giữa tín hiệu lớn nhất có thể đo được và tín hiệu nhỏ nhất có thể phát hiện.

Dynamic range cao giúp giữ chi tiết ở cả vùng sáng và vùng tối.

---

# 10. Vì sao cách thu nhận ảnh rất quan trọng?

Ảnh không chỉ phụ thuộc vào vật thể, mà còn phụ thuộc vào thiết bị và cách chụp.

Ví dụ cùng một người:

- Ảnh RGB nhìn như ảnh chụp bình thường.
- Ảnh X-ray cho thấy xương.
- Ảnh MRI cho thấy mô mềm.
- Ảnh hồng ngoại cho thấy nhiệt.

Do đó, trước khi xây dựng model cần hiểu:

- Ảnh đến từ sensor nào?
- Ảnh có bao nhiêu channel?
- Độ phân giải bao nhiêu?
- Bit depth bao nhiêu?
- Có nhiễu không?
- Có bị blur không?
- Có bias do thiết bị không?
- Dữ liệu train có giống dữ liệu deploy không?

Một lỗi rất phổ biến là **measurement bias**.

Ví dụ:

```text
Train model bằng ảnh camera xịn độ phân giải cao
Deploy trên camera giám sát rẻ, ảnh mờ, ánh sáng kém
→ Model hoạt động tệ
```

Kết luận quan trọng:

> Nên thu thập dữ liệu càng giống môi trường triển khai thật càng tốt.

---

# 11. Tiền xử lý ảnh

Tiền xử lý ảnh là các thao tác chuẩn bị hoặc cải thiện ảnh trước khi đưa vào model.

Các nhóm thao tác chính:

- Logical operations.
- Statistical operations.
- Geometrical operations.
- Mathematical operations.
- Transform operations.

---

## 11.1. Resize

CNN thường yêu cầu input có kích thước cố định.

Ví dụ:

```python
import cv2

img = cv2.imread("image.jpg")
resized = cv2.resize(img, (224, 224))
```

Kích thước phổ biến:

```text
224x224
256x256
384x384
512x512
```

Trade-off:

- Ảnh lớn: giữ nhiều chi tiết, nhưng tốn tài nguyên.
- Ảnh nhỏ: train nhanh hơn, nhưng mất chi tiết.

---

## 11.2. Normalize pixel

Ảnh thường có pixel từ 0 đến 255. Neural network thường học tốt hơn khi đưa về khoảng nhỏ hơn.

Ví dụ đưa về `[0, 1]`:

```python
img = img.astype("float32") / 255.0
```

Hoặc chuẩn hóa theo mean/std:

```python
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

---

## 11.3. Spatial Filtering

Spatial filtering thay đổi giá trị pixel dựa trên vùng lân cận.

### Low-pass filter

Dùng để làm mờ, giảm nhiễu.

Ví dụ Gaussian blur:

```python
blurred = cv2.GaussianBlur(img, (5, 5), 0)
```

### High-pass filter

Dùng để làm nổi cạnh, làm nét ảnh.

Ví dụ Laplacian:

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
```

---

## 11.4. Morphological Operations

Morphology thường dùng với ảnh nhị phân/mask.

Các phép phổ biến:

- Erosion: làm mỏng vùng foreground.
- Dilation: làm dày vùng foreground.
- Opening: erosion rồi dilation, giúp loại nhiễu nhỏ.
- Closing: dilation rồi erosion, giúp lấp lỗ nhỏ.

Ví dụ:

```python
import cv2
import numpy as np

kernel = np.ones((3, 3), np.uint8)

eroded = cv2.erode(mask, kernel, iterations=1)
dilated = cv2.dilate(mask, kernel, iterations=1)
opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```

---

# 12. Data Augmentation

Data augmentation là tạo thêm biến thể từ dữ liệu train.

Mục tiêu:

- Tăng độ đa dạng dữ liệu.
- Giảm overfitting.
- Giúp model tổng quát tốt hơn.
- Giảm chi phí thu thập dữ liệu.

Các phép augmentation ảnh phổ biến:

- Flip.
- Rotate.
- Crop.
- Resize.
- Shift.
- Zoom.
- Brightness adjustment.
- Contrast adjustment.
- Gaussian noise.
- Blur.
- Perspective transform.
- Color jitter.

Ví dụ với `torchvision`:

```python
import torchvision.transforms as T

train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    ),
    T.ToTensor(),
])
```

Lưu ý quan trọng:

Augmentation phải hợp lý với bài toán.

Ví dụ:

- Với nhận diện mèo/chó, horizontal flip thường ổn.
- Với nhận diện chữ số, flip có thể làm sai nhãn.
- Với ảnh y tế, xoay/lật tùy tiện có thể không hợp lệ về mặt lâm sàng.
- Với biển báo giao thông, thay đổi màu quá mạnh có thể phá ý nghĩa.

---

# 13. Image restoration, enhancement và reconstruction

## Image Enhancement

Enhancement là cải thiện ảnh theo mục tiêu thị giác, thường mang tính chủ quan.

Ví dụ:

- Tăng độ sáng.
- Tăng tương phản.
- Làm nét.
- Cân bằng histogram.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
```

---

## Image Restoration

Restoration cố gắng khôi phục ảnh bị suy giảm dựa trên mô hình suy giảm.

Ví dụ:

- Khử nhiễu.
- Deblur.
- Loại bỏ artifacts.
- Khôi phục ảnh cũ.

Restoration thường kỹ thuật hơn enhancement vì cần hiểu quá trình làm hỏng ảnh.

---

## Image Reconstruction

Reconstruction tạo ảnh từ dữ liệu đo gián tiếp.

Ví dụ:

- CT reconstruction từ nhiều projection.
- MRI reconstruction.
- 3D reconstruction từ nhiều ảnh 2D.

---

# 14. Màu sắc trong xử lý ảnh

Màu là đặc trưng rất quan trọng.

## RGB

RGB có 3 channel:

```text
Red, Green, Blue
```

Mỗi channel 8-bit:

```text
R: 0-255
G: 0-255
B: 0-255
```

Tổng số màu:

```text
256 × 256 × 256 = 16,777,216 màu
```

---

## CMY/CMYK

Thường dùng trong in ấn.

```text
Cyan
Magenta
Yellow
Black
```

---

## Pseudo-color

Pseudo-color là gán màu giả cho ảnh grayscale để dễ quan sát.

Ví dụ ảnh nhiệt:

```text
giá trị thấp → xanh
giá trị cao → đỏ
```

---

# 15. Nén ảnh

Nén ảnh giảm số lượng dữ liệu cần lưu hoặc truyền.

Có ba loại redundancy chính:

## 15.1. Coding redundancy

Một số giá trị pixel xuất hiện nhiều hơn nhưng vẫn được mã hóa bằng số bit như các giá trị hiếm.

Giải pháp: mã hóa giá trị phổ biến bằng code ngắn hơn.

Ví dụ: Huffman coding.

---

## 15.2. Spatial redundancy

Các pixel lân cận thường giống nhau.

Ví dụ một vùng trời xanh có nhiều pixel gần giống nhau.

Kỹ thuật như run-length encoding có thể tận dụng điều này.

---

## 15.3. Temporal redundancy

Trong video, frame kế tiếp thường rất giống frame trước.

Nén video tận dụng điều này bằng cách lưu khác biệt giữa các frame thay vì lưu toàn bộ frame.

---

## 15.4. Irrelevant information

Một số thông tin không quan trọng với người xem hoặc mục đích bài toán có thể bị loại bỏ.

Đây thường là nén mất mát — lossy compression.

Ví dụ: JPEG.

---

# 16. Feature Extraction

Feature là đặc trưng giúp mô hình nhận biết dữ liệu.

Ví dụ trong ảnh:

- Edge.
- Corner.
- Texture.
- Color histogram.
- Shape.
- Keypoint.
- Embedding từ CNN.

---

## 16.1. Đặc trưng tốt cần có gì?

Một descriptor tốt nên có:

### Invariance

Ít thay đổi khi ảnh bị:

- Xoay.
- Dịch chuyển.
- Scale.
- Thay đổi ánh sáng.

### Distinctiveness

Phải đủ khác biệt để phân biệt object/vùng ảnh này với object/vùng ảnh khác.

### Compactness

Không quá lớn, để dễ lưu trữ và tính toán.

### Locality

Nên mô tả được các vùng cục bộ quan trọng.

### Repeatability

Cùng một vật thể trong nhiều ảnh khác nhau vẫn tạo descriptor tương tự.

### Robustness to noise

Không bị nhiễu nhỏ làm thay đổi quá nhiều.

### Computational efficiency

Phải đủ nhanh nếu dùng trong real-time system.

---

# 17. SIFT

SIFT là viết tắt của:

```text
Scale-Invariant Feature Transform
```

SIFT dùng để phát hiện và mô tả local features trong ảnh.

Pipeline chính:

```text
Scale-space extrema detection
→ Keypoint localization
→ Orientation assignment
→ Descriptor generation
→ Descriptor matching
```

## 17.1. Scale-space extrema detection

SIFT tìm điểm nổi bật ở nhiều scale khác nhau.

Điều này giúp robust với thay đổi kích thước.

---

## 17.2. Keypoint localization

Loại bỏ keypoint yếu:

- Độ tương phản thấp.
- Nằm trên cạnh không ổn định.

---

## 17.3. Orientation assignment

Gán hướng chính cho mỗi keypoint dựa trên gradient cục bộ.

Nhờ vậy descriptor ít bị ảnh hưởng bởi xoay ảnh.

---

## 17.4. Descriptor generation

Tạo vector mô tả vùng quanh keypoint dựa trên histogram gradient.

---

## 17.5. Matching

So sánh descriptor giữa hai ảnh để tìm điểm tương ứng.

Ví dụ:

```python
import cv2

img1 = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("image2.jpg", cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

matched_img = cv2.drawMatchesKnn(
    img1, kp1,
    img2, kp2,
    good,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
```

---

# 18. SURF

SURF là:

```text
Speeded-Up Robust Features
```

SURF tương tự SIFT nhưng tối ưu tốc độ hơn.

Các ý chính:

- Dùng integral image để tính nhanh.
- Dùng Hessian matrix để phát hiện blob/keypoint.
- Dùng Haar wavelet response để mô tả hướng và descriptor.
- Robust với scale, rotation, illumination.
- Phù hợp hơn cho real-time so với SIFT trong nhiều trường hợp.

---

# 19. ORB

ORB là descriptor nhị phân, thường nhanh và miễn phí hơn SIFT/SURF trong thực tế.

ORB dùng Hamming distance để matching.

Ví dụ:

```python
import cv2

img1 = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("image2.jpg", cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

matched_img = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    matches[:50],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
```

---

# 20. Feature Matching

Feature matching là tìm điểm tương ứng giữa hai ảnh.

Ứng dụng:

- Image stitching.
- Object tracking.
- Scene understanding.
- 3D reconstruction.
- Visual localization.
- SLAM.
- Panorama.

---

## 20.1. Brute-force matching

So sánh từng descriptor của ảnh 1 với từng descriptor của ảnh 2.

Ưu điểm:

- Đơn giản.
- Dễ hiểu.
- Chính xác nếu dữ liệu nhỏ.

Nhược điểm:

- Chậm khi có nhiều descriptor.

---

## 20.2. Ratio Test

Ratio test giúp lọc match kém.

Ý tưởng:

```text
Nếu match tốt nhất chỉ tốt hơn match thứ hai một chút
→ match đó không đáng tin
```

Công thức thường dùng:

```python
if m.distance < 0.75 * n.distance:
    good.append(m)
```

---

## 20.3. FLANN

FLANN là:

```text
Fast Library for Approximate Nearest Neighbors
```

Thay vì tìm chính xác bằng brute-force, FLANN tìm gần đúng nhưng nhanh hơn.

Phù hợp khi:

- Nhiều descriptor.
- Ảnh lớn.
- Cần tốc độ tốt.

Ví dụ với SIFT:

```python
import cv2

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1

index_params = dict(
    algorithm=FLANN_INDEX_KDTREE,
    trees=5,
)

search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
```

---

# 21. LoFTR — matching bằng Transformer

LoFTR là phương pháp matching hiện đại dựa trên Deep Learning/Transformer.

Khác với SIFT/SURF/ORB:

```text
SIFT/SURF/ORB: detect keypoint trước, rồi match
LoFTR: detector-free, học trực tiếp correspondence
```

Ưu điểm:

- Robust hơn trong nhiều trường hợp khó.
- Xử lý tốt ảnh ít texture hơn.
- Có thể match ảnh có góc nhìn/ánh sáng khác nhau.

Ví dụ dùng Kornia:

```python
import kornia as K
import kornia.feature as KF
import torch
from kornia.feature import LoFTR

img1 = K.io.load_image("image1.jpg", K.io.ImageLoadType.RGB32)[None, ...]
img2 = K.io.load_image("image2.jpg", K.io.ImageLoadType.RGB32)[None, ...]

img1 = K.geometry.resize(img1, (512, 512), antialias=True)
img2 = K.geometry.resize(img2, (512, 512), antialias=True)

matcher = LoFTR(pretrained="outdoor")

input_dict = {
    "image0": K.color.rgb_to_grayscale(img1),
    "image1": K.color.rgb_to_grayscale(img2),
}

with torch.inference_mode():
    correspondences = matcher(input_dict)

mkpts0 = correspondences["keypoints0"]
mkpts1 = correspondences["keypoints1"]
confidence = correspondences["confidence"]
```

Sau đó thường dùng RANSAC để lọc outlier:

```python
import cv2

mkpts0_np = mkpts0.cpu().numpy()
mkpts1_np = mkpts1.cpu().numpy()

F, inliers = cv2.findFundamentalMat(
    mkpts0_np,
    mkpts1_np,
    cv2.USAC_MAGSAC,
    0.5,
    0.999,
    100000,
)
```

---

# 22. Ứng dụng thực tế của Computer Vision

## 22.1. Xe tự lái

Computer Vision giúp xe:

- Phát hiện làn đường.
- Nhận diện biển báo.
- Phát hiện người đi bộ.
- Theo dõi xe khác.
- Ước lượng khoảng cách.
- Ra quyết định phanh/tăng tốc/chuyển làn.

Yêu cầu kỹ thuật:

- Real-time.
- Robust với ánh sáng, mưa, sương mù.
- Sai số thấp.
- Tích hợp camera, radar, lidar.

---

## 22.2. Retail và E-commerce

Ứng dụng:

- Nhận diện sản phẩm.
- Gợi ý sản phẩm tương tự.
- Theo dõi tồn kho.
- Phân tích hành vi khách hàng.
- Tối ưu bố trí cửa hàng.

---

## 22.3. Quality Control trong sản xuất

Computer Vision dùng để:

- Phát hiện lỗi sản phẩm.
- Kiểm tra lắp ráp.
- Đo kích thước.
- Tự động loại sản phẩm lỗi.
- Cảnh báo lỗi dây chuyền theo thời gian thực.

Ví dụ:

```text
Ảnh linh kiện → detect scratch / dent / missing part
```

---

## 22.4. Medical Image Analysis

Ứng dụng:

- Phát hiện tumor.
- Phân đoạn cơ quan.
- Hỗ trợ chẩn đoán X-ray, CT, MRI.
- Theo dõi tiến triển bệnh.
- Hỗ trợ lập kế hoạch điều trị.
- Phân tích tế bào/mô học.

Rủi ro quan trọng:

- False negative trong y tế có thể gây hậu quả nghiêm trọng.
- Dữ liệu y tế thường nhỏ, nhạy cảm và lệch phân phối.
- Cần human oversight.

---

## 22.5. Facial Recognition

Dùng trong:

- Mở khóa điện thoại.
- Kiểm soát truy cập.
- Sân bay.
- Giám sát.
- Một số ứng dụng y tế.

Vấn đề cần chú ý:

- Privacy.
- Bias theo chủng tộc, giới tính, độ tuổi.
- Consent.
- Sai số trong môi trường thực tế.

---

## 22.6. Object Tracking

Ứng dụng:

- Thể thao: tracking bóng.
- Xe tự lái: tracking xe/người đi bộ.
- Wildlife conservation: tracking động vật.
- Camera giám sát.

Kỹ thuật có thể gồm:

- Color tracking.
- Kalman filter.
- Optical flow.
- CNN-based tracker.
- Transformer-based tracker.

---

## 22.7. Anomaly Detection

Tìm những thứ bất thường trong ảnh/video.

Ứng dụng:

- Phát hiện hành vi đáng ngờ.
- Phát hiện đồ vật bị bỏ quên.
- Phát hiện lỗi sản xuất.
- Phát hiện bất thường trong ảnh y tế.

Một hướng phổ biến:

```text
Train model trên dữ liệu normal
→ Nếu ảnh mới khác normal quá nhiều
→ anomaly
```

Ví dụ autoencoder:

```text
Ảnh normal → encode → decode → reconstruction tốt
Ảnh anomaly → reconstruction kém → error cao
```

---

# 23. Thách thức lớn trong Computer Vision

Các thách thức chính:

## Data variability

Dữ liệu thực tế thay đổi rất nhiều:

- Ánh sáng.
- Góc nhìn.
- Nền.
- Occlusion.
- Camera.
- Thời tiết.
- Domain.

---

## Scalability

Hệ thống phải xử lý nhiều ảnh/video.

Ví dụ:

```text
Hàng triệu ảnh sản phẩm
Video 24/7 từ hàng nghìn camera
```

---

## Accuracy

Cần độ chính xác cao, đặc biệt với:

- Y tế.
- Xe tự lái.
- Giám sát an ninh.
- Công nghiệp.

---

## Robustness to noise

Ảnh thực tế thường có:

- Blur.
- Compression artifacts.
- Sensor noise.
- Low light.
- Motion blur.

---

## Real-time processing

Một số ứng dụng cần xử lý tức thời:

- Xe tự lái.
- Robotics.
- AR/VR.
- Camera an ninh.

---

## Generalization

Model phải hoạt động tốt trên dữ liệu mới, không chỉ train set.

Đây là vấn đề rất lớn.

---

## Calibration và Maintenance

Camera/sensor cần được hiệu chỉnh và bảo trì.

Nếu camera lệch, bẩn, thay đổi thông số, model có thể giảm hiệu năng.

---

# 24. Đạo đức trong Computer Vision

Các vấn đề đạo đức quan trọng:

## Privacy

Computer Vision thường xử lý dữ liệu cá nhân:

- Khuôn mặt.
- Biển số xe.
- Hành vi.
- Dữ liệu y tế.

Cần quan tâm:

- Consent.
- Lưu trữ dữ liệu.
- Ai có quyền truy cập?
- Dữ liệu có bị dùng sai mục đích không?

---

## Bias và Fairness

Bias có thể xuất hiện ở:

- Dữ liệu train.
- Quy trình label.
- Thiết kế model.
- Cách triển khai.
- Ngữ cảnh sử dụng.

Ví dụ:

```text
Dataset mặt người chủ yếu từ một nhóm dân số
→ Model nhận diện kém nhóm khác
```

---

## Accountability

Nếu model sai, ai chịu trách nhiệm?

Đặc biệt trong:

- Y tế.
- Pháp lý.
- Tài chính.
- Xe tự lái.

---

## Transparency và Explainability

Người dùng cần hiểu:

- Model dùng dữ liệu gì?
- Model có giới hạn gì?
- Khi nào không nên tin model?
- Model có bias nào đã biết?

Một công cụ quan trọng là **model card**, dùng để ghi lại:

- Mục đích model.
- Dataset.
- Metrics.
- Giới hạn.
- Rủi ro.
- Bias đã biết.

---

# 25. Những điểm kỹ thuật quan trọng cần nhớ

## Ảnh là dữ liệu có cấu trúc không gian

Không thể xử lý ảnh như một vector bình thường mà bỏ qua không gian.

Pixel lân cận thường có liên hệ với nhau.

Đó là lý do CNN hiệu quả với ảnh.

---

## Độ phân giải cao không phải lúc nào cũng tốt

Cần cân bằng:

```text
chi tiết ảnh
vs
nhiễu
vs
chi phí tính toán
vs
khả năng deploy
```

---

## Dữ liệu train nên giống dữ liệu deploy

Nếu deploy trên camera thật, hãy train/validate với dữ liệu giống camera thật.

---

## Preprocessing cần nhất quán

Nếu train dùng:

```text
resize 224x224
normalize mean/std
RGB order
```

Thì inference cũng phải dùng đúng như vậy.

---

## Augmentation phải đúng ngữ nghĩa

Không phải phép biến đổi nào cũng hợp lệ.

---

## Feature extraction có hai hướng

### Cổ điển

- SIFT.
- SURF.
- ORB.
- HOG.
- Harris corner.
- Color histogram.

### Deep Learning

- CNN embeddings.
- Vision Transformer features.
- Learned descriptors.
- LoFTR-style matching.

---

## Matching cần lọc outlier

Feature matching thường sinh match sai.

Các kỹ thuật lọc:

- Ratio test.
- Cross-check.
- RANSAC.
- Fundamental matrix estimation.

---

# 26. Mini pipeline thực tế cho bài toán Computer Vision

Một pipeline cơ bản:

```text
1. Xác định task
2. Hiểu nguồn dữ liệu ảnh
3. Khám phá dữ liệu
4. Tiền xử lý
5. Data augmentation
6. Chọn model
7. Train
8. Validate
9. Error analysis
10. Deploy
11. Monitor drift/bias/performance
```

Ví dụ code đơn giản với PyTorch transform:

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

Điểm cần nhớ:

- Train có augmentation.
- Validation/test không nên dùng augmentation ngẫu nhiên.
- Normalize phải giống nhau giữa train và inference.

---

# Kết luận ngắn gọn

Unit 1 đặt nền móng cho Computer Vision bằng các ý chính:

1. Computer Vision là quá trình biến ảnh/video thành thông tin có ý nghĩa.
2. Ảnh là hàm/tensor có cấu trúc không gian, không chỉ là tập pixel rời rạc.
3. Cách thu nhận ảnh ảnh hưởng mạnh đến chất lượng dữ liệu và hiệu năng model.
4. Tiền xử lý, resize, normalize, filtering và augmentation là các bước rất quan trọng.
5. Feature extraction có thể dùng phương pháp cổ điển như SIFT/SURF/ORB hoặc deep learning.
6. Feature matching là nền tảng cho stitching, tracking, 3D reconstruction và localization.
7. Ứng dụng Computer Vision rất rộng, nhưng luôn đi kèm rủi ro kỹ thuật, bias, privacy và đạo đức.