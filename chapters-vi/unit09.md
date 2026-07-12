# Unit 9 — Tối ưu mô hình và triển khai mô hình Computer Vision

Sau khi train xong một mô hình Computer Vision, công việc chưa kết thúc. Mô hình cần được **triển khai** để người dùng hoặc hệ thống khác có thể sử dụng. Tuy nhiên, mô hình train tốt trên GPU mạnh chưa chắc chạy tốt khi đưa vào production, mobile, IoT hoặc edge device.

Các vấn đề thường gặp khi deployment:

- Model quá lớn.
- Inference chậm.
- Tốn RAM hoặc VRAM.
- Không tương thích phần cứng.
- Tiêu thụ nhiều điện năng.
- Latency không đáp ứng yêu cầu real-time.

Vì vậy, trước khi deployment thường cần thêm bước **model optimization**.

---

# 1. Model Optimization là gì?

**Model optimization** là quá trình điều chỉnh mô hình đã train để mô hình chạy hiệu quả hơn khi inference.

Mục tiêu chính:

- Giảm kích thước model.
- Giảm thời gian inference.
- Giảm RAM/VRAM cần dùng.
- Giảm chi phí tính toán.
- Giảm tiêu thụ điện.
- Tăng khả năng chạy trên edge/mobile/embedded device.

Ví dụ:

- Train model bằng GPU mạnh như NVIDIA A100.
- Deploy model lên điện thoại, Raspberry Pi, camera AI, drone hoặc thiết bị IoT.

Phần cứng deployment thường yếu hơn rất nhiều phần cứng training, nên mô hình cần được tối ưu.

---

# 2. Vì sao optimization quan trọng trong Computer Vision?

Computer Vision thường xử lý ảnh hoặc video, dữ liệu có kích thước lớn hơn nhiều so với text đơn giản. Vì vậy inference có thể rất nặng.

## 2.1 Giới hạn tài nguyên

Các mô hình CV thường cần nhiều:

- CPU/GPU.
- RAM/VRAM.
- Storage.
- Memory bandwidth.

Nếu deploy lên mobile hoặc edge device, tài nguyên này rất hạn chế.

Ví dụ:

```text
Model gốc: 500 MB, FP32
Sau quantization INT8: khoảng 125 MB
```

Model nhỏ hơn giúp:

- Load nhanh hơn.
- Chạy được trên thiết bị RAM thấp.
- Giảm chi phí lưu trữ.
- Giảm băng thông khi tải model.

---

## 2.2 Yêu cầu latency thấp

Nhiều ứng dụng cần phản hồi gần như real-time:

- Xe tự lái.
- Camera giám sát.
- AR/VR.
- Face detection.
- Object detection trên video.
- Robot.

Ví dụ nếu camera chạy 30 FPS, hệ thống chỉ có khoảng:

```text
1 giây / 30 frame ≈ 33 ms/frame
```

Nếu model mất 200 ms để xử lý một ảnh thì không thể chạy real-time.

Optimization giúp giảm thời gian inference.

---

## 2.3 Tiêu thụ điện năng

Thiết bị dùng pin như:

- Drone.
- Wearable device.
- Mobile phone.
- Camera thông minh.

Nếu model quá nặng, CPU/GPU hoạt động nhiều hơn, làm pin tụt nhanh hơn và thiết bị nóng hơn.

---

## 2.4 Tương thích phần cứng

Một số phần cứng hỗ trợ tốt định dạng hoặc kiểu tính toán nhất định.

Ví dụ:

- NVIDIA GPU tối ưu tốt với TensorRT.
- Intel CPU/VPU tối ưu tốt với OpenVINO.
- Google Edge TPU yêu cầu model được compile theo định dạng riêng.
- Mobile có thể cần Core ML hoặc TensorFlow Lite.

Không phải cứ có model PyTorch là chạy tốt ở mọi nơi. Thường cần convert hoặc optimize theo target hardware.

---

# 3. Các kỹ thuật tối ưu mô hình quan trọng

## 3.1 Pruning

**Pruning** là kỹ thuật loại bỏ các kết nối, weight, neuron hoặc channel ít quan trọng trong mô hình.

Ý tưởng:

> Không phải toàn bộ tham số trong neural network đều quan trọng như nhau. Một số weight gần như không ảnh hưởng nhiều đến kết quả.

Sau pruning:

- Model nhỏ hơn.
- Ít phép tính hơn.
- Có thể inference nhanh hơn.
- Nhưng accuracy có thể giảm nếu prune quá mạnh.

Ví dụ đơn giản:

```text
Trước pruning:
[0.91, 0.02, -0.003, 0.75, 0.0001]

Sau pruning các weight rất nhỏ:
[0.91, 0, 0, 0.75, 0]
```

Có hai hướng pruning phổ biến:

### Unstructured pruning

Loại bỏ từng weight riêng lẻ.

Ưu điểm:

- Dễ đạt sparsity cao.
- Ít ảnh hưởng accuracy hơn.

Nhược điểm:

- Không phải phần cứng nào cũng tăng tốc tốt với sparse matrix.

### Structured pruning

Loại bỏ cả channel, filter, layer hoặc block.

Ưu điểm:

- Dễ tăng tốc thực tế hơn.
- Phù hợp phần cứng phổ thông.

Nhược điểm:

- Dễ ảnh hưởng accuracy hơn nếu prune mạnh.

---

## 3.2 Quantization

**Quantization** là chuyển weight và/hoặc activation từ độ chính xác cao sang độ chính xác thấp hơn.

Ví dụ:

```text
FP32 -> FP16
FP32 -> INT8
```

FP32 dùng 32 bit cho mỗi số. INT8 chỉ dùng 8 bit.

Vì vậy INT8 có thể giảm kích thước model khoảng 4 lần:

```text
32 bit / 8 bit = 4
```

Lợi ích:

- Model nhỏ hơn.
- Dùng ít RAM hơn.
- Giảm memory bandwidth.
- Inference nhanh hơn trên hardware hỗ trợ INT8.
- Tiết kiệm điện hơn.

Rủi ro:

- Accuracy có thể giảm.
- Một số layer nhạy cảm với quantization.
- Cần calibration hoặc quantization-aware training nếu post-training quantization không đủ tốt.

### Các loại quantization phổ biến

#### Post-training quantization

Train model bình thường bằng FP32, sau đó convert sang INT8 hoặc FP16.

Ưu điểm:

- Dễ làm.
- Không cần train lại hoặc chỉ cần calibration nhẹ.

Nhược điểm:

- Accuracy có thể giảm nếu model nhạy.

#### Quantization-aware training

Trong lúc train/fine-tune, mô phỏng lỗi quantization bằng fake quantization.

Ưu điểm:

- Accuracy thường tốt hơn post-training quantization.
- Phù hợp khi cần INT8 nhưng vẫn giữ chất lượng.

Nhược điểm:

- Cần training hoặc fine-tuning lại.
- Pipeline phức tạp hơn.

Ví dụ PyTorch minh họa ý tưởng:

```python
import torch

model = ...  # model đã train FP32
model.eval()

# Dynamic quantization, thường dùng tốt cho Linear/LSTM hơn CNN,
# nhưng minh họa API quantization của PyTorch.
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

Với Computer Vision CNN/Transformer, quantization thực tế thường cần chuẩn bị kỹ hơn tùy backend.

---

## 3.3 Knowledge Distillation

**Knowledge Distillation** là kỹ thuật dùng một model lớn, mạnh làm **teacher**, sau đó train một model nhỏ hơn làm **student** để bắt chước teacher.

Mục tiêu:

- Student nhỏ hơn.
- Inference nhanh hơn.
- Accuracy gần với teacher.

Teacher không chỉ cung cấp nhãn cứng như:

```text
cat
```

Mà còn cung cấp phân phối xác suất mềm:

```text
cat: 0.82
dog: 0.12
fox: 0.04
other: 0.02
```

Phân phối này chứa nhiều thông tin hơn nhãn one-hot.

Ví dụ loss đơn giản:

```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, true_labels, temperature=4.0, alpha=0.5):
    hard_loss = F.cross_entropy(student_logits, true_labels)

    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)

    soft_loss = F.kl_div(
        soft_student,
        soft_teacher,
        reduction="batchmean"
    ) * (temperature ** 2)

    return alpha * hard_loss + (1 - alpha) * soft_loss
```

Trong đó:

- `hard_loss`: học từ ground truth label.
- `soft_loss`: học cách teacher phân phối xác suất.
- `temperature`: làm mềm phân phối xác suất.
- `alpha`: cân bằng giữa học từ label thật và học từ teacher.

---

## 3.4 Low-rank approximation

**Low-rank approximation** xấp xỉ ma trận lớn bằng các ma trận nhỏ hơn.

Ý tưởng:

Thay vì lưu và tính toán một ma trận lớn:

```text
W: m x n
```

Ta xấp xỉ:

```text
W ≈ A x B
A: m x k
B: k x n
k nhỏ hơn nhiều so với m, n
```

Lợi ích:

- Giảm số tham số.
- Giảm chi phí tính toán.
- Giảm memory.

Kỹ thuật này liên quan đến các phương pháp decomposition như SVD.

---

## 3.5 Hardware-specific optimization

Một số framework tối ưu model theo phần cứng cụ thể:

- NVIDIA GPU → TensorRT.
- Intel CPU/GPU/VPU → OpenVINO.
- Google Edge TPU → Edge TPU compiler.
- ONNX Runtime → nhiều backend khác nhau.
- Hugging Face Optimum → tối ưu theo backend/hardware.

Khi deploy production, cần biết model chạy trên hardware nào trước khi chọn công cụ tối ưu.

---

# 4. Trade-off giữa accuracy, latency và resource usage

Không có tối ưu nào miễn phí. Thường phải đánh đổi giữa:

## 4.1 Accuracy

Accuracy là độ đúng của mô hình.

Model lớn thường có accuracy cao hơn nhưng:

- Chậm hơn.
- Tốn RAM hơn.
- Khó deploy edge hơn.
- Chi phí inference cao hơn.

## 4.2 Performance / Latency

Latency là thời gian model xử lý một request hoặc một ảnh.

Latency thấp quan trọng với:

- Real-time detection.
- Video streaming.
- AR.
- Autonomous systems.

Tăng tốc model thường có thể làm giảm accuracy.

## 4.3 Resource usage

Bao gồm:

- CPU usage.
- GPU usage.
- RAM/VRAM.
- Disk storage.
- Battery.
- Network bandwidth.

Thiết bị càng nhỏ thì resource usage càng quan trọng.

---

## 4.4 Cách tư duy thực tế

Không nên chỉ hỏi:

> Model nào accuracy cao nhất?

Mà nên hỏi:

> Model nào đủ chính xác, đủ nhanh, đủ nhỏ cho target deployment?

Ví dụ:

| Use case | Ưu tiên |
|---|---|
| Medical imaging offline | Accuracy |
| Face unlock mobile | Latency + security |
| Drone tracking object | Latency + power |
| Batch processing ảnh trên cloud | Throughput |
| Camera IoT | Model size + memory + latency |

---

# 5. Công cụ và framework tối ưu mô hình

## 5.1 TensorFlow Model Optimization Toolkit

TensorFlow Model Optimization Toolkit, thường gọi là TMO, cung cấp công cụ tối ưu model TensorFlow.

Cài đặt:

```bash
pip install -U tensorflow-model-optimization
```

Hỗ trợ:

- Post-training quantization.
- Quantization-aware training.
- Pruning.
- TensorFlow Lite optimization.

Điểm đáng chú ý:

- TensorFlow Lite post-training quantization có thể convert weight sang 8-bit.
- Model size có thể giảm khoảng 4 lần.
- Phù hợp deployment trên mobile/edge.

---

## 5.2 PyTorch Quantization

PyTorch hỗ trợ INT8 quantization trực tiếp trong package `torch`.

Cài đặt:

```bash
pip install torch
```

Import:

```python
import torch.quantization
```

PyTorch hỗ trợ nhiều hướng:

### 1. Train FP32 rồi convert INT8

Đây là post-training quantization.

### 2. Quantization-aware training

Train với fake quantization để model thích nghi với lỗi lượng tử hóa.

### 3. Dùng quantized tensor/operator

Có thể xây dựng model hoặc một phần model chạy với tensor độ chính xác thấp.

Lợi ích chính:

- Model size giảm khoảng 4 lần khi từ FP32 sang INT8.
- Memory bandwidth cũng giảm khoảng 4 lần.
- Có thể tăng tốc inference nếu backend hỗ trợ tốt.

---

## 5.3 ONNX Runtime

**ONNX Runtime** là runtime tăng tốc inference đa nền tảng.

Cài đặt CPU:

```bash
pip install onnxruntime
```

Cài đặt GPU:

```bash
pip install onnxruntime-gpu
```

Lưu ý: chỉ nên cài **một trong hai** package trong cùng environment.

ONNX Runtime dùng được với model từ:

- PyTorch.
- TensorFlow/Keras.
- TFLite.
- scikit-learn.
- Các framework khác hỗ trợ ONNX.

Lợi ích:

- Tăng performance inference.
- Chạy được trên nhiều OS/hardware.
- Train bằng Python nhưng deploy trong C#/C++/Java.
- Giúp tách framework training khỏi framework serving.

Ví dụ export PyTorch model sang ONNX:

```python
import torch

model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17
)
```

Inference bằng ONNX Runtime:

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")

input_name = session.get_inputs()[0].name

x = np.random.randn(1, 3, 224, 224).astype(np.float32)

outputs = session.run(None, {input_name: x})
```

---

## 5.4 TensorRT

**TensorRT** là SDK của NVIDIA để tối ưu inference cho GPU NVIDIA.

Cài đặt:

```bash
pip install tensorrt
```

TensorRT phù hợp khi:

- Deploy trên NVIDIA GPU.
- Cần throughput cao.
- Cần latency thấp.
- Muốn dùng FP16 hoặc INT8 inference.

TensorRT thường nhận model từ ONNX rồi build thành TensorRT engine.

Pipeline phổ biến:

```text
PyTorch / TensorFlow
        ↓
      ONNX
        ↓
    TensorRT engine
        ↓
 NVIDIA GPU inference
```

Lưu ý:

- TensorRT phụ thuộc mạnh vào GPU, CUDA, driver, version.
- Engine build cho một môi trường có thể không portable hoàn toàn sang môi trường khác.
- INT8 thường cần calibration data.

---

## 5.5 OpenVINO

**OpenVINO** là toolkit tối ưu inference của Intel.

Cài đặt:

```bash
pip install openvino
```

Phù hợp với:

- Intel CPU.
- Intel integrated GPU.
- Intel VPU.
- Một số hardware platform khác.

Lợi ích:

- Tối ưu performance trên Intel hardware.
- Có OpenVINO Runtime để chạy local.
- Có OpenVINO Model Server để phục vụ inference trong server/Kubernetes.
- Ít dependency ngoài.
- Hỗ trợ deploy đa OS/ngôn ngữ.

OpenVINO phù hợp khi production chủ yếu chạy trên Intel CPU và muốn tối ưu latency/throughput mà không cần GPU NVIDIA.

---

## 5.6 Hugging Face Optimum

**Optimum** là extension của Hugging Face Transformers để tối ưu training và inference trên các phần cứng khác nhau.

Cài đặt:

```bash
pip install optimum
```

Optimum hỗ trợ nhiều backend/hardware:

- Habana.
- Intel.
- NVIDIA.
- AWS Trainium/Inferentia.
- AMD.
- FuriosaAI.
- ONNX Runtime.
- BetterTransformer.

Ý nghĩa thực tế:

Nếu đang dùng Hugging Face model, Optimum giúp tối ưu mà vẫn giữ trải nghiệm gần với Transformers.

Ví dụ các hướng thường gặp:

- Export sang ONNX.
- Quantization.
- Tối ưu inference với ONNX Runtime.
- Tối ưu cho Intel/OpenVINO.

---

## 5.7 Edge TPU

**Edge TPU** là ASIC của Google cho AI inference ở edge.

Phù hợp với:

- Thiết bị nhỏ.
- Cần tiêu thụ điện thấp.
- Cần inference nhanh tại chỗ.
- Không muốn gửi dữ liệu lên cloud.

Lợi ích:

- Hiệu năng cao trong kích thước nhỏ.
- Tiết kiệm điện.
- Phù hợp edge AI.

Nhược điểm:

- Model phải tương thích với Edge TPU.
- Thường cần TensorFlow Lite INT8.
- Không phải operator nào cũng được hỗ trợ.
- Cần compile model cho Edge TPU.

Pipeline thường gặp:

```text
TensorFlow model
      ↓
TensorFlow Lite INT8 model
      ↓
Edge TPU compiler
      ↓
Run on Edge TPU device
```

---

# 6. Deployment platform

Có ba hướng deployment chính: cloud, edge và mobile.

---

## 6.1 Cloud deployment

Deploy model trên AWS, Google Cloud, Azure hoặc server riêng.

Ưu điểm:

- Dễ scale.
- Có nhiều CPU/GPU/RAM.
- Tích hợp tốt với cloud services.
- Dễ logging, monitoring, CI/CD.
- Dễ update model.

Nhược điểm:

- Chi phí có thể cao.
- Network latency.
- Phụ thuộc internet.
- Vấn đề privacy nếu gửi dữ liệu người dùng lên cloud.

Phù hợp với:

- Batch inference.
- API server.
- Model lớn.
- Ứng dụng không yêu cầu offline.
- Cần quản lý tập trung.

---

## 6.2 Edge deployment

Edge là deploy model gần nơi dữ liệu được sinh ra, ví dụ:

- Camera AI.
- IoT device.
- Edge server trong nhà máy.
- Raspberry Pi.
- Jetson Nano.
- Thiết bị nhúng.

Ưu điểm:

- Latency thấp.
- Có thể hoạt động offline.
- Giảm dữ liệu gửi lên cloud.
- Tốt hơn cho privacy.

Nhược điểm:

- Tài nguyên hạn chế.
- Khó update hàng loạt.
- Cần optimize kỹ.
- Có thể phụ thuộc phần cứng cụ thể.

Điểm quan trọng:

> Edge không chỉ là thiết bị nhỏ. Edge nghĩa là inference được đưa gần người dùng hoặc gần nguồn dữ liệu hơn.

Thông thường:

```text
Train model trên cloud/server mạnh
        ↓
Optimize model
        ↓
Deploy model lên edge device
```

---

## 6.3 Mobile deployment

Deploy trên điện thoại Android/iOS.

Framework phổ biến:

- Core ML cho iOS.
- TensorFlow Lite cho Android/iOS.
- PyTorch Mobile từng được dùng, nhưng hiện nay cần kiểm tra kỹ trạng thái hỗ trợ theo version thực tế.
- ONNX Runtime Mobile cũng là lựa chọn.

Mobile deployment cần chú ý:

- Model size.
- RAM.
- Battery.
- Nhiệt độ thiết bị.
- Khả năng chạy offline.
- Tốc độ startup.
- Privacy.

---

# 7. Serialization và Packaging

## 7.1 Serialization là gì?

Serialization là chuyển model thành định dạng có thể lưu trữ hoặc truyền đi.

Ví dụ:

- `.pt`, `.pth` cho PyTorch.
- `SavedModel` cho TensorFlow.
- `.onnx` cho ONNX.
- `.tflite` cho TensorFlow Lite.

Hiểu đơn giản:

> Serialization là đóng băng model thành file.

---

## 7.2 ONNX

**ONNX — Open Neural Network Exchange** là định dạng trung gian giúp model có thể chuyển giữa nhiều framework/runtime.

Ví dụ:

```text
PyTorch model -> ONNX -> ONNX Runtime
TensorFlow model -> ONNX -> ONNX Runtime
ONNX -> TensorRT
```

PyTorch export:

```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx"
)
```

TensorFlow có thể dùng công cụ như `tf2onnx`.

ONNX hữu ích khi:

- Train bằng một framework nhưng deploy bằng runtime khác.
- Cần tối ưu trên TensorRT/OpenVINO/ONNX Runtime.
- Cần deploy sang C++, C#, Java.

---

## 7.3 Packaging là gì?

Packaging là đóng gói toàn bộ những thứ cần để chạy model.

Một package production thường gồm:

- Serialized model file.
- Preprocessing code.
- Postprocessing code.
- Label mapping.
- Config.
- Runtime dependencies.
- Dockerfile hoặc container image.
- API server code.
- Version metadata.

Ví dụ với image classification:

```text
model.onnx
preprocess.py
postprocess.py
labels.json
requirements.txt
app.py
Dockerfile
```

Điểm rất quan trọng:

> Model file một mình thường chưa đủ để production chạy đúng.

Nếu preprocessing khác lúc train, accuracy production có thể sai nghiêm trọng.

Ví dụ sai phổ biến:

- Train normalize ảnh bằng ImageNet mean/std.
- Deploy quên normalize.
- Train input RGB.
- Deploy đọc ảnh BGR bằng OpenCV nhưng không convert.
- Train resize/crop theo một cách.
- Deploy resize/crop khác.

---

# 8. Model Serving và Inference

## 8.1 Model Serving

Model serving là làm cho model có thể nhận request và trả prediction.

Các cách serving phổ biến:

### HTTP REST API

Dễ dùng, phổ biến.

Framework:

- Flask.
- FastAPI.
- TensorFlow Serving.
- TorchServe.
- BentoML.
- KServe.

Ví dụ FastAPI đơn giản:

```python
from fastapi import FastAPI, UploadFile
import numpy as np

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):
    image_bytes = await file.read()

    # preprocess image_bytes
    # run model
    # postprocess output

    return {"label": "cat", "score": 0.98}
```

### gRPC

gRPC phù hợp khi cần:

- Hiệu năng cao.
- Contract rõ ràng qua protobuf.
- Giao tiếp service-to-service.
- Streaming hoặc multi-language client.

Nhược điểm:

- Phức tạp hơn REST.
- Debug thủ công khó hơn HTTP JSON.

### Cloud managed service

AWS, Azure, GCP cung cấp dịch vụ deploy model được quản lý sẵn.

Ưu điểm:

- Dễ scale.
- Ít quản lý infrastructure.
- Tích hợp monitoring/logging.

Nhược điểm:

- Vendor lock-in.
- Chi phí.
- Ít kiểm soát hơn tự quản lý.

---

## 8.2 Inference

Inference là quá trình dùng model đã deploy để tạo output từ input mới.

Vòng đời đơn giản:

```text
Client gửi ảnh
      ↓
Server nhận request
      ↓
Preprocess ảnh
      ↓
Run model
      ↓
Postprocess output
      ↓
Trả kết quả
```

Với Computer Vision, postprocess có thể gồm:

- Softmax classification.
- Non-Maximum Suppression cho object detection.
- Decode bounding boxes.
- Resize mask về ảnh gốc cho segmentation.
- Convert keypoint coordinates.

---

# 9. Kubernetes trong deployment

Kubernetes là nền tảng orchestration container.

Dùng Kubernetes giúp:

- Scale nhiều replica model server.
- Rolling update.
- Rollback.
- Health check.
- Load balancing.
- Quản lý resource CPU/GPU.
- Deploy trong production ổn định hơn.

Phù hợp khi:

- Có nhiều service.
- Cần scale linh hoạt.
- Có DevOps/MLOps pipeline.
- Chạy nhiều model hoặc nhiều version model.

Không nên dùng Kubernetes nếu hệ thống rất nhỏ và team chưa cần độ phức tạp đó.

---

# 10. Hugging Face Inference Endpoints

Hugging Face Inference Endpoints là dịch vụ managed deployment.

Ý nghĩa:

- Deploy model mà không cần tự quản lý container/GPU.
- Hỗ trợ Transformers, Diffusers hoặc custom model.
- Có thể dùng cho production.
- Giảm công sức infrastructure.

Phù hợp khi:

- Muốn deploy nhanh.
- Đang dùng model trên Hugging Face Hub.
- Không muốn tự quản lý GPU server.

---

# 11. Best practices cho production deployment

## 11.1 MLOps

MLOps áp dụng nguyên tắc DevOps cho Machine Learning.

Bao gồm:

- Version control code.
- Version control dataset.
- Version control model.
- CI/CD.
- Automated testing.
- Model registry.
- Monitoring.
- Rollback.
- Re-training pipeline.
- Documentation.

Điểm quan trọng:

> Production ML không chỉ là deploy model. Nó là vận hành một hệ thống có dữ liệu thay đổi theo thời gian.

---

## 11.2 Load testing

Load testing mô phỏng tải thực tế để kiểm tra hệ thống.

Cần đo:

- Latency trung bình.
- P95/P99 latency.
- Throughput request/second.
- CPU/GPU utilization.
- Memory usage.
- Error rate.

Ví dụ:

```text
Average latency: 40 ms
P95 latency: 90 ms
P99 latency: 180 ms
```

P95/P99 thường quan trọng hơn latency trung bình vì người dùng sẽ cảm nhận các request chậm nhất.

---

## 11.3 Anomaly detection

Cần phát hiện bất thường trong input hoặc output.

Ví dụ:

- Input data khác phân phối train.
- Camera bị mờ.
- Ảnh ban đêm nhiều hơn dữ liệu train.
- Tỷ lệ một class tăng đột ngột.
- Confidence giảm bất thường.

Một khái niệm quan trọng là **distribution shift**.

Distribution shift xảy ra khi dữ liệu production khác dữ liệu training.

Ví dụ:

```text
Train: ảnh ban ngày, rõ nét
Production: ảnh ban đêm, mưa, rung, thiếu sáng
```

Khi đó accuracy có thể giảm mạnh dù model không hề thay đổi.

---

## 11.4 Real-time monitoring

Cần monitor liên tục:

- Latency.
- Throughput.
- Error rate.
- Model confidence.
- Input statistics.
- Output distribution.
- Resource usage.
- Drift signal.

Ví dụ cảnh báo:

```text
Nếu average confidence giảm từ 0.92 xuống 0.55 trong 10 phút,
gửi alert cho team.
```

---

## 11.5 Security và Privacy

Cần bảo vệ dữ liệu và model.

Các điểm cần chú ý:

- Mã hóa dữ liệu khi truyền qua mạng.
- HTTPS/TLS.
- Access control.
- Authentication/authorization.
- Không log dữ liệu nhạy cảm.
- Giới hạn quyền truy cập model endpoint.
- Bảo vệ model khỏi abuse hoặc extraction attack.

Với Computer Vision, ảnh có thể chứa:

- Khuôn mặt.
- Biển số xe.
- Tài liệu cá nhân.
- Không gian riêng tư.

Vì vậy privacy rất quan trọng.

---

## 11.6 A/B testing

A/B testing dùng để so sánh model mới với model cũ.

Ví dụ:

```text
90% traffic -> model hiện tại
10% traffic -> model mới
```

Sau đó so sánh:

- Accuracy proxy.
- User engagement.
- Latency.
- Error rate.
- Business metric.

Nếu model mới tốt và ổn định, tăng dần traffic.

---

## 11.7 Continuous evaluation

Model cần được đánh giá liên tục sau deployment.

Lý do:

- Dữ liệu thay đổi.
- Người dùng thay đổi.
- Môi trường thay đổi.
- Camera/sensor thay đổi.
- Model có thể xuống cấp theo thời gian.

Cần chuẩn bị rollback nhanh nếu model mới gây lỗi.

---

## 11.8 Documentation

Cần ghi lại:

- Model architecture.
- Dataset train/validation.
- Preprocessing.
- Postprocessing.
- Metrics.
- Hardware target.
- Dependency versions.
- Optimization đã dùng.
- Known limitations.
- Deployment date.
- Owner/team responsible.

Thiếu documentation khiến việc debug production rất khó.

---

# 12. Quy trình deployment gợi ý cho Computer Vision

Một pipeline thực tế có thể như sau:

```text
1. Train model
2. Evaluate trên validation/test set
3. Xác định target deployment:
   - Cloud?
   - Edge?
   - Mobile?
   - NVIDIA GPU?
   - Intel CPU?
4. Chọn kỹ thuật optimization:
   - Quantization
   - Pruning
   - Distillation
   - TensorRT/OpenVINO/ONNX Runtime
5. Export/serialize model
6. Kiểm tra parity:
   - Output model gốc vs model optimized
7. Benchmark:
   - Latency
   - Throughput
   - Memory
   - Accuracy
8. Package:
   - Model
   - Preprocess
   - Postprocess
   - Dependencies
9. Deploy serving API hoặc deploy lên device
10. Monitor production
11. A/B test hoặc canary rollout
12. Rollback/retrain khi cần
```

---

# 13. Các điểm kỹ thuật cần nắm chắc

Nếu chỉ nhớ các ý quan trọng nhất của Unit 9, hãy nhớ:

1. **Model train tốt chưa chắc deploy tốt.**
2. Optimization giúp model nhỏ hơn, nhanh hơn, ít tốn tài nguyên hơn.
3. Ba yếu tố phải cân bằng là:
   - Accuracy.
   - Latency.
   - Resource usage.
4. Quantization từ FP32 sang INT8 có thể giảm model size khoảng 4 lần.
5. Pruning loại bỏ phần ít quan trọng của model.
6. Knowledge distillation dùng model lớn dạy model nhỏ.
7. ONNX là định dạng trung gian quan trọng để chuyển model giữa framework/runtime.
8. TensorRT mạnh cho NVIDIA GPU.
9. OpenVINO mạnh cho Intel hardware.
10. TensorFlow Lite/Edge TPU phù hợp edge/mobile.
11. Deployment không chỉ là model file; cần preprocessing, postprocessing, dependencies và serving.
12. Production cần monitoring, load testing, anomaly detection, A/B testing và rollback.
13. Distribution shift là rủi ro rất lớn trong ML production.
14. Với Computer Vision, sai khác nhỏ trong preprocessing có thể làm model production sai nghiêm trọng.

---

# 14. Checklist ngắn trước khi deploy model CV

```text
[ ] Model đã được evaluate trên test set phù hợp chưa?
[ ] Target hardware là gì?
[ ] Latency requirement là bao nhiêu?
[ ] Memory limit là bao nhiêu?
[ ] Có cần real-time không?
[ ] Có cần chạy offline không?
[ ] Đã thử quantization/pruning/distillation chưa?
[ ] Đã benchmark model optimized chưa?
[ ] Accuracy sau optimization giảm bao nhiêu?
[ ] Preprocessing production có giống training không?
[ ] Postprocessing đã được kiểm thử chưa?
[ ] Model đã được serialize đúng format chưa?
[ ] Có Docker/container/package đầy đủ chưa?
[ ] Có health check không?
[ ] Có monitoring latency/error/resource không?
[ ] Có plan rollback không?
[ ] Có version model và config không?
```

---