# 1. Synthetic Data là gì?

**Synthetic data** là dữ liệu được tạo ra bằng mô hình toán học, thuật toán, mô phỏng vật lý, mô hình thống kê hoặc mô hình học sâu, thay vì được thu thập trực tiếp từ sự kiện thật ngoài đời.

Ví dụ:

- Ảnh người tạo bởi GAN hoặc diffusion model.
- Ảnh xe, đường, pedestrian tạo từ game/simulator như GTA5, CARLA.
- Ảnh y tế giả lập để huấn luyện mô hình phát hiện bệnh.
- Point cloud tạo từ mô phỏng LiDAR.
- Ảnh object render từ Blender với nhãn segmentation, depth, normal, pose.

Điểm quan trọng:

> Synthetic data **không phải dữ liệu thật**, nhưng cố gắng bắt chước phân phối thống kê của dữ liệu thật.

Một synthetic dataset tốt nên giống dữ liệu thật ở các khía cạnh:

- Phân phối màu sắc.
- Hình dạng.
- Texture.
- Background.
- Lighting.
- Camera angle.
- Noise.
- Occlusion.
- Tỉ lệ class.
- Các trường hợp hiếm/outlier.

---

# 2. Tại sao cần Synthetic Data?

Dữ liệu thật thường có nhiều vấn đề:

## 2.1 Thiếu dữ liệu

Một số lĩnh vực rất khó thu thập đủ dữ liệu, ví dụ:

- Ảnh y tế.
- Tai nạn giao thông.
- Vũ khí trong CCTV.
- Động vật quý hiếm.
- Lỗi công nghiệp hiếm gặp.
- Dữ liệu robot trong môi trường nguy hiểm.

Synthetic data có thể bổ sung thêm mẫu huấn luyện.

---

## 2.2 Annotation đắt và chậm

Trong computer vision, nhãn có thể rất tốn công:

- Bounding box.
- Semantic segmentation mask.
- Instance segmentation.
- Depth map.
- Optical flow.
- Pose 6D.
- Surface normal.
- Point cloud label.

Với dữ liệu render từ 3D scene, ta có thể tự động lấy nhãn chính xác gần như “miễn phí” vì hệ thống biết sẵn:

- Object ở đâu.
- Camera ở đâu.
- Depth từng pixel.
- Class từng pixel.
- Pose object.
- Normal surface.

---

## 2.3 Quyền riêng tư

Dữ liệu y tế, khuôn mặt, camera giám sát thường không thể chia sẻ công khai.

Synthetic data có thể giúp giảm rủi ro lộ thông tin cá nhân, nhưng cần nhớ:

> Synthetic data không tự động private. Nếu mô hình sinh học thuộc dữ liệu train quá kỹ, nó vẫn có thể leak thông tin gốc.

Nếu cần bảo mật nghiêm túc, nên dùng các cơ chế như **Differential Privacy** ở quá trình sinh dữ liệu, không chỉ kiểm tra dataset cuối.

---

## 2.4 Cân bằng dữ liệu và giảm bias

Nếu real dataset bị lệch class, synthetic data có thể bổ sung class thiếu.

Ví dụ:

- Dataset bệnh A có quá ít ảnh positive.
- Dataset autonomous driving thiếu ảnh ban đêm/mưa/tuyết.
- Dataset động vật thiếu loài hiếm.
- Dataset face thiếu nhóm tuổi hoặc màu da nào đó.

Synthetic data có thể giúp cân bằng lại, nhưng nếu sinh không cẩn thận, nó cũng có thể sao chép hoặc khuếch đại bias.

---

# 3. Các cách tạo Synthetic Data

Unit 10 nhắc đến ba hướng chính:

1. **3D rendering / Physically Based Rendering**
2. **Point cloud / 3D data**
3. **Generative models: GAN, Diffusion**

---

# 4. Physically Based Rendering, Blender và BlenderProc

## 4.1 Physically Based Rendering là gì?

**Physically Based Rendering**, viết tắt là **PBR**, là kỹ thuật render ảnh bằng cách mô phỏng cách ánh sáng tương tác với vật liệu ngoài đời.

PBR quan tâm đến:

- Ánh sáng chiếu vào object.
- Bóng đổ.
- Phản xạ.
- Độ nhám bề mặt.
- Metallic/non-metallic material.
- Texture.
- Normal map.
- Transparency.
- Refraction.
- Multiple light bounces.

Ví dụ quả táo bóng sẽ có highlight phản chiếu ánh sáng, trong khi vải nhám sẽ tán xạ ánh sáng mềm hơn.

Mục tiêu của PBR:

> Tạo ảnh synthetic càng giống ảnh thật càng tốt để giảm “reality gap”.

---

## 4.2 Reality gap là gì?

**Reality gap** là khoảng cách giữa dữ liệu synthetic và dữ liệu thật.

Ví dụ mô hình được train trên ảnh render quá sạch:

- Background luôn đẹp.
- Object luôn nằm giữa ảnh.
- Lighting luôn hoàn hảo.
- Không có blur.
- Không có noise.
- Không có occlusion.

Khi đem ra đời thật, mô hình có thể fail vì dữ liệu thật phức tạp hơn.

Cách giảm reality gap:

- Random camera position.
- Random lighting.
- Random object pose.
- Random background.
- Random texture.
- Thêm noise, blur, occlusion.
- Domain randomization.
- Dùng PBR.
- Mix synthetic + real data.
- Fine-tune trên real data.

---

## 4.3 Blender

**Blender** là phần mềm 3D mã nguồn mở dùng để:

- Tạo 3D object.
- Tạo scene.
- Render ảnh.
- Mô phỏng ánh sáng.
- Gán material/texture.
- Xuất annotation như depth, normal, segmentation.

Ví dụ workflow tạo ảnh synthetic bằng Blender:

1. Tạo hoặc import object 3D.
2. Tạo background/scene.
3. Gán material và texture.
4. Đặt camera.
5. Đặt light.
6. Random vị trí/rotation/scale object.
7. Render ảnh.
8. Xuất nhãn đi kèm.

Có thể dùng Python API của Blender là `bpy` để tự động hóa.

Ví dụ ý tưởng script:

```python
import bpy
import random
import math

obj = bpy.data.objects["Elephant"]

for i in range(1000):
    obj.location = (
        random.uniform(-2, 2),
        random.uniform(-2, 2),
        0,
    )

    obj.rotation_euler = (
        0,
        0,
        random.uniform(0, 2 * math.pi),
    )

    bpy.context.scene.render.filepath = f"outputs/image_{i:04d}.png"
    bpy.ops.render.render(write_still=True)
```

Đoạn này minh họa cách random vị trí và rotation để render nhiều ảnh.

---

## 4.4 BlenderProc

Blender mạnh nhưng khó học. **BlenderProc** là thư viện xây trên Blender để tạo synthetic dataset dễ hơn.

Cài đặt:

```bash
pip install blenderProc
```

Chạy script:

```bash
blenderproc run your_script.py
```

BlenderProc hỗ trợ:

- Procedural scene generation.
- Randomization.
- Physics simulation.
- Segmentation mask.
- Depth map.
- Normal map.
- Pose estimation.
- Large-scale rendering.
- Parallel processing.

Các loại output thường có:

- RGB image.
- Semantic segmentation.
- Instance segmentation.
- Depth.
- Surface normal.
- Object pose.
- Camera intrinsic/extrinsic.

Điểm kỹ thuật cần nhớ:

> BlenderProc phải chạy trong môi trường Python của Blender vì nó cần truy cập Blender API.

---

# 5. Synthetic Dataset trong Computer Vision

Unit 10 liệt kê nhiều synthetic dataset nổi bật theo nhóm bài toán.

---

## 5.1 Low-level vision

### Optical flow

Optical flow là bài toán ước lượng chuyển động pixel giữa hai frame.

Dataset tiêu biểu:

- **Middlebury**
- **MPI-Sintel**
- **Playing for Benchmarks**

Synthetic data rất hữu ích vì optical flow ground truth ngoài đời rất khó lấy chính xác.

---

### Stereo matching

Stereo matching tìm disparity giữa ảnh trái/phải từ stereo camera.

Dataset:

- Flying Chairs
- Flying Chairs 3D
- Driving
- Monkaa
- Middlebury 2014
- Tsukuba Stereo

Nhãn thường gồm:

- Disparity map.
- Occlusion map.
- Depth.
- Scene flow.

---

## 5.2 Autonomous driving

Dataset/simulator:

- **CARLA**
- **GTA5**
- **SYNTHIA**
- **Virtual KITTI 2**
- **ApolloScape**
- **Driving in the Matrix**
- **ProcSy**

Các task:

- Semantic segmentation.
- Object detection.
- Lane detection.
- Depth estimation.
- Tracking.
- 3D object detection.
- Autonomous navigation.

Ví dụ CARLA cung cấp:

- RGB camera.
- Depth map.
- Semantic segmentation.
- Bounding box.
- Vehicle pose.
- Sensor customization.

Điểm quan trọng:

> Simulator giúp tạo tình huống nguy hiểm hoặc hiếm gặp mà ngoài đời khó thu thập, như tai nạn, thời tiết xấu, ban đêm.

---

## 5.3 Indoor simulation/navigation

Dataset/platform:

- Habitat
- Minos
- House3D

Dùng cho:

- Robot navigation.
- Embodied AI.
- Human-robot interaction.
- Indoor scene understanding.

---

## 5.4 Human action / human body

Dataset:

- PHAV
- SURREAL

SURREAL tạo hàng triệu frame người synthetic từ motion capture, có:

- Pose.
- Depth map.
- Segmentation mask.
- Human part segmentation.

---

## 5.5 Face recognition

Dataset:

- FaceSynthetics
- FFHQ

FaceSynthetics dùng synthetic face với ground truth label.

Ứng dụng:

- Face landmark.
- Face parsing.
- Recognition.
- Bias analysis.

---

## 5.6 3D object dataset

Dataset:

- ShapeNetCore
- PartNet
- Falling Things
- SceneNet RGB-D
- YCB-Video
- ABO
- Pix3D

Task:

- 3D reconstruction.
- 6D pose estimation.
- 3D object understanding.
- Part segmentation.
- Multi-view retrieval.
- Material prediction.

---

# 6. Point Cloud

## 6.1 Point cloud là gì?

**Point cloud** là tập hợp các điểm trong không gian 3D. Mỗi điểm thường có tọa độ:

```text
[x, y, z]
```

Ngoài tọa độ, mỗi điểm có thể có thêm:

- RGB color.
- Normal vector.
- Albedo.
- BRDF.
- Intensity.
- Class label.
- Instance label.

Point cloud dùng nhiều trong:

- Autonomous driving.
- Robotics.
- AR/VR.
- 3D reconstruction.
- LiDAR perception.
- Mapping.
- Scene understanding.

---

## 6.2 LiDAR

**LiDAR** đo khoảng cách bằng cách phát tia laser và đo thời gian tia phản xạ quay lại.

Nguyên lý đơn giản:

```text
distance = speed_of_light * time_of_flight / 2
```

Chia 2 vì tia đi từ sensor đến object rồi quay lại.

LiDAR tạo ra point cloud biểu diễn bề mặt vật thể và môi trường.

---

## 6.3 Các định dạng file 3D phổ biến

Unit 10 nhắc các định dạng:

### PLY

Chứa:

- Vertex list.
- Face list.
- Có thể thêm property như color, normal.

Phù hợp cho point cloud/mesh có metadata.

---

### STL

Dùng nhiều trong 3D printing.

Đặc điểm:

- Biểu diễn surface bằng triangle facets.
- Không lưu color/texture.
- Mỗi facet có normal và 3 vertex.

---

### OFF

Biểu diễn geometry bằng polygon surface.

Có thể ở dạng ASCII hoặc binary.

---

### 3DS

Định dạng của Autodesk 3D Studio.

Dữ liệu dạng binary chunk, lưu:

- Shape.
- Lighting.
- Camera/view.

---

### X3D

Định dạng XML cho 3D graphics.

Có thể lưu:

- Vector/raster graphics.
- Transparency.
- Lighting.
- Animation.
- Color.

---

### DAE / COLLADA

Định dạng XML trao đổi asset 3D giữa phần mềm khác nhau.

Ưu điểm:

- Tính tương thích cao.
- Dùng để chuyển scene/model/material giữa tool.

---

## 6.4 Thư viện Python cho point cloud

Cài đặt:

```bash
pip install point-cloud-utils
pip install open3d
```

Hoặc bản CPU nhẹ hơn:

```bash
pip install open3d-cpu
```

Ví dụ đọc và hiển thị point cloud bằng Open3D:

```python
import open3d as o3d

pcd = o3d.io.read_point_cloud("sample.ply")

print(pcd)
print("Number of points:", len(pcd.points))

o3d.visualization.draw_geometries([pcd])
```

Ví dụ lấy numpy array:

```python
import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("sample.ply")
points = np.asarray(pcd.points)

print(points.shape)  # (N, 3)
print(points[:5])
```

---

# 7. Synthetic Data bằng GAN/DCGAN

## 7.1 GAN là gì?

**GAN – Generative Adversarial Network** gồm hai mạng:

1. **Generator**
2. **Discriminator**

Generator tạo ảnh giả từ random noise.

Discriminator phân biệt ảnh thật và ảnh giả.

Hai mạng học đối kháng:

- Generator cố lừa Discriminator.
- Discriminator cố phát hiện ảnh giả.
- Qua training, Generator tạo ảnh ngày càng thật hơn.

---

## 7.2 DCGAN

**DCGAN – Deep Convolutional GAN** là GAN dùng convolutional layers, thường dùng cho ảnh.

Ứng dụng trong Unit 10:

> Tạo ảnh X-ray phổi synthetic.

Dataset thật dùng:

- Chest X-Ray Images Pneumonia.
- Khoảng 5,863 ảnh JPEG.
- Hai class: Pneumonia và Normal.
- Ảnh bệnh nhi 1–5 tuổi.

---

## 7.3 Kiến trúc Generator trong DCGAN

Input:

```text
random noise vector, ví dụ kích thước 100
```

Output:

```text
ảnh 128 x 128 x 3
```

Các block thường gồm:

- ConvTranspose2D hoặc upsampling convolution.
- BatchNorm.
- ReLU.
- Layer cuối dùng Tanh.

Tanh đưa pixel về khoảng:

```text
[-1, 1]
```

---

## 7.4 Kiến trúc Discriminator

Input:

```text
ảnh thật hoặc ảnh giả
```

Output:

```text
xác suất ảnh là real
```

Các block thường gồm:

- Conv2D.
- LeakyReLU.
- BatchNorm ở một số layer.
- Layer cuối Sigmoid.

Sigmoid trả về xác suất trong khoảng:

```text
[0, 1]
```

---

## 7.5 Preprocessing ảnh

Ví dụ dùng PyTorch transform:

```python
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

image_size = 128

transform = Compose([
    Resize(image_size),
    CenterCrop(image_size),
    ToTensor(),
    Normalize([0.5], [0.5]),
])
```

Nếu ảnh RGB:

```python
Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
```

Vì Generator dùng Tanh, ảnh thật cũng nên normalize về `[-1, 1]`.

---

## 7.6 Vấn đề của GAN

GAN có thể tạo ảnh khá thật nhưng có nhiều rủi ro:

### Mode collapse

Generator chỉ tạo vài kiểu ảnh lặp lại.

Ví dụ trong ảnh X-ray phổi, các ảnh synthetic nhìn tương tự nhau.

### Ảnh mờ/hazy

Một số ảnh có noise không tự nhiên hoặc thiếu chi tiết.

### Khó đánh giá chất lượng y tế

Trong medical imaging, ảnh nhìn “ổn” với người thường chưa chắc đúng chuyên môn.

Vì vậy cần:

> Human-in-the-middle, ví dụ bác sĩ/radiologist kiểm tra ảnh synthetic trước khi dùng.

---

# 8. Synthetic Data bằng Diffusion Models

## 8.1 Diffusion model hoạt động thế nào?

Diffusion model học cách biến noise thành ảnh.

Training gồm hai ý tưởng:

1. Thêm Gaussian noise dần vào ảnh thật.
2. Huấn luyện model học cách denoise từng bước.

Quá trình generation:

```text
random noise -> denoise step 1 -> denoise step 2 -> ... -> final image
```

Diffusion model thường tạo ảnh chất lượng cao hơn GAN, nhưng inference chậm hơn vì cần nhiều bước denoising.

---

## 8.2 Stable Diffusion

Stable Diffusion gồm ba thành phần chính:

### 1. Diffusion process

Học cách khử noise từng bước.

### 2. Image encoder/decoder

Stable Diffusion hoạt động trong latent space thay vì pixel space.

Ảnh được encode sang latent nhỏ hơn, giúp:

- Giảm chi phí tính toán.
- Tăng tốc training/inference.
- Giữ thông tin quan trọng.

### 3. Conditional encoder

Điều kiện hóa quá trình sinh ảnh bằng:

- Text prompt.
- Image.
- Depth.
- Mask.
- Audio hoặc representation khác.

Bản gốc Stable Diffusion dùng text encoder.

---

## 8.3 Diffusers pipelines

Thư viện `diffusers` hỗ trợ nhiều pipeline:

| Task | Ý nghĩa |
|---|---|
| Unconditional image generation | Sinh ảnh từ Gaussian noise |
| Text-to-image | Sinh ảnh từ prompt |
| Image-to-image | Biến đổi ảnh theo prompt |
| Inpainting | Điền vùng bị mask |
| Depth-to-image | Sinh/chỉnh ảnh giữ cấu trúc depth |

Ví dụ text-to-image:

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

prompt = "a chest x-ray image, medical scan, realistic"
image = pipe(prompt).images[0]

image.save("synthetic_xray.png")
```

Lưu ý: prompt kiểu này chỉ minh họa kỹ thuật. Với medical data thật, cần kiểm định chuyên môn rất nghiêm ngặt.

---

## 8.4 Các tình huống dùng diffusion để tạo synthetic data

### Mở rộng dataset có sẵn

Ví dụ dataset y tế chỉ có vài nghìn ảnh, diffusion có thể sinh thêm biến thể.

### Tạo dataset từ đầu

Ví dụ không có ảnh CCTV chứa loại vũ khí cụ thể, có thể dùng image-to-image/style transfer để mô phỏng camera giám sát.

### Bảo vệ riêng tư

Sinh dữ liệu thay thế dữ liệu nhạy cảm, ví dụ ảnh bệnh nhân hoặc ảnh giám sát.

Nhưng cần kiểm soát privacy leakage.

---

## 8.5 Các kỹ thuật cá nhân hóa diffusion model

### Textual Inversion

Thêm token mới vào vocabulary rồi học embedding cho token đó từ vài ảnh mẫu.

Ví dụ học token:

```text
<sks_dog>
```

Sau đó prompt:

```text
a photo of <sks_dog> in a park
```

Điểm kỹ thuật:

- Chỉ học embedding.
- Nhẹ hơn fine-tune toàn model.
- Phù hợp học một concept cụ thể.

---

### LoRA

**LoRA – Low-Rank Adaptation** fine-tune model bằng cách thêm các ma trận nhỏ có rank thấp vào weight update.

Ưu điểm:

- Ít tham số.
- Train nhanh.
- File checkpoint nhỏ.
- Base model giữ nguyên.
- Có thể kết hợp nhiều LoRA.

Ý tưởng:

```text
W' = W + ΔW
ΔW ≈ A @ B
```

Trong đó `A` và `B` nhỏ hơn nhiều so với `W`.

---

### DreamBooth

DreamBooth fine-tune text-to-image model với vài ảnh của một subject cụ thể.

Ý tưởng:

- Gán subject với rare token.
- Fine-tune để model hiểu subject đó.
- Prompt có thể đặt subject vào nhiều bối cảnh.

Ví dụ:

```text
a photo of sks_object on a table
```

Cần chọn rare token để tránh trùng nghĩa phổ biến.

---

### Custom Diffusion

Custom Diffusion học nhiều concept cùng lúc và chỉ update một phần model/text encoder.

Ưu điểm:

- Ít tham số hơn full fine-tuning.
- Có thể học nhiều concept.
- Fine-tune nhanh.

---

# 9. Các thách thức khi dùng Synthetic Data

## 9.1 Overfitting vào pattern giả

Nếu synthetic data quá đều, model học nhầm shortcut.

Ví dụ:

- Tất cả circle đều đỏ.
- Tất cả square đều xanh.
- Object luôn nằm giữa ảnh.
- Background của class A luôn khác class B.
- Object class A luôn lớn hơn class B.
- Lighting quá nhất quán.

Khi ra dữ liệu thật, model fail.

Cần kiểm tra:

- Màu sắc có đa dạng không?
- Kích thước có đa dạng không?
- Background có đa dạng không?
- Vị trí object có random không?
- Camera angle có thay đổi không?
- Lighting/weather/occlusion có thay đổi không?

---

## 9.2 Bias trong synthetic data

Synthetic data có thể:

- Thiếu diversity.
- Sao chép bias từ real data.
- Tạo class không đại diện đúng thực tế.
- Làm mô hình quá tự tin vào pattern sai.

Ví dụ nếu muốn nhận dạng aye-aye lemur nhưng synthetic dataset chỉ có ring-tailed lemur, model sẽ không học đúng loài cần nhận dạng.

---

## 9.3 Chi phí tính toán

Tạo synthetic data chất lượng cao có thể tốn:

- GPU.
- Thời gian render.
- Thời gian fine-tune.
- Lưu trữ.
- Nhân lực kiểm định.

Quy tắc thực dụng:

> Chỉ nên dùng synthetic data nếu lợi ích cuối cùng lớn hơn chi phí tạo và kiểm định nó.

---

## 9.4 Chất lượng ảnh synthetic

Ảnh synthetic chất lượng thấp có thể làm mô hình tệ hơn.

Các lỗi thường gặp:

- Texture không thật.
- Background sai.
- Lighting sai.
- Noise lạ.
- Anatomy sai trong ảnh y tế.
- Object méo.
- Sai label.
- Thiếu variation.

---

# 10. Đánh giá chất lượng Synthetic Data

Unit 10 nhắc ba metric quan trọng:

## 10.1 FID – Frechet Inception Distance

FID so sánh phân phối feature giữa real images và generated images.

Quy trình:

1. Dùng pretrained Inception model extract feature.
2. Tính mean và covariance của feature real.
3. Tính mean và covariance của feature synthetic.
4. Đo khoảng cách giữa hai phân phối.

Ý nghĩa:

```text
FID thấp hơn -> synthetic gần real hơn
```

Cẩn thận:

- FID không thay thế đánh giá downstream task.
- Với medical image, Inception pretrained trên ImageNet có thể không đại diện tốt.

---

## 10.2 IS – Inception Score

IS đo ảnh generated có dễ được Inception nhận dạng rõ không.

Ý nghĩa:

```text
IS cao hơn thường tốt hơn
```

Nhược điểm:

- Không so trực tiếp với real dataset.
- Phụ thuộc vào pretrained classifier.
- Không phù hợp mọi domain.

---

## 10.3 CAS – Classification Accuracy Score

CAS đánh giá synthetic data thông qua performance của classifier.

Một cách dùng phổ biến:

1. Train classifier trên synthetic data.
2. Test trên real data.
3. Nếu accuracy tốt, synthetic có ích.

Hoặc:

1. Train trên real + synthetic.
2. So với train real-only.
3. Kiểm tra synthetic có cải thiện không.

Đây thường là đánh giá thực tế hơn cho bài toán cụ thể.

---

# 11. Checklist kỹ thuật khi dùng Synthetic Data

Trước khi đưa synthetic data vào training, nên kiểm tra:

## Dataset design

- Synthetic data đại diện đúng task không?
- Có bao phủ case hiếm không?
- Có cân bằng class không?
- Có đa dạng background, pose, scale, lighting không?
- Có tránh shortcut không?

## Label quality

- Segmentation mask đúng không?
- Bounding box khớp object không?
- Depth có scale đúng không?
- Pose/camera parameter có chính xác không?

## Domain gap

- Synthetic có quá sạch không?
- Real data có blur/noise/occlusion mà synthetic thiếu không?
- Có cần domain randomization không?
- Có cần fine-tune trên real data không?

## Privacy

- Model sinh có memorize ảnh gốc không?
- Có cần differential privacy không?
- Có kiểm tra nearest neighbors giữa synthetic và real không?

## Evaluation

- Có đánh giá FID/IS/CAS không?
- Có test trên real validation/test set không?
- Có human expert review không, nhất là y tế?
- Synthetic có thật sự cải thiện downstream metric không?

---

# 12. Kết luận quan trọng

Synthetic data rất mạnh trong computer vision vì nó giúp:

- Giải quyết thiếu dữ liệu.
- Tự động tạo nhãn chính xác.
- Bổ sung class hiếm.
- Giảm chi phí annotation.
- Mô phỏng tình huống nguy hiểm/hiếm.
- Hỗ trợ privacy nếu làm đúng.

Nhưng synthetic data không phải “thuốc chữa bách bệnh”.

Các rủi ro chính:

- Reality gap.
- Bias.
- Overfitting vào pattern giả.
- Mode collapse với GAN.
- Privacy leakage.
- Chi phí tính toán cao.
- Chất lượng synthetic không ổn định.

Cách dùng thực tế nhất thường là:

```text
Real data baseline
-> thêm synthetic data có kiểm soát
-> đánh giá trên real validation/test set
-> chỉ giữ synthetic nếu downstream performance cải thiện
```

Nếu dùng cho domain nhạy cảm như y tế, bắt buộc cần chuyên gia đánh giá và kiểm định nghiêm ngặt.