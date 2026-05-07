# ts2img-lightcnn

Repo mẫu cho bài nghiên cứu:

**So sánh hiệu quả các kỹ thuật biểu diễn ảnh 2D từ dữ liệu chuỗi thời gian 1D cho bài toán phân loại bằng mạng nơ-ron tích chập nhẹ**.

Repo này hỗ trợ:

- Chạy thử trên laptop bằng VS Code.
- Chạy thực nghiệm cuối trên Google Colab GPU.
- Lưu checkpoint vào Google Drive khi chạy trên Colab.
- Tự khôi phục trạng thái huấn luyện bằng `BackupAndRestore` khi Colab bị ngắt.
- Xuất kết quả vào `results/summary_results.csv` để đưa vào bài báo.

---

## 1. Cấu trúc thư mục

```text
ts2img-lightcnn/
├── configs/
│   └── default.yaml
├── data/
│   └── .gitkeep
├── notebooks/
│   └── colab_run.ipynb
├── results/
│   └── .gitkeep
├── runs/
│   └── .gitkeep
├── scripts/
│   ├── run_colab_example.sh
│   └── run_local_test.ps1
├── src/
│   ├── __init__.py
│   ├── data_ucr.py
│   ├── evaluate.py
│   ├── models.py
│   ├── run_experiments.py
│   ├── train.py
│   ├── transforms.py
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 2. Cài đặt trên laptop Windows

Mở PowerShell tại thư mục repo:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Chạy test nhanh 2 epoch:

```powershell
python -m src.train --dataset GunPoint --representation gaf --model_type light2dcnn --epochs 2 --batch_size 16 --image_size 64 --seed 42
```

Hoặc chạy script:

```powershell
.\scripts\run_local_test.ps1
```

---

## 3. Chạy trên Google Colab

### Bước 1: bật GPU

Vào menu:

```text
Runtime > Change runtime type > GPU
```

### Bước 2: mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Bước 3: clone repo

```python
%cd /content
!git clone https://github.com/USERNAME/ts2img-lightcnn.git
%cd ts2img-lightcnn
```

Thay `USERNAME` bằng tài khoản GitHub của anh.

### Bước 4: cài thư viện

```python
!pip install -r requirements.txt
```

### Bước 5: chạy thực nghiệm

```python
!python -m src.train \
  --dataset GunPoint \
  --representation gaf \
  --model_type light2dcnn \
  --epochs 50 \
  --batch_size 32 \
  --image_size 64 \
  --seed 42
```

Khi chạy trên Colab, checkpoint và kết quả tự lưu vào:

```text
/content/drive/MyDrive/ts2img-lightcnn/
```

Vì vậy khi Colab bị ngắt hoặc hết GPU, lần sau mount Drive và chạy lại đúng lệnh cũ, chương trình sẽ cố gắng khôi phục từ trạng thái backup gần nhất.

---

## 4. Các lệnh thực nghiệm cơ bản

### 1D-CNN baseline

```bash
python -m src.train --dataset GunPoint --representation none --model_type cnn1d --epochs 50 --seed 42
```

### GAF + Light 2D-CNN

```bash
python -m src.train --dataset GunPoint --representation gaf --model_type light2dcnn --epochs 50 --seed 42
```

### MTF + Light 2D-CNN

```bash
python -m src.train --dataset GunPoint --representation mtf --model_type light2dcnn --epochs 50 --seed 42
```

### RP + Light 2D-CNN

```bash
python -m src.train --dataset GunPoint --representation rp --model_type light2dcnn --epochs 50 --seed 42
```

### STFT + Light 2D-CNN

```bash
python -m src.train --dataset GunPoint --representation stft --model_type light2dcnn --epochs 50 --seed 42
```

### Depthwise Separable 2D-CNN

```bash
python -m src.train --dataset GunPoint --representation gaf --model_type depthwise2dcnn --epochs 50 --seed 42
```

---

## 5. Chạy nhiều thí nghiệm

```bash
python -m src.run_experiments \
  --datasets GunPoint \
  --representations gaf,mtf,rp,stft \
  --model_type light2dcnn \
  --seeds 42,2024,2026 \
  --epochs 50 \
  --batch_size 32 \
  --image_size 64
```

Nên chạy từng nhóm nhỏ trên Colab để tránh quá thời lượng GPU.

---

## 6. Đưa dữ liệu UCR khác vào repo

Nếu muốn dùng ECG200, Coffee, FordA, Wafer..., đặt dữ liệu theo cấu trúc:

```text
data/UCR/ECG200/ECG200_TRAIN.tsv
data/UCR/ECG200/ECG200_TEST.tsv
```

Sau đó chạy:

```bash
python -m src.train --dataset ECG200 --representation gaf --model_type light2dcnn --epochs 50 --seed 42
```

---

## 7. File kết quả

Kết quả tổng hợp nằm tại:

```text
results/summary_results.csv
```

Trên Colab:

```text
/content/drive/MyDrive/ts2img-lightcnn/results/summary_results.csv
```

Các chỉ số được lưu:

- `accuracy`
- `macro_f1`
- `params`
- `training_time_sec`
- `inference_time_per_sample_sec`
- `best_model_path`
- `run_dir`

---

## 8. Gợi ý workflow chuẩn

1. Sửa code trên laptop bằng VS Code.
2. Chạy local 2 epoch với `GunPoint`.
3. Commit và push lên GitHub.
4. Clone/pull code trên Colab.
5. Mount Google Drive.
6. Chạy thực nghiệm chính.
7. Nếu Colab ngắt, chạy lại cùng lệnh để tiếp tục.
8. Lấy `summary_results.csv` để cập nhật bảng kết quả trong bài báo.
