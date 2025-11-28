# 📸 Face Attendance System (Hệ thống Điểm danh bằng Khuôn mặt)

Ứng dụng web tương tác này được xây dựng bằng **Streamlit** và **OpenCV** để thực hiện điểm danh tự động bằng công nghệ nhận dạng khuôn mặt. Mục tiêu là cung cấp một giải pháp điểm danh nhanh chóng và hiệu quả, giảm thiểu các quy trình thủ công.

Nó cho phép người dùng đăng ký khuôn mặt mới (Enrollment) và sau đó sử dụng camera để điểm danh (Attendance) các khuôn mặt đã đăng ký.

---

## 💡 Công Nghệ Sử Dụng

Dự án này là một ví dụ điển hình về ứng dụng Khoa học Dữ liệu trong đời sống, sử dụng các thư viện Python chuyên dụng:

* **Python:** Ngôn ngữ lập trình chính.
* **Streamlit:** Xây dựng giao diện web tương tác (Front-end) chỉ bằng Python.
* **OpenCV (cv2):** Xử lý hình ảnh, truy cập camera và nhận dạng khuôn mặt.
* **face_recognition (hoặc dlib):** Thư viện mạnh mẽ cho việc encoding (mã hóa) và so sánh khuôn mặt, dựa trên thuật toán Deep Learning.
* **Pandas:** Quản lý và lưu trữ dữ liệu điểm danh trong file CSV.

---

## ⚙️ Cấu Trúc File Chính & Cài Đặt

### 1. Cấu Trúc Dự Án
FaceRecognitionProject/  
├── app.py               # File chính  
├── requirements.txt     # Các thư viện cần thiết   
├── README.md            #   
├── ImageAttendace/      # Thư mục chứa ảnh  
│   ├── Nguyen_Van_A.jpg  
│   └── Tran_Thi_B.jpg   
├── encodings.pkl  # Chứa vector mã hóa  
└── attendance.csv

### 2. Yêu Cầu

* **Python 3.x** đã được cài đặt.
* **Camera (Webcam)** hoạt động tốt để nhận diện.

### 3. Thiết Lập Môi Trường (Setup)

**a. Clone Kho Lưu Trữ:**
Sử dụng Git để tải mã nguồn về máy của bạn:

```bash
git clone
cd [Tên thư mục dự án]

python -m venv venv
# Trên Windows
.\venv\Scripts\activate
# Trên macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

streamlit run app.py