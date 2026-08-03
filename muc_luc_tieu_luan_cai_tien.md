# MỤC LỤC TIỂU LUẬN TỐT NGHIỆP (ĐỀ XUẤT CẢI TIẾN)

**Đề tài:** Nghiên cứu cải tiến TickNets bằng cơ chế attention CBAM cho bài toán phân loại ảnh

---

**DANH SÁCH CHỮ VIẾT TẮT**
**DANH MỤC HÌNH ẢNH**
**DANH MỤC BẢNG**
**TÓM TẮT**
**MỤC LỤC**

**MỞ ĐẦU**
1. LÝ DO CHỌN ĐỀ TÀI
2. MỤC TIÊU VÀ PHẠM VI NGHIÊN CỨU
3. Ý NGHĨA KHOA HỌC VÀ THỰC TIỄN CỦA ĐỀ TÀI

---

### CHƯƠNG 1: TỔNG QUAN ĐỀ TÀI

---

### CHƯƠNG 2: CƠ SỞ LÝ THUYẾT
* **2.1. MẠNG NƠ-RON TÍCH CHẬP SIÊU NHẸ TICKNETS:**
  * **2.1.1. Khái niệm và đặc điểm của họ mạng tích chập siêu nhẹ TickNets**
  * **2.1.2. Các phép tích chập tối giản (Depthwise Convolution và Pointwise Convolution)**
  * **2.1.3. Kiến trúc khối Full-Residual Point-Depth-Point (FR-PDP)**
  * **2.1.4. Cấu trúc xương sống co giãn kênh hình dấu tích (Tick-shape Backbone) và mô hình TickNet-small**
  * **2.1.5. Cơ chế lan truyền đặc trưng và cách thức hoạt động của TickNets**
* **2.2. CƠ CHẾ CHÚ Ý (ATTENTION MECHANISM):**
  * **2.2.1. Cơ chế chú ý theo kênh Squeeze-and-Excitation (SE) mặc định trong TickNets**
  * **2.2.2. Mô-đun chú ý theo kênh (Channel Attention Module - CAM) của CBAM**
  * **2.2.3. Mô-đun chú ý theo không gian (Spatial Attention Module - SAM) của CBAM**
  * **2.2.4. Mô-đun chú ý khối tích chập CBAM hoạt động tuần tự (CAM + SAM)**
* **2.3. PHƯƠNG PHÁP NGHIÊN CỨU**

---

### CHƯƠNG 3: BÀI TOÁN PHÂN LOẠI HÌNH ẢNH VÀ ĐỀ XUẤT CẢI TIẾN PHÂN CẤP
* **3.1. GIỚI THIỆU BÀI TOÁN PHÂN LOẠI HÌNH ẢNH**
  * **3.1.1. Khái niệm và đặc điểm của bài toán phân loại ảnh**
  * **3.1.2	Biểu diễn hình ảnh trong máy tính**
  * **3.1.3	Thách thức của bài toán**
  * **3.1.4	Ứng dụng thực tiễn**
  * **3.1.5	Các phương pháp tiếp cận bài toán phân loại hình ảnh**
  * **3.1.6	Vai trò của cơ chế chú ý trong bài toán phân loại ảnh**

* **3.2. ĐỀ XUẤT MÔ HÌNH TICKNETS KẾT HỢP CƠ CHẾ CHÚ Ý PHÂN CẤP (HIERARCHICAL ATTENTION):**
  * **3.2.1. Thiết kế kiến trúc đề xuất TickNet-small CBAM-Hook**
  * **3.2.2. Cơ chế chú ý phân cấp (Giữ SE nội khối FR-PDP, chèn CBAM ngoại khối tại các điểm nối Hooking points và cuối mạng)**
  * **3.2.3. Phân tích toán học giải quyết sự chồng chéo đặc trưng không gian giữa Depthwise Convolution và Spatial Attention Module (SAM)**
* **3.3. HIỆN THỰC GIẢI PHÁP VÀ THỰC NGHIỆM:**
  * **3.3.1. Môi trường thực nghiệm và xây dựng mô hình:**
    * *3.3.1.1. Môi trường thực nghiệm (Nền tảng Kaggle, GPU T4, PyTorch)*
    * *3.3.1.2. Mô tả 2 bộ dữ liệu thực nghiệm:*
    * *3.3.1.3. Các phương pháp Tiền xử lý dữ liệu và Tăng cường dữ liệu (Data Augmentation) cho từng nhóm ảnh*
    * *3.3.1.4. Hiện thực hóa 3 mô hình đối chứng thực nghiệm:*
      * 3.3.1.4.1. Mô hình 1 (SE Baseline): TickNet-small nguyên bản sử dụng chú ý SE trong khối
      * 3.3.1.4.2. Mô hình 2 (CBAM-Local): Chèn trực tiếp CBAM thay SE cục bộ trong khối FR-PDP
      * 3.3.1.4.3. Mô hình 3 (CBAM-Hook - Đề xuất cải tiến): Chèn CBAM phân cấp tại 3 điểm nối chiến lược
    * *3.3.1.5. Cấu hình quá trình huấn luyện và tối ưu hóa:*
  * **3.3.2. Kết quả thực nghiệm, phân tích và đánh giá:**
    * *3.3.2.1. Các độ đo hiệu năng chuẩn mực (Accuracy, Precision, Recall, F1-score) và cấu trúc Confusion Matrix*
    * *3.3.2.2. Kết quả thực nghiệm và biểu đồ Loss/Accuracy trên tập dữ liệu CIFAR-10 (ảnh nhỏ 32×32)*
    * *3.3.2.3. Kết quả thực nghiệm và biểu đồ Loss/Accuracy trên tập dữ liệu PlantVillage (ảnh lớn 224×224)*
    * *3.3.2.4. Phân tích hiện tượng suy giảm hiệu năng cục bộ (CBAM-Local) và ưu thế của giải pháp đề xuất (CBAM-Hook)*
    * *3.3.2.5. Giải thích mô hình bằng trực quan hóa bản đồ nhiệt Grad-CAM (Sự tập trung vùng không gian đặc trưng thực tế)*

---

### CHƯƠNG 4: KẾT QUẢ, KẾT LUẬN VÀ KIẾN NGHỊ
* **4.1. KẾT QUẢ ĐẠT ĐƯỢC CỦA ĐỀ TÀI**
* **4.2. KẾT LUẬN (Đánh giá mức độ hoàn thành so với mục tiêu và ý nghĩa thực tiễn)**
* **4.3. KIẾN NGHỊ VÀ CÁC HƯỚNG PHÁT TRIỂN TIẾP THEO**

---

**TÀI LIỆU THAM KHẢO**
