# MỤC LỤC DỰ KIẾN TIỂU LUẬN TỐT NGHIỆP

**ĐỀ TÀI: NGHIÊN CỨU CẢI TIẾN TICKNETS BẰNG CƠ CHẾ ATTENTION CBAM CHO BÀI TOÁN PHÂN LOẠI ẢNH**

---

## MỤC LỤC

**DANH SÁCH CHỮ VIẾT TẮT**  
**DANH MỤC HÌNH ẢNH**  
**DANH MỤC BẢNG**  
**TÓM TẮT**  

**MỞ ĐẦU**  
1. Lý do chọn đề tài  
2. Mục tiêu và phạm vi nghiên cứu  
3. Ý nghĩa khoa học và thực tiễn của đề tài  

---

### CHƯƠNG 1: TỔNG QUAN ĐỀ TÀI
1.1. Tầm quan trọng của bài toán phân loại ảnh trong thị giác máy tính  
1.2. Tổng quan các kiến trúc mạng nơ-ron tích chập (CNN) và cơ chế chú ý (Attention)  
1.3. Tổng quan về dòng mạng tối ưu cho thiết bị di động (Mobile-friendly Networks)  
1.4. Phát biểu bài toán nghiên cứu  

### CHƯƠNG 2: MẠNG TICKNETS VÀ CƠ CHẾ ATTENTION
2.1. Tổng quan về kiến trúc mạng TickNets  
  2.1.1. Khái niệm và đặc điểm kênh đàn hồi (Elastic Channels) của TickNets  
  2.1.2. Khối tích chập cốt lõi FR-PDP (Flexible Residual - Pointwise Depthwise Pointwise)  
  2.1.3. Nguyên lý hoạt động và ưu điểm của TickNets so với CNN truyền thống  
2.2. Cơ chế chú ý (Attention Mechanism) trong thị giác máy tính  
  2.2.1. Cơ chế chú ý kênh Squeeze-and-Excitation (SE) - Tiền đề của TickNet gốc  
  2.2.2. Channel Attention Module (CAM) của CBAM  
  2.2.3. Spatial Attention Module (SAM) của CBAM  
  2.2.4. Convolutional Block Attention Module (CBAM) - Sự kết hợp Kênh và Không gian  
2.3. Đánh giá lý thuyết về sự phù hợp của CBAM đối với cấu trúc khối FR-PDP  

### CHƯƠNG 3: BÀI TOÁN PHÂN LOẠI HÌNH ẢNH VÀ ĐỀ XUẤT CẢI TIẾN
3.1. Đề xuất kiến trúc TickNets kết hợp cơ chế CBAM cải tiến (TickNetA-CBAM)  
  3.1.1. Phân tích vị trí tích hợp CBAM tối ưu trong khối FR-PDP (Trước residual connection)  
  3.1.2. Tối ưu hóa siêu tham số của module CBAM (Reduction ratio & Spatial Kernel size)  
  3.1.3. Thiết lập lớp Dropout chống overfitting cho dữ liệu phức tạp  
3.2. Thiết lập môi trường và bộ dữ liệu thực nghiệm  
  3.2.1. Môi trường phần cứng và phần mềm (PyTorch, GPU Kaggle, Mixed Precision)  
  3.2.2. Các bộ dữ liệu thực nghiệm đa kích thước  
    3.2.2.1. Bộ dữ liệu ảnh cỡ nhỏ: CIFAR-10 và Fashion-MNIST (32x32)  
    3.2.2.2. Bộ dữ liệu nông nghiệp: PlantVillage (224x224)  
    3.2.2.3. Bộ dữ liệu y tế thực tế: Chest X-Ray (Pneumonia) (224x224)  
3.3. Quy trình huấn luyện và các độ đo đánh giá  
  3.3.1. Chiến lược huấn luyện (SGD, Cosine Annealing learning rate)  
  3.3.2. Các độ đo đánh giá (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)  
3.4. Kết quả thực nghiệm và thảo luận  
  3.4.1. Kết quả thực nghiệm trên ảnh kích thước nhỏ (CIFAR-10 & Fashion-MNIST)  
  3.4.2. Kết quả thực nghiệm trên ảnh nông nghiệp (PlantVillage)  
  3.4.3. Kết quả thực nghiệm trên ảnh y tế (Chest X-Ray)  
  3.4.4. Phân tích tác động của kích thước không gian (Spatial Resolution) đến hiệu quả của Spatial Attention  
  3.4.5. Trực quan hóa vùng chú ý bằng kỹ thuật Grad-CAM (So sánh SE vs CBAM)  

### CHƯƠNG 4: KẾT LUẬN VÀ KIẾN NGHỊ
4.1. Những kết quả đạt được của đề tài  
4.2. Hạn chế và nguyên nhân  
4.3. Hướng nghiên cứu tiếp theo  

---

**TÀI LIỆU THAM KHẢO**  
**PHỤ LỤC**  