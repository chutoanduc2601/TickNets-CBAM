# TickNets-CBAM
Tôi là Chu Toàn Đức, sinh viên năm 4 học công nghệ thông tin, hiện tại tôi sẽ thực hiện đề cương chi tiết tiểu luận tốt nghiệp với đề tài "Nghiên cứu cải tiến TickNets bằng cơ chế attention CBAM cho bài toán phân loại ảnh". Dự kiến đề cương dài khoảng 51-60 trang.

 NNPhuong-NguyenNgocPhuong.docx : đây là tài liệu tiểu luận tham khảo của Nguyễn Ngọc Phương đã đạt điểm cao trong kì vừa rồi nên tôi sẽ dùng nó làm tài liệu tham khảo cho đề tài của mình vì tôi thấy nó có nhiều điểm chung là :
 1. Cùng giải quyết bài toán phân loại hình ảnh (Image Classification) Cả hai đề tài đều nhắm đến việc giải quyết bài toán cốt lõi trong thị giác máy tính là phân loại hình ảnh
 2. Sử dụng chung cơ chế cốt lõi: Attention CBAM Cả hai đề tài đều tích hợp module CBAM vào kiến trúc mạng để cải thiện khả năng học biểu diễn của mô hình
 3. Mục tiêu cải thiện hiệu năng Cả hai đều tập trung vào việc nâng cao độ chính xác phân loại và tối ưu hóa mô hình
 Điểm khác biệt nhỏ: Là đề tài của tôi sẽ áp dụng trên kiến trúc TickNets.
 Hãy sử dụng NNPhuong-NguyenNgocPhuong.docx làm tài liệu tham khảo phần nội dung và mục lục để viết tài liệu tiểu luận của tôi nhé. Hãy viết đầy đủ nội dung vào file word, sắp xếp lại ý sao cho hợp lý. Viết thật trôi chảy, văn phong hàn lâm, không sao chép nguyên bản, dùng từ ngữ của mình nhé, tránh lặp từ nhé. Đặc biệt quan trọng là : Cần phải viết lại và mở rộng dựa trên tài liệu ticknets.

 Huong dan viet de cuong chi tiet Tieu luan KCNTT.doc : đây là tài liệu hướng dẫn viết đề cương chi tiết tiểu luận khoa công nghệ thông tin. Dựa vào đó để viết đề cương chi tiết tiểu luận cho tôi nhé.
    Lưu ý: nội dung nghiên cứu 
    (mục 6.	
    Nội dung của đề tài: 
-	Các nội dung cần nghiên cứu, thử nghiệm
-	Các công việc và các bước cần thực hiện để đạt được các mục tiêu đã nêu trên
) + sản phẩm của đề tài 
(mục 8 Sản phẩm của đề tài: 
-Đối với sản phẩm nghiên cứu cơ bản: Cần trình bày các phương pháp tiến hành thực nghiệm, kết quả thực nghiệm, so sánh và đánh giá kết quả thực nghiệm.
-Đối với sản phẩm ứng dụng: Cần mô tả chi tiết các chức năng của sản phẩm sau khi hoàn tất đề tài.
-Đối với sản phẩm nghiên cứu triển khai theo hướng quản trị mạng: Cần đánh giá giải pháp và tài liệu hướng dẫn sử dụng chi tiết.) 
sẽ là tiêu chí để đánh giá mức độ hoàn thành tiểu luận tốt nghiệp.

Muclucdukienchotieuluan.md: Đây là mục lục dự kiến tôi sẽ triển khai cho tiểu luận.

LƯU Ý : Những điểm bạn cần chú ý khi viết theo mục lục này:
Ở Chương 2, tài liệu NNPhuong đi rất sâu vào lý thuyết CNN (Convolution layer, Pooling, Stride, Padding, hàm kích hoạt) và mô hình VGG16
. Bạn cần thay thế phần này bằng các nền tảng lý thuyết tương ứng của TickNets.
Ở Chương 3 (mục 3.2), tác giả NNPhuong chèn CBAM vào sau các Convolutional block hoặc trước Fully Connected layer của CNN
. Bạn cần trình bày rõ kiến trúc cụ thể khi chèn CBAM vào TickNets của mình ở vị trí nào để đạt hiệu quả tối ưu nhất.
Các bộ dữ liệu: Nếu bạn dùng lại Fashion-MNIST và CIFAR-10 như tác giả
, bạn có thể giữ nguyên cấu trúc mô tả tập dữ liệu. Nếu bạn dùng tập dữ liệu khác, hãy cập nhật lại ở mục 3.3.1.2.

NeuroNetwork_TickNets_2023_revised_with_marked.pdf : đây là tài liệu về ticknets

https://www.kaggle.com/code/farzadnekouei/cifar-10-image-classification-with-cnn : nguồn dataset cifar-10

https://www.kaggle.com/code/gpreda/cnn-with-tensorflow-keras-for-fashion-mnist : nguồn dataset fashion-mnist

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia : nguồn dataset chest-xray-pneumonia

https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset : nguồn dataset plantvillage

Mục lục viết cho tiểu luận này là 
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
* **1.1. Bối cảnh lĩnh vực phân loại hình ảnh và sự phát triển của học sâu (Deep Learning)**
* **1.2. Thách thức về tài nguyên tính toán và sự cần thiết của mạng nơ-ron siêu nhẹ (Light-weight CNNs)**
* **1.3. Các nghiên cứu liên quan về cơ chế chú ý (Attention Mechanisms) trong thị giác máy tính**
* **1.4. Phát biểu bài toán, giới hạn nghiên cứu và giải pháp đề xuất cải tiến của tiểu luận**

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
* **2.3. PHƯƠNG PHÁP NGHIÊN CỨU LÝ THUYẾT VÀ THỰC NGHIỆM**

---

### CHƯƠNG 3: BÀI TOÁN PHÂN LOẠI HÌNH ẢNH VÀ ĐỀ XUẤT CẢI TIẾN PHÂN CẤP
* **3.1. GIỚI THIỆU BÀI TOÁN PHÂN LOẠI HÌNH ẢNH**
* **3.2. ĐỀ XUẤT MÔ HÌNH TICKNETS KẾT HỢP CƠ CHẾ CHÚ Ý PHÂN CẤP (HIERARCHICAL ATTENTION):**
  * **3.2.1. Thiết kế kiến trúc đề xuất TickNet-small CBAM-Hook**
  * **3.2.2. Cơ chế chú ý phân cấp (Giữ SE nội khối FR-PDP, chèn CBAM ngoại khối tại các điểm nối Hooking points và cuối mạng)**
  * **3.2.3. Phân tích toán học giải quyết sự chồng chéo đặc trưng không gian giữa Depthwise Convolution và Spatial Attention Module (SAM)**
* **3.3. HIỆN THỰC GIẢI PHÁP VÀ THỰC NGHIỆM:**
  * **3.3.1. Môi trường thực nghiệm và xây dựng mô hình:**
    * *3.3.1.1. Môi trường thực nghiệm (Môi trường Google Colab, GPU T4, TensorFlow và Keras)*
    * *3.3.1.2. Mô tả 4 bộ dữ liệu thực nghiệm:*
      * 3.3.1.2.1. Nhóm dữ liệu chuẩn học thuật: Fashion-MNIST và CIFAR-10
      * 3.3.1.2.2. Nhóm dữ liệu ứng dụng thực tiễn: PlantVillage (bệnh lá cây) và Chest-Xray-Pneumonia (viêm phổi)
    * *3.3.1.3. Các phương pháp Tiền xử lý dữ liệu và Tăng cường dữ liệu (Data Augmentation) cho từng nhóm ảnh*
    * *3.3.1.4. Hiện thực hóa 3 mô hình đối chứng thực nghiệm:*
      * 3.3.1.4.1. Mô hình 1 (SE Baseline): TickNet-small nguyên bản sử dụng chú ý SE trong khối
      * 3.3.1.4.2. Mô hình 2 (CBAM-Local): Chèn trực tiếp CBAM thay SE cục bộ trong khối FR-PDP
      * 3.3.1.4.3. Mô hình 3 (CBAM-Hook - Đề xuất cải tiến): Chèn CBAM phân cấp tại các điểm nối và cuối mạng
    * *3.3.1.5. Cấu hình quá trình huấn luyện và tối ưu hóa:*
      * 3.3.1.5.1. Bộ tối ưu SGD và các hàm Loss tương ứng
      * 3.3.1.5.2. Hàm gọi lại tự động giảm tốc độ học (ReduceLROnPlateau) và dừng sớm (EarlyStopping)
  * **3.3.2. Kết quả thực nghiệm, phân tích và đánh giá:**
    * *3.3.2.1. Các độ đo hiệu năng chuẩn mực (Accuracy, Precision, Recall, F1-score) và cấu trúc Confusion Matrix*
    * *3.3.2.2. Kết quả thực nghiệm và biểu đồ Loss/Accuracy trên tập dữ liệu chuẩn (Fashion-MNIST, CIFAR-10)*
    * *3.3.2.3. Kết quả thực nghiệm và biểu đồ Loss/Accuracy trên tập dữ liệu thực tiễn (PlantVillage, Chest-Xray-Pneumonia)*
    * *3.3.2.4. Phân tích hiện tượng suy giảm hiệu năng cục bộ (CBAM-Local) và ưu thế của giải pháp đề xuất (CBAM-Hook)*
    * *3.3.2.5. Giải thích mô hình bằng trực quan hóa bản đồ nhiệt Grad-CAM (Sự tập trung vùng không gian đặc trưng thực tế)*

---

### CHƯƠNG 4: KẾT QUẢ, KẾT LUẬN VÀ KIẾN NGHỊ
* **4.1. KẾT QUẢ ĐẠT ĐƯỢC CỦA ĐỀ TÀI**
* **4.2. KẾT LUẬN (Đánh giá mức độ hoàn thành so với mục tiêu và ý nghĩa thực tiễn)**
* **4.3. KIẾN NGHỊ VÀ CÁC HƯỚNG PHÁT TRIỂN TIẾP THEO**

---

**TÀI LIỆU THAM KHẢO**
