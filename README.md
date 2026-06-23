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

Mục lục dự kiến cho đề cương như sau:
MỤC LỤC
DANH SÁCH CHỮ VIẾT TẮT
DANH MỤC HÌNH ẢNH
DANH MỤC BẢNG
TÓM TẮT
MỞ ĐẦU
1. LÝ DO CHỌN ĐỀ TÀI
2. MỤC TIÊU VÀ PHẠM VI NGHIÊN CỨU
3. Ý NGHĨA KHOA HỌC VÀ THỰC TIỄN CỦA ĐỀ TÀI
CHƯƠNG 1
TỔNG QUAN ĐỀ TÀI
CHƯƠNG 2
MẠNG TICKNETS VÀ CƠ CHẾ ATTENTION
2.1. TỔNG QUAN VỀ KIẾN TRÚC TICKNETS
2.1.1. Khái niệm và đặc điểm của mạng TickNets
2.1.2. Kiến trúc và các thành phần cốt lõi của mạng TickNets
2.1.3. Cách thức hoạt động của TickNets 
2.2. CƠ CHẾ CHÚ Ý (ATTENTION MECHANISM)
2.2.1. Channel Attention Module (CAM)
2.2.2. Spatial Attention Module (SAM)
2.2.3. Convolutional Block Attention Module (CBAM)
2.3. PHƯƠNG PHÁP NGHIÊN CỨU
CHƯƠNG 3 
BÀI TOÁN PHÂN LOẠI HÌNH ẢNH VÀ ĐỀ XUẤT CẢI TIẾN
3.1. GIỚI THIỆU BÀI TOÁN PHÂN LOẠI HÌNH ẢNH
3.2. MÔ HÌNH TICKNETS KẾT HỢP CƠ CHẾ ATTENTION CBAM (Phần này bạn trình bày cách chèn module CBAM vào kiến trúc khối của TickNets)
3.3. HIỆN THỰC GIẢI PHÁP
3.3.1. Môi trường thực nghiệm và xây dựng mô hình
3.3.1.1. Môi trường thực nghiệm
3.3.1.2. Xây dựng mô hình TickNets cải tiến và huấn luyện
 (Bao gồm chuẩn bị dữ liệu, tiền xử lý, thêm CBAM vào mô hình)
3.3.2. Kết quả thực nghiệm và đánh giá
 (Sử dụng các độ đo như Accuracy, Precision, Recall, F1-score và Confusion Matrix tương tự tài liệu tham khảo)
CHƯƠNG 4: KẾT QUẢ, KẾT LUẬN VÀ KIẾN NGHỊ
4.1. KẾT QUẢ
4.2. KẾT LUẬN
4.3. KIẾN NGHỊ
TÀI LIỆU THAM KHẢO

LƯU Ý : Những điểm bạn cần chú ý khi viết theo mục lục này:
Ở Chương 2, tài liệu NNPhuong đi rất sâu vào lý thuyết CNN (Convolution layer, Pooling, Stride, Padding, hàm kích hoạt) và mô hình VGG16
. Bạn cần thay thế phần này bằng các nền tảng lý thuyết tương ứng của TickNets.
Ở Chương 3 (mục 3.2), tác giả NNPhuong chèn CBAM vào sau các Convolutional block hoặc trước Fully Connected layer của CNN
. Bạn cần trình bày rõ kiến trúc cụ thể khi chèn CBAM vào TickNets của mình ở vị trí nào để đạt hiệu quả tối ưu nhất.
Các bộ dữ liệu: Nếu bạn dùng lại Fashion-MNIST và CIFAR-10 như tác giả
, bạn có thể giữ nguyên cấu trúc mô tả tập dữ liệu. Nếu bạn dùng tập dữ liệu khác, hãy cập nhật lại ở mục 3.3.1.2.