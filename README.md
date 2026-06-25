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