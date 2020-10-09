**Cài đặt và tối ưu thuật toán Radix Sort trên GPU với CUDA**
================
***Đồ án cuối kì của môn học "Lập trình song song trên GPU".  
Lớp 18HCB, Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên - ĐHQG TP.HCM.***

# 1. Yêu cầu
**Làm việc nhóm:**  
**Chạy đúng và tổng quát hóa:**  
**Thời gian chạy:**

# 2. Các mốc thời gian
**Bắt đầu thực hiện**: Ngày 13/09/2020  
**Thuyết trình và vấn đáp:** Dự kiến vào 03/10/2020
|**STT**|**Tóm tắt công việc**|**Thời gian hoàn thành**|
|---|:---|:---|
|1|**Cài đặt phiên bản baseline #1**: Cài đặt thuật toán Radix Sort tuần tự.|13/09/2020|
|2|**Cài đặt phiên bản baseline #2**: Cài đặt song song hai phần histogram và scan của thuật toán Radix Sort tuần tự.|16/09/2020|
|3|**Cài đặt phiên bản baseline #3**: Cài đặt song song thuật toán Radix Sort với kích thước digit là 1-bit.|17/09/2020|
|4|**Cài đặt phiên bản baseline #4**: Cài đặt tuần tự thuật toán Radix Sort song song.|23/09/2020|
|5|**Cài đặt phiên bản parallel #1**: Dựa trên baseline #4. Cài đặt song song phần sắp xếp cục bộ từng block dữ liệu.|04/10/2020|
|6|**Cài đặt phiên bản parallel #2**: Dựa trên parallel #1. Cài đặt song song phần tính rank và scatter|04/10/2020|
|7|**Cài đặt phiên bản parallel #3**: Dựa trên parallel #2. Cài đặt song song phần tính toán bảng histogram.|04/10/2020|
|8|**Cài đặt phiên bản parallel #4**: Dựa trên parallel #3. Cài đặt song song phần scan bảng histogram.|07/10/2020|
|9|**Cài đặt phiên bản parallel #5**||
|10|**Cài đặt phiên bản parallel #6**||
|11|**Cài đặt phiên bản parallel #7**||
|12|**Cài đặt phiên bản parallel #8**||

# 3. Quá trình thực hiện
## 3.1. Các phiên bản bản Baseline
### 3.1.1. Cài đặt phiên bản Baseline #1
**Nội dung:** Cài đặt thuật toán Radix Sort tuần tự.  
_(thuật toán đã được cung cấp sẵn trong file "Docs/RadixSortTemplate.cu")_  
**Tập tin mã nguồn:** [SourceCode/Baseline1.cu](./SourceCode/Baseline1.cu)  

### 3.1.2. Cài đặt phiên bản Baseline #2
**Nội dung:** Cài đặt song song hai bước histogram và scan của thuật toán Radix Sort tuần tự.  
* Thuật toán histogram có sử dụng shared memory ([Docs/Snippets/Histogram.cu](./Docs/Snippets/Histogram.cu)).
* Thuật toán prefix-sum work-efficient ([Docs/Snippets/PrefixSum-WorkEfficient](./Docs/Snippets/PrefixSum-WorkEfficient.cu)).

**Tập tin mã nguồn:** [SourceCode/Baseline2.cu](./SourceCode/Baseline2.cu)  

### 3.1.3. Cài đặt phiên bản Baseline #3
**Nội dung:** Cài đặt song song thuật toán Radix Sort với k = 1 bit.  
**Tập tin mã nguồn:** [SourceCode/Baseline3.cu](./SourceCode/Baseline3.cu)

### 3.1.4. Cài đặt phiên bản Baseline #4
**Nội dung:** Cài đặt tuần tự ý tưởng của thuật toán Radix Sort song song trong bài báo của NVIDIA.  
**Tập tin mã nguồn:** [SourceCode/Baseline4.cu](./SourceCode/Baseline4.cu)

### 3.1.5. So sánh các phiên bản Baseline

## 3.2. Các phiên bản tối ưu
### **Quá trình tối ưu:**  

**Bước 1 - Phân tích:**
* Xác định đoạn code cần tối ưu, bằng cách đo thời gian.
* Đề xuất ý tưởng để tối ưu.

**Bước 2 - Thiết kế:** Cụ thể hóa ý tưởng ở bước 1.  

**Bước 3. Cài đặt:** Cài đặt thiết kế ở bước 2.

### 3.2.1. Phiên bản tối ưu #1
1. **Phân tích:** Ở phiên bản Baseline #4, tốn nhiều thời gian ở phần "sắp xếp cục bộ" một block.
2. **Thiết kế:** Cài đặt song song thuật toán Radix Sort với $$k = 1 bit$$, để thay thế phần sắp xếp tuần tự trong dữ liệu cục bộ của một block.
3. **Cài đặt:** Sử dụng lại cách cài đặt trong Baseline #3.

### 3.2.2. Phiên bản tối ưu #2
1. **Phân tích:** Ở phiên bản Parallel #1, tốn nhiều thời gian ở phần tìm phần thiếu của rank và scatter.
2. **Thiết kế:**  Cài đặt song song phần tìm phần thiếu của rank và scatter cải thiện thời gian chạy.
3. **Cài đặt:** Hai phần này có thể gộp chung với nhau vào trong một hàm kernel.

### 3.2.3. Phiên bản tối ưu #3
1. **Phân tích:** Ở phiên bản Parallel #2, tốn nhiều thời gian ở phần tính histogram.
2. **Thiết kế:** Cài đặt song song phần tính histogram để cải thiện thời gian chạy.
3. **Cài đặt:** Sử dụng lại thuật toán histogram song song, có sử dụng SMEM.

### 3.2.4. Phiên bản tối ưu #4
1. **Phân tích:** Ở phiên bản Parallel #3, còn một phần chưa được cài đặt song song, là phần scan bảng histogram (tạo thành từ các histogram cục bộ của mỗi block). Tiến hành cài đặt song song phần này để cải thiện thời gian chạy, cũng như để dễ dàng hơn trong việc quản lí bộ nhớ ở các bản tối ưu về sau, khi giảm bớt chi phí cấp phát bộ nhớ ở device và sao chép dữ liệu từ host sang deivce và ngược lại.
2. **Thiết kế:** Scan bảng histogram theo thứ tự từng cột nối với nhau (từ bin thấp đến bin cao) - hay scan theo column-major order.

   Một vấn đề phát sinh khi duyệt bảng histogram (một ma trận lưu trữ theo kiểu row-major order trong bộ nhớ) theo column-major order, thì pattern truy xuất bộ nhớ Global Memory không được hiệu quả. Lí do là không đảm bảo các truy xuất có tính coalesced (liền mạch), nên không tận dụng hiểu quả băng thông bộ nhớ (cần nhiều transaction hơn, dữ liệu đọc lên nhưng không cần đến rất nhiều).

   Để giải quyết vấn đề pattern truy xuất bộ nhớ, có thể chuyển vị ma trận histogram trước khi scan, khi đó duyệt cột ở ma trận cũ chính là duyệt dòng ở ma trận chuyển vị.
3. **Cài đặt:**  
   * Cài đặt song song thuật toán chuyển vị ma trận ([Docs/Snippets/MatrixTranspose](./Docs/Snippets/MatrixTranspose.cu)).  
   * Trước khi thực hiện scan ma trận histogram, tiến hành chuyển vị ma trận histogram để có được pattern truy xuất Global Memory hiệu quả. 
   Sau khi scan xong: 
      * Để giữ cho hàm kernel scatter hoạt động mà không phải chỉnh sửa gì thêm, thì phải chuyển vị ma trận scan histogram. Hàm kernel scatter vẫn xem bảng scan histogram là ma trận có số cột là số bin, số dòng là số block.  
      (tạm thời chọn cách này)
      * Nếu không muốn chuyển vị thì phải chỉnh sửa dòng lệnh tìm index của giá trị prefix-sum của một bin.

### 3.2.5. Phiên bản tối ưu #5
1. **Phân tích:** Do cài đặt từng phần một song song trên device, nên mỗi khi chạy hàm kernel đều có phần sao chép dữ liệu từ host sang device và ngược lại, để cho những phần chưa được cài đặt song song chạy. Bây giờ, các phần đều được cài đặt song song trên device, nên có thể bỏ đi những thao tác cấp phát, sao chép dữ liệu không cần thiết.
2. **Thiết kế:** Bỏ đi những thao tác cấp phát, sao chép dữ liệu giữa host và device không cần thiết. Công việc cấp phát dữ liệu chỉ nên thực hiện 1 lần, và hạn chế tối đa việc sao chép dữ liệu.
3. **Cài đặt:**
   * Bỏ những thao tác cấp phát, sao chép dữ liệu không cần thiết trong phần sắp xếp cục bộ, tính toán histogram, chuyển vị và scan, scatter.
   * Trong phần thực hiện scan bảng histogram, để không phải cấp phát lại nhiều lần mảng lưu trữ tổng của mỗi block, sử dụng biến static và thêm cơ chế tự phát hiện khi có thay đổi về kích thước (trong trường hợp không thoát chương trình và chạy hàm với input khác hoặc block size khác).
   * Ngoài ra, còn có cài đặt thêm một số chỉ thị tiền xử lí để bỏ đi những câu lệnh kiểm tra lỗi khi gọi CUDA API và kernel khi không có nhu cầu kiểm lỗi.

### 3.2.6. Phiên bản tối ưu #6
1. **Phân tích:** Ở phiên bản Parallel #5, phần sắp xếp nội bộ dữ liệu của một block chiếm nhiều thời gian nhất. Nếu tối ưu được thời gian chạy của phần này sẽ cải thiện thời gian chạy của chương trình lên rất nhiều.
2. **Thiết kế:**
3. **Cài đặt:**
