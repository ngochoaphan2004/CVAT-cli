# HƯỚNG DẪN GÁN NHÃN SEMANTIC SEGMENTATION
**Tài liệu:** Student Annotation Guideline • CVAT (AI20K)  
**Mục tiêu:** Gán đúng class cho từng pixel trong ảnh giao thông, với biên semantic nhất quán, không chồng lấn và có quy trình xử lý rõ ràng cho vùng không chắc chắn.

---

## 1. Phạm vi và Nguyên tắc Chung

- **Dữ liệu:** Ảnh `.jpg` trong thư mục `segmentation/images`. Học viên không sử dụng annotation/ground truth có sẵn trong khi thực hành.
- **Bản chất bài toán:** Là **Semantic Segmentation** — mỗi pixel được gán theo class ngữ nghĩa, không giữ identity riêng cho từng instance cùng class.
- **Giới hạn nhãn:** Chỉ sử dụng đúng **19 class** đã quy định. Tuyệt đối không tự tạo class, không đổi tên class, không ghép class theo cảm tính.
- **Quy ước màu:** Mã màu RGB chỉ dùng để visualize/overlay trực quan. Quyết định gán nhãn phải dựa trên **ngữ nghĩa thực tế của class**, không dựa vào màu hiển thị.

### 🔴 3 Quy Tắc Cốt Lõi (Core Rules):

> **RULE 01:** Mỗi pixel thuộc tối đa **một class semantic**. Không được tạo hai mask/class chồng lên cùng một vùng ảnh.  
> **RULE 02:** Biên mask phải **bám theo biên nhìn thấy** của vật thể/vùng trên ảnh. Không “vẽ ước lượng” ra ngoài phần có bằng chứng hình ảnh.  
> **RULE 03:** Nếu pixel/vùng không thể gán chắc chắn vào một trong 19 class, đánh dấu để **review theo SOP của lớp**; không ép vào class gần giống chỉ để lấp kín ảnh.

---

## 2. Danh Sách 19 Class và Bảng Mã Màu (Color Map)

> *Lưu ý:* Bảng dưới đây là palette hiển thị chuẩn của bài tập. Khi cấu hình CVAT, cần đặt màu label tương ứng để overlay hiển thị nhất quán giữa các nhóm.

| STT | Class | Tên tiếng Việt / Mô tả | Màu hiển thị | Mã HEX | Mã RGB (R, G, B) |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | `road` | Mặt đường xe chạy | 🟪 | `#804080` | `128, 64, 128` |
| 2 | `sidewalk` | Vỉa hè, lối đi bộ | 🌸 | `#F423E8` | `244, 35, 232` |
| 3 | `building` | Tòa nhà, công trình | ⬛ | `#464646` | `70, 70, 70` |
| 4 | `wall` | Tường rào độc lập | 🟪 | `#66669C` | `102, 102, 156` |
| 5 | `fence` | Hàng rào (thanh/lưới) | 🟫 | `#BE9999` | `190, 153, 153` |
| 6 | `pole` | Cột điện, cột đèn, trụ | ⬜ | `#999999` | `153, 153, 153` |
| 7 | `traffic_light` | Đèn giao thông | 🟧 | `#FAAA1E` | `250, 170, 30` |
| 8 | `traffic_sign` | Biển báo giao thông | 🟨 | `#DCDC00` | `220, 220, 0` |
| 9 | `vegetation` | Cây cối, bụi rậm, tán lá | 🟩 | `#6B8E23` | `107, 142, 35` |
| 10 | `terrain` | Mặt đất tự nhiên, bãi cỏ thấp | 🟩 | `#98FB98` | `152, 251, 152` |
| 11 | `sky` | Bầu trời | 🟦 | `#4682B4` | `70, 130, 180` |
| 12 | `person` | Người đi bộ / đứng / ngồi | 🟥 | `#DC143C` | `220, 20, 60` |
| 13 | `rider` | Người lái/ngồi trên xe 2 bánh | 🔴 | `#FF0000` | `255, 0, 0` |
| 14 | `car` | Ô tô con | 🟦 | `#00008E` | `0, 0, 142` |
| 15 | `truck` | Xe tải | 🟦 | `#000046` | `0, 0, 70` |
| 16 | `bus` | Xe buýt / xe khách | 🟦 | `#003C64` | `0, 60, 100` |
| 17 | `train` | Tàu hỏa, tàu điện | 🟦 | `#005064` | `0, 80, 100` |
| 18 | `motorcycle` | Xe máy, mô tô | 🟦 | `#0000E6` | `0, 0, 230` |
| 19 | `bicycle` | Xe đạp | 🟫 | `#770B20` | `119, 11, 32` |

---

## 3. Quy Tắc Vẽ Mask và Xử Lý Biên

| Tiêu chí | Hướng dẫn chi tiết |
| :--- | :--- |
| **BIÊN CLASS** | Đi theo đường biên nhìn thấy giữa hai semantic region. **Zoom khi cần**, đặc biệt chú ý các chi tiết như: người, rider, xe hai bánh, cột (`pole`), biển báo và đèn tín hiệu. |
| **KHÔNG CHỒNG LẤN** | Hai class khác nhau tuyệt đối không được cùng chiếm một pixel. Nếu hai object chồng nhau theo phối cảnh thị giác, **pixel hiển thị thuộc về object ở phía trước**. |
| **KHÔNG TẠO LỖ GIẢ** | Không để khoảng trống lủng lỗ giữa các vùng kề nhau do thao tác vẽ mask/polygon ẩu. Tuy nhiên, **không được phép “lấp bừa”** vùng chưa rõ bằng class phỏng đoán. |
| **OBJECT MẢNH** | Đối với cột (`pole`), chân đỡ biển báo/đèn, nan hoa xe đạp/xe máy và chi tiết tứ chi người: cần giữ hình dạng mảnh mai hợp lý, **tránh tô phình to quá mức**. |
| **OBJECT NHỎ / XA** | Nếu vật thể ở xa nhưng mắt thường vẫn nhận dạng được class rõ ràng thì tiến hành gán nhãn. Nếu quá nhỏ/mờ không thể xác định chắc chắn: **chuyển sang review**, không đoán mò. |
| **OCCLUSION (Bị che khuất)** | Trong Semantic Segmentation, **chỉ gán những pixel đang thực sự nhìn thấy**. Không suy đoán và không tô phần vật thể bị che khuất đằng sau vật khác. |
| **TRUNCATION (Bị cắt bởi viền)** | Chỉ gán phần vật thể nằm bên trong ảnh. Mask **dừng ngay tại biên ảnh**, không suy đoán phần cơ thể hay vật thể nằm ngoài khung hình. |
| **REFLECTION / SHADOW** | **Không gán** bóng đổ trên mặt đường, hình ảnh phản chiếu qua kính hoặc hình vẽ trên billboard/màn hình thành object thật, trừ khi guideline riêng của batch có quy định khác. |

---

## 4. Bảng Phân Biệt Các Cặp Class Dễ Nhầm Lẫn

| Cặp dễ nhầm | Quy tắc thực hành phân định |
| :--- | :--- |
| **`road` vs `sidewalk`** | • `road`: Phần lòng đường chính dành cho phương tiện cơ giới lưu thông.<br>• `sidewalk`: Vỉa hè, lối đi bộ được ngăn cách hoặc có gờ phân tách với lòng đường. |
| **`building` vs `wall`** | • `building`: Bề mặt tường gắn liền với công trình, tòa nhà kiến trúc.<br>• `wall`: Bức tường xây độc lập, tường bao quanh, ranh giới rào đặc không phải mặt chính tòa nhà. |
| **`wall` vs `fence`** | • `wall`: Bề mặt xây kín, đặc (bê tông, gạch).<br>• `fence`: Hàng rào có kết cấu khe hở, thanh chấn song, lưới sắt hoặc cọc rào. |
| **`vegetation` vs `terrain`** | • `vegetation`: Cây cối, thân gỗ, bụi rậm, cành lá tầng cao.<br>• `terrain`: Mặt đất đồi, bãi cỏ thấp mọc sát đất, bề mặt tự nhiên phẳng không thành bụi cây. |
| **`person` vs `rider`** | • `rider`: Người đang cưỡi, ngồi lái hoặc điều khiển xe hai bánh (xe đạp, xe máy) hoặc súc vật kéo.<br>• `person`: Người đi bộ, người đứng chờ, người ngồi ghế công viên (không điều khiển xe). |
| **`car` vs `truck` vs `bus`** | Chọn đúng loại phương tiện theo thực tế cấu tạo. Nếu xe ở quá xa/quá mờ để nhận dạng chắc chắn: **đưa lên review**, không võ đoán. |
| **`motorcycle` vs `bicycle`** | • Phân biệt xe máy (phương tiện gắn động cơ, pô, máy) với xe đạp (chạy bằng sức người).<br>• **Lưu ý đặc biệt:** Người lái được gán nhãn `rider`, còn phương tiện bên dưới vẫn gán `motorcycle` hoặc `bicycle` riêng rẽ. |

---

## 5. Quy Trình Thao Tác Khuyến Nghị Trên CVAT

1. **Khởi đầu:** Mở đúng Job / đúng Organization được phân công và kiểm tra kỹ label set 19 class trước khi bắt đầu.
2. **Công cụ gán:** 
   - Dùng **Mask/Brush** cho các vùng pixel có hình dáng phức tạp, hữu cơ (cây cối, con người,...).
   - Có thể dùng **Polygon** cho các vùng lớn có ranh giới thẳng và rõ ràng (mặt đường, bầu trời, vách nhà phẳng).
3. **Thứ tự ưu tiên:** Luôn ưu tiên gán **vùng nền lớn trước** (`road`, `sky`, `building`, `vegetation`), sau đó mới gán đè/cắt các **object nhỏ, mảnh ở phía trên**.
4. **Kiểm tra biên:** Zoom phóng to để chỉnh boundary sắc nét; thường xuyên kéo thanh trượt **giảm opacity** của lớp overlay để soi ảnh gốc bên dưới.
5. **Chống vẽ nhầm:** Dùng tính năng **Lock / Hide / Filter label** khi khung hình có quá nhiều lớp chồng chất để tránh vô tình kéo sửa nhầm class khác.
6. **Xử lý khúc mắc:** Nếu không chắc chắn về class hoặc đường biên: **tạo Issue trên CVAT** hoặc đánh dấu theo SOP review của lớp; tuyệt đối không tự đặt quy tắc cá nhân.
7. **Lưu dữ liệu:** Bấm **Save (Ctrl + S)** thường xuyên; tự rà soát toàn ảnh trước khi chuyển Job sang trạng thái **Completed**.

---

## 6. Xử Lý Vùng Không Chắc Chắn & Escalation

- **Không đoán mò:** Không cố gắng đạt 100% diện tích bằng cách đoán mò. Vùng nào không đủ bằng chứng rõ ràng trên ảnh thì ưu tiên đưa lên review.
- **Quy định Ignore/Unlabeled:** Nếu SOP có nhãn/cờ `ignore` hoặc `unlabeled`, chỉ sử dụng đúng theo cấu hình của batch. Không tự ý tạo thêm ignore class.
- **Quy chuẩn đặt tên Issue trên CVAT:** Ghi ngắn gọn loại vấn đề theo mẫu:
  - `UNCERTAIN_CLASS`: Không rõ đối tượng thuộc class nào.
  - `UNCERTAIN_BOUNDARY`: Không phân định được ranh giới giữa 2 vùng.
  - `UNCERTAIN_SMALL_OBJECT`: Vật thể quá nhỏ hoặc quá nhòe ở hậu cảnh.
- **Tính đồng nhất (Decision Log):** Các trường hợp đặc biệt gặp lặp đi lặp lại phải được Mentor / Project Lead thống nhất và ghi chép thành **Decision Log** để tất cả thành viên làm theo đồng nhất.

---

## 7. Quality Checklist Trước Khi Nộp Bài

Annotator / Reviewer phải tích đủ các tiêu chí trước khi bấm nộp:

- [ ] **Không chồng lấn:** Không có pixel nào bị chồng 2 lớp mask khác nhau.
- [ ] **Không lỗ thủng:** Không xuất hiện khoảng trống thủng giữa các vùng đã xác định rõ ràng.
- [ ] **Bám biên chuẩn:** Biên của các vùng lớn (`road`, `sidewalk`, `building`, `sky`, `vegetation`) khít với thực tế ảnh.
- [ ] **Zoom chi tiết:** Các đối tượng `person`, `rider`, `car`, `bicycle`, `motorcycle` và vật thể mảnh (`pole`, biển báo) đã được kiểm tra ở mức zoom hợp lý.
- [ ] **Đúng ranh giới nhìn thấy:** Không tô phần cơ thể hoặc vật thể bị che khuất hoặc tràn ra ngoài ảnh.
- [ ] **Phân tách đúng `person` / `rider`:** Đã rà soát kỹ các đối tượng ngồi trên phương tiện.
- [ ] **Không thừa/thiếu nhãn:** Không dùng class nào ngoài 19 class quy định.
- [ ] **Màu chuẩn:** Bảng màu hiển thị đúng theo cấu hình color map.
- [ ] **Xử lý Issue:** Toàn bộ các Issue trên Job đã được giải quyết hoặc chuyển tiếp cho Reviewer.
- [ ] **Tự kiểm tra ngẫu nhiên:** Đã tự soi lại kỹ ít nhất 10 ảnh ngẫu nhiên trong batch trước khi submit cuối cùng.

---

## 8. Tiêu Chí Đánh Giá Nghiệm Thu

1. **Độ chính xác ngữ nghĩa (Semantic Accuracy):** Mọi pixel đều được gán đúng nhãn theo guideline.
2. **Độ chính xác biên (Boundary Accuracy):** Đường viền dứt khoát, không cắt lẹm vào vật thể và không lấy thừa background.
3. **Tính nhất quán (Consistency):** Cùng một tình huống/vật thể phải được xử lý giống nhau giữa các ảnh và giữa các annotator.
4. **Chỉ số định lượng (Quantitative Metrics):** Đánh giá dựa trên **per-class IoU** và **mIoU (Mean Intersection over Union)** so với Ground Truth. Điểm đạt (pass) theo ngưỡng quy định của từng đợt đào tạo/mentor.

---

## 9. Bảng Tra Cứu Nhanh (Quick Reference)

| Nếu gặp tình huống... | Hành động chuẩn cần thực hiện |
| :--- | :--- |
| **Object bị vật khác che khuất** | 👉 Chỉ tô pixel nhìn thấy; tuyệt đối không suy đoán phần bị che lấp. |
| **Object bị tràn/cắt bởi mép ảnh** | 👉 Tô chạm tới biên ảnh là dừng; không vẽ tràn ra ngoài viền. |
| **Không chắc chắn về class** | 👉 Không đoán mò; gắn cờ tạo Issue (`UNCERTAIN_CLASS`) để đưa review. |
| **Hai class chồng mask lên nhau** | 👉 Cắt/sửa lại mask sao cho mỗi pixel chỉ thuộc duy nhất một class (ưu tiên vật ở trước). |
| **Vùng lớn có ranh giới rõ ràng** | 👉 Dùng Mask/Brush hoặc Polygon theo quy ước chuẩn của lớp để tối ưu tốc độ. |
| **Vật thể nhỏ, thanh mảnh** | 👉 Zoom lớn bám sát biên; nếu nhòe không đủ bằng chứng hình ảnh thì chuyển review. |
| **Bóng đổ / Hình phản chiếu kính** | 👉 Bỏ qua, không gán nhãn như vật thật (trừ khi có quy định riêng). |
