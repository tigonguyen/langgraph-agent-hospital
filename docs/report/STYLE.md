# Quy tắc văn phong — Báo cáo tiếng Việt

Quy tắc để **bất kỳ agent nào viết tiếp báo cáo cũng giữ đúng một giọng**. Đọc file này trước khi
soạn hoặc sửa `report.tex`. Mục tiêu: tiếng Việt **tự nhiên, đơn giản, đi thẳng vào ý**, giữ nguyên
các **thuật ngữ tiếng Anh là khái niệm trong hệ thống**.

## 1. Nguyên tắc cốt lõi
- **Dẫn bằng kết luận.** Nói ý chính trước, giải thích sau. Cắt rào đón, đừng nhắc lại đề.
- **Đơn giản hơn là hoa mỹ.** Chọn từ đời thường thay cho từ Hán-Việt nặng hoặc dịch máy móc.
- **Trung thực về trạng thái.** Số liệu chưa đo → đánh dấu `\todo{...}` (hiện màu đỏ `[CẦN LÀM: …]`).
  Không viết như thể đã đo khi chưa đo.
- **Không phiên âm.** Dùng nguyên gốc từ tiếng Anh, KHÔNG phiên âm gạch nối: viết `module`,
  KHÔNG viết "mô-đun".

## 2. Xưng hô & trình bày
- Chủ ngữ nhóm tác giả là **"Nhóm"** — không dùng "chúng tôi", "tôi", "chúng ta".
- Trang tiêu đề: **không để ngày** (`\date{}`). Tác giả là danh sách nhóm.
- Câu vừa phải, tránh câu lồng nhiều mệnh đề. Ưu tiên câu chủ động.

## 3. Giữ nguyên tiếng Anh (thuật ngữ hệ thống)
Không dịch các khái niệm sau — giữ nguyên: `retriever`, `embedder`, `prompt`, `baseline`, `node`,
`gate`, `seed`, `offline`, `pipeline`, `RAG`, `token`, `bootstrap`, `McNemar`, `temperature`,
`greedy`, `multi-agent`, `ablation`, `attending`, `verifier`, `reasoner`, `decider`, `panel`,
tên mô hình/tệp/hàm (`qwen2.5:14b`, `build_variant`, `RunConfig`…). Bọc tên mã bằng `\texttt{}`.

Có thể dịch (đã quen tai): `độ chính xác`, `độ trễ`, `ngữ liệu` (corpus), `bằng chứng` (evidence),
`truy hồi` (retrieval), `nhiệt độ` (temperature khi trong câu văn), `khoảng tin cậy` (CI).

## 4. Bảng từ nên tránh → nên dùng
Rút ra từ các lần sửa thực tế. Khi gặp cột trái, đổi sang cột phải.

| Tránh (dịch cứng) | Dùng (tự nhiên) | Gốc |
|---|---|---|
| mô-đun | module | module |
| chưng cất (truy vấn) | rút gọn | distillation |
| mức lợi | mức tăng | gain |
| bóc tách / bóc đáp án | trích xuất / trích đáp án | parse |
| nhà máy tạo nút | hàm tạo node | node factory |
| truy hồi trơ | truy hồi không có tác dụng | inert |
| lưỡng thái | chia hai nhánh rõ rệt | bimodal |
| đóng góp biên / giá trị biên | đóng góp riêng | marginal |
| bị chi phối bởi lập luận | chủ yếu đòi hỏi lập luận | reasoning-bound |
| vệ sinh đo độ trễ | lưu ý khi đo độ trễ | latency hygiene |
| hạt giống (ngẫu nhiên) | seed | seed |
| lười (khởi tạo) | khởi tạo trễ | lazy |
| bề mặt linh hoạt | nơi tập trung mọi tuỳ chỉnh | flexibility surface |
| làm dịch chuyển | làm thay đổi | move |
| tiêu thụ (kết quả) | xử lý | consume |
| âm thầm (sai) | bị tính nhầm | silent(ly) |
| nửa mang tính xây dựng | mặt tích cực | constructive half |
| hành xử như | tức là / hoạt động như | behaves like |

Nguyên tắc chung: nếu một cụm nghe như dịch word-by-word từ tiếng Anh, viết lại bằng cách một
người Việt sẽ nói.

## 4b. Dấu câu
- **Không dùng em-dash** (`---` / dấu gạch ngang dài) trong phần văn bản. Thay bằng:
  dấu hai chấm cho mục liệt kê/nhãn (`\item \textbf{X}: mô tả`, `V0: LLM trực tiếp`), dấu phẩy hoặc
  ngoặc đơn cho ý chèn, dấu chấm phẩy cho hai vế nối. Ví dụ: viết "Cổng chủ yếu ảnh hưởng tốc độ, chưa
  chứng minh được chất lượng", KHÔNG viết "Cổng là ... --- chưa ...".
- **Giữ en-dash** (`--`) cho khoảng/dải: `V0--V4`, `A--D`, `\ref{a}--\ref{b}`, `0.60--0.65`.
- Dòng comment trang trí `% ----------` không tính (không hiển thị ra PDF).

## 4c. Viết hoa
- Tiêu đề mục/tiểu mục dùng **sentence case** (chỉ hoa chữ đầu + danh từ riêng), KHÔNG Title Case
  kiểu tiếng Anh. Ví dụ: "Kết quả đánh giá", KHÔNG "Kết quả Đánh giá"; "Thiết kế hệ thống", KHÔNG
  "Thiết kế Hệ thống". Giữ hoa cho danh từ riêng/nhãn: `MedQA-USMLE`, `RAG`, `Thắng/Thua/Hòa`.

## 5. Chống nhập nhằng
- **"nút" chỉ dùng cho `node`** (nút đồ thị). KHÔNG dùng "nút" cho *knob* (cơ chế/tham số điều chỉnh).
  Ví dụ: "self-consistency là một cơ chế tăng độ chính xác", KHÔNG "là nút tăng độ chính xác";
  "Cổng chủ yếu ảnh hưởng tốc độ", KHÔNG "cổng là nút tốc độ".
- Một khái niệm → một cách gọi xuyên suốt (đã chọn: `verifier` = "bộ kiểm chứng"; `attending` =
  "bác sĩ chính"; `panel` = "hội đồng"). Đừng đổi qua lại.

## 6. Quy ước LaTeX / biên dịch
- Biên dịch bằng **XeLaTeX** (dòng `% !TEX program = xelatex` ở đầu file). Trên Overleaf phải đổi
  compiler sang XeLaTeX.
- Font nạp theo tệp OTF Latin Modern (`\setmainfont{lmroman10-regular.otf}[…]`) để đủ dấu tiếng Việt
  và chạy được cả trên `tectonic` lẫn Overleaf.
- Hình: convert SVG → PDF (`rsvg-convert`), `\includegraphics` file PDF — **không** dùng gói `svg`.
  Nội dung trong sơ đồ được phép để tiếng Anh.
- Sau khi sửa: chạy `tectonic report.tex` để kiểm lỗi, xoá file trung gian
  (`.aux .log .bbl .blg .out .toc`), rồi làm mới `report-overleaf.zip`.
- Kết quả/số liệu lấy từ `docs/REPORT_NOTES.md`, `docs/ARCHITECTURE.md`, `docs/TECHNICAL_DECISIONS.md`
  — **không bịa số**. Chỗ chưa có → `\todo{}`.

## 7. Kiểm nhanh trước khi xong
- [ ] Không còn phiên âm gạch nối (grep: `mô-đun`, v.v.).
- [ ] Không còn "chúng tôi/tôi"; chỉ có "Nhóm".
- [ ] "nút" chỉ mang nghĩa node.
- [ ] Không còn em-dash `---` trong văn bản (`grep -- "---" report.tex` chỉ còn dòng `%`).
- [ ] Số liệu chưa đo đều có `\todo{}`.
- [ ] `tectonic report.tex` biên dịch không lỗi.
