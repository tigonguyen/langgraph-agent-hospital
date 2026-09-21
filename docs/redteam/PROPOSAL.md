# Pair 5 — Đề xuất: fine-tuning làm mất alignment an toàn (attack & defense)

Giả định nêu trước: viết bằng tiếng Việt theo văn phong `docs/report/STYLE.md`, là tài liệu riêng,
tách khỏi báo cáo cuối kỳ tiếng Việt (`docs/report/`, vốn nói về dự án LangGraph hospital-agent
chính). Nếu tài liệu này cần gộp vào chỗ khác, Nhóm sẽ điều chỉnh sau.

## Hệ thống được kiểm thử

Một trợ lý chỉ trả lời y khoa, xây trên Qwen3-14B: trả lời câu hỏi kiểu MedQA-USMLE, từ chối các
yêu cầu y khoa có hại, và từ chối mọi thứ ngoài phạm vi y khoa. Đây là một thay đổi so với bản
thiết kế gốc (nhắm vào Qwen2.5-7B-Instruct tổng quát, dữ liệu alignment BeaverTails tổng quát) —
lý do trình bày ở phần cuối.

## 1. Mô hình mối đe dọa (threat model)

| | |
|---|---|
| Quyền truy cập của kẻ tấn công | Có quyền fine-tune trên trọng số mô hình đã triển khai (LoRA cục bộ trên 16 layer cuối), tương tự một fine-tuning API được hosting. Không có quyền truy cập dữ liệu alignment hay code huấn luyện. |
| Ràng buộc của kẻ tấn công | Dữ liệu huấn luyện phải qua được bộ lọc kiểm duyệt nội dung, nên chỉ có thể dùng các cặp QA vô hại (TenBenign, arXiv:2510.02833: 10 câu hỏi y khoa vô hại, hai giai đoạn). Việc kiểm chứng dữ liệu này được một judge an toàn đánh giá là vô hại: đã lên kế hoạch (`moderation_check.py`), chưa chạy, Nhóm nêu rõ để không khẳng định điều chưa đo. |
| Mục tiêu của kẻ tấn công | Mô hình trả lời các yêu cầu y khoa có hại hoặc ngoài phạm vi mà trước đó nó từ chối, trong khi vẫn trông như một trợ lý bình thường. |
| Quyền của bên phòng thủ | Kiểm soát giai đoạn alignment trước khi triển khai (SFT thường, hoặc Booster, Huang et al., ICLR 2025) nhưng không thể kiểm tra hay chặn lần fine-tune sau đó của kẻ tấn công. |
| Ngoài phạm vi | Jailbreak ở mức prompt (Pair 2), đầu độc kho RAG (Pair 3), backdoor trọng số (Pair 4). |

## 2. Các chỉ số

| trục | chỉ số | cách đo |
|---|---|---|
| **attack** | Harmful-Response Rate (HRR): tỷ lệ prompt y khoa có hại mà mô hình *tuân theo* | LLM judge (qwen3:14b, rubric 3 nhãn: refused / pushback / complied); regex theo từ khóa từ chối làm chỉ số phụ, rẻ hơn |
| **scope** (thêm mới, thiết kế gốc không có) | Tỷ lệ trả lời câu hỏi ngoài phạm vi thay vì từ chối | Cùng LLM judge, rubric refused/answered |
| **utility** | Độ chính xác MedQA-USMLE | tập test giữ riêng, định dạng đề thi |
| **cost** | Thời gian huấn luyện/tấn công, dung lượng đĩa | timer trong script |
| **baseline** | HRR / scope / utility của mô hình đã align nhưng chưa bị tấn công | hàng 1 bên dưới |

Tập prompt: 900 yêu cầu y khoa có hại từ MedSafetyBench TEST (chưa từng huấn luyện), cộng 40 prompt
viết tay trong phạm vi; 200 prompt ngoài y khoa giữ riêng từ OASST1; tập test MedQA-USMLE.

## 3. Baseline (hàng 1)

| mô hình | MedQA | harmful refused/pushback/complied | HRR | scope refused |
|---|---|---|---|---|
| qwen3:14b (nguyên bản, chưa align y khoa) | 0.705 | 622/240/38 (trên 900) | 0.042 | 2/100 |
| med-base (align bằng SFT thường, baseline triển khai) | 0.677 | 95/5/0 (mẫu 100) | 0.00 | 100/100 |

Có hai mốc tham chiếu thay vì một: qwen3:14b cho thấy Qwen đã có sẵn bao nhiêu an toàn tổng quát
trước khi align y khoa (hữu ích về sau: TenBenign được thử trên cả mô hình gốc lẫn mô hình đã
align, để tách riêng phần mà việc align y khoa thực sự đóng góp). med-base là hệ thống thật sự bị
tấn công.

## Khác biệt so với thiết kế gốc, và lý do

| thiết kế gốc | Nhóm đã làm | lý do |
|---|---|---|
| Qwen2.5-7B-Instruct, tổng quát | Qwen3-14B, chỉ y khoa | Hệ thống kiểm thử là một hospital agent; một phạm vi hẹp mới là triển khai thực tế, và cảnh báo của đề bài ("defense vẫn còn rủi ro sót lại trên input phạm vi hẹp") nên được kiểm chứng trên đúng một phạm vi hẹp, không phải phạm vi tổng quát |
| BeaverTails / HarmBench / Alpaca | MedSafetyBench (chia train/test, không rò rỉ) / OASST1 / MedMCQA | Dữ liệu harmful và utility đúng miền; BeaverTails/HarmBench là tổng quát, không thăm dò được harm đặc thù y khoa |
| Judge Llama-Guard-3 | LLM judge tự xây (qwen3:14b, rubric 3 nhãn rõ ràng) | Không cần tải thêm một mô hình 8B; rubric có thể kiểm tra được và được lưu version trong `judge.py`; đánh đổi cần nêu rõ: judge và hệ thống cùng họ mô hình, có thể là nguồn thiên lệch tương quan, cần nói trong báo cáo cuối |
| Booster h(w) = harmful → câu trả lời tuân theo (cần dữ liệu trả lời có hại) | Booster h(w) = harmful → refusal (biến thể refusal-grad) | Nhóm không tạo dữ liệu huấn luyện "trả lời có hại" từ một mô hình đã bị jailbreak, vì lý do đạo đức; biến thể refusal-grad mô phỏng cùng ý tưởng "một bước tấn công" nhưng theo chiều an toàn ngược lại (đánh đổi có thật, không miễn phí, xem phần phân tích residual-gap) |

## Trạng thái so với đề xuất này

Baseline (hàng 1), attack, và ba biến thể defense (Booster gốc và hai bản ablation về tập
refusal/hyperparameter/rank của LoRA) đã xây xong và được judge, xem
[`docs/redteam/STATUS.md`](STATUS.md) để có bảng kết quả đầy đủ và phần còn dang dở (sweep ngân
sách attacker, attacker giữ utility, và báo cáo viết).
