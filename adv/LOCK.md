# Khóa cấu hình – Thử thách nâng cao (ghi TRƯỚC khi chạy test)

Ngày: 2026-11-16. Tài liệu: Gemini 1.5 technical report (arXiv 2403.05530v5, 154 trang). Benchmark: `adv/benchmark.json` (test = 30 câu: 24 có đáp án + 6 không có bằng chứng).

## Cấu hình khóa: `D_hybrid_k5_b6000` (adv/run_adv.py)
- Retrieval: hybrid RRF (dense all-MiniLM-L6-v2 trên pgvector + BM25), 20 ứng viên mỗi nhánh, top_k = 5
- Context: 6000 ký tự, chính sách `latest_first`; tối đa 4 lần search; LLM `gpt-6-luna`, reasoning_effort=none
- Index: `ADV_gemini15` (766 chunk: 623 text, 79 bảng, 41 biểu đồ, 23 hình)

## Lý do chọn (chỉ dựa trên dev, 18 câu, 2 lượt mỗi cấu hình ứng viên)
Quy tắc đặt trước: nhiều câu đạt hơn trên dev thắng; hòa thì chọn rẻ hơn.
- C_hybrid_latest (top-3, 3000): lượt 2 sai d12 (multi-hop hai bảng); ~50K token/18 câu
- D_hybrid_k5_b6000: dev 15/15 trả lời, 15/15 citation phủ bằng chứng, 3/3 từ chối (lượt 2); ~67K token/18 câu
- Dev đã sửa 2 câu hỏi mơ hồ (d07, d09; bằng chứng không đổi). Test KHÔNG bị chạm.

## Tiêu chí chấm test (cố định trước khi chạy)
- Câu có đáp án = ĐẠT khi: (1) đáp án đúng và đủ (giá trị + đơn vị; câu so sánh có hiệu/đại lượng đúng; multi-hop có đủ cả hai nguồn), (2) trích dẫn chunk thuộc đúng trang/vùng gold, (3) không khẳng định sai nào kèm theo. Thiếu một điều kiện = KHÔNG ĐẠT. Câu hệ thống lỗi (error) = KHÔNG ĐẠT, vẫn nằm trong mẫu số.
- Câu không có bằng chứng = ĐẠT khi từ chối (không đủ bằng chứng) và không đưa ra con số/khẳng định không có cơ sở.
- Ngưỡng: >= 27/30 (90%), không làm tròn. Chạy test đúng 1 lần, không chỉnh sau khi xem kết quả.
