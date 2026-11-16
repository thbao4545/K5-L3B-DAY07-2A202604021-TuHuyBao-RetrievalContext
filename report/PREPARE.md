# Day 7 – Chuẩn bị (GUIDE.md §1)

## Tái sử dụng Day 6

| Mục | Giá trị |
|---|---|
| Corpus | `2026Reimagining-Text-to-SQL.pdf` (arXiv 2611.00171v1), 47 trang, 47 trang nội dung (≥ 30), không trang trống/OCR |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (local, không tốn phí API); sentence-transformers 6.3.0, torch 2.15.0 |
| Số chiều | 384, vector chuẩn hóa, khoảng cách cosine (`<=>`) |
| Vector store | PostgreSQL 16.16 + pgvector 0.8.7 (Docker `pgvector/pgvector:0.8.7-pg16`) |
| Kết nối | `docker compose -p day6_student_starter up -d --wait` trong `K5-L3-Day06-DataIngestionEngineering/starter`; `DATABASE_URL` trong `.env` của Day 6 (`127.0.0.1:55433/day6_lab`, chỉ local) |
| Bảng | `chunks(strategy, chunk_id, document_id, source, page, text, metadata jsonb, embedding vector(384))` |
| Index chính | `B_paragraph`: 199 chunk, 47 trang. Còn `A_fixed`: 252 chunk (800/100) |
| ID chunk | `<document_id>:p<page>:<paragraph\|fixed>:<index>` (duy nhất, giữ nguyên từ Day 6) |
| Metadata | `section` (heading đánh số gần nhất, 28 giá trị), `page`, `strategy`, `chunk_index`, `char_count`, `empty`, `needs_ocr` |

**Tương thích query/index:** dense retrieval phải embed query bằng đúng model trên (`MyEmbedder.embed_query` của Day 6, không có prefix vì không phải họ e5). `starter/export_day6.py` kiểm tra: số chiều model = số chiều vector lưu, và embed lại một chunk đã lưu cho cosine = 1.000000. Không đổi model nên không phát sinh index mới hay chi phí embed thêm (chi phí embed Day 6: 7.89 s cho A, 6.67 s cho B, 0 USD).

## Dữ liệu xuất cho Day 7 (`starter/data/`, bị git bỏ qua)

- `day6/chunks_A_fixed.json`, `day6/chunks_B_paragraph.json`, `day6/chunks.json` (= B), `day6/index_info.json`
- `converted/{chunks,questions}.json` – đổi tên trường bằng `convert_day6.py`
- `benchmark.json` – **file chấm điểm; chứa gold, không được đưa vào agent/retriever**

Tái tạo (từ thư mục gốc repo, dùng venv của Day 6 vì cần psycopg + sentence-transformers):

```bash
D6=../K5-L3-Day06-DataIngestionEngineering/starter
$D6/.venv/Scripts/python.exe -I starter/export_day6.py --day6 $D6 --output starter/data/day6
python -I starter/convert_day6.py --chunks starter/data/day6/chunks.json --questions $D6/data/benchmark.json --output starter/data/converted
python -I starter/build_benchmark.py
```

## Benchmark (theo BENCHMARK.md §1)

| | Có đáp án | Thiếu bằng chứng | Tối thiểu |
|---|---:|---:|---:|
| Dev | 20 (q01–q20) | 2 (u01, u02) | 17 ✓ |
| Test | 6 (t01–t06, mới) | 2 (u03, u04) | 7 ✓ |

- 20 câu có đáp án lấy từ Day 6; đã kiểm tra mỗi `evidence` nằm trọn trong **một** chunk B_paragraph đúng trang. 5 câu có `filter` theo `section` do người dùng cung cấp (q07, q08, q12, q14, q19).
- 4 câu thiếu bằng chứng là câu gần chủ đề, nhưng đối tượng hỏi không có trong tài liệu: KaggleDBQA/GPT-4o, giờ công annotation của Spider 2.0, Dr.Spider, EHRSQL. `build_benchmark.py` quét toàn văn mọi chunk của cả hai chiến lược và **dừng nếu thấy từ khóa**; cách kiểm tra ghi trong `absence_reason`. Quét từ khóa chưa thay thế việc đọc lại: đã tìm thêm các từ liên quan (`annotat`, `hour`, `perturb`, `health`) và không thấy thông tin trả lời.

## Giới hạn cần ghi trong báo cáo

1. **Test Day 6 đã dùng để báo cáo và chấm tay** (q16–q20), nên đã chuyển sang dev (có `day6_split` trong benchmark.json). Test Day 7 là 6 câu mới t01–t06 (trang 17–25, 1 câu có filter `section`), viết trước khi chạy bất kỳ phương pháp Day 7 nào; `build_benchmark.py` kiểm tra mỗi evidence nằm nguyên văn trong đúng 1 chunk B_paragraph của trang đó và filter khớp metadata. Hạn chế: người viết test cũng là người đã đọc tài liệu và biết các chunk, test chỉ 6 câu.
2. Chỉ có một tài liệu (47 trang), toàn văn bản; không bảng/ảnh nên không thuộc ADVANCED.md.
3. Câu hỏi tiếng Anh và model embedding chủ yếu tiếng Anh, vì vậy chưa kiểm tra được ảnh hưởng của tách từ tiếng Việt.
4. `tests/test_baseline.py::test_cli_uses_requested_split_only` lỗi trên Windows (đọc file UTF-8 bằng cp1252). Lỗi có sẵn trong repo gốc, không liên quan đến thay đổi này; chạy với `PYTHONUTF8=1` để tránh.

## Kiểm tra nhanh

TF-IDF baseline trên 15 câu dev (có đáp án), Hit@1 = 10/15, Hit@3 = 13/15, Context = 2/15 với budget 800 ký tự mặc định. Chỉ để xác nhận dữ liệu đúng định dạng, chưa phải kết quả nộp.
