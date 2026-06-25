# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Lê Thanh Điệp
**Submission option**: <!-- TODO: Điền Option nộp bài bạn chọn (A / B / C) -->

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (hoặc thay đổi nếu bạn chọn model khác trong notebook)
- **Dataset**: Vietnamese Alpaca (Custom subset `data_500_p95.csv`), 500 samples (450 train + 50 eval)
- **max_seq_length**: 512 hoặc 1024 (p95 thực tế của file này là 548 tokens, làm tròn lên)
- **GPU**: Tesla T4, 15/16 GB VRAM (Google Colab Free)
- **Training cost**: <!-- TODO: Điền chi phí thực tế và quy đổi. Vì dùng Google Colab Free nên chi phí thực tế là $0.00. Tuy nhiên, bạn hãy quy đổi tương đương theo giá thị trường: Tổng thời gian chạy cả 3 rank (ví dụ: ~50 phút) nhân với giá thuê GPU T4 thông dụng (khoảng $0.20 đến $0.35/giờ). Công thức: <Số phút train / 60> x <Giá GPU/giờ>. Ví dụ: 50/60 * 0.30 = ~0.25 USD -->
- **HF Hub link** (Nếu chọn Option B): <!-- TODO: Điền link repo adapter của bạn trên HuggingFace Hub nếu có (ví dụ: https://huggingface.co/username/adapter-name). Nếu không có (hoặc chọn Option A/C) thì ghi 'N/A' -->

## 2. Rank Experiment Results

<!-- TODO: Điền các thông số bạn thu được sau khi hoàn thành 3 đợt huấn luyện trong notebook vào bảng bên dưới. Hướng dẫn lấy dữ liệu:
  * Trainable Params: Lấy từ dòng in ra của notebook (ví dụ: "✓ Trainable: 3,686,400...") hoặc từ key "trainable_params" trong object kết quả của hàm train_one_rank().
  * Train Time: Thời gian huấn luyện tính bằng phút. Lấy từ cell huấn luyện baseline r=16 (tính thủ công bằng time.time()) hoặc từ key "train_time_min" trong object kết quả của r=8, r=64.
  * Peak VRAM: Bộ nhớ GPU đỉnh điểm (GB). Lấy từ cell baseline r=16 (chạy torch.cuda.max_memory_allocated() / 1e9) hoặc từ key "peak_vram_gb" trong object kết quả của r=8, r=64.
  * Eval Loss & Perplexity: Kết quả loss sau khi chạy safe_evaluate() và Perplexity = exp(eval_loss) (hoặc lấy từ key "eval_loss" và "eval_perplexity").
  * Base: Đối với Base model gốc, chỉ cần chạy safe_evaluate() trên base_model để lấy Eval Loss & Perplexity mà không cần train. -->

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | <!-- TODO -->   | ... min    | ... GB    | ...       | ...        |
| 16   | <!-- TODO -->   | ... min    | ... GB    | ...       | ...        |
| 64   | <!-- TODO -->   | ... min    | ... GB    | ...       | ...        |
| Base | -               | -          | -         | ...       | ...        |

---

## 3. Loss Curve Analysis

<!-- TODO: Chèn ảnh biểu đồ loss_curve.png của bạn vào đây (nếu chạy local hoặc tải về máy). Có thể dùng markdown cú pháp: ![Loss Curve](results/loss_curve.png) -->

* **Quan sát**: 
  <!-- TODO: Viết nhận xét về biểu đồ loss (đường training loss và eval loss). Có xảy ra hiện tượng overfitting không (eval loss tăng trong khi train loss vẫn giảm)? Giải thích lý do tại sao có hoặc không. -->

---

## 4. Qualitative Comparison (5 examples)

<!-- TODO: Chèn 5 ví dụ side-by-side so sánh kết quả sinh văn bản của Base Model và Fine-tuned Model (r=16) trên cùng một Prompt. -->

### Example 1
* **Prompt**: <!-- TODO: Nhập câu Prompt thử nghiệm 1 -->
* **Base Model**: <!-- TODO: Câu trả lời của mô hình base gốc -->
* **Fine-tuned (r=16)**: <!-- TODO: Câu trả lời của mô hình sau fine-tune -->
* **Nhận xét**: <!-- TODO: Nhận xét chất lượng (improved - tốt hơn / same - như nhau / degraded - tệ đi) -->

### Example 2
* **Prompt**: <!-- TODO: Nhập câu Prompt thử nghiệm 2 -->
* **Base Model**: <!-- TODO: Câu trả lời của mô hình base gốc -->
* **Fine-tuned (r=16)**: <!-- TODO: Câu trả lời của mô hình sau fine-tune -->
* **Nhận xét**: <!-- TODO: Nhận xét chất lượng (improved / same / degraded) -->

### Example 3
* **Prompt**: <!-- TODO: Nhập câu Prompt thử nghiệm 3 -->
* **Base Model**: <!-- TODO: Câu trả lời của mô hình base gốc -->
* **Fine-tuned (r=16)**: <!-- TODO: Câu trả lời của mô hình sau fine-tune -->
* **Nhận xét**: <!-- TODO: Nhận xét chất lượng (improved / same / degraded) -->

### Example 4
* **Prompt**: <!-- TODO: Nhập câu Prompt thử nghiệm 4 -->
* **Base Model**: <!-- TODO: Câu trả lời của mô hình base gốc -->
* **Fine-tuned (r=16)**: <!-- TODO: Câu trả lời của mô hình sau fine-tune -->
* **Nhận xét**: <!-- TODO: Nhận xét chất lượng (improved / same / degraded) -->

### Example 5
* **Prompt**: <!-- TODO: Nhập câu Prompt thử nghiệm 5 -->
* **Base Model**: <!-- TODO: Câu trả lời của mô hình base gốc -->
* **Fine-tuned (r=16)**: <!-- TODO: Câu trả lời của mô hình sau fine-tune -->
* **Nhận xét**: <!-- TODO: Nhận xét chất lượng (improved / same / degraded) -->

---

## 5. Conclusion về Rank Trade-off

<!-- TODO: Viết kết luận tối thiểu 100 từ, trả lời đầy đủ 3 câu hỏi sau:
1. Rank nào mang lại ROI (tỷ lệ hiệu quả trên chi phí/thời gian) tốt nhất trên tập dữ liệu này? Tại sao?
2. Có quan sát thấy hiện tượng "diminishing returns" không (tức là tăng rank từ 16 lên 64 nhưng chỉ số perplexity/eval loss hầu như không cải thiện hoặc cải thiện cực ít, trong khi tốn thêm nhiều thời gian và VRAM)?
3. Nếu phải triển khai sản phẩm thực tế (production deployment), bạn sẽ khuyến nghị chọn rank nào? Tại sao? -->

---

## 6. What I Learned

<!-- TODO: Viết ít nhất 2-3 điểm đúc kết, bài học kinh nghiệm cá nhân của bạn sau khi hoàn thành bài lab này (Ví dụ: về tối ưu hóa VRAM, cách chọn rank, tầm quan trọng của chất lượng dữ liệu vs số lượng dữ liệu...). -->

- <!-- TODO: Bài học 1 -->
- <!-- TODO: Bài học 2 -->
- <!-- TODO: Bài học 3 (Tùy chọn) -->
