# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Lê Thanh Điệp
**Submission option**: Option B (GitHub + HuggingFace Hub)

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: Vietnamese Alpaca (Custom subset `data_500_p95.csv`), 500 samples (450 train + 50 eval)
- **max_seq_length**: 1024 (chọn từ phân tích p95 thực tế của token length)
- **GPU**: Tesla T4, 15/16 GB VRAM (Google Colab Free)
- **Training cost**: $0.00 USD (Quy đổi tương đương: ~0.25 USD với tổng 43.6 phút huấn luyện trên GPU T4 của Google Colab Free, đơn giá thị trường khoảng $0.35/giờ)
- **HF Hub link**: [thanhdiepnltd/qwen2.5-3b-vi-lab21-r16](https://huggingface.co/thanhdiepnltd/qwen2.5-3b-vi-lab21-r16)

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 1,843,200       | 14.26 min  | 3.15 GB   | 1.0769    | 2.94       |
| 16   | 3,686,400       | 14.93 min  | 2.55 GB   | 1.0699    | 2.91       |
| 64   | 14,745,600      | 14.38 min  | 3.93 GB   | 1.0670    | 2.91       |
| Base | -               | -          | -         | N/A       | N/A        |

*Ghi chú*: Chỉ số Peak VRAM đo được trên môi trường Colab sử dụng hàm `torch.cuda.max_memory_allocated()`. Mức VRAM của r=16 thấp hơn r=8 có thể do cơ chế thu hồi bộ nhớ (garbage collection) kích hoạt khác nhau giữa các cells, nhưng nhìn chung cả 3 đợt chạy đều tiêu thụ lượng VRAM cực kỳ tiết kiệm (< 4 GB) nhờ cấu hình QLoRA 4-bit và tối ưu hóa từ Unsloth.

---

## 3. Loss Curve Analysis

* **Quan sát**: 
  Tiến trình huấn luyện cho thấy đường Training Loss giảm đều đặn từ khoảng **1.29** ở các bước đầu tiên (step 5) xuống còn **1.03** ở bước cuối cùng (step 170). Vì cấu hình huấn luyện trên GPU T4 tắt tính năng đánh giá giữa chừng (`eval_strategy = "no"`) để tối ưu hóa bộ nhớ và tăng tốc độ, chúng ta chỉ quan sát được đường giảm của Training Loss. Tuy nhiên, giá trị Eval Loss sau khi kết thúc quá trình huấn luyện đạt **1.0699** (đối với r=16) - rất gần với giá trị Training Loss cuối cùng. Điều này chứng tỏ mô hình học tốt và **không xảy ra hiện tượng overfitting** (quá khớp), mô hình giữ được khả năng tổng quát hóa tốt trên tập eval.

---

## 4. Qualitative Comparison (5 examples)

Dưới đây là so sánh kết quả sinh văn bản giữa Base Model gốc và Fine-tuned Model (cấu hình baseline r=16) trên cùng một tập prompt kiểm thử:

### Example 1
* **Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
* **Base Model**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, n...
* **Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo)...
* **Nhận xét**: **Improved** (Tốt hơn). Mô hình sau khi fine-tune giải thích rõ ràng hơn về khía cạnh "không cần sự hướng dẫn trực tiếp" và có cách diễn đạt tự nhiên hơn bằng tiếng Việt.

### Example 2
* **Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
* **Base Model**: Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n:
```python
def fibonacci(n):
    if n <= 0:
        return "N p..."
```
* **Fine-tuned (r=16)**: Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau:
```python
def fibonacci(n):
    if n < 0:
        raise ValueError("Input phải là một số nguyên dương.")
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
```
* **Nhận xét**: **Improved** (Tốt hơn rõ rệt). Bản Base sinh code bị cụt và sử dụng đệ quy chưa tối ưu (hoặc bị lỗi logic khi n <= 0). Bản Fine-tuned trả về đoạn code sử dụng vòng lặp với độ phức tạp thời gian tối ưu $O(n)$, kiểm tra đầu vào đầy đủ và có cấu trúc mạch lạc hơn.

### Example 3
* **Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
* **Base Model**: 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục, màu sắc,...
* **Fine-tuned (r=16)**: 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản: UI/UX th...
* **Nhận xét**: **Improved** (Tốt hơn). Bản Base lặp đi lặp lại từ "thân thiện" và bị cụt ý. Bản Fine-tuned đưa ra các nguyên tắc cô đọng, chuyên nghiệp và có tính định hướng thực tế tốt hơn.

### Example 4
* **Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
* **Base Model**: LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp cải thiện hi...
* **Fine-tuned (r=16)**: LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai kỹ thuật khác nhau được sử dụng để cải thiện hiệu suất của mô hình máy học dựa trên ma trận. LoRA sử dụng ma trận bổ sung (add-on matrix) để thay đổi các ma trận trọng số của mô hình. Trong khi đó, QLoRA sử dụng kỹ thuật quantization (định lượng)...
* **Nhận xét**: **Improved** (Tốt hơn). Bản Fine-tuned giải thích chính xác bản chất kỹ thuật của QLoRA là tích hợp thêm quantization (lượng tử hóa ma trận trọng số gốc xuống 4-bit) so với LoRA thông thường, tránh được các diễn giải mơ hồ của bản Base.

### Example 5
* **Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
* **Base Model**: Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất của ...
* **Fine-tuned (r=16)**: Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp mô hình hiểu yêu cầu. RAG kết hợp việc tìm kiếm thông tin bên ngoài. Fine-tuning huấn luyện lại các trọng số...
* **Nhận xét**: **Improved** (Tốt hơn). Bản Fine-tuned trình bày rõ ràng và phân biệt mạch lạc hơn hẳn về vai trò của từng phương pháp trong thực tế phát triển ứng dụng LLM.

---

## 5. Conclusion về Rank Trade-off

Dựa trên kết quả thực nghiệm huấn luyện mô hình `Qwen2.5-3B` trên tập dữ liệu tiếng Việt nhỏ (500 mẫu), ta rút ra các kết luận sau:

1. **Rank mang lại ROI tốt nhất**: **Rank $r=16$** mang lại ROI tốt nhất. Nó đạt mức perplexity rất thấp (**2.91**), tương đương với Rank $r=64$ nhưng tiết kiệm hơn về mặt tham số huấn luyện (chỉ 3.6 triệu tham số so với 14.7 triệu của $r=64$) và giảm nguy cơ quá khớp dữ liệu khi tập mẫu nhỏ.
2. **Hiện tượng "Diminishing Returns" (Hiệu suất cận biên giảm dần)**: Hiện tượng này được quan sát rất rõ ràng khi so sánh giữa $r=16$ và $r=64$. Khi tăng rank từ 16 lên 64 (gấp 4 lần số lượng tham số huấn luyện), chỉ số Perplexity chỉ giảm cực kỳ nhỏ (từ **2.915** xuống **2.907**, tức là chỉ cải thiện khoảng 0.27%). Trong khi đó, tài nguyên tính toán và bộ nhớ GPU đỉnh điểm (Peak VRAM) tăng từ 2.55 GB lên 3.93 GB. Điều này chứng minh rằng đối với các tập dữ liệu nhỏ hoặc tác vụ chuyên biệt hóa không quá phức tạp, việc tiếp tục nâng cao rank sẽ mang lại lợi ích rất ít so với chi phí tài nguyên tiêu thụ.
3. **Khuyến nghị cho triển khai thực tế (Production)**: Khuyến nghị lựa chọn **Rank $r=16$** làm cấu hình mặc định. Mức rank này đảm bảo hiệu năng tối ưu nhất về mặt chất lượng câu trả lời trong khi giữ cho kích thước file adapter nhỏ nhẹ (~14.7 MB), giúp hệ thống dễ dàng phục vụ nhiều tenant cùng lúc (multi-tenant serving) bằng cách hoán đổi động các adapter trên cùng một base model.

---

## 6. What I Learned

Sau khi hoàn thành bài lab này, tôi đã rút ra được một số bài học kinh nghiệm cá nhân quan trọng:

- **Tối ưu hóa tài nguyên phần cứng**: Nhờ có bitsandbytes (NF4 quantization) và thư viện Unsloth, việc fine-tune một mô hình ngôn ngữ lớn 3B đã trở nên khả thi ngay trên các dòng GPU phổ thông/miễn phí như Tesla T4 với chưa đầy 15 phút huấn luyện và tiêu tốn chỉ ~3 GB VRAM.
- **Tầm quan trọng của Rank và Alpha**: Việc thiết lập tỷ lệ $\alpha/r = 2$ (ví dụ: $r=8, \alpha=16$ hoặc $r=16, \alpha=32$) là một quy tắc kinh nghiệm tốt giúp ổn định quá trình tối ưu hóa. Cần đánh giá kỹ lưỡng để tránh lãng phí tài nguyên cho việc tăng rank vô ích.
- **Chất lượng dữ liệu quyết định đầu ra**: Với các tập dữ liệu nhỏ dưới 1000 dòng, chất lượng câu trả lời (qualitative output) phụ thuộc rất lớn vào việc làm sạch dữ liệu, lọc bỏ các câu cụt và giới hạn token hợp lý (`max_seq_length`) để tránh mô hình học phải các mẫu lỗi.
