import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

def main():
    # Đường dẫn file
    parquet_path = os.path.join("data", "data.parquet")
    csv_output_path = os.path.join("data", "data_500_p95.csv")
    
    if not os.path.exists(parquet_path):
        print(f"❌ Không tìm thấy file {parquet_path}. Vui lòng kiểm tra lại đường dẫn.")
        return

    print(f"1. Đang đọc dữ liệu từ {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"✓ Đã đọc {len(df)} dòng dữ liệu.")
    
    # In ra các cột hiện có để kiểm tra
    print(f"✓ Các cột có trong file: {list(df.columns)}")
    
    # Xác định các cột tiếng Việt
    instruction_col = 'instruction_vi' if 'instruction_vi' in df.columns else 'instruction'
    input_col = 'input_vi' if 'input_vi' in df.columns else 'input'
    output_col = 'output_vi' if 'output_vi' in df.columns else 'output'
    
    print(f"✓ Sử dụng cột: instruction='{instruction_col}', input='{input_col}', output='{output_col}'")

    # 2. Khởi tạo tokenizer của Qwen2.5-3B để tính token length chính xác
    # Bạn có thể thay đổi tên model tùy theo model bạn chọn huấn luyện
    model_name = "unsloth/Qwen2.5-3B-bnb-4bit"
    print(f"2. Đang tải tokenizer cho {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"⚠️ Không tải được tokenizer online. Sẽ fallback sang tokenizer mặc định 'gpt2'. Chi tiết lỗi: {e}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Template format prompt Alpaca
    alpaca_template = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
    alpaca_template_no_input = "### Instruction:\n{instruction}\n\n### Response:\n{output}"

    # 3. Tính độ dài token cho từng dòng dữ liệu
    print("3. Đang phân tích độ dài token từng mẫu dữ liệu...")
    token_lengths = []
    
    # Điền giá trị Na/Null bằng chuỗi rỗng
    df[instruction_col] = df[instruction_col].fillna("")
    df[input_col] = df[input_col].fillna("")
    df[output_col] = df[output_col].fillna("")

    for idx, row in df.iterrows():
        inst = row[instruction_col]
        inp = row[input_col]
        out = row[output_col]
        
        if inp.strip():
            text = alpaca_template.format(instruction=inst, input=inp, output=out)
        else:
            text = alpaca_template_no_input.format(instruction=inst, output=out)
            
        length = len(tokenizer.encode(text))
        token_lengths.append(length)

    df["token_length"] = token_lengths

    # 4. Tính toán bách phân vị p95
    p95_threshold = int(np.percentile(token_lengths, 95))
    print(f"✓ Ngưỡng p95 độ dài token: {p95_threshold} tokens.")

    # 5. Lọc các mẫu <= p95 (loại bỏ 5% mẫu quá dài)
    df_filtered = df[df["token_length"] <= p95_threshold]
    print(f"✓ Số lượng mẫu an toàn (<= {p95_threshold} tokens): {len(df_filtered)}")

    # 6. Trích xuất ngẫu nhiên 500 mẫu
    # Nếu tập dữ liệu lọc xong vẫn nhiều hơn 500 dòng
    if len(df_filtered) >= 500:
        df_subset = df_filtered.sample(n=500, random_state=42)
    else:
        df_subset = df_filtered
        print(f"⚠️ Tập dữ liệu sau lọc chỉ có {len(df_filtered)} mẫu, lấy toàn bộ.")

    # Loại bỏ các cột không cần thiết (giữ lại các cột tiếng Việt gốc và cột tiếng Anh nếu muốn)
    columns_to_keep = [instruction_col, input_col, output_col]
    for col in ['instruction_en', 'input_en', 'output_en']:
        if col in df.columns:
            columns_to_keep.append(col)
            
    df_subset = df_subset[columns_to_keep]

    # 7. Xuất file CSV
    df_subset.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"🎉 Đã tạo thành công file: {csv_output_path} với {len(df_subset)} dòng!")

if __name__ == "__main__":
    main()
