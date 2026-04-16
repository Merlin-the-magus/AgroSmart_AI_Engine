import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    ExtraTreesClassifier, 
    HistGradientBoostingClassifier
)
from sklearn.metrics import accuracy_score
import time
import os

# Tên file dữ liệu tải từ Kaggle
DATA_FILE = 'Crop_recommendation.csv' 

def main():
    # 1. Kiểm tra sự tồn tại của file dữ liệu
    if not os.path.exists(DATA_FILE):
        print(f"[LỖI] Không tìm thấy file '{DATA_FILE}'.")
        print("Hãy đảm bảo file .csv nằm cùng cấp thư mục với script chạy.")
        return

    print(f"[*] Đang tải dữ liệu từ {DATA_FILE}...\n")
    df = pd.read_csv(DATA_FILE)

    # 2. Xử lý đặc trưng (Features) và Nhãn (Labels)
    # Tự động lấy tên cột theo file Kaggle tiêu chuẩn
    if 'nitrogen' in df.columns:
        features = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
    else:
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    X = df[features]
    y = df['label']

    # Chia tập dữ liệu: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"[-] Dữ liệu huấn luyện: {len(X_train)} mẫu")
    print(f"[-] Dữ liệu kiểm thử: {len(X_test)} mẫu\n")

    # 3. Khởi tạo danh sách 5 mô hình
    # Thiết lập n_jobs=-1 ở các mô hình hỗ trợ để tận dụng tối đa đa luồng (multithreading) của CPU
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
    }

    # 4. Vòng lặp huấn luyện và so sánh
    print("=" * 65)
    print(f"{'SO SÁNH HIỆU NĂNG CÁC THUẬT TOÁN (AI ENGINE)':^65}")
    print("=" * 65)

    best_model_name = ""
    best_accuracy = 0

    for name, model in models.items():
        start_time = time.time()
        
        # Huấn luyện mô hình
        model.fit(X_train, y_train)
        
        # Dự đoán
        y_pred = model.predict(X_test)
        
        # Đo lường thời gian và độ chính xác
        execution_time = time.time() - start_time
        acc = accuracy_score(y_test, y_pred)
        
        # In kết quả
        print(f"▶ {name.upper()}")
        print(f"  - Độ chính xác: {acc * 100:.2f}%")
        print(f"  - Thời gian chạy: {execution_time:.4f} giây")
        print("-" * 65)

        # Cập nhật thuật toán dẫn đầu
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name

    print(f"\n=> KẾT LUẬN: Mô hình đề xuất đưa vào hệ thống là [{best_model_name}] với độ chính xác {best_accuracy * 100:.2f}%.")

if __name__ == "__main__":
    main()