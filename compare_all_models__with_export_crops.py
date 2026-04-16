import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    ExtraTreesClassifier, 
    HistGradientBoostingClassifier
)
from sklearn.metrics import accuracy_score

DATA_FILE = 'export_crops.csv' 
SAMPLES_PER_CROP = 200  

def generate_synthetic_data(file_path):
    print(f"[*] Đang đọc file {file_path} và dọn dẹp dữ liệu...")
    
    # 1. Đọc file (hỗ trợ cả dấu phẩy và chấm phẩy đề phòng Excel)
    try:
        df_rules = pd.read_csv(file_path)
        if len(df_rules.columns) < 5:
            df_rules = pd.read_csv(file_path, sep=';')
    except Exception as e:
        print(f"[LỖI] Không thể đọc file: {e}")
        return pd.DataFrame()

    synthetic_data = []
    valid_crops = 0
    
    # Hàm ép kiểu số cực kỳ an toàn
    def safe_float(val):
        if pd.isna(val): return None
        # Đổi dấu phẩy thành dấu chấm (ví dụ: 5,5 -> 5.5) 
        cleaned = str(val).replace(',', '.').strip()
        try:
            return float(cleaned)
        except ValueError:
            return None

    # Duyệt từng dòng một cách an toàn
    for index, row in df_rules.iterrows():
        crop_name = row.get('Crop Name', f'Unknown_{index}')
        
        try:
            n_min = safe_float(row['N_min']); n_max = safe_float(row['N_max'])
            p_min = safe_float(row['P_min']); p_max = safe_float(row['P_max'])
            k_min = safe_float(row['K_min']); k_max = safe_float(row['K_max'])
            t_min = safe_float(row['Temp_min']); t_max = safe_float(row['Temp_max'])
            h_min = safe_float(row['Humidity_min']); h_max = safe_float(row['Humidity_max'])
            ph_min = safe_float(row['pH_min']); ph_max = safe_float(row['pH_max'])
            r_min = safe_float(row['Rain_min']); r_max = safe_float(row['Rain_max'])

            # Bỏ qua chỉ riêng những dòng bị lệch cột/chữ rác
            if None in [n_min, n_max, p_min, p_max, k_min, k_max, t_min, t_max, h_min, h_max, ph_min, ph_max, r_min, r_max]:
                print(f"  -> [BỎ QUA] Cây '{crop_name}' do dữ liệu cột số bị lỗi hoặc trống.")
                continue
                
            # Tự động sửa lỗi gõ ngược Min/Max
            if n_min > n_max: n_min, n_max = n_max, n_min
            if p_min > p_max: p_min, p_max = p_max, p_min
            if k_min > k_max: k_min, k_max = k_max, k_min
            if t_min > t_max: t_min, t_max = t_max, t_min
            if h_min > h_max: h_min, h_max = h_max, h_min
            if ph_min > ph_max: ph_min, ph_max = ph_max, ph_min
            if r_min > r_max: r_min, r_max = r_max, r_min

            # Sinh dữ liệu
            for _ in range(SAMPLES_PER_CROP):
                n = np.random.uniform(n_min, n_max)
                p = np.random.uniform(p_min, p_max)
                k = np.random.uniform(k_min, k_max)
                temp = np.random.uniform(t_min, t_max)
                hum = np.random.uniform(h_min, h_max)
                ph = np.random.uniform(ph_min, ph_max)
                rain = np.random.uniform(r_min, r_max)
                
                synthetic_data.append([n, p, k, temp, hum, ph, rain, crop_name])
            
            valid_crops += 1
            
        except KeyError as e:
            print(f"[LỖI] File bị thiếu cột: {e}. Vui lòng kiểm tra lại dòng tiêu đề.")
            return pd.DataFrame()
    
    print(f"[*] Đã quét xong. Tổng số loại cây cứu được: {valid_crops}/{len(df_rules)}")
    columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
    return pd.DataFrame(synthetic_data, columns=columns)

def main():
    if not os.path.exists(DATA_FILE):
        print(f"[LỖI] Không tìm thấy '{DATA_FILE}'.")
        return

    df = generate_synthetic_data(DATA_FILE)
    
    # Chặn lỗi văng code nếu DataFrame vẫn rỗng
    if df.empty:
        print("\n[LỖI NGHIÊM TRỌNG] Không có dữ liệu hợp lệ nào được sinh ra!")
        return

    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"[-] Đã sinh thành công tổng cộng: {len(df)} mẫu dữ liệu.")
    print(f"[-] Dữ liệu huấn luyện: {len(X_train)} mẫu")
    print(f"[-] Dữ liệu kiểm thử: {len(X_test)} mẫu\n")

    # Khởi tạo danh sách 5 mô hình (Đã tối ưu tốc độ cho 100 loại cây)
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(random_state=42, n_jobs=-1),
        
        # Giới hạn số cây (n_estimators=10) thay vì 100 để nó chạy xong trong vài giây
        "Gradient Boosting (Fast)": GradientBoostingClassifier(random_state=42, n_estimators=10),
        
        # Thuật toán Boosting thế hệ mới, sinh ra để xử lý data khổng lồ
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42, max_iter=50)
    }

    print("=" * 65)
    print(f"{'SO SÁNH HIỆU NĂNG CÁC THUẬT TOÁN':^65}")
    print("=" * 65)

    best_model_name = ""
    best_accuracy = 0

    for name, model in models.items():
        start_time = time.time()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        execution_time = time.time() - start_time
        acc = accuracy_score(y_test, y_pred)
        
        print(f"▶ {name.upper()}")
        print(f"  - Độ chính xác: {acc * 100:.2f}%")
        print(f"  - Thời gian chạy: {execution_time:.4f} giây")
        print("-" * 65)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name

    print(f"\n=> KẾT LUẬN: Mô hình tối ưu trên tập dữ liệu này là [{best_model_name}] với độ chính xác {best_accuracy * 100:.2f}%.")

if __name__ == "__main__":
    main()