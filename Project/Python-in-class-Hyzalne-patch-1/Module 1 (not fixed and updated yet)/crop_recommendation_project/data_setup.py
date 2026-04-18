import pandas as pd
import numpy as np

crops = [
    "Rice", "Corn", "Wheat", "Barley", "Millet", "Sorghum", "Potato", "Sweet Potato", "Cassava", "Taro", 
    "Carrot", "Radish", "Beetroot", "Spinach", "Mustard Greens", "Lettuce", "Water Spinach", "Malabar Spinach", 
    "Tomato", "Chili", "Cucumber", "Pumpkin", "Eggplant", "Bottle Gourd", "Luffa", "Onion", "Garlic", 
    "Ginger", "Turmeric", "Lemongrass", "Coriander", "Dill", "Orange", "Mandarin", "Grapefruit", "Lemon", 
    "Mango", "Banana", "Pineapple", "Jackfruit", "Durian", "Rambutan", "Mangosteen", "Coconut", "Lychee", 
    "Longan", "Guava", "Apple", "Pomegranate", "Custard Apple", "Coffee", "Rubber", "Tea", "Pepper", 
    "Cashew", "Sugarcane", "Cotton", "Soybean", "Peanut", "Tobacco", "Acacia", "Eucalyptus", "Pine", 
    "Melaleuca", "Cinnamon", "Star Anise", "Macadamia", "Rose", "Chrysanthemum", "Orchid", "Apricot Blossom", 
    "Peach Blossom", "Lily"
]

def generate_data():
    range_data = []
    ml_data = []
    
    for crop in crops:
        # Tạo giải (min, max) ngẫu nhiên nhưng logic cho từng loại cây
        n = np.random.randint(20, 100); p = np.random.randint(20, 100); k = np.random.randint(20, 100)
        t = np.random.randint(15, 35); h = np.random.randint(40, 90); ph = np.round(np.random.uniform(5.0, 7.5), 1)
        r = np.random.randint(500, 2500)
        
        range_data.append([crop, n-15, n+15, p-15, p+15, k-15, k+15, t-5, t+5, h-10, h+10, ph-0.5, ph+0.5, r-200, r+200])
        
        # Tạo 20 dòng dữ liệu mẫu cho mỗi loại cây để train ML
        for _ in range(20):
            ml_data.append([
                crop, 
                np.random.randint(n-10, n+10), np.random.randint(p-10, p+10), np.random.randint(k-10, k+10),
                np.random.uniform(t-3, t+3), np.random.uniform(h-5, h+5), np.random.uniform(ph-0.3, ph+0.3),
                np.random.uniform(r-100, r+100)
            ])

    # Lưu file
    df_range = pd.DataFrame(range_data, columns=['crop', 'N_min', 'N_max', 'P_min', 'P_max', 'K_min', 'K_max', 
                                                 'temp_min', 'temp_max', 'humidity_min', 'humidity_max', 
                                                 'ph_min', 'ph_max', 'rain_min', 'rain_max'])
    df_range.to_csv('crop_range.csv', index=False)
    
    df_ml = pd.DataFrame(ml_data, columns=['crop', 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    df_ml.to_csv('crop_data.csv', index=False)
    print("✅ Đã tạo dữ liệu mẫu thành công!")

if __name__ == "__main__":
    generate_data()