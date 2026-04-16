# AgroSmart Predictor - Phân hệ AI Engine

Đây là dự án mô phỏng hệ thống hỗ trợ ra quyết định dựa trên mô hình Hybrid AI. Hệ thống thực hiện việc phân loại và đề xuất nhãn phù hợp dựa trên vector đặc trưng gồm 7 thông số môi trường (N, P, K, pH, nhiệt độ, độ ẩm, lượng mưa).

Kho lưu trữ này tập trung triển khai **Lớp xử lý thông minh (Intelligence Layer). Mục tiêu chính là xây dựng, huấn luyện và so sánh hiệu năng của các thuật toán Machine Learning (Decision Tree, Random Forest, Gradient Boosting, v.v.) nhằm chọn ra mô hình tối ưu nhất cho hệ thống phân loại. Mô hình AI này sau đó sẽ được kết hợp với Rule-based Engine để đưa ra kết quả dự đoán cuối cùng (Fusion Module).

**Công nghệ & Thuật toán sử dụng:**
- **Ngôn ngữ:** Python
- **Thư viện:** Scikit-learn, Pandas
- **Thuật toán cốt lõi:** Random Forest, Gradient Boosting, Decision Tree, Extra Trees.
- **Kiến trúc:** Phân tầng (Presentation, Application, Intelligence, Data).
