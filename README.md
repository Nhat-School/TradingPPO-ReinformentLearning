# 📈 Reinforcement Trading Bot (PPO)

This project implements a Forex trading bot using Reinforcement Learning (PPO - Proximal Policy Optimization) from the `stable-baselines3` library. The bot is trained on EUR/USD hourly candlestick data.

---

## 🇻🇳 Hướng dẫn sử dụng 

### ⚠️ Lưu ý quan trọng
Dự án này yêu cầu **Python 3.9** (hoặc cao hơn nhưng dưới 3.14) để tương thích với thư viện `pandas-ta`. Nếu bạn dùng Python 3.14, thư viện sẽ bị lỗi.

### Các bước cài đặt:
1. **Di chuyển vào thư mục dự án**:
   ```bash
   cd ReinforcementTrading_Part_1
   ```
2. **Tạo môi trường ảo (Venv) sử dụng Python 3.9**:
   *(Đảm bảo bạn đã cài Python 3.9 trên máy)*
   ```bash
   /usr/bin/python3 -m venv venv
   ```
3. **Kích hoạt môi trường ảo**:
   ```bash
   source venv/bin/activate
   ```
4. **Cài đặt các thư viện cần thiết**:
   ```bash
   pip install -r Requirements.txt
   pip install pandas-ta-classic torch stable-baselines3 matplotlib tensorboard shimmy
   ```

### Cách chạy Bot:
*   **Để Train mới**: `python train_agent.py`
    *   Kết quả sẽ lưu model vào `model_eurusd_best.zip` và ảnh đồ thị vào `equity_curve.png`.
*   **Để Test model đã có**: `python test_agent.py`
    *   Kết quả sẽ lưu lịch sử giao dịch vào `trade_history_output.csv` và ảnh đồ thị vào `test_evaluation_curve.png`.

### Cách xem tensorboard:
```bash
tensorboard --logdir=ReinforcementTrading_Part_1/tensorboard_log
```