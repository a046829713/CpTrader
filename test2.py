import torch
import torch.nn as nn
import torch.optim as optim

# 定義一個簡單的線性模型
model = nn.Linear(in_features=1, out_features=1)

# 定義 MSE 損失函數
criterion = nn.MSELoss()

# 優化器（以隨機梯度下降為例）
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 模擬數據
x = torch.randn(10, 1)  # 假設有 10 個輸入樣本
y_actual = 2 * x + 3    # 真實值，基於某個線性關係

# 模型預測
y_pred = model(x)

print(y_actual)
print(y_pred)
# 計算 MSE 損失
loss = criterion(y_pred, y_actual)

print("MSE Loss:", loss.item())