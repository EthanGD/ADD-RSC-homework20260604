import re
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 從日誌中提取訓練數據
log_file = r'C:\Ethan\Project\Project\test\requests_curl\project\CDSS\ADD-RSC-homework20260604\2nd\main.log'

epochs = []
train_loss_list = []
train_acc_list = []
test_epochs = []
test_acc_list = []

with open(log_file, 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# 提取每個 epoch 的最後一個 batch 的訓練 loss 和 accuracy
for i, line in enumerate(lines):
    # 匹配訓練結束行
    if re.match(r'Train epoch \d+', line) and 'total time' in line:
        match = re.search(r'Train epoch (\d+).*?accuracy:([\d.]+)', line)
        if match:
            epoch = int(match.group(1))
            acc = float(match.group(2))
            epochs.append(epoch)
            train_acc_list.append(acc)
            
            # 往前找最後一個 batch 的 loss (epoch 索引和實際 epoch 號碼一致)
            for j in range(i-1, max(0, i-20), -1):
                if f'Train: [{epoch}][260/263]' in lines[j]:
                    loss_match = re.search(r'loss\s+([\d.]+)', lines[j])
                    if loss_match:
                        train_loss_list.append(float(loss_match.group(1)))
                    break

# 提取測試結果 (每 2 個 epoch 一次)
for i, line in enumerate(lines):
    if '* S_p:' in line and 'Best S_p:' in line:
        # 往前找 epoch 號碼
        for j in range(i-1, max(0, i-10), -1):
            if 'Train epoch' in lines[j]:
                match = re.search(r'Train epoch (\d+)', lines[j])
                if match:
                    test_epochs.append(int(match.group(1)))
                break
        # 往前找測試 accuracy
        for j in range(i+1, min(len(lines), i+5)):
            if '* Acc@1' in lines[j]:
                acc_match = re.search(r'\* Acc@1\s+([\d.]+)', lines[j])
                if acc_match:
                    test_acc_list.append(float(acc_match.group(1)))
                break

# 保存數據
data = {
    'epochs': epochs,
    'Train_Loss': train_loss_list,
    'Train_Accuracy': train_acc_list,
    'Test_Epochs': test_epochs,
    'Test_Accuracy': test_acc_list
}

with open(r'C:\Ethan\Project\Project\test\requests_curl\project\CDSS\ADD-RSC-homework20260604\2nd\loss_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"Extracted {len(epochs)} epochs")
print(f"Train Loss: {len(train_loss_list)}, Train Acc: {len(train_acc_list)}")
print(f"Test Acc: {len(test_acc_list)}")

if train_loss_list:
    print(f"Final Train Loss: {train_loss_list[-1]:.4f}, Train Acc: {train_acc_list[-1]:.2f}%")
if test_acc_list:
    print(f"Final Test Acc: {test_acc_list[-1]:.2f}%")

# 繪製 Loss 和 Accuracy 曲線
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss 曲線
if train_loss_list:
    axes[0].plot(epochs[:len(train_loss_list)], train_loss_list, 'b-o', label='Training Loss', linewidth=2, markersize=5)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss over Epochs', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(min(epochs)-2, max(epochs)+2)

# Accuracy 曲線
if train_acc_list:
    axes[1].plot(epochs[:len(train_acc_list)], train_acc_list, 'g-s', label='Training Accuracy', linewidth=2, markersize=5)
if test_acc_list and test_epochs:
    axes[1].plot(test_epochs, test_acc_list, 'r-^', label='Test Accuracy', linewidth=2, markersize=5, alpha=0.7)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Accuracy over Epochs', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(min(epochs)-2, max(epochs)+2)
axes[1].set_ylim(0, 100)

plt.tight_layout()
plt.savefig(r'C:\Ethan\Project\Project\test\requests_curl\project\CDSS\ADD-RSC-homework20260604\2nd\loss_curves.png', 
           dpi=300, bbox_inches='tight')
plt.close()

print(f"\nCharts saved to: 2nd/loss_curves.png")
print(f"Data saved to: 2nd/loss_data.json")
