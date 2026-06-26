import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 讀取數據
with open(r'C:\Ethan\Project\Project\test\requests_curl\project\CDSS\ADD-RSC-homework20260604\2nd\loss_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

epochs = data['epochs']
train_acc = data['Train_Accuracy']
test_acc = data['Test_Accuracy']
test_epochs = data['Test_Epochs']

# 計算差距
gap = []
for i, epoch in enumerate(epochs):
    if epoch in test_epochs:
        idx = test_epochs.index(epoch)
        gap.append(train_acc[i] - test_acc[idx])

# 繪製過擬合分析圖
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Train vs Test Accuracy 對比
axes[0].plot(epochs, train_acc, 'g-o', label='Training Accuracy', linewidth=2, markersize=4)
axes[0].plot(test_epochs, test_acc, 'r-s', label='Test Accuracy', linewidth=2, markersize=4)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].set_title('Training vs Test Accuracy', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 100)

# 標記最大差距點
max_gap_epoch = None
max_gap_value = 0
for i, epoch in enumerate(epochs):
    if epoch in test_epochs:
        idx = test_epochs.index(epoch)
        current_gap = train_acc[i] - test_acc[idx]
        if current_gap > max_gap_value:
            max_gap_value = current_gap
            max_gap_epoch = epoch

if max_gap_epoch:
    idx = epochs.index(max_gap_epoch)
    test_idx = test_epochs.index(max_gap_epoch)
    axes[0].annotate(f'Max Gap: {max_gap_value:.1f}%', 
                    xy=(max_gap_epoch, train_acc[idx]),
                    xytext=(max_gap_epoch+3, train_acc[idx]+10),
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='purple'))

# 2. 差距曲線 (Gap = Train - Test)
axes[1].plot(test_epochs, gap, color='purple', marker='o', linewidth=2, markersize=5, label='Gap (Train - Test)')
axes[1].axhline(y=15, color='orange', linestyle='--', alpha=0.7, label='Warning: 15%')
axes[1].axhline(y=25, color='red', linestyle='--', alpha=0.7, label='Critical: 25%')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Gap (%)', fontsize=12)
axes[1].set_title('Overfitting Gap Analysis', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 40)

# 3. 收斂情況分析
axes[2].plot(epochs, train_acc, 'g-o', label='Training Acc', linewidth=2, markersize=4, alpha=0.6)
axes[2].plot(test_epochs, test_acc, 'r-s', label='Test Acc', linewidth=2, markersize=4, alpha=0.6)

# 填充差距區域
for i in range(len(test_epochs)-1):
    epoch_start = test_epochs[i]
    epoch_end = test_epochs[i+1]
    train_subset = [train_acc[epochs.index(e)] for e in range(epoch_start, epoch_end+1)]
    axes[2].fill_between(range(epoch_start, epoch_end+1), 
                         [test_acc[i]]*len(range(epoch_start, epoch_end+1)),
                         train_subset,
                         alpha=0.3, color='purple')

axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylabel('Accuracy (%)', fontsize=12)
axes[2].set_title('Overfitting Visualization (Purple Area = Gap)', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(0, 100)

plt.tight_layout()
plt.savefig(r'C:\Ethan\Project\Project\test\requests_curl\project\CDSS\ADD-RSC-homework20260604\2nd\overfitting_analysis.png', 
           dpi=300, bbox_inches='tight')
plt.close()

# 輸出分析報告
print("=" * 60)
print("Overfitting Analysis Report")
print("=" * 60)
print(f"\n1. 最終差距 (Epoch {epochs[-1]}):")
print(f"   訓練準確率：{train_acc[-1]:.2f}%")
print(f"   測試準確率：{test_acc[-1]:.2f}%")
print(f"   差距：{train_acc[-1] - test_acc[-1]:.2f}%")

print(f"\n2. 最大差距:")
print(f"   發生在 Epoch {max_gap_epoch}: {max_gap_value:.2f}%")

print(f"\n3. 過擬合判斷:")
if train_acc[-1] - test_acc[-1] < 5:
    print("   ✓ 沒有明顯過擬合 (差距 < 5%)")
elif train_acc[-1] - test_acc[-1] < 15:
    print("   ⚠ 輕微過擬合 (差距 5-15%)")
elif train_acc[-1] - test_acc[-1] < 25:
    print("   ⚠ 明顯過擬合 (差距 15-25%)")
else:
    print("   ✗ 嚴重過擬合 (差距 > 25%)")

print(f"\n4. 收斂情況:")
print(f"   訓練準確率提升：{train_acc[0]:.2f}% → {train_acc[-1]:.2f}% (+{train_acc[-1]-train_acc[0]:.2f}%)")
print(f"   測試準確率提升：{test_acc[0]:.2f}% → {test_acc[-1]:.2f}% (+{test_acc[-1]-test_acc[0]:.2f}%)")

# 檢查 Test Acc 是否在下降
last_5_test = test_acc[-5:]
if len(last_5_test) >= 5:
    if last_5_test[-1] < last_5_test[0]:
        print(f"   ⚠ 測試準確率最近 5 個 epoch 下降：{last_5_test[0]:.2f}% → {last_5_test[-1]:.2f}%")
    else:
        print(f"   ✓ 測試準確率最近 5 個 epoch 上升：{last_5_test[0]:.2f}% → {last_5_test[-1]:.2f}%")

print(f"\n5. 建議:")
if train_acc[-1] - test_acc[-1] > 15:
    print("   - 考慮使用早期停止 (Early Stopping)")
    print("   - 增加正則化 (Dropout, L2 正則化)")
    print("   - 增加數據增強 (Data Augmentation)")
    print("   - 減少模型複雜度")
    print("   - 使用權重衰減 (Weight Decay)")

print(f"\n圖表已保存：2nd/overfitting_analysis.png")
print("=" * 60)
