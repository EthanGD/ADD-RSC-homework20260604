import re
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 從日誌中提取評估數據
log_file = r'C:\Ethan\Project\Project\test\requests_curl\project\CDSS\ADD-RSC-homework20260604\2nd\main.log'

epochs = []
sp_list = []
se_list = []
score_list = []
best_sp_list = []
best_se_list = []
best_score_list = []

with open(log_file, 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    if '* S_p:' in line and 'Best S_p:' in line:
        # 往前找 epoch 號碼
        for j in range(i-1, max(0, i-10), -1):
            if 'Train epoch' in lines[j]:
                match = re.search(r'Train epoch (\d+)', lines[j])
                if match:
                    epoch = int(match.group(1))
                    # 解析測試結果
                    test_match = re.search(r'\* S_p:\s*([\d.]+),\s*S_e:\s*([\d.]+),\s*Score:\s*([\d.]+)\s*\(Best S_p:\s*([\d.]+),\s*S_e:\s*([\d.]+),\s*Score:\s*([\d.]+)\)', line)
                    if test_match:
                        sp = float(test_match.group(1))
                        se = float(test_match.group(2))
                        score = float(test_match.group(3))
                        best_sp = float(test_match.group(4))
                        best_se = float(test_match.group(5))
                        best_score = float(test_match.group(6))
                        
                        epochs.append(epoch)
                        sp_list.append(sp)
                        se_list.append(se)
                        score_list.append(score)
                        best_sp_list.append(best_sp)
                        best_se_list.append(best_se)
                        best_score_list.append(best_score)
                    break

# 保存數據
data = {
    'epochs': epochs,
    'Sp': sp_list,
    'Se': se_list,
    'Score': score_list,
    'Best_Sp': best_sp_list,
    'Best_Se': best_se_list,
    'Best_Score': best_score_list
}

with open(r'C:\Ethan\Project\Project\test\requests_curl\project\CDSS\ADD-RSC-homework20260604\2nd\evaluation_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"Extracted {len(epochs)} evaluation points")
print(f"Final: Sp={sp_list[-1]:.2f}%, Se={se_list[-1]:.2f}%, Score={score_list[-1]:.2f}%")
print(f"Best: Sp={best_sp_list[-1]:.2f}%, Se={best_se_list[-1]:.2f}%, Score={best_score_list[-1]:.2f}%")

# 繪製曲線圖
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Sp 曲線
axes[0].plot(epochs, sp_list, 'b-o', label='Specificity', linewidth=2, markersize=6)
axes[0].plot(epochs, best_sp_list, 'b--', label='Best Specificity', linewidth=1.5, alpha=0.7)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Specificity (%)', fontsize=12)
axes[0].set_title('Specificity (Sp) over Epochs', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(min(epochs)-2, max(epochs)+2)
axes[0].set_ylim(0, 100)

# Se 曲線
axes[1].plot(epochs, se_list, 'r-s', label='Sensitivity', linewidth=2, markersize=6)
axes[1].plot(epochs, best_se_list, 'r--', label='Best Sensitivity', linewidth=1.5, alpha=0.7)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Sensitivity (%)', fontsize=12)
axes[1].set_title('Sensitivity (Se) over Epochs', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(min(epochs)-2, max(epochs)+2)
axes[1].set_ylim(0, 100)

# Score 曲線
axes[2].plot(epochs, score_list, 'g-^', label='Score', linewidth=2, markersize=6)
axes[2].plot(epochs, best_score_list, 'g--', label='Best Score', linewidth=1.5, alpha=0.7)
axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylabel('Score (%)', fontsize=12)
axes[2].set_title('Score over Epochs', fontsize=14)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(min(epochs)-2, max(epochs)+2)
axes[2].set_ylim(0, 100)

# 標記最佳點
best_epoch = epochs[best_score_list.index(max(best_score_list))]
best_score = max(best_score_list)
best_sp = best_sp_list[best_score_list.index(max(best_score_list))]
best_se = best_se_list[best_score_list.index(max(best_score_list))]

axes[2].axvline(x=best_epoch, color='orange', linestyle=':', alpha=0.7, linewidth=2, label=f'Best Epoch ({best_epoch})')
axes[2].annotate(f'Best: {best_score:.2f}%', 
                xy=(best_epoch, best_score), 
                xytext=(best_epoch+5, best_score+8),
                fontsize=11,
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))

# 添加論文基線參考線
axes[2].axhline(y=58.97, color='gray', linestyle='-.', alpha=0.5, linewidth=1.5, label='Paper Baseline (58.97%)')

plt.tight_layout()
plt.savefig(r'C:\Ethan\Project\Project\test\requests_curl\project\CDSS\ADD-RSC-homework20260604\2nd\evaluation_curves.png', 
           dpi=300, bbox_inches='tight')
plt.close()

print(f"\nBest performance at Epoch {best_epoch}:")
print(f"Sp={best_sp:.2f}%, Se={best_se:.2f}%, Score={best_score:.2f}%")
print(f"\nCharts saved to: 2nd/evaluation_curves.png")
print(f"Data saved to: 2nd/evaluation_data.json")
