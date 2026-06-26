✅ **已完成修改！**

## 使用方式

### 1. 使用默认值（预测 `data/ICBHI/ICBHI_final_database/101_1b1_Al_sc_Meditron.wav`）
```bash
python predict_single.py
```

### 2. 指定音频文件
```bash
python predict_single.py --audio "data/ICBHI/ICBHI_final_database/your_audio.wav"
```

### 3. 指定模型 checkpoint
```bash
python predict_single.py --checkpoint "other_folder/best.pth"
```

### 4. 同时指定音频和模型
```bash
python predict_single.py --audio "data/ICBHI/ICBHI_final_database/test.wav" --checkpoint "icbhi_resnet50_train_resnet/best.pth"
```

---

## 路径说明
- **音频路径**: 相对项目根目录 (`ADD-RSC-homework20260604/`)
- **Checkpoint 路径**: 相对脚本所在目录 (`2nd/`)

## 服务器运行
```bash
# 在服务器上设置 GPU
export CUDA_VISIBLE_DEVICES=6

# 运行预测
cd 2nd
python predict_single.py --audio "data/ICBHI/ICBHI_final_database/101_1b1_Al_sc_Meditron.wav"
```