## Specificity（特异性，Sp）是什么？

**特异性（Specificity）**衡量模型把“负类/正常”识别正确的能力，也可以理解为“正常样本不被误报为异常”的能力。

- 二分类定义：  
  \[
  \text{Specificity}=\frac{TN}{TN+FP}
  \]
  - TN（True Negative）：真实为负类、预测也为负类  
  - FP（False Positive）：真实为负类、但预测成正类（误报）

**在呼吸音分类（ICBHI lungsound）里**，通常把：
- **Normal（正常）当作负类**  
- **Abnormal（异常：crackle/wheeze/both）当作正类（合并）**

因此 Sp 高，代表模型很少把正常误判为异常；Sp 低，代表误报多。

---

## Sensitivity（敏感性，Se）是什么？

**敏感性（Sensitivity）**衡量模型把“正类/异常”识别出来的能力，也可称为“召回率（Recall）/真阳性率（TPR）”。

- 二分类定义：  
  \[
  \text{Sensitivity}=\frac{TP}{TP+FN}
  \]
  - TP（True Positive）：真实为正类、预测也为正类  
  - FN（False Negative）：真实为正类、但预测成负类（漏检）

在医疗场景里，Se 往往更关键：Se 低意味着异常被大量漏检。

---

## 本项目里 Sp/Se 是怎么计算的？

本仓库的实现见 [icbhi_util.py](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/util/icbhi_util.py#L186-L197)：

- \[
  Sp = \frac{\text{hits}[0]}{\text{counts}[0]}
  \]
  其中 label=0 表示 normal（正常）
- \[
  Se = \frac{\sum_{c=1}^{C-1}\text{hits}[c]}{\sum_{c=1}^{C-1}\text{counts}[c]}
  \]
  其中 label>0（crackle/wheeze/both）都被当作 abnormal（异常）合并计算
- 综合分数：
  \[
  Score=\frac{Sp+Se}{2}
  \]

也就是说：即使任务是 4 类分类，最终的 Sp/Se/Score 评价是把 **normal vs abnormal** 按二分类方式汇总计算的。

---

## 如何解读训练结果（常见现象）

- **Sp 很高但 Se 很低**：模型倾向“全判 normal”，正常识别看起来很好，但异常几乎都漏掉。  
  在不平衡数据集上很常见（normal 占比更高时尤其明显）。
- **Se 很高但 Sp 很低**：模型倾向“全判 abnormal”，异常抓得多，但误报严重。
- **Score** 是 Sp 和 Se 的平均：任何一项很低都会把 Score 拉下来。

