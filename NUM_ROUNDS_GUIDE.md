# NUM_ROUNDS 参数说明与设置指南

## 1. NUM_ROUNDS 是什么？

`NUM_ROUNDS` = 训练的**迭代轮数**，每轮包含完整的 RL 训练循环：

```
┌─ Round 0 ─────────────────────────────────┐
│ 1️⃣ 生成数据 (CityFlow 模拟 3600秒)         │
│ 2️⃣ 构造训练样本                             │
│ 3️⃣ 训练神经网络 (1个epoch内的多次反向传播) │
│ 4️⃣ 评测 (在同一数据集上测试)                │
│ → 输出: ATT/AWT 指标                        │
└────────────────────────────────────────────┘
     ↓ 经验池积累
┌─ Round 1 ─────────────────────────────────┐
│ 1️⃣ 生成更多数据                            │
│ 2️⃣ 经验池 = [Round0的样本] + [新样本]     │
│ 3️⃣ 从经验池中采样 3000 个样本训练          │
│ 4️⃣ 评测                                    │
└────────────────────────────────────────────┘
     ... (继续积累)
┌─ Round 99 ────────────────────────────────┐
│ 经验池充满(MAX_MEMORY_LEN=12000)           │
│ 模型收敛，取最后10轮平均结果作为最终指标  │
└────────────────────────────────────────────┘
```

---

## 2. NUM_ROUNDS 与其他参数的关系

### 参数对照表

| 参数 | 当前值 | 含义 | 与 NUM_ROUNDS 的关系 |
|------|--------|------|----------------------|
| **NUM_ROUNDS** | 100 | 训练轮数 | 核心参数 |
| **RUN_COUNTS** | 3600 | 每轮模拟时长(秒) | ⊥ 独立；RUN_COUNTS越长数据越丰富 |
| **MAX_MEMORY_LEN** | 12000 | 经验池最大容量 | 每轮新数据会加入；100轮可充满 |
| **SAMPLE_SIZE** | 3000 | 每轮训练样本数 | 从经验池中采样；NUM_ROUNDS越多样本质量越好 |
| **BATCH_SIZE** | 20 | 每个mini-batch大小 | ⊥ 独立，与NUM_ROUNDS无关 |
| **EPOCHS** | 100 | 每轮训练内的epoch数 | ⊥ 独立，与NUM_ROUNDS无关 |
| **LEARNING_RATE** | 1e-3 | 学习率 | ⊥ 独立；NUM_ROUNDS长收敛更好 |
| **D_DENSE** | 20 | 隐藏层大小 | ⊥ 独立，固定网络结构 |

### 关键：数据积累与遗忘机制

```python
# 伪代码 - 每轮的样本管理逻辑
for round in range(NUM_ROUNDS):
    # 1. 生成新数据
    new_samples = generator.generate()
    experience_pool.append(new_samples)
    
    # 2. 遗忘旧样本（防止内存溢出）
    if len(experience_pool) > MAX_MEMORY_LEN:
        # 只保留最近的 MAX_MEMORY_LEN 个样本
        experience_pool = experience_pool[-MAX_MEMORY_LEN:]
    
    # 3. 采样训练
    batch = random.sample(experience_pool, SAMPLE_SIZE)
    model.train(batch, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # 4. 评测
    metrics = model.evaluate()
    print(f"Round {round}: ATT={metrics['ATT']}, AWT={metrics['AWT']}")

# 最后，取最后10轮的平均结果
final_metrics = average(results[-10:])
```

---

## 3. 设置建议

### 📋 不同场景的 NUM_ROUNDS 设置

#### **场景 1：快速测试/Debug** ⚡ **5-10 分钟**
```bash
NUM_ROUNDS = 5
```
- ✅ 快速验证代码能否运行
- ✅ 检查数据格式/模型结构
- ❌ 结果不可靠（模型未充分训练）
- **使用场景**：首次跑通代码、问题排查

#### **场景 2：中等实验** ⏱️ **30-60 分钟/方法**
```bash
NUM_ROUNDS = 20
```
- ✅ 快速得到相对比较
- ✅ 适合初步对比多个方法
- ⚠️ 结果稳定性一般
- **使用场景**：快速原型/方法对比验证

#### **场景 3：标准基线实验** ✅ **推荐用于论文** **1-2 小时/方法**
```bash
NUM_ROUNDS = 100
```
- ✅ 模型收敛充分（～50轮已接近收敛）
- ✅ 最后10轮平均结果稳定可靠
- ✅ 符合主流文献标准
- ✅ 6个方法并行运行≈3-4小时
- **使用场景**：论文实验、最终上报

#### **场景 4：严谨的长期实验** 🔬 **4-8 小时/方法**
```bash
NUM_ROUNDS = 200-300
```
- ✅ 极端严谨
- ✅ 确保收敛
- ❌ 边际收益低（50-100轮后改进≈1%）
- ❌ 耗时巨大
- **使用场景**：顶会投稿最后冲刺

---

## 4. 如何设置 NUM_ROUNDS？

### 方法 A：修改单个脚本 (不推荐，重复多次)

编辑 [/home/wen/tr/LLMTSCS/run_mplight.py](run_mplight.py) 等：

```python
# 第 33 行左右
if in_args.dataset == 'jinan':
    num_rounds = 100  # ← 改这里
elif in_args.dataset == 'hangzhou':
    num_rounds = 100  # ← 改这里
```

**缺点**：要改 6 个文件，容易不一致。

### 方法 B：使用批量脚本设置 (推荐)

运行脚本时，一键设置所有方法的 NUM_ROUNDS：

```bash
# 生成一个临时修改脚本
cat > /tmp/set_rounds.py << 'EOF'
import sys
import re

target_rounds = int(sys.argv[1])
scripts = [
    "run_mplight.py",
    "run_attendlight.py", 
    "run_presslight.py",
    "run_colight.py",
    "run_efficient_colight.py",
    "run_advanced_colight.py"
]

for script in scripts:
    with open(script, 'r') as f:
        content = f.read()
    # 替换 num_rounds = 100
    content = re.sub(r'num_rounds = \d+', f'num_rounds = {target_rounds}', content)
    with open(script, 'w') as f:
        f.write(content)
    print(f"✓ Updated {script} to NUM_ROUNDS={target_rounds}")
EOF

# 用法
python /tmp/set_rounds.py 100  # 设置为 100 轮
```

### 方法 C：在启动脚本中自动设置 (最推荐)

修改 `run_RL_Baseline_train.sh`：

```bash
#!/bin/bash

# 允许通过命令行参数设置 NUM_ROUNDS
NUM_ROUNDS=${1:-100}  # 默认100，可传参

echo "[INFO] Setting NUM_ROUNDS=${NUM_ROUNDS}"

# 临时修改所有脚本
python << PYTHON
import re
target = $NUM_ROUNDS
for script in ["run_mplight.py", "run_attendlight.py", "run_presslight.py",
               "run_colight.py", "run_efficient_colight.py", "run_advanced_colight.py"]:
    with open(script, 'r') as f:
        content = f.read()
    content = re.sub(r'num_rounds = \d+', f'num_rounds = {target}', content)
    with open(script, 'w') as f:
        f.write(content)
PYTHON

# 然后运行训练...
```

**使用**：
```bash
bash run_RL_Baseline_train.sh 100   # 100轮
bash run_RL_Baseline_train.sh 50    # 50轮
bash run_RL_Baseline_train.sh 200   # 200轮
```

---

## 5. 论文中通常怎么设置？

根据扫过的相关论文（TSC领域）：

| 论文类型 | NUM_ROUNDS | 说明 |
|---------|-----------|------|
| **DRL-based TSC** (如CoLight/MPLight) | 100 | 业界标准 |
| **Efficient-*系列** | 100 | 保持一致 |
| **Advanced-* 系列** | 100 | 保持一致 |
| **Transfer Learning** | 100(source) + 1(eval) | 训练用100轮，评测1轮 |

**结论**：100轮已成为**事实标准**，除非论文明确说明其他值。

---

## 6. 快速总结

| 要做什么 | NUM_ROUNDS | 脚本改法 |
|---------|-----------|---------|
| 第一次跑通代码 | 5 | 改成5 |
| 快速对比方法 | 20 | 改成20 |
| **写论文提交** | **100** | 保持默认 ✅ |
| 最终冲刺版本 | 200+ | 改大 |

---

## 7. 常见问题

**Q: NUM_ROUNDS=100 要跑多久？**  
A: 单个方法约 **1.5-2 小时**；6个方法并行约 **3-4 小时**（取决于CPU/IO）

**Q: 前50轮和后50轮的结果差异大吗？**  
A: 前50轮下降快（收敛），后50轮平缓。这就是为什么取"最后10轮平均"。

**Q: 能改成NUM_ROUNDS=300吗？**  
A: 可以，但通常在50-100轮已基本收敛，后面改进≤1%，不推荐。

**Q: 不同方法的NUM_ROUNDS要不要一样？**  
A: **一定要一样**！这样才能公平对比。

**Q: NUM_ROUNDS与SAMPLE_SIZE/BATCH_SIZE/EPOCHS有什么关系？**  
A: 无直接关系。SAMPLE_SIZE是从经验池采样数（NUM_ROUNDS越多体验越丰富），BATCH_SIZE/EPOCHS是单轮训练的细节，LEARNING_RATE控制收敛速度。

---

## 总结

- **NUM_ROUNDS = 训练周期数**
- **默认 100 轮是行业标准**
- **后面 10 轮取平均作为最终指标**
- **不同方法必须设成一样**
- **快速测试用 5-20，正式实验用 100**

现在你可以开始设置实验了！🚀
