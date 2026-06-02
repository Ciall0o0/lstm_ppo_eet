# EET - LSTM+PPO Elevator Scheduling System

基于 LSTM+PPO 强化学习的电梯群控调度系统，用于西门子杯 CIMC 离散行业自动化赛题。支持 3 部 10 层高速电梯的智能调度，并可集成西门子 S7-1200 PLC 实现实时控制。

## 特性

- **LSTM+PPO 强化学习**：双层 LSTM 编码状态序列，双优化器 PPO 算法优化调度策略
- **7 种交通场景**：早高峰、晚高峰、午间交叉、层间交互、极端闲时、会议集散
- **事件驱动仿真**：高效仿真引擎，支持从历史数据回放乘客流
- **多环境并行训练**：MultiEnvRunner 支持 16 个并行环境，批量 GPU 推理
- **Optuna 超参搜索**：自动搜索最优 PPO/模型/奖励参数，支持早停和剪枝
- **效能评估指标**：平均候梯时间、乘梯时间、长候梯率、空载率、运行效率、能耗
- **PLC 集成**：通过 python-snap7 与西门子 S7-1200 PLC 通信（支持 Profibus）
- **7 段数码管解码**：直接从 PLC LED 位读取电梯楼层显示

## 项目结构

```
eet/
├── config/
│   └── config.yaml            # 超参数配置（数据、环境、模型、PPO、训练、奖励、PLC）
├── datasets/                   # NPZ 格式训练数据（由 eet_dataset.py 生成）
├── src/
│   ├── utils.py               # 共享工具（PROJ_ROOT, load_config, load_optuna_params）
│   ├── data/
│   │   └── dataset.py         # 数据加载、分层划分 train/val/test (70/15/15)
│   ├── env/
│   │   ├── elevator_env.py    # Gymnasium 电梯群控仿真环境
│   │   └── metrics.py         # 效能指标计算
│   ├── models/
│   │   ├── policy.py          # LSTMActorCritic 网络（LSTM + Actor/Critic）
│   │   └── lstm_ppo.py        # PPOTrainer（GAE、PPO 更新、RolloutBuffer）
│   ├── runner.py              # MultiEnvRunner 多环境并行批量推理
│   ├── train.py               # 训练主循环
│   ├── evaluate.py            # 测试集评估（按场景分组指标 + 图表）
│   ├── inference.py           # 实时推理模块（加载模型、查询调度决策）
│   ├── optuna_search.py       # Optuna 超参数搜索
│   └── plc/
│       ├── snap7_client.py    # PLC 通信客户端（读/写 DB/IQ 区域）
│       └── bridge.py          # PLC↔推理桥接（主循环：读状态→推理→写指令）
├── checkpoints/                # 模型检查点 + Optuna 结果 + 评估图表
├── main.py                    # 管道编排：Optuna → Train → Evaluate
├── apply_best_params.py       # 将 Optuna 最佳参数写入 config.yaml
├── monitor.py                 # 训练进度监控与异常检测
├── fill_report.py             # 学术报告自动生成（python-docx）
├── eet_parser.py              # .eet 二进制文件解析器
└── eet_dataset.py             # 从 .eet 提取训练数据集
```

## 安装

### 前置条件

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) 包管理器
- （可选）CUDA 兼容 GPU（推荐，训练速度提升显著）
- （可选）西门子 S7-1200 PLC 或 PLC 模拟器

### 安装步骤

```bash
cd eet/

# 使用 uv 同步依赖
uv sync
```

### 依赖项

```
torch>=2.12.0
numpy>=2.4.6
pyyaml>=6.0.3
gymnasium>=1.3.0
python-snap7>=3.0.0
matplotlib>=3.10.9
scikit-learn>=1.8.0
optuna>=4.8.0
tqdm>=4.67.0
swanlab>=0.5.0
```

## 使用方法

### 1. 准备数据集

将 .eet 文件放入 `datasets/` 目录（已按交通场景分文件夹存放），然后运行：

```bash
uv run python eet_dataset.py
```

这将生成 NPZ 格式的训练数据到 `datasets/` 目录。

### 2. 一键运行完整流程

```bash
# 默认：训练 + 评估（自动加载 Optuna 最佳参数，如已存在）
uv run python main.py

# 完整流程：超参搜索 → 训练 → 评估
uv run python main.py --optuna --optuna-trials 30 --train --eval

# 仅训练，指定轮数
uv run python main.py --train --epochs 200

# 仅评估，指定检查点
uv run python main.py --eval --checkpoint checkpoints/best_model.pt
```

### 3. 超参数搜索（Optuna）

```bash
uv run python -m src.optuna_search --n-trials 20
```

搜索空间覆盖 PPO 参数、模型架构和奖励塑形权重，支持：
- MedianPruner 自动剪枝无前景 trial
- 每个 trial 独立早停（验证集奖励 10 轮无改善）
- 结果保存到 `checkpoints/optuna_best_params.json`

将最佳参数应用到配置：

```bash
uv run python apply_best_params.py
```

### 4. 训练模型

```bash
uv run python -m src.train
```

训练配置在 `config/config.yaml` 中：

```yaml
training:
  total_epochs: 200       # 训练轮数
  eval_every: 5           # 每 N 轮验证一次
  early_stop_patience: 20 # 早停耐心值
  num_envs: 16            # 并行环境数
```

训练过程中会：
- 保存最佳模型到 `checkpoints/best_model.pt`
- 每 50 轮保存检查点到 `checkpoints/checkpoint_epoch{N}.pt`
- 生成训练曲线图 `checkpoints/plots/training_curve.png`

### 5. 监控训练

```bash
# 单次查看训练状态
uv run python monitor.py

# 实时监控（每 30 秒刷新）
uv run python monitor.py watch
```

监控程序自动检测：
- 熵值停滞在 log(3) 超过 15 轮
- Value loss 反复飙升 >200
- NaN/Inf 出现
- Advantage 均值持续异常

### 6. 评估模型

```bash
uv run python -m src.evaluate --checkpoint checkpoints/best_model.pt
```

评估结果包含：
- 按场景分组的效能指标表格
- 平均候梯时间、乘梯时间、长候梯率、空载率、运行效率、能耗
- 生成柱状图到 `checkpoints/plots/`

### 7. 实时推理

```python
from src.inference import ElevatorScheduler

scheduler = ElevatorScheduler()
scheduler.reset()

state = {
    "elevators": [
        {"floor": 1, "direction": 0, "load_ratio": 0.0, "is_moving": False, "door_open": False},
        {"floor": 5, "direction": 0, "load_ratio": 0.3, "is_moving": False, "door_open": False},
        {"floor": 8, "direction": -1, "load_ratio": 0.2, "is_moving": True, "door_open": False},
    ],
    "floor_up_calls": [True] + [False] * 9,
    "floor_down_calls": [False] * 10,
}

elevator_id = scheduler.query(state)
print(f"推荐电梯: {elevator_id}")
```

### 8. PLC 集成

编辑 `config/config.yaml` 配置 PLC 连接：

```yaml
plc:
  ip: "192.168.0.1"      # PLC IP 地址
  rack: 0
  slot: 1
  timeout_ms: 1000
  poll_interval_ms: 100
  db_input: 10            # 输入 DB 号
  db_output: 11           # 输出 DB 号
```

运行 PLC 桥接：

```bash
uv run python -m src.plc.bridge
```

桥接程序将循环读取电梯状态 → 运行 LSTM+PPO 推理 → 将调度决策写入 PLC。

## 模型架构

```
输入: state [batch, seq_len, 109]
  → LSTM (hidden=256, num_layers=2, dropout=0.1)
  → 取最后时刻 hidden state [batch, 256]
  → LayerNorm
  → Actor head: Linear(256→64) → ReLU → Linear(64→3) → Softmax
  → Critic head: Linear(256→64) → ReLU → Linear(64→1)
```

**状态空间 (109 维)：**
- 每部电梯 (3 部) × 状态编码 = 3×33 = 99
- 每层呼梯 (10 层) × [上行(1) + 下行(1)] = 20

**动作空间：** 离散动作 — 当有新呼梯时，分配电梯 ID (0/1/2)

**奖励函数（含分配塑形）：**
- `+3.0` 每位乘客送达
- `-0.001 × wait_time` 候梯时间惩罚（秒）
- `-0.005 × empty_floors` 空载运行距离惩罚（层）
- `-0.002 × start_stop` 启停能耗惩罚
- `-0.005 × idle_time` 闲时惩罚（秒）
- 分配时塑形：距离、方向对齐、负载均衡、预估等待时间

## 配置说明

所有超参数集中在 `config/config.yaml`：

| 部分 | 关键参数 | 说明 |
|------|----------|------|
| `data` | train_ratio, val_ratio, random_seed | 数据划分比例和随机种子 |
| `env` | num_elevators, num_floors, floor_travel_time, idle_timeout | 仿真环境参数 |
| `model` | lstm_hidden, lstm_layers, actor_hidden, critic_hidden, use_layer_norm | 网络结构 |
| `ppo` | learning_rate, gamma, clip_epsilon, ppo_epochs, kl_target, burn_in_steps | PPO 算法参数 |
| `training` | total_epochs, eval_every, early_stop_patience, num_envs | 训练控制 |
| `reward` | passenger_delivered, assignment_*, normalize, clip_range | 奖励函数权重 |
| `plc` | ip, rack, slot, poll_interval_ms, db_input, db_output | PLC 连接参数 |

## 效能指标

评估使用以下指标：

- **平均候梯时间 (AvgWait)**：乘客从按呼梯按钮到进入电梯的平均时间
- **平均乘梯时间 (AvgRide)**：乘客从进入电梯到到达目标楼层的平均时间
- **长候梯率 (LongWait%)**：候梯时间超过 60 秒的乘客比例
- **空载率 (Empty%)**：空载运行距离占总运行距离的比例
- **运行效率 (OpEff%)**：有效载客时间 / 总运行时间
- **能耗 (EnergyWh)**：启停次数 × 单次能耗

## 技术细节

### 事件驱动仿真

环境使用事件驱动仿真而非固定时间步长，大幅提升效率：
- 当有待处理呼梯时：时间不推进（dt=0）
- 当无呼梯时：时间跳到下一个事件或电梯到达时间

### 多环境并行

MultiEnvRunner 管理 N 个并行 ElevatorEnv 实例，将观测批量堆叠后送入 GPU 进行一次前向推理，避免逐环境串行调用。支持 pinned memory 加速 CPU→GPU 数据传输。

### LSTM 序列处理

- 使用滑动窗口（seq_len=32）构建状态序列
- 支持 burn-in：前 N 步只更新 LSTM 隐藏状态，不计算损失
- 推理时维护最近状态的 deque 作为 LSTM 输入
- 隐藏状态在单个 episode 内传递，episode 结束时重置

### KL 散度早停

PPO 更新时监控 KL 散度，当超过 kl_target 阈值时提前终止当前 epoch 的更新轮次，防止策略更新过大致使性能崩溃。

## 已知限制

- PLC 通信需要西门子 S7-1200 硬件或模拟器 S7 PLCSIM
- 7 段数码管解码假设 LED 位布局固定，可能需要根据实际 PLC 程序调整

## 相关资源

- [西门子杯 CIMC 离散行业自动化赛题](https://www.siemenscup-cimc.org.cn/)
- [PPO 算法论文](https://arxiv.org/abs/1707.06347)
- [python-snap7 文档](https://python-snap7.readthedocs.io/)
- [Gymnasium 文档](https://gymnasium.farama.org/)

## 许可证

本项目为西门子杯参赛项目，仅供学习和竞赛使用。
