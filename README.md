# EET - LSTM+PPO Elevator Scheduling System

基于 LSTM+PPO 强化学习的电梯群控调度系统，用于西门子杯 CIMC 离散行业自动化赛题。支持 3 部 10 层高速电梯的智能调度，并可集成西门子 S7-1200 PLC 实现实时控制。

## 特性

- **LSTM+PPO 强化学习**：使用 LSTM 编码状态序列，PPO 算法优化调度策略
- **7 种交通场景**：早高峰、晚高峰、午间交叉、层间交互、极端闲时、会议集散
- **事件驱动仿真**：高效仿真引擎，支持从历史数据回放乘客流
- **效能评估指标**：平均候梯时间、乘梯时间、长候梯率、空载率、运行效率、能耗
- **PLC 集成**：通过 python-snap7 与西门子 S7-1200 PLC 通信（支持 Profibus）
- **7 段数码管解码**：直接从 PLC LED 位读取电梯楼层显示

## 项目结构

```
eet/
├── config/
│   └── config.yaml          # 超参数配置（数据、环境、模型、PPO、训练、奖励、PLC）
├── datasets/                 # NPZ 格式训练数据（由 eet_parser.py 生成）
├── src/
│   ├── utils.py             # 共享工具（PROJ_ROOT, load_config）
│   ├── data/
│   │   └── dataset.py      # 数据加载、分层划分 train/val/test (70/15/15)
│   ├── env/
│   │   ├── elevator_env.py # Gymnasium 电梯群控仿真环境
│   │   └── metrics.py      # 效能指标计算
│   ├── models/
│   │   ├── policy.py       # LSTMActorCritic 网络（LSTM + Actor/Critic）
│   │   └── lstm_ppo.py    # PPOTrainer（GAE、PPO 更新、RolloutBuffer）
│   ├── inference.py        # 实时推理模块（加载模型、查询调度决策）
│   ├── train.py            # 训练主循环
│   ├── evaluate.py         # 测试集评估（按场景分组指标 + 图表）
│   └── plc/
│       ├── snap7_client.py # PLC 通信客户端（读/写 DB/IQ 区域）
│       └── bridge.py       # PLC↔推理桥接（主循环：读状态→推理→写指令）
├── checkpoints/             # 模型检查点 + 评估图表
├── eet_parser.py           # .eet 二进制文件解析器
├── eet_dataset.py          # 从 .eet 提取训练数据集
└── README.md
```

## 安装

### 前置条件

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) 包管理器
- （可选）西门子 S7-1200 PLC 或 PLC 模拟器

### 安装步骤

```bash
# 克隆项目
cd eet/

# 使用 uv 创建虚拟环境并安装依赖
uv sync

# 或手动安装依赖
uv add torch numpy pyyaml gymnasium python-snap7 matplotlib scikit-learn
```

### 依赖项

```
torch>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
gymnasium>=0.29.0
python-snap7>=3.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

## 使用方法

### 1. 准备数据集

将 .eet 文件放入 `datasets/` 目录，然后运行：

```bash
uv run python eet_dataset.py
```

这将生成 NPZ 格式的训练数据：`global_features.npz`, `event_sequences.npz`, `labels.npz`, `file_ids.npz`, `event_lengths.npz`

### 2. 训练模型

```bash
uv run python -m src.train
```

训练配置在 `config/config.yaml` 中：

```yaml
training:
  total_epochs: 3        # 训练轮数
  eval_every: 2           # 每 N 轮验证一次
  early_stop_patience: 5  # 早停耐心值
```

训练过程中会：
- 保存最佳模型到 `checkpoints/best_model.pt`
- 生成训练曲线图 `checkpoints/plots/training_curve.png`

### 3. 评估模型

```bash
uv run python -m src.evaluate --checkpoint checkpoints/best_model.pt
```

评估结果包含：
- 按场景分组的效能指标表格
- 平均候梯时间、乘梯时间、长候梯率、空载率、运行效率、能耗
- 生成柱状图到 `checkpoints/plots/`

**示例输出：**
```
======================================================================
Evaluation Results — 12 test files
======================================================================
Avg reward: -2516.7436

Scenario           Files  AvgWait  AvgRide LongWait%   Empty%   OpEff%  EnergyWh
----------------------------------------------------------------------
Morning Peak           1     0.00     0.00      0.0%   100.0%    10.4%     9.84
Evening Peak           1   315.25     8.00    100.0%    26.4%     2.7%   218.29
...
----------------------------------------------------------------------
OVERALL               12   149.27     7.75         —    45.1%    12.7%         —
```

### 4. 实时推理

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

### 5. PLC 集成

#### 配置 PLC 连接

编辑 `config/config.yaml`：

```yaml
plc:
  ip: "192.168.0.1"      # PLC IP 地址
  rack: 0
  slot: 1
  timeout_ms: 1000
  poll_interval_ms: 100
  db_number: 1
```

#### 运行 PLC 桥接

```bash
uv run python -m src.plc.bridge
```

桥接程序将：
1. 连接 S7-1200 PLC
2. 循环读取电梯状态（楼层、方向、载重、门状态）
3. 读取楼层呼梯信号
4. 运行 LSTM+PPO 推理得到调度决策
5. 将决策写入 PLC（分配电梯、目标楼层、方向）

## 模型架构

```
输入: state [batch, seq_len, 73]
  → LSTM (hidden=64, num_layers=1)
  → 取最后时刻 hidden state [batch, 64]
  → Actor head: Linear(64→32) → ReLU → Linear(32→3) → Softmax
  → Critic head: Linear(64→32) → ReLU → Linear(32→1)
```

**状态空间 (73 维)：**
- 每部电梯 (3 部) × [楼层 one-hot(10) + 方向(3) + 载重比(1) + 运行中(1) + 门状态(1) + 轿厢呼叫(1)] = 3×17 = 51
- 每层呼梯 (10 层) × [上行(1) + 下行(1)] = 20
- 全局: [时间归一化(1) + 活跃呼叫数(1)] = 2

**动作空间：** 离散动作 — 当有新呼梯时，分配电梯 ID (0/1/2)

**奖励函数：**
- `+2.0` 每位乘客送达
- `-0.05 × wait_time` 候梯时间惩罚（秒）
- `-0.1 × empty_floors` 空载运行距离惩罚（层）
- `-0.05 × start_stop` 启停能耗惩罚

## 配置说明

所有超参数集中在 `config/config.yaml`：

| 部分 | 关键参数 | 说明 |
|------|----------|------|
| `data` | train_ratio, val_ratio, random_seed | 数据划分比例和随机种子 |
| `env` | num_elevators, num_floors, floor_travel_time | 仿真环境参数 |
| `model` | lstm_hidden, lstm_layers, actor_hidden, critic_hidden | 网络结构 |
| `ppo` | learning_rate, gamma, clip_epsilon, ppo_epochs | PPO 算法参数 |
| `training` | total_epochs, eval_every, early_stop_patience | 训练控制 |
| `reward` | passenger_delivered, wait_time_per_sec | 奖励函数权重 |
| `plc` | ip, rack, slot, poll_interval_ms | PLC 连接参数 |

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

### 时间归一化

不同 .eet 文件格式（Group A/B/C）产生不同范围的时间戳。训练时自动将每个文件的时间戳归一化到 [0, 3600] 秒范围。

### LSTM 序列处理

- 使用滑动窗口（seq_len=32）构建状态序列
- 推理时维护最近 32 个状态的 deque 作为 LSTM 输入
- 隐藏状态在单个 episode 内传递，episode 结束时重置

## 已知限制

- PLC 通信需要西门子 S7-1200 硬件或模拟器
- 当前模型训练轮数较少（默认 3 轮），建议增加至 50-200 轮以获得更好效果
- 7 段数码管解码假设 LED 位布局固定，可能需要根据实际 PLC 程序调整

## 相关资源

- [西门子杯 CIMC 离散行业自动化赛题](https://www.siemenscup-cimc.org.cn/)
- [PPO 算法论文](https://arxiv.org/abs/1707.06347)
- [python-snap7 文档](https://python-snap7.readthedocs.io/)
- [Gymnasium 文档](https://gymnasium.farama.org/)

## 许可证

本项目为西门子杯参赛项目，仅供学习和竞赛使用。
