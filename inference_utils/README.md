# 统一模型加载器

## ✅ 已实现功能

### 支持的模型
- **DP3**: 3D Diffusion Policy
- **GHOST**: 所有变体（baseline/keyframe/beacon/beacon_key）

### 核心特性
1. **统一接口**：一行代码加载任意模型
2. **自包含配置**：所有配置从 checkpoint 读取，无需外部 yaml
3. **Normalizer 集成**：自动从模型 state_dict 恢复
4. **EMA 优先**：优先加载 EMA 权重以获得更好性能
5. **自动设备管理**：自动检测并使用 GPU/CPU

## 📖 使用方法

### 基础用法

```python
from inference_utils.model_loader import load_policy_model
import torch

# 1. 加载模型
policy = load_policy_model(
    policy_name='DP3',           # 'DP3' 或 'GHOST'
    task_name='pick_place_d405', # 任务名称
    ckpt_name='750.ckpt'         # checkpoint 文件名
)

# 2. 准备输入
obs_dict = {
    'point_cloud': torch.randn(1, 2, 512, 6),  # (B, To, N, 6) - xyz+rgb
    'agent_pos': torch.randn(1, 2, 14)         # (B, To, D)
}

# 3. 推理
actions = policy.predict_action(obs_dict)
# 返回: (B, horizon, action_dim) numpy.ndarray
```

### 命令行测试

```bash
# 测试 DP3
python inference_utils/model_loader.py DP3 pick_place_d405 750.ckpt

# 测试 GHOST
python inference_utils/model_loader.py GHOST pick_place_d405 latest.ckpt
```

## 📊 模型对比

| 模型 | 参数量 | 观察步数 | 动作步数 | agent_pos 维度 | 特殊特性 |
|------|--------|----------|----------|----------------|----------|
| **DP3** | 262.6M | 3 | 6 | 14 | 纯点云编码 |
| **GHOST** | 80.3M | 2 | 8 | 32 (VGC格式) | Trident辅助点云 |

## 🔍 关于配置文件的说明

### ❓ 为什么不需要加载 yaml？

**答：checkpoint 中已包含完整配置！**

训练时保存结构：
```python
checkpoint = {
    'cfg': cfg,                    # ← 完整的 OmegaConf 配置
    'state_dicts': {
        'model': model.state_dict(),      # 包含 normalizer 参数
        'ema_model': ema_model.state_dict(),  # EMA 模型
        'optimizer': optimizer.state_dict()
    },
    'epoch': epoch,
    'global_step': global_step
}
```

加载时直接使用：
```python
checkpoint = torch.load(ckpt_path, weights_only=False)
config = checkpoint['cfg']  # 直接读取，无需 yaml
```

### ✅ 优势
1. **版本一致性**：配置与权重完全匹配
2. **无依赖**：不需要配置文件目录
3. **简化部署**：只需要一个 .ckpt 文件

### ⚠️ 注意事项
- `deploy_policy.py` 中重新加载 yaml 是为了灵活调试
- 纯推理时使用 checkpoint 内配置更可靠
- 确保使用 `weights_only=False`（因为包含 OmegaConf 对象）

## 🎯 输入格式说明

### DP3
```python
obs_dict = {
    'point_cloud': (B, To, N, 6),  # To=3, N=512~1024
    'agent_pos': (B, To, 14)       # 14个关节角度
}
# 返回: (B, 6, 14) - horizon=6, action_dim=14
```

### GHOST
```python
obs_dict = {
    'point_cloud': (B, To, N, 6),  # To=2, N=512~1024
    'agent_pos': (B, To, 32)       # VGC格式: 14关节+9左手+9右手
}
# 返回: (B, 16, 14) - horizon=16, action_dim=14
```

## 📁 文件结构

```
weights/
├── pick_place_d405/
│   ├── DP3/
│   │   ├── 750.ckpt          # DP3 checkpoint
│   │   └── latest.ckpt
│   └── GHOST/
│       └── latest.ckpt       # GHOST checkpoint (包含 EMA)
```

## 🔧 故障排除

### 1. ConfigKeyError: Key 'xxx' is not in struct
**原因**：OmegaConf struct mode 限制  
**解决**：使用 `OmegaConf.to_container()` 转换为普通字典

### 2. ModuleNotFoundError: No module named 'ghost_policy'
**原因**：路径未添加  
**解决**：检查 `sys.path` 是否包含 `policy/GHOST`

### 3. CUDA out of memory
**解决**：
```python
device = torch.device('cpu')  # 强制使用 CPU
model = model.to(device)
```

## 📝 代码示例

### 完整推理示例
```python
from inference_utils.model_loader import load_policy_model
import torch

# 加载模型
policy = load_policy_model('GHOST', 'pick_place_d405', 'latest.ckpt')

# 准备观察数据
obs_dict = {
    'point_cloud': torch.randn(1, 2, 512, 6),
    'agent_pos': torch.randn(1, 2, 32)
}

# 推理
with torch.no_grad():
    actions = policy.predict_action(obs_dict)
    
print(f"Actions shape: {actions.shape}")
print(f"First action: {actions[0, 0]}")
```

### 批量推理
```python
# 批量处理
batch_size = 4
obs_dict = {
    'point_cloud': torch.randn(batch_size, 2, 512, 6),
    'agent_pos': torch.randn(batch_size, 2, 32)
}

actions = policy.predict_action(obs_dict)
# actions.shape: (4, 16, 14)
```
