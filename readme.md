# PLED 图像增强模型

本项目包含用于训练和测试图像增强模型的代码。

## 目录结构

- `train.py`：模型训练脚本
- `test.py`：模型测试脚本
- `sum_para.py`：统计模型参数和FLOPs
- `model/`：模型结构与损失函数定义
- `DataProcess/`：数据加载与处理
- `utils/`：工具函数
- `checkpoints/`：模型权重保存目录
- `log/`：训练日志

## 环境依赖

- Python 3.8+
- torch2.4.1 
- torchvision

## 训练模型

1. **准备数据集**  
   修改 [train.py](train.py) 中的 `train_dir` 和 `val_dir` 路径，指向你的训练和验证集。

2. **开始训练**  
   运行以下命令：

   ```sh
   python train.py --epochs 50000 --batch-size 50 --learning-rate 1e-4
   ```

   你可以根据需要调整参数，具体参数说明见 [train.py](train.py) 的 [`get_args`](d:/Common/Decetion/fuwu/PLED/train.py) 函数。

3. **训练日志与模型保存**  
   - 日志保存在 `log/` 目录
   - 模型权重保存在 `checkpoints/` 目录

## 测试模型

1. **准备测试集**  
   修改 [test.py](test.py) 中的数据路径，确保指向你的测试集。

2. **加载模型权重**  
   默认会从 `checkpoints/model_bestPSNR.pth` 或 `model_bestSSIM.pth` 加载最优模型。

3. **运行测试**  
   ```sh
   python test.py
   ```

   测试结果会输出到控制台或指定目录。

## 统计模型参数和FLOPs

运行：

```sh
python sum_para.py
```

会输出模型的参数量和FLOPs信息。

---

如需详细配置或自定义训练/测试流程，请参考各脚本内注释和函数说明。