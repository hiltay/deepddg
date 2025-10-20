#!/usr/bin/env python3
"""
MacroDDG 推理脚本
支持命令行参数输入，可以对新数据进行DDG预测
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from autogluon.tabular import TabularPredictor

from macroddg.feature_generator import FeatureGenerator


def setup_device(device_arg):
    """设置和检测设备可用性"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"✓ 检测到CUDA可用")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            print(f"  当前GPU: {torch.cuda.current_device()}")
            print(f"  GPU名称: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            print("CUDA不可用，使用CPU")
    elif device_arg == "gpu" or device_arg == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"✓ 使用指定的GPU设备")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            print(f"  GPU名称: {torch.cuda.get_device_name()}")
        else:
            print("指定使用GPU但CUDA不可用，回退到CPU")
            device = "cpu"
    else:
        device = "cpu"
        print("✓ 使用CPU设备")

    return device


def load_input_data(input_path):
    """加载输入数据"""
    if not Path(input_path).exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    try:
        if input_path.endswith(".csv"):
            data_df = pd.read_csv(input_path)
        else:
            raise ValueError("不支持的文件格式，请使用.csv文件")

        print(f"✓ 成功加载数据: {len(data_df)} 行")

        # 检查必需的列
        required_columns = ["ori_seq", "mut_seq"]
        missing_columns = [col for col in required_columns if col not in data_df.columns]

        if missing_columns:
            raise ValueError(f"缺少必需的列: {missing_columns}")

        print(f"✓ 数据格式验证通过")
        return data_df

    except Exception as e:
        raise ValueError(f"读取输入文件失败: {str(e)}")


def load_model(model_path):
    """加载训练好的模型"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    try:
        model = TabularPredictor.load(model_path, require_py_version_match=False)
        print(f"✓ 成功加载模型: {model_path}")
        return model
    except Exception as e:
        raise ValueError(f"加载模型失败: {str(e)}")


def save_results(predictions, output_path, input_df):
    """保存预测结果"""
    try:
        # 创建输出目录
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 合并输入数据和预测结果
        result_df = input_df.copy()
        result_df["predicted_ddg"] = predictions

        # 保存结果
        if output_path.endswith(".csv"):
            result_df.to_csv(output_path, index=False)
        elif output_path.endswith(".xlsx"):
            result_df.to_excel(output_path, index=False)
        else:
            # 默认保存为CSV
            output_path = output_path + ".csv"
            result_df.to_csv(output_path, index=False)

        print(f"✓ 结果已保存到: {output_path}")
        return output_path

    except Exception as e:
        raise ValueError(f"保存结果失败: {str(e)}")


def run_inference(input_path, output_path, model_path, device, batch_size):
    """运行推理流程"""
    print("开始DDG推理...")
    print("=" * 60)

    total_start_time = time.time()

    # 1. 设置设备
    print("\n 设备设置")
    device = setup_device(device)

    # 2. 加载输入数据
    print("\n 加载输入数据")
    input_df = load_input_data(input_path)

    # 3. 加载模型
    print("\n 加载模型")
    ddg_model = load_model(model_path)

    # 4. 初始化特征生成器
    print("\n 初始化特征生成器")
    init_start_time = time.time()
    feature_generator = FeatureGenerator(device=device, batch_size=batch_size)
    init_time = time.time() - init_start_time
    print(f"✓ 特征生成器初始化完成，耗时: {init_time:.2f}秒")

    # 5. 特征提取
    print("\n 特征提取")
    feature_start_time = time.time()
    feature_df = feature_generator.make_ddg_features(input_df)
    feature_time = time.time() - feature_start_time
    print(f"✓ 特征提取完成，耗时: {feature_time:.2f}秒")

    # 6. 模型推理
    print("\n 模型推理")
    pred_start_time = time.time()
    predictions = ddg_model.predict(feature_df)
    pred_time = time.time() - pred_start_time
    print(f"✓ 模型推理完成，耗时: {pred_time:.2f}秒")

    # 7. 保存结果
    print("\n 保存结果")
    output_file = save_results(predictions, output_path, input_df)

    # 8. 时间统计
    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print("推理完成！")
    print("\n  时间统计:")
    print(f"  初始化时间: {init_time:.2f}秒")
    print(f"  特征提取时间: {feature_time:.2f}秒")
    print(f"  模型推理时间: {pred_time:.2f}秒")
    print(f"  总时间: {total_time:.2f}秒")

    print(f"\n 处理统计:")
    print(f"  输入样本数量: {len(input_df)}")
    print(f"  成功预测数量: {len(predictions)}")
    print(f"  平均处理时间: {total_time/len(input_df):.3f}秒/样本")

    # 9. 显示预测结果摘要
    print(f"\n 预测结果摘要:")
    print(f"  DDG范围: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"  DDG均值: {predictions.mean():.3f}")
    print(f"  DDG标准差: {predictions.std():.3f}")

    return output_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="MacroDDG 推理脚本 - 预测蛋白质突变的自由能变化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python inference.py -i data.csv -o results.csv
  python inference.py -i test_data.xlsx -o predictions/ --device gpu --batch-size 32
  python inference.py -i mutations.csv -o output.csv --model custom_model/

输入文件格式:
  CSV或Excel文件，必须包含以下列:
  - ori_seq: 原始蛋白质序列
  - mut_seq: 突变后蛋白质序列
        """,
    )

    parser.add_argument("-i", "--input", type=str, required=True, help="输入文件路径 (.csv)")

    parser.add_argument("-o", "--output", type=str, required=True, help="输出文件路径 (.csv)")

    parser.add_argument("--model", type=str, default="checkpoints", help="模型路径 (默认: checkpoints)")

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "gpu", "cuda", "cpu"],
        default="auto",
        help="计算设备 (默认: auto，自动检测GPU可用性)",
    )

    parser.add_argument("--batch-size", type=int, default=16, help="批处理大小 (默认: 16)")

    args = parser.parse_args()

    try:
        # 运行推理
        output_file = run_inference(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            device=args.device,
            batch_size=args.batch_size,
        )

        print(f"\n✅ 推理成功完成！结果保存在: {output_file}")

    except Exception as e:
        print(f"\n❌ 推理失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
