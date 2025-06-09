import json
import time
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
import pandas as pd

def read_gsm8k_jsonl(file_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """
    读取GSM8K JSONL文件并解析为字典列表
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if max_samples and i >= max_samples:
                    break
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"解析第{i+1}行JSON时出错: {e}")
                    continue
        print(f"成功读取 {len(data)} 条数据")
        return data
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return []
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []

def prepare_prompts(data: List[Dict[str, Any]]) -> List[str]:
    """
    准备推理提示词，根据GSM8K数据结构调整
    """
    prompts = []
    for item in data:
        # GSM8K通常包含'question'字段，根据需要调整
        if 'question' in item:
            prompt = f"请解决以下数学问题：\n{item['question']}\n\n请逐步推理并给出最终答案。"
        elif 'input' in item:
            prompt = f"请解决以下数学问题：\n{item['input']}\n\n请逐步推理并给出最终答案。"
        else:
            # 如果数据结构不同，使用第一个字符串字段
            text_fields = [v for k, v in item.items() if isinstance(v, str)]
            if text_fields:
                prompt = f"请解决以下数学问题：\n{text_fields[0]}\n\n请逐步推理并给出最终答案。"
            else:
                prompt = str(item)
        prompts.append(prompt)
    return prompts

def batch_process(data: List[Any], batch_size: int = 256) -> List[List[Any]]:
    """
    将数据分批处理
    """
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def setup_vllm_model(model_path: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> LLM:
    """
    设置vLLM模型和推理参数
    """
    # 配置采样参数
    sampling_params = SamplingParams(
        temperature=0.3,        # 较低温度保证推理稳定性
        top_p=0.9,             # 核采样参数
        max_tokens=1024,        # 数学问题需要较长推理
        presence_penalty=0.1,  # 轻微惩罚重复内容
        frequency_penalty=0.1,
    )
    
    # 初始化vLLM引擎
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,           # 根据GPU数量调整
        pipeline_parallel_size=2,
        max_num_seqs=256,                # 最大序列数，与批量大小匹配
        max_model_len=4096,              # 最大模型长度
        gpu_memory_utilization=0.9,     # GPU内存利用率
        trust_remote_code=True,           # 信任远程代码（如需要）
        # enforce_eager=True,
        # disable_pp_async_send=True,
    )
    
    return llm, sampling_params

def process_with_vllm(llm: LLM, sampling_params: SamplingParams, 
                     prompts: List[str], batch_size: int = 256) -> List[Dict[str, Any]]:
    """
    使用vLLM进行批量推理
    """
    results = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    
    print(f"开始处理 {len(prompts)} 条提示词，共 {total_batches} 个批次")
    
    # 分批处理
    prompt_batches = batch_process(prompts, batch_size)
    
    for i, batch_prompts in enumerate(prompt_batches):
        start_time = time.time()
        print(f"处理批次 {i+1}/{total_batches}, 本批数量: {len(batch_prompts)}")
        
        try:
            # 使用vLLM进行推理
            outputs = llm.generate(batch_prompts, sampling_params)
            
            # 处理输出结果
            batch_results = []
            for j, output in enumerate(outputs):
                result = {
                    "original_prompt": batch_prompts[j],
                    "generated_text": output.outputs[0].text,
                    "prompt_tokens": len(output.prompt_token_ids) if output.prompt_token_ids else 0,
                    "generated_tokens": len(output.outputs[0].token_ids) if output.outputs[0].token_ids else 0,
                    "finish_reason": output.outputs[0].finish_reason
                }
                batch_results.append(result)
            
            results.extend(batch_results)
            batch_time = time.time() - start_time
            print(f"批次 {i+1} 完成，耗时: {batch_time:.2f}秒")
            
        except Exception as e:
            print(f"处理批次 {i+1} 时出错: {e}")
            # 记录错误但继续处理
            for prompt in batch_prompts:
                results.append({
                    "original_prompt": prompt,
                    "generated_text": f"ERROR: {str(e)}",
                    "prompt_tokens": 0,
                    "generated_tokens": 0,
                    "finish_reason": "error"
                })
    
    return results

def save_results(results: List[Dict[str, Any]], output_file: str):
    """
    保存推理结果到JSONL文件
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存结果时出错: {e}")

def main():
    """
    主函数：完整的GSM8K数据处理流程
    """
    # 配置参数
    JSONL_FILE_PATH = "/home/g00955623/benchmark/ais_bench/datasets/gsm8k/test.jsonl"  # 替换为实际文件路径
    OUTPUT_FILE = "gsm8k_vllm_results.jsonl"
    MODEL_PATH = "/mnt/nfs/weight/Qwen3-32B"  # 替换为实际模型路径
    BATCH_SIZE = 1000
    MAX_SAMPLES = 1000  # 限制处理样本数，设为None处理全部
    
    print("开始GSM8K数据处理流程...")
    
    # 1. 读取数据
    print("步骤1: 读取JSONL文件")
    data = read_gsm8k_jsonl(JSONL_FILE_PATH, MAX_SAMPLES)
    if not data:
        print("没有读取到数据，程序退出")
        return
    
    # 2. 准备提示词
    print("步骤2: 准备推理提示词")
    prompts = prepare_prompts(data)
    
    # 3. 设置vLLM模型
    print("步骤3: 初始化vLLM模型")
    llm, sampling_params = setup_vllm_model(MODEL_PATH)
    
    # 4. 批量推理
    print("步骤4: 开始批量推理")
    results = process_with_vllm(llm, sampling_params, prompts, BATCH_SIZE)
    
    # 5. 保存结果
    print("步骤5: 保存推理结果")
    save_results(results, OUTPUT_FILE)
    
    # 统计信息
    successful = len([r for r in results if not r["generated_text"].startswith("ERROR")])
    print(f"\n处理完成! 成功处理: {successful}/{len(results)} 条数据")

if __name__ == "__main__":
    main()
