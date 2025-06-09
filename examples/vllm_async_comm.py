import os
import time
import argparse
import random
import string

from vllm import LLM, SamplingParams
from datasets import load_dataset, Features, Value, Sequence
from transformers import AutoTokenizer

os.environ["HCCL_BUFFSIZE"] = "1024"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TASK_QUEUE_ENABLE"] = "1"
os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM1"] = "1"
os.environ["HCCL_OP_EXPANSION_MODE"] = "AIV"
# os.environ["ASCEND_RT_VISIBLE_DEVICES"]="0,1,2,3"
# os.environ["DYNAMIC_EPLB"] = "false"
os.environ["VLLM_TORCH_PROFILER_DIR"] = "/home/g00955623/profiling/async_cpp_lite_1"
# os.environ["VLLM_TORCH_PROFILER_WITH_STACK"] = "0"

def generate_prompts_128K(model_path, target_length):
    # 定义特征 schema
    ft = Features({
        "id": Value("int64"),
        "context": Value("string"),
        "input": Value("string"),
        "answer": Sequence(Value("string")),
        "options": Sequence(Value("string"))
    })

    # 加载数据集 (InfiniteBench 包含长上下文样本)
    dataset_dict = load_dataset("/home/l00889328/datasets/InfiniteBench", features=ft)
    dataset = dataset_dict["train"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    token_ids = []
    target_length = target_length  # 目标128K tokens
    current_length = 0

    # 遍历数据集样本，累积到接近128K tokens
    for i in range(len(dataset)):
        try:
            # 构建当前样本的提示
            prompt = f"{dataset['context'][i]}\n{dataset['input'][i]}"

            # 编码提示
            encoded = tokenizer(
                prompt,
                truncation=False,  # 不截断，我们自己控制长度
                return_tensors="pt"
            )

            # 获取当前样本的token长度
            sample_length = encoded["input_ids"].shape[1]

            # 检查添加后是否超过目标长度
            if current_length + sample_length <= target_length:
                # 添加当前样本的tokens
                token_ids.extend(encoded["input_ids"].squeeze(0).tolist())
                current_length += sample_length
                print(f"已添加样本 {i}，当前总长度: {current_length}/{target_length}")
            else:
                # 如果添加后会超过，尝试填充剩余部分
                remaining = target_length - current_length
                if remaining > 0:
                    token_ids.extend(encoded["input_ids"].squeeze(0)[:remaining].tolist())
                    current_length = target_length
                    print(f"已达到目标长度 {target_length}")
                break  # 达到目标长度，退出循环

            # 如果已达到目标长度，退出循环
            if current_length >= target_length:
                break

        except Exception as e:
            print(f"处理样本 {i} 时出错: {str(e)}")
            continue

    # 将token ids解码为文本
    token_ids_text = tokenizer.decode(token_ids)
    print(f"最终生成的提示词长度: {len(token_ids)}/131072 tokens")
    return token_ids_text

# Performance testing function
def run_performance(args):
    """Run performance tests and return timing results."""

    sampling_params = SamplingParams(temperature = 0.0, top_p = 0.95, ignore_eos=True, max_tokens=args.output_len)

    prompt_text = [generate_prompts_128K(args.model_path, args.input_len)]
    # prompt_text = generate_prompts(args.input_len, 1)
    # Create an LLM
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        data_parallel_size=args.dp,
        pipeline_parallel_size=args.pp,
        prefill_context_parallel_size=args.pcp,
        decode_context_parallel_size=args.dcp,
        # enable_expert_parallel=True,
        enable_prefix_caching=False,
        max_num_batched_tokens=args.chunk_size + args.output_len + 10,  # 8192 + 11 = 8203
        max_model_len=args.input_len + args.output_len + 10,
        # quantization="ascend",
        async_scheduling=False,
        additional_config={"ascend_scheduler_config": {"enabled": False, "dynamic_eplb":False}},
        max_num_seqs=8,
        block_size=128,
        gpu_memory_utilization=0.8,
        enable_dcpp=True,
        # dcpp_min_chunk=2024
    )

    print("========================= First Infer =========================")
    t0 = time.time()
    llm.generate(prompts=prompt_text, sampling_params=sampling_params)
    t1 = time.time()
    dt0 = t1 - t0
    print(f"E2E: {dt0} s")
    print("============================= First Infer finished. ============================")

    # Second run for comparison
    print("========================= Second Infer ===========================")
    # llm.start_profile()
    t2 = time.time()
    for _ in range(args.iter_times):
        outputs = llm.generate(prompts=prompt_text, sampling_params=sampling_params)
    t3 = time.time()
    # llm.stop_profile()
    # Give engines time to pause their processing loops before exiting.
    time.sleep(1)
    dt1 = t3 - t2
    print(f"E2E: {dt1} s")
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print(f"req_num: {i}\nGenerated text: {generated_text!r}")
    print("============================= Second Infer finished. ============================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_len', type=int, default=32*1024)
    parser.add_argument('--chunk_size', type=int, default=4*1024)
    # current output_len only suppot 1 for long_seq prefill stage
    parser.add_argument('--output_len', type=int, default=1)
    parser.add_argument('--bs', type=int, default=1)
    # /home/t00608739/vllm_models/Qwen3-32B /home/z00911889/data/model_from_hf/DeepSeek-V2-Lite
    # /mnt/nfs/vllm/DeepSeek-V3.1-Terminus-w8a8-QuaRot-lfs /mnt/nfs/weight/Qwen3-235B-A22B-Instruct-2507
    # parser.add_argument('--model_path', type=str, default="/mnt/nfs/vllm/DeepSeek-V3.1-Terminus-w8a8-QuaRot-lfs")
    parser.add_argument('--model_path', type=str, default="/home/t00608739/vllm_models/Qwen3-32B")
    parser.add_argument('--tp', type=int, default=4)
    parser.add_argument('--pcp', type=int, default=1)
    parser.add_argument('--dcp', type=int, default=1)
    parser.add_argument('--dp', type=int, default=1)
    parser.add_argument('--pp', type=int, default=2)
    parser.add_argument('--iter_times', type=int, default=1)

    args = parser.parse_args()
    # Run performance test using our new function
    run_performance(args)
    # from torch_npu.profiler.profiler import analyse
    # analyse(os.environ["VLLM_TORCH_PROFILER_DIR"])


