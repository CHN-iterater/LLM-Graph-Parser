"""
Subprocess script for CPU-only kernel profiling.
Run by run.py as a subprocess to avoid CUDA context contamination.
Usage: python profile_kernels.py <model_path> <prompt_text> <phase>
Phase: "prefill" or "decode"
Outputs JSON to stdout with category time ratios.
"""
import sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = sys.argv[1]
prompt = sys.argv[2]
phase = sys.argv[3]

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=False)
model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].cuda()
attn_mask = inputs.get("attention_mask")
if attn_mask is not None:
    attn_mask = attn_mask.cuda()

if phase == "decode":
    input_ids = input_ids[:, -1:]
    if attn_mask is not None:
        attn_mask = attn_mask[:, -1:]

kwargs = {}
if attn_mask is not None:
    kwargs["attention_mask"] = attn_mask

with torch.no_grad():
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        model(input_ids, **kwargs)
        torch.cuda.synchronize()

cat_times = {"compute_bound": 0.0, "memory_bound": 0.0, "data_movement": 0.0, "communication": 0.0}
for ev in prof.key_averages():
    name = ev.key.lower()
    d = 0
    for a in ("cpu_time_total", "self_cpu_time_total", "cpu_time"):
        v = getattr(ev, a, None)
        if v is not None and isinstance(v, (int, float)) and v > 0:
            d = v
            break
    if not d:
        continue
    if any(k in name for k in ("nccl","allreduce","allgather","broadcast","reduce_scatter")):
        cat_times["communication"] += d
    elif any(k in name for k in ("memcpy","memset","aten::copy_","aten::to","aten::cat","aten::transpose","aten::permute","aten::reshape","aten::view","aten::expand","aten::slice","aten::split","aten::clone")):
        cat_times["data_movement"] += d
    elif any(k in name for k in ("cublas","cutlass","gemm","aten::mm","aten::addmm","aten::bmm","aten::matmul","aten::linear","aten::_convolution","aten::conv","aten::softmax","aten::layer_norm","aten::native_layer_norm","aten::rms_norm","aten::gelu","aten::silu","aten::relu","aten::tanh","aten::sigmoid","flash","attention")):
        cat_times["compute_bound"] += d
    else:
        cat_times["memory_bound"] += d

total = sum(cat_times.values()) or 1
for k in cat_times:
    cat_times[k] /= total

print(json.dumps(cat_times))
