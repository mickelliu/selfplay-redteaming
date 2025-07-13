## How to run eval

Unfortunately for this project, we have to adopt three separate libraries for evaluations. That means you need to install three separate conda environments due to each library has vastly different dependencies.

### Step 0: Install all three environments

Note: since the `full_evals.sh` will activate specific environment when start running specific benchmark, if you decide to change the environment name, please make them consistent with you have in  `full_evals.sh`

**Safety Eval**
```
cd eval/benchmarks/safety-eval-fork
conda create -n safety-eval python=3.10 && conda activate safety-eval
pip install -e .
pip install -r requirements.txt
pip install vllm==0.4.2
```

**OLMES**
```
cd eval/benchmarks/olmes
conda create -n olmes python=3.10
conda activate olmes
pip install -e .
```

**lm-eval**
```
cd eval/benchmarks/lm-evaluation-harness
conda create -n lm-eval python=3.10.6
pip install -e .
```

### Step 1: Run full eval
After finish running the experiments, grab the checkpoint path and prepare to run `launch_full_evals.sh`.
```
bash launch_full_evals.sh
```
Inside the `launch_full_evals.sh` file, change the following field if you want to just evaluate one model with the "llama3_cot" chat template. "llama3_cot" is just a template name that we assigned to the R1 template across these libraries. You can view or change this template (or add you own template) in `eval/benchmarks/safety-eval-fork/src/templates/single_turn.py` or `eval/benchmarks/olmes/oe_eval/tasks/chat_templates.py`

```
MODEL_TEMPLATES=(
    ["/path/to/model/1"]="llama3_cot"
)

# If you just want to use the chat template that comes with the model's tokenizer configuration file, leave blank.
# ["/path/to/model/1"]="" or "hf"
```



