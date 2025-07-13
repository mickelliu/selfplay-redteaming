MODEL_CONFIGS: dict = {}

# Add models in alphabetical order of the model name key
# If model size isn't apparent from the key, add it to the metadata dict as "model_size"
# Default "model" is the location in HF hub, "model_path" can be a list and include
# a local path and a remote path (beaker:// or hf:// for HF hub)
# "gantry_args" can be used to pass additional arguments to gantry

MODEL_CONFIGS.update(
    {
        "dclm-1b": {
            "model": "TRI-ML/DCLM-1B",
            "gantry_args": {
                "install": "pip install -e . && pip install git+https://github.com/mlfoundations/open_lm.git"
            },
            "metadata": {},
        },
        "llama-7b": {
            "model": "llama-7b",
            "model_path": "beaker://oyvindt/llama-7b",
            "metadata": {},
        },
        "llama2-7b": {
            "model": "meta-llama/Llama-2-7b-hf",
            "model_path": [
                "/net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B",
                "hf://meta-llama/Llama-2-7b-hf",
            ],
            "gantry_args": {"hf_token": True},
            "metadata": {},
        },
        "llama2-13b": {
            "model": "meta-llama/Llama-2-13b-hf",
            "model_path": [
                "/net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/13B",
                "hf://meta-llama/Llama-2-13b-hf",
            ],
            "gantry_args": {"hf_token": True},
            "metadata": {},
        },
        "llama3-8b": {
            "model": "meta-llama/Meta-Llama-3-8B",
            "model_path": [
                "beaker://davidw/Meta-Llama-3-8B",
                "hf://meta-llama/Meta-Llama-3-8B",
            ],
            "gantry_args": {"hf_token": True},
            "metadata": {},
        },
        "llama3-8b-instruct": {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "model_path": [
                "hf://meta-llama/Meta-Llama-3-8B-Instruct",
                # "beaker://davidw/Meta-Llama-3-8B-Instruct",  # dealing with <|eot_id|> issue
            ],
            "chat_model": True,
            "gantry_args": {"hf_token": True},
            "metadata": {},
        },
        "llama3-70b": {
            "model": "meta-llama/Meta-Llama-3-70B",
            "model_path": [
                "beaker://davidw/Meta-Llama-3-70B",
                "hf://meta-llama/Meta-Llama-3-70B",
            ],
            "gantry_args": {"hf_token": True},
            "metadata": {},
        },
        "olmo-1.7-flanfix-7b": {
            "model": "OLMo-1.7-7B-flanfix-hf",
            "model_path": "/net/nfs.cirrascale/allennlp/davidw/checkpoints/mitchish7-anneal-from477000-50B-flan_fix",
            "metadata": {},
        },
        "olmo-1.7-2.7T-S3-7b": {
            "model": "OLMo-1.7-2.7T-S3-7B",
            "model_path": "s3://ai2-llm/checkpoints/OLMo-medium/mitchish7-datafix/step639650-huggingface",
            "metadata": {},
        },
        "olmo-1.7-2.75T-anneal-7b": {
            "model": "OLMo-1.7-2.75T-anneal-7B",
            # s3://ai2-llm/checkpoints/davidw/annealing/mitchish7-datafix-anneal-from639650-50B/step11931-huggingface
            "model_path": "/net/nfs.cirrascale/allennlp/davidw/checkpoints/mitchish7-datafix-anneal-from639650-50B",
            "metadata": {},
        },
        "olmo-7b-amberish7-anneal-from477850-50B": {
            "model": "olmo-7b-amberish7-anneal-from477850-50B",
            "model_path": "/net/nfs.cirrascale/allennlp/davidw/checkpoints/amberish7-anneal-from477850-50B/step11931-huggingface-oldstyle",
            "metadata": {},
        },
        "openelm-3b-BUGGY": {
            "model": "apple/OpenELM-3B",
            "trust_remote_code": True,
            "tokenizer": "meta-llama/Llama-2-7b-hf",
            "gantry_args": {"hf_token": True},
            "metadata": {
                "description": "This variant doesn't work it seem, gets random performance on arc_easy"
            },
        },
        "olmo-7b-amberish7-step477850-hf-olmo": {
            "model": "olmo-7b-amberish7-step477850-hf-olmo",
            "model_path": "/net/nfs.cirrascale/allennlp/davidw/checkpoints/amberish7/step477850/hf-olmo",
            "metadata": {},
        },
        "olmo-1b-newhp-newds-hf-olmo": {
            "model": "olmo-1b-newhp-newds-hf-olmo",
            "model_path": "/net/nfs.cirrascale/allennlp/davidw/checkpoints/olmo-1b-newhp-newds/step20000/hf-olmo",
            "metadata": {},
        },
        "olmo-1b-newhp-newds-cx5-hf-olmo": {
            "model": "olmo-1b-newhp-newds-cx5-hf-olmo",
            "model_path": "/net/nfs.cirrascale/allennlp/davidw/checkpoints/olmo-1b-newhp-newds-cx5/step30993/hf-olmo",
            "metadata": {},
        },
        "olmo-1b-newhp-newds-cx5-datafix-hf-olmo": {
            "model": "olmo-1b-newhp-newds-cx5-datafix-hf-olmo",
            "model_path": "/net/nfs.cirrascale/allennlp/davidw/checkpoints/olmo-1b-newhp-newds-cx5-datafix/step30993/hf-olmo",
            "metadata": {},
        },
        "olmo-1b-newhp-newds-cx5-flan-hf-olmo": {
            "model": "olmo-1b-newhp-newds-cx5-flan-hf-olmo",
            "model_path": "/net/nfs.cirrascale/allennlp/davidw/checkpoints/olmo-1b-newhp-newds-cx5-flan/step30993/hf-olmo",
            "metadata": {},
        },
        "olmo-1b-newhp-newds-cx5-reddit-hf-olmo": {
            "model": "olmo-1b-newhp-newds-cx5-reddit-hf-olmo",
            "model_path": "/net/nfs.cirrascale/allennlp/davidw/checkpoints/olmo-1b-newhp-newds-cx5-reddit/step30993/hf-olmo",
            "metadata": {},
        },
        "olmo-1b-newhp-oldds-hf-olmo": {
            "model": "olmo-1b-newhp-oldds-hf-olmo",
            "model_path": "/net/nfs.cirrascale/allennlp/davidw/checkpoints/olmo-1b-newhp-oldds/step15000/hf-olmo",
            "metadata": {},
        },
        "olmo-1b-newhp-oldds-cx5-hf-olmo": {
            "model": "olmo-1b-newhp-oldds-cx5-hf-olmo",
            "model_path": "/net/nfs.cirrascale/allennlp/davidw/checkpoints/olmo-1b-newhp-oldds-cx5/step30993/hf-olmo",
            "metadata": {},
        },
        "olmo-7b-amberish": {
            "model": "olmo-7b-amberish",
            "model_path": "s3://ai2-llm/checkpoints/davidw/annealing/amberish7-anneal-from477850-50B/step11931-huggingface-oldstyle",
            "metadata": {},
        },
        "olmo-7b-peteish-anneal-from-928646-50B-no-warmup": {
            "model": "olmo-7b-peteish-anneal-from-928646-50B-no-warmup",
            "model_path": "weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7-anneal-from-928646-50B-no-warmup/step11931-hf-olmo",
            "metadata": {},
        },
        "olmo-7b-peteish-anneal-from-928646-50B-nowup-dclm07-fw2": {
            "model": "olmo-7b-peteish-anneal-from-928646-50B-nowup-dclm07-fw2",
            "model_path": "weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7-anneal-from-928646-50B-nowup-dclm07-fw2/step11931-hf",
            "metadata": {},
        },
        "olmo-7b-peteish-anneal-from-928646-50B-nowup-dclm07-flan": {
            "model": "olmo-7b-peteish-anneal-from-928646-50B-nowup-dclm07-flan",
            "model_path": "weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7-anneal-from-928646-50B-nowup-dclm07-flan/step11931-hf",
            "metadata": {},
        },
    }
)
