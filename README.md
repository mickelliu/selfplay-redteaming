# Self-RedTeam

This code supplements our recently released paper: [Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models](https://arxiv.org/abs/2506.07468)

This codebase is built on OpenRLHF: a high-performance RLHF framework built on Ray, DeepSpeed and HF Transformers. To visit the original repo:  [GitHub Repo](https://github.com/OpenRLHF/OpenRLHF/tree/main) | [Slides](https://docs.google.com/presentation/d/1JRhB1d7csofx0PIZBmfyBdMluxNd5JLPpUHrrvVhGnk/edit?usp=sharing) | [Technical Report](https://arxiv.org/abs/2405.11143) | [Documentations](https://openrlhf.readthedocs.io/)

The reward model part is built on WildGuard: [GitHub Repo](https://github.com/allenai/wildguard) | [Paper](https://arxiv.org/abs/2406.18495) | [Model](https://huggingface.co/allenai/wildguard)


## Quick Start

### Installation

Although OpenRLHF recommends using Docker, we ran our experiments in a conda manage environment. 

```bash
cd selfplay-openrlhf
conda create --name openrlhf python=3.10
pip install -e .
pip install openrlhf[vllm]
```

> [!NOTE]
>We recommend using vLLM 0.8.2 or higher.
>`export VLLM_USE_V1=1` requires vLLM 0.8.2 or the Nightly version and enable `export VLLM_ENABLE_V1_MULTIPROCESSING=0`.

## Training

### Step 0: Start `ray` cluster
```bash
# We use 4 A100-80GB-PCIe to run our experiments
ray start --head
```

### Step 1: Host Reward model
```bash
# We use 4 L40-48GB to host reward model inferences
# This GPU setup is more flexible depending how much inference workload you need to handle simutaneously
# We recommand num-gpus here = number of actors in the training process
bash scripts/serve_remote_wildguard.sh --num-gpus 4 --tensor-parallel-size 1
```

### Step 2: Run REINFORCE++ to reproduce our Self-play + SFT checkpoints
Before you start, please ensure that you have done the following:
- Unzip `red_team/data/data.zip` to the same directory if you need the dataset used in our experiment.
- Get the hostname of your remote reward model process, it will usually print in the console when it first initializes, change `REMOTE_RM_URL="http://0.0.0.0:5000/classify"` to whatever that hostname is.
```bash
# Change you experiment setting inside the shell scripts
bash ./scripts/red_team_game_reinforce_8b.sh
```

## Cite This
If you help our work helpful, please consider citing this work!
```
@misc{liu2025chasingmovingtargetsonline,
      title={Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models}, 
      author={Mickel Liu and Liwei Jiang and Yancheng Liang and Simon Shaolei Du and Yejin Choi and Tim Althoff and Natasha Jaques},
      year={2025},
      eprint={2506.07468},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.07468}, 
}
```

