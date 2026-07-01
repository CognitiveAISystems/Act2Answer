<div align="center">

# Does VLA Even Know the Basics? Measuring Commonsense and World Knowledge Retention in Vision-Language-Action Models

<div align="center">

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.19297) [![Project-Page](https://img.shields.io/badge/Project--Page-%2300B4AB?style=for-the-badge&logo=logolol&logoColor=white&labelColor=000000)](https://tttonyalpha.github.io/act2answer/)  
[![HF Papers](https://img.shields.io/badge/Model--Dataset-%23FFD14D?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/papers/2606.19297)

</div>
</div>

<p align="center">
  <img src="figs/a2a_preview.png" width="100%" alt="Act2Answer overview and evaluation results">
</p>

**Act2Answer** is an embodied evaluation protocol for testing whether Vision-Language-Action
(VLA) models retain commonsense and world knowledge after robotics adaptation. Instead of asking a
model to answer in text, each VLM-style question becomes a short tabletop episode: the agent reads a
natural-language instruction and answers by placing a cube on the image tile it believes is correct.

The goal is to keep the motor problem deliberately simple, so failures are more informative about
missing, forgotten, or action-inaccessible knowledge rather than long-horizon control difficulty.

<!-- ## News:

- **[2026/06]** Act2Answer is released on arXiv: [2606.19297](https://arxiv.org/abs/2606.19297).
- **[2026/06]** Code and evaluation scripts for the Act2Answer benchmark suite are available in this repository. -->

## Contents

- [**✨ Overview**](#overview)
- [**📚 Act2Answer**](#act2answer)
- [**🔍 Key Findings**](#keyfindings)
- [**⚙️ Installation**](#installation)
- [**📈 Evaluation**](#runningevaluations)
- [**❤️ Citation**](#citation)

<h2 id="overview">✨ Overview</h2>

Act2Answer adapts established VLM benchmarks into an embodied binary-choice format. Each task has a
short action-compatible instruction, two visual answer options, and a common selection action in
simulation. The benchmark focuses on knowledge categories that matter for everyday embodied agents:
social, physical, quantitative, temporal, normative, cultural, and biological knowledge.

<p align="center">
  <img src="figs/preview_video_grid_v2.gif" width="95%" alt="Act2Answer task examples">
</p>

<h2 id="act2answer">📚 Act2Answer</h2>

The Act2Answer suite contains **1,720 unique binary questions** and **3,440 evaluation episodes** after including
the original and swapped layouts. It covers 12 categories adapted from five source benchmarks.

<p align="center">
  <img src="figs/a2a_construction.png" width="95%" alt="Act2Answer data curation pipeline">
</p>

<h2 id="keyfindings">🔍 Key Findings</h2>

<p align="center">
  <img src="figs/knowledge_probing.png" width="95%" alt="Layerwise probing results">
</p>

- Current VLAs usually preserve simple perceptual distinctions such as **Color** and **Shape**.
- Richer semantic categories are much harder: **Emotion**, **Attribute**, **State**, **Time**,
  **Counting**, **Symmetry**, **Traffic**, **Public Info**, **Celebrity**, and **Living World** often
  remain near chance for many models.
- Strong VLM baselines can outperform their VLA counterparts by roughly **20-40 points** on many
  knowledge-sensitive categories, suggesting a substantial VLM-to-VLA gap.
- Layerwise probing shows that answer-relevant information often remains recoverable in intermediate
  backbone layers, but weakens near the layers used for action prediction.
- VLA models trained with continued vision-language supervision tend to do better on knowledge-sensitive
  tasks than models trained mainly on robotics data.
- Downstream action fine-tuning can improve control while further weakening some forms of
  knowledge-sensitive behavior.

<h2 id="installation">⚙️ Installation</h2>

Prerequisites:

- Linux with an NVIDIA GPU and working CUDA driver.
- Conda or Miniconda available at `/opt/conda`, or configured with `CONDA_ROOT`.
- `git`, `tmux`, and enough disk for model weights.
- Optional but recommended: `huggingface-cli login` before the first model download.

Clone the repository and external model repos:

```bash
git clone <this-repo-url> Act2Answer
cd Act2Answer

bash scripts/setup/clone_external_repos.sh
```

Build the conda environment for the model you want to run. For example, SpatialVLA:

```bash
bash scripts/setup/setup_spatialvla_env.sh
```

Full setup instructions for all supported model stacks are in
[SETUP_README.md](SETUP_README.md).

<h2 id="runningevaluations">📈 Evaluation</h2>

Every evaluation wrapper sources [scripts/env.sh](scripts/env.sh), activates the expected conda
environment, runs both `noswap` and `swap`, and writes `FINAL_STATS` to
`$A2A_LOG_DIR/<model>_<asset>_eval.log`.

In-process models:

```bash
ASSETS=test_colors COUNT=6 EVAL_GPU=3 bash scripts/eval_pi0.sh
ASSETS=test_colors COUNT=6 EVAL_GPU=3 bash scripts/eval_magma.sh
ASSETS=test_colors COUNT=6 EVAL_GPU=3 bash scripts/eval_openvla.sh
ASSETS=test_colors COUNT=6 EVAL_GPU=3 bash scripts/eval_spatialvla.sh
```

Server-based models need their policy server first:

```bash
GPU=0 bash scripts/servers/run_xiaomi_policy_server.sh
ASSETS=test_colors COUNT=6 EVAL_GPU=3 bash scripts/eval_xiaomi.sh
```

Available evaluation wrappers:

| VLA | Script | Environment |
|-----|--------|-------------|
| pi0 | `scripts/eval_pi0.sh` | `pi0_act2answer` |
| Magma | `scripts/eval_magma.sh` | `magma_act2answer` |
| OpenVLA | `scripts/eval_openvla.sh` | `openvla_rl4vla` |
| SpatialVLA | `scripts/eval_spatialvla.sh` | `spatialvla_act2answer` |
| Xiaomi-Robotics-0 | `scripts/eval_xiaomi.sh` | server `mibot`, client `act2ans` |
| InternVLA-M1 | `scripts/eval_internvla.sh` | server `internvla`, client `act2ans` |
| MolmoAct2 | `scripts/eval_molmoact.sh` | server `molmoact2`, client `act2ans` |

You can also run the combined helper:

```bash
bash scripts/eval_all3.sh test_colors 6 3
```

## Assets

Act2Answer assets live under:

```text
ManiSkill/mani_skill/assets/carrot/<asset_name>/
```

Each asset set contains `pairs.json`, tile models/textures, and metadata. Use `ASSETS=<asset_name>`
and `COUNT=<n>` to select an evaluation slice. `COUNT=0` means all tasks.

## Outputs

Evaluation creates local files only:

- Videos and per-run YAML: `$A2A_OUTPUT_DIR/<run-name>/glob/` (default: `outputs/`).
- Logs: `$A2A_LOG_DIR/` (default: `logs/`).
- No wandb initialization, runs, or artifacts are created by evaluation.

<h2 id="citation">❤️ Citation</h2>

If you find Act2Answer useful, please cite our paper:

```bibtex
@misc{kachaev2026doesvlaknowbasics,
  title={Does VLA Even Know the Basics? Measuring Commonsense and World Knowledge Retention in Vision-Language-Action Models},
  author={Nikita Kachaev and Andrey Moskalenko and Matvey Skripkin and Nikita Kurlaev and Daria Pugacheva and Albina Burlova and Mikhail Kolosov and Denis Shepelev and Andrey Kuznetsov and Elena Tutubalina and Aleksandr I. Panov and Alexey K. Kovalev and Vlad Shakhuro},
  year={2026},
  eprint={2606.19297},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2606.19297}
}
```

## Acknowledgements

Act2Answer builds on [SimplerEnv](https://github.com/simpler-env/SimplerEnv) and
[ManiSkill](https://github.com/haosulab/ManiSkill), with evaluation harness pieces derived from
[RL4VLA](https://github.com/gen-robot/RL4VLA). The README structure follows the public
[BlindVLA](https://github.com/CognitiveAISystems/BlindVLA) project style.
