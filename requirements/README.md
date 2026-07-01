# Curated Requirements

These files list the main direct packages needed for each Act2Answer environment. They are not full
`pip freeze` dumps. Transitive dependencies are intentionally left to pip/conda so the release stays
readable and portable.

CUDA-sensitive packages are installed in the matching `scripts/setup/setup_*.sh` before each
requirements file:

- `torch`, `torchvision`, `torchaudio`: installed from the PyTorch CUDA wheel index.
- `flash-attn`: installed separately after torch for models that need it.
- `ManiSkill/`, `SimplerEnv/`, and `openvla/`: installed as editable local packages by setup scripts.
- External repos (`RL4VLA`, `Xiaomi-Robotics-0`, `InternVLA-M1`, `molmoact2`) are cloned by
  `scripts/setup/clone_external_repos.sh`.

| Role | conda env | requirements file |
|------|-----------|-------------------|
| eval client for server models | `act2ans` | `eval_client.txt` |
| pi0 in-process eval | `pi0_act2answer` | `pi0.txt` |
| Magma in-process eval | `magma_act2answer` | `magma.txt` |
| OpenVLA in-process eval | `openvla_rl4vla` | `openvla.txt` |
| SpatialVLA in-process eval | `spatialvla_act2answer` | `spatialvla.txt` |
| InternVLA-M1 policy server | `internvla` | `internvla_server.txt` |
| Xiaomi-Robotics-0 policy server | `mibot` | `xiaomi_server.txt` |
| MolmoAct2 policy server | `molmoact2` | `molmoact_server.txt` |
