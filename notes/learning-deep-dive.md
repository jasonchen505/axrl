# axrl 深入学习文档

> 上游：`XYZ-AI-Lab/axrl`（AxisRL），Agentic RL post-training 框架：SGLang 管 rollout，Megatron 管训练，axrl 做其间的系统层。
> Fork：`jasonchen505/axrl`。本文只写在 fork 的 `notes/learning` 分支，`main` 未动，未来任何上游贡献分支必须从上游 `main` 起，个人笔记绝不进上游 PR diff。
> 调研时间：2026-10-14。调研方式：GitHub API 只读查询 + 关键事实复核。凡标注“未验证”的，均指本轮未逐行读到证据，不要当成结论。
> 一个纠正：用户原话“72 天前更新”是误读。最后提交是 2026-07-24，距调研日约 82 天；`updated_at=2026-10-13` 是仓库元数据，不是提交。
> 勘误（2026-10-28）：§5 原疑点 1（mismatch mask 只进 metrics、loss 是 no-op）经逐行核验为误报，见 §5 修正版。不要以该理由提 issue。

---

## 1. 基本面：单日快照式开源，82 天零维护

| 项 | 结果 |
|---|---|
| 上游 | `XYZ-AI-Lab/axrl`，描述：“AxisRL is an agentic RL post-training framework built on SGLang rollout, Megatron training, and real-world agent workflows.” |
| Stars / Forks | 612 ⭐ / 16 forks（API 返回） |
| License | Apache-2.0，但附录模板仍是占位符 `Copyright [yyyy] [name of copyright owner]`，未填年份与作者 |
| 默认分支 | `main` |
| 上游 HEAD | `ad7659cf0b42da6c0576292194cc27a9f3ea45aa`，`docs: simplify Docker installation instructions`，作者 Junjie Li，2026-07-24T16:50:50Z（本轮复核通过） |
| 提交节奏 | 全部 9 个 commit 都在 2026-07-24 一天内：`initial commit`（`1c144b59`）+ logo / 中文 README / CI 徽章等 4 个 docs PR + 2 个直接 docs commit。仅 2 位作者：Junjie Li、Jinyu Wang。实质是单日快照式开源 + 文档微调，之后零代码提交 |
| Fork 状态 | Fork HEAD 与上游完全一致（`ad7659cf…`），0 ahead / 0 behind；fork 根目录各条目 SHA 与上游逐个相同；fork main 是干净镜像，无个人笔记 commit |
| Open issues | 0（本轮复核通过，`totalCount=0`） |
| Closed issues | 0，从未有过 issue |
| Open PRs | 0 |
| Closed PRs | 4 个，全是 docs：#1 logo、#2 中文 README、#3 README 徽章 + CI、#4 简化 README |
| CI | 仅 `.github/workflows/ci.yml`：`ruff` job（`ruff==0.14.11`，只跑 `ruff check .`，无 `ruff format --check`、无 mypy/pyright、无 pytest）+ `shellcheck` job（扫 `axis_recipe` / `benchmark` / `scripts` 下 `*.sh`）。触发 push/PR 到 main |
| CONTRIBUTING / PR 模板 / DCO / CLA / CODEOWNERS / SECURITY | 全部不存在。`AGENTS.md`（562 B）仅含 Git 安全规则，无其他贡献指南 |
| 本地校验比 CI 严 | `.pre-commit-config.yaml`：`check-added-large-files`（5000 KB）、`shellcheck`、`ruff` + `ruff-format`；`skills/axrl-fix-linting` 称 `scripts/run-precommit-check.sh` 还跑 pyright + mypy（该脚本本轮未全文核验） |

**基本面判断：** 612 星更多是“发布即走红”而非持续维护的结果。0 issue + 82 天零维护 + 无贡献指南，意味着任何贡献都要先过“maintainer 是否还在看”这一关，且无法通过 issue 对齐语义，只能直接提 PR。提 PR 前必须自己把验证做足。

---

## 2. 定位与顶层结构

### 2.1 README 核心主张

- 定位：SGLang 管高吞吐 rollout / 服务，Megatron 管大规模训练 / 并行 / checkpoint，AxisRL 只管其间的系统层（agent 编排、权重同步、资源调度、数据移动、可观测性、配方）。
- Highlights：300+ turn 轨迹、数百 B 参数规模训练；PPO、GRPO / GRPO2、GSPO、TOPR、TIS 等 objective；白盒环境 + 黑盒 harness 捕获（OpenAI 兼容代理）；partial rollout + 轻量控制面；handle-based 数据移动、context packing、routing replay、mismatch 分析、spike replay。
- **数字主张均无可验证数据**：博客 `docs/axrl-blog-en.md` 全文仅有“300+ turns”“hundreds-of-billions 参数规模”两句量化表述，无准确率 / 奖励 / 吞吐 / 延迟表、对比基线、消融结果。文档中必须标注为“官方主张，未附可验证 benchmark”，不要当成实测结论。
- Roadmap：更多真实 agent 配方 / 公开案例；改善 rollout / training 间异步执行；扩展多模态 rollout 产物。

### 2.2 目录树（根目录）

```
.github/workflows/ci.yml  .pre-commit-config.yaml  AGENTS.md  LICENSE  README.md
axrl/            # 核心库
axis_recipe/     # 公开配方（5 个：grpo_gsm8k、ppo_gsm8k、search_r1、grpo_dapo17k_moe、blackbox_rl）
benchmark/       # 内存 / 并行 / Magi 前向 / 权重同步基准
docker/  docs/   scripts/  skills/  tests/
pyproject.toml   setup.py
```

注意：README 配方表只列 4 个，遗漏 `grpo_dapo17k_moe`（表 4 行 vs 目录 5 个配方），原因未查到。

### 2.3 依赖：几乎全靠 Docker 镜像

- `pyproject.toml`：无 `[project]` / `[build-system]` / `dependencies` 段，只有 `[tool.*]`（pyright、mypy 均大量关闭错误码、ruff `line-length=150`、`select=["ALL"]` + 长 ignore 列表、pydocstyle google 约定）。
- `setup.py`：包名 `axrl` 0.1.0，Python ≥ 3.12，唯一声明的 pip 依赖是 `e2b>=2.34.0`。SGLang / Megatron / Ray / CUDA 均不在此声明。
- 推荐环境：Docker 镜像 `leejunjie/sglang-mcore:cu130-sgl0.5.14-mcore0.18-magi`，Dockerfile `docker/cuda/cu130-sgl0.5.14-mcore0.18-magi.Dockerfile`；容器内 `pip install -e .`。

### 2.4 配置系统（Pydantic + OmegaConf 三层合并）

- 全部配置是 Pydantic `StrictBaseModel`（`axrl/configs.py:62-65`，`extra="forbid"`，未知字段直接报错，fail fast）。不是 dataclass（仅 `DirInfo` 是 `@dataclass`，`configs.py:12-17`）。
- 顶层是 `PipelineExperimentConfig`（`axrl/pipeline/config.py:57-82`）：`controller` / `eval_only` / `megatron_worker` / `value_worker`（PPO 用）/ `rollout_worker` / `train_datasets` / `test_datasets` / `eval_sampling_config` / `train_sampling_config` / `grpo` / `sft` / `online_rl_train` / `replay_rl_train` / `mismatch_test` / `logger`。
- `run_mode`（`pipeline/config.py:19-25`）：`online_rl_train`（新 rollout 再训）/ `replay_rl_train`（用已保存 rollout 训）/ `eval_only` / `sft_train` / `mismatch_test`。
- 合并优先级（`axrl/utils/config_utils.py:124-189`）：YAML 文件 < CLI `--path.to.field=value` < 环境变量 `AXRL__path__to__field=value`（`AXRL__config_path` 最高）。CLI 值解析靠 OmegaConf。目录默认值来自环境变量（`AXRL_OUTPUT_DIR_NAME` 默认 `default`、`AXRL_DATA_DIR` 默认 `~/axrl-data/datasets` 等，`configs.py:20-39`）。
- 关键默认值：`OnlineRLTrainConfig`（`configs.py:568-588`）：`num_rollouts_per_conversation=8`、`model_sync_every_n_global_updates=4`、`max_global_updates=4000`、`reward_mean/std_type="group"`、`filter_zero_std=False`、`sort_sampled_prompts_by_response_length=True`；`GrpoTrainerConfig`（`:534-565`）：默认 `loss_type="grpo2"`、`clip 0.28/0.2`、`dual_clip_neg_adv_factor=3.0`；`MegatronWorkerConfig`（`:370-497`）：`tp/pp/dp/cp` 默认 1、`global_batch_size=32`、`checkpoint_dir="checkpoints/megatron"`、`most_recent_checkpoint_k=2`、`deterministic_mode=True`。
- 旧版顶层 `GrpoExperimentConfig`（`axrl/trainer/grpo_exp_config.py:13-25`）+ legacy `GrpoConfig`（`configs.py:591-611`）仅供旧 controller 用，新配方均用 `PipelineExperimentConfig`。

### 2.5 训练入口全链路

```
bash axis_recipe/<recipe>/run_train*.sh [--path.to.field=value ...]
  │  grpo/ppo/search 单机 / dapo/blackbox/search 分布式：先跑 generate*.py 重生成 YAML（blackbox 若 yaml 已存在则跳过）
  │  search：先 start_retriever.sh 起 BM25 检索服务；dapo/blackbox/search 分布式：先 scripts/start_ray.sh，仅 head 跑 controller，worker 等 Ray 退出
  ▼
python -u axis_recipe/<recipe>/train_*.py --config_path=<recipe>.yaml [--controller.* / --online_rl_train.* ...]
  │  load_and_validate_config(<Recipe>Config, print_configs=True) → Pydantic strict 校验
  ▼
PipelineController(config, <Recipe>(config))
  │  Recipe.initialize_local_processors() /（blackbox：start_services → OpenAI proxy + tunnel）
  │  Recipe.run_rollout(conversation, runtime)：白盒 MathEnv/SearchEnv + RolloutAgent → rollout_from_env；
  │    黑盒 OpenHandsEnv(E2B) + OpenAI proxy 回调 + RolloutAgent 循环
  ▼
controller.start() → rollout (SGLang) ⇄ 训练 (Megatron) ⇄ 权重同步 → 下一迭代
```

5 个配方（`axis_recipe/`）：`grpo_gsm8k`（Qwen2.5-1.5B-Instruct，`loss_type=grpo`，max_updates 600）、`ppo_gsm8k`（同模型，`loss_type=ppo` + value worker，max_updates 128）、`search_r1`（Qwen2.5-3B-Instruct，`grpo2`，N Q 数据集 + BM25 检索，另有 MoE 分布式变体脚本，R3 开关是 moe 版与 moe-r3 版的唯一差异）、`grpo_dapo17k_moe`（Qwen3-30B-A3B-Base，`max_length=16384`，max_updates 30000，无 `run_train.sh`，仅分布式脚本）、`blackbox_rl`（**WIP**，Qwen3-4B-Instruct-2507，`max_length=65536`，默认 `run_mode=eval_only`，OpenHands + E2B + Cloudflare Quick Tunnel，checkpoint 默认关闭）。

共同点：所有 `train_*.py` 均以 `asyncio.run(run_controller())` 收尾；差异只在 Recipe 的环境 / verifier / 采样 / 默认值与 launcher 的 Ray / 检索 / 隧道前置步骤。注意 `search_r1/run_train.sh` **不透传 `"$@"`**（与 grpo/ppo 不同），CLI 覆盖在该配方单机脚本中无效。

### 2.6 SGLang rollout 与 Megatron 训练如何衔接

**不是 HTTP 服务调用，是进程内 + Ray actor + phase 分时：**

- SGLang 侧：`SGLangWorker`（`axrl/worker/sglang_worker.py:49`）直接 `from sglang.srt.entrypoints.engine import Engine`（`:12`），`Engine(...)` 进程内构造（`:107`），生成走 `engine_tokenizer_manager.generate_request`（`:534`），无 HTTP。
- 训练侧：`MegatronWorker`（`axrl/worker/megatron_worker.py:173`），`train_step`（`:1060-1124`）调 Megatron `get_forward_backward_func`。
- Ray 层：每 rank 一个 `RemoteMegatronWorker`（`axrl/ray/ray_megatron_worker.py:33-34`），每 bundle 一个 `RemoteSGLangWorker`（`ray_rollout_worker.py:28-29`）；rollout actor 用 `NodeAffinitySchedulingStrategy` 绑定到 rollout bundle 节点（`pipeline/controller.py:711-721`）。
- Placement：`get_worker_placement`（`pipeline/utils.py:242-280`）强制 rollout 与 Megatron GPU 总数相等；colocated 时两者共享同一个 `ResourceGroup`（`:270-272`），disaggregated 时分别建组。PPO 的 value worker 与 actor 共享同一个 megatron `ResourceGroup`（`:301`），靠 `to_cpu` / `to_gpu` 分时复用同一批 GPU。

**Phase 切换：** `StageManager`（`axrl/controller/stage_manager.py:10-25`）收敛为 4 个方法。`ColocatedStageManager`（`:27-69`）：切权重同步 = pause rollout → 释放 rollout GPU → 仅恢复权重 → value `to_cpu` → megatron `to_gpu`；切 rollout = megatron/value `to_cpu` → rollout 恢复权重 + KV → 恢复生成。`DisaggregatedStageManager`（`:72-110`）不碰 rollout GPU，只切 megatron/value。

**权重同步（两条路径，无显式版本号 / fence）：** `WeightUpdater` 工厂（`axrl/utils/megatron/weight_update.py:33-68`）按 `(engine_type, colocated)` 分发。Colocated：`SGLangTensorWeightUpdater`（`sglang_weight_updater.py:60-272`），每 rank 把桶 pack 成单个 `FlattenedTensorBucket`（一个 CUDA IPC handle），经 Gloo `gather_object` 汇到 gather_src，再调 engine 的 `update_weights_from_tensor`，每桶后 `dist.barrier` + `torch.cuda.ipc_collect()`。Disaggregated：`SGLangDistributedWeightUpdater`（`:274-338`），建 NCCL `weight_update_group`，`update_weights_from_distributed` + `dist.broadcast`。SGLang 侧实现只断言 `is_paused()`（`sglang_worker.py:778,788`），权重“新旧”由 controller 调用时序 + `strict_on_policy` 语义保证，而非版本号校验（见 §5 可疑点）。

**数据流全链路：** `RemoteRolloutActor.run`（`pipeline/rollout_actor.py:93-99`）起 `max_running_tasks` 个并行 worker，各自阻塞 `rollout_queue.get` 拉任务 → `recipe.run_rollout` → OPD teacher 标注（`:127-136`）→ actor 侧打包 → `result_queue.put`。打包后 `ray.put` 存 plasma，`packed_samples_ref` 传引用，并 `_strip_heavy_rollout_payload`（`:165-179`）清空 trace / messages / input_ids 等，只留轻量字段。`TensorHandle`（`axrl/utils/tensor_store.py:28-`）包 Ray `ObjectRef` + 行切片，R3 路由回放数据经此传递，用完由 `delete_r3_handles_and_caches`（`pipeline/controller.py:1406-1432`）去重删除。Group 组装：`stream_rollout_groups`（`:1019-1045`）按 `group_id` 攒够 `num_rollouts_per_conversation` 条才放行，攒满即 `normalize_group_rewards` + 过滤。DP batch 只存一次 plasma、各 rank 只收自己 ref（`ray_megatron_worker.py:330-355`）。

**新版主循环** `run_online_rl_train`（`pipeline/controller.py:1617-1653`）：`switch_to_train → load_checkpoint → prepare_for_weight_updates → update_rollout_model_weights → run_evals → while global_step < max: 取 batch → prepare_online_training_samples → train_online_sample_dataset → delete R3 handles → save_rollouts_snapshot → save_checkpoint_if_needed → sync_rollout_after_online_train`。PPO 分支先训 critic（可纯 value warmup），再训 actor。

---

## 3. 核心模块

### 3.1 `axrl/trainer/`：loss 与 forward 解耦，objective 并列实现

| 类 | 文件 | 职责 |
|---|---|---|
| `BaseTrainer` | `trainer/base_trainer.py:19` | `loss_func` / `forward_step` 抽象；`aggregate_with_mask`（`:50-107`，支持 `token-mean` / `seq-mean-token-mean` / `token-max/min/std/p05/p95` / `seq-mean-token-std`）；`compute_denominator_count`（`:110-119`）；`logprob_forward_step`（`:141-162`，零 loss、只回传 logprob + index） |
| `GrpoTrainer` | `trainer/grpo_trainer.py:50` | 全部 RL objective |
| `SftTrainer` | `trainer/sft_trainer.py:20` | `-log_prob` 按 `token-mean` 聚合（`:62`），可选 entropy / accuracy（`:49-58`） |
| `ValueTrainer` | `trainer/value_trainer.py:23` | PPO critic；`value_forward_step`（`:122-141`）只回传 values |

Objective 分发 `get_loss_func`（`grpo_trainer.py:569-587`）：`grpo` → `grpo_loss`（`:197-227`，标准 PPO-clip，`dual_clip_neg_adv_factor` 非空时对负 advantage 再 dual-clip）；`ppo` → **同样返回 `grpo_loss`**（`:574-576`，且断言 `is_base_logprobs=="old_logprobs"`，PPO 与 GRPO 在 actor loss 上无独立实现，区别只在 advantage 来源）；`tis` → `tis_loss`（`:229-258`）；`kimi2_5` → `kimi2_5_loss`（`:260-288`，`hard_clip=True`）；`topr` → `topr_loss`（`:290-329`，正 / 负 advantage 分两段，clip 区间分别为 `[1,1]` 与 `[0,1]`）；`grpo2` → `grpo2_loss`（`:383-449`，按 advantage 正负 × ratio 大于 / 小于 1 切 4 段，hard-clip 与 soft-clip 混用，依赖 `dual_soft_clip`）；`gspo` → `gspo`（`:352-375`，序列级 `seq_ratio`，并断言 `loss_agg_type=="seq-mean-token-mean"`）。共享 helper `calculate_pg_loss_and_clip_ratio`（`:331-350`），`hard_clip` 时被 clip 的 token 的 loss 被 detach（`:345-347`）。

`loss_func`（`:698-803`）：总 loss `= pg_loss + entropy_loss*alpha + kl_loss*kl_control_alpha`（`:775`，`kl_control_alpha` 默认 0.0）。KL：`get_kl_loss`（`:483-485`），`kl_utils.py:15-32` 提供 k1/k2/k3，**loss 内部用哪种 kl_type 未完整读取，未验证**。Entropy：`get_entropy_loss`（`:470-481`）是到目标值的平方误差 `(target_entropy - batch_entropy)²`，默认目标 0.1，支持 top-quantile 只约束高熵 token。

Mismatch / 过滤三件套（`torch.no_grad()` 下算，`:727-747`）：`truncate_mismatch_token_ratio`（`:605-630`，`old/rollout` logprob 的 token 级 importance ratio）；`get_icepop_token_mask`（`:632-654`，ratio 超界的 token 被 mask）；`get_mismatch_sequence_mask`（`:656-696`，序列级 ratio 超界的序列被 mask + veto）。注释明确 mismatch 过滤只 gate PG 项，KL / entropy 仍用原始 `loss_mask`（`:753-755`）。**§5 的原疑点（“`policy_loss_mask` 只进 metrics”）已于 2026-10-28 逐行核验为误报**：过滤机制走的不是这个 mask 变量，而是乘法因子 `policy_mismatch_tir`，详见 §5 修正版。

OPD：`apply_opd_to_advantages`（`:488-567`），见 §3.6。

细节：`logits_processor`（`:805-825`）注释承认 `vocab_parallel_log_prob()` 会原地改 logits，为避免影响 entropy 反传先 `clone()`（`:821-824`）。

### 3.2 Reward / Advantage

**Reward 全部是 verifier 在轨迹结束时给一个标量，无模型 reward，无 shaping / 归一化 / 过程 reward。** 通用接口 `BaseVerifier`（`verifier/base_verifier.py:7-34`）：`VerifierInput{label, output_text, verbose, question, metadata}` → `VerifierOutput{score: float, infos}`。`MathEnv` 每步调 verifier，score 直接作 reward（`envs/math_env.py:38-53`）。`SearchEnv` 中间轮 reward 恒 `0.0`（`search_env.py:121`），结束时 verifier 给最终分（`:123-128`）。`BlackBoxEnv` 超时则 `score=0.0`（`blackbox_env.py:518-520`）。本轮未查到 learned RM / LLM-as-judge。

**GRPO group baseline 不在 trainer 里**：`GrpoTrainer.loss_func` 直接读 `batch["advantage"]`（`grpo_trainer.py:705`）。旧版 `GrpoController.normalize_group_rewards`（`controller/grpo_controller.py:364-407`）：`advantage=clip((score-mean)/(std+1e-6), -5, 5)`，`std<1e-8` 时置 0；新版 `PipelineController.normalize_group_rewards`（`pipeline/controller.py:867-905`）同款公式 + `finalize_packed_sample_batch_reward`（`:924-937`）写回 packed batch。

**GAE 只在 PPO 路径**：`prepare_ppo_training_samples`（`pipeline/controller.py:1310-1339`）先由 value worker 算 `old_values`，再 `build_terminal_token_rewards`（标量 reward 放到最后一个有效 token）+ `compute_gae`，`gamma/gae_lambda` 默认均为 1.0（`configs.py:526-527`）。`compute_gae` 在 `trainer/ppo_utils.py`，逆序递推。`ppo_utils.py` 另有 `normalize_over_valid_tokens_in_batch`（仅 PPO actor 侧）与 `clipped_value_loss_per_token`（`value_clip` 默认 0.2，由 `ValueTrainer.loss_func` 在 `value_trainer.py:77-83` 调用）。

过滤：`GroupFilterType = "pass" | "zero_std_all_fail" | "zero_std_all_success"`（`pipeline/rollout_data.py:14-18`），`filter_zero_std` 门控（`configs.py:583`）。

### 3.3 Rollout 调度

- `RolloutAgent.act`（`agent/rollout_agent.py:17-44`）极薄：从 `conversation.gen_state` 取 `session_id/input_ids/sampling_config/tools/tool_choice/tool_call_parser/capture_routing` 组装 `GenerationInput`（`:32-41`），调 `rollout_worker.generate`（`:43`），断言 `output_ids` 非空。**无多轮循环、无 tool 执行、无 context 管理**，多轮逻辑全在 env + 通用循环。
- 通用循环 `rollout_from_env`（`pipeline/utils.py:86-94`）：裸 `while True`，`agent.act → env.step`，`done` 即返回。**循环内无 turn 计数、无超时、无 max-turn 守卫**（见 §5 可疑点）。终止全靠各 env：`MathEnv` 单步 `done=True`（`math_env.py:41`）；`SearchEnv` 无 tool call / 超 `search_limit`（默认 5，`search_env.py:52`）/ 触答案 token 预算（`max_length - answer_token_reserve`，reserve 默认 128，`:63,208-209`）即 `_finish()`；`BlackBoxEnv` 收到 finish action 或 `len(outputs) >= max_model_calls`（默认 16）即结束（`blackbox_env.py:160-165`）。
- **Partial rollout**：`SGLangWorker.generate`（`sglang_worker.py:312-469`），`while True` 循环，每次把已产 `output_ids` 拼回 `input_ids`（`:355`）续生成；`finish_reason=="abort"`（权重更新打断）且 `continue_generation_after_abort` 为真则续写；`clear_partial_outputs_after_abort` 为真则丢弃已产 token 重来（`:398-402`）。预算由 `_compute_remaining_max_new_tokens` 控制，超预算则 `finish_reason="length"`。**该循环内未见显式最大重试次数 / 退避**（`retry` 只计数，`:405`），持续 abort 时理论上可无限循环，外层兜底未验证。
- 推理路由：`InferenceRouter`（`worker/infer_router.py:47`），同 `session_id` 优先路由到同一 worker 复用 KV cache，受 `max_imbalance` 约束（`:126-148`），idle 会话映射定期清理（`:161-179`）。

### 3.4 Token / Mask 处理

- `Sample`（`data/sample.py:23-52`）：`input_ids / labels / loss_mask / attention_mask / position_ids / reward / reward_baseline / advantage` + 可选 `rollout_logprobs / old_logprobs / ref_logprobs / teacher_logprobs / old_values / returns / turn_index / turn_reward / routing_handles_per_path / merge_info / trajectory_id`。`__post_init__`（`:54-68`）统一转 `i32/f32/bool`。
- `TokenTrace.to_last_turn_sample`（`data/token_trace.py:158-`）：`loss_mask` 只在最新一轮 assistant token 上为 True（`:177-185`），首 token 强制非 loss（`:185`），随后**左移一位对齐 label**（`_shift_left`，`:153-156,190-192`），padding 时右 pad：`input_ids` 补 `pad_token_id`、`labels` 补 `IGNORE_INDEX`、`loss_mask/attention_mask` 补 False、`turn_index` 补 -1（`:195-201`）。
- `Sample` 级 padding（`sample.py:80-111`，`_pad_sample_to`）：右 pad，`labels` 补 `IGNORE_INDEX`，`attention_mask/loss_mask` 补 False；**注意 `position_ids` 对 pad 位置继续 `arange(cur, max_length)`（`:96`），而非置 0/-1**，靠 `attention_mask=False` 使其不被读取（docstring `:81-84` 自述）。
- Truncation 三处不一致：训练轨迹超 `max_length` 直接 `assert` 失败（`rollout_trace.py:67,103,172`），不做静默截断；SFT 转换两次 `apply_chat_template(..., max_length, truncation=True)` 并断言完整 `input_ids` 以 `prompt_ids` 开头（`sft_sample_converter.py:44-61,73-75`），若截断破坏前缀则断言失败；Verifier 侧只看末 300 字符（见 §3.5）。
- 300+ turn 的实际支撑（代码中**无任何 `300` / `max_turn` 常量**，`search_code` 仅命中 README 与中文 README 的 Highlights）：Token 级轨迹（`RolloutTrace` 持 chunk 式 `TokenTrace`，`rollout_trace.py:36-68`；`TokenTrace` 按 `token_type` 分 chunk，`token_trace.py:109-136`）+ 每轮即时成样本（`rollout_trace.py:144-177`，`turn_samples` 声明 immutable，`:94-97`）+ Context 压缩 `compact()`（`:376-397`：摘要替换整条 live prompt，或保留最近 `max_recent_tool_results` 个 tool 结果、其余换成 placeholder，并切掉受影响的 R3 handle，`:427-462`）+ 打包切分 `to_packed_samples(max_pack_length)`（`:261-317`，贪心按 turn 合并，单 turn 超长直接 `raise ValueError`，`:299-302`）+ `pack_rollout_traces_for_train_batches` 打 `trajectory_id` 并断言打包前后可训练 token 总数不变（`:551-582`，`:581`）。`global_batch_iterator.py:1-22` 保证同一轨迹的 packed samples 必进同一 global batch。
- 前缀复用 / 合并：`prepare_generation_input` 计算新 prompt 与 live trace 的公共前缀 `_common_prefix_len`（`rollout_trace.py:543-548`），只保留该前缀的 routing handle；`to_sample()` 调 `merge_trajectory_samples` 合并全部 `turn_samples`（`:250-259`）；`to_packed_samples` 在 `allow_prefix_sharing=False` 时要求严格线性前缀链，否则拒绝切分（`:284-285,319-374`）。
- Prefix-tree merge：`utils/megatron/prefix_tree.py`：`MergingTree`（`:126`）+ `add_path` / `to_packed_sequence`、`extract_merge_info_from_batch`（`:264`）、`build_prefix_merge_info`（`:538`）、`merge_trajectory_samples`（`:620`）、`scatter_packed_to_batch`（`:511`）、`gather_merged_routing_per_path`（`:802`）。开关：`megatron_worker.use_magi_merged_forward`（`configs.py:458-464`，一次 `calc_attn`）与 `use_magi_flat_forward`（`:466-471`，互斥，bit-exact 基线）；`controller.allow_prefix_merging`（`pipeline/config.py:41`）与 `allow_prefix_sharing = allow_prefix_merging and use_magi_merged_forward`（`pipeline/controller.py:980`，`rollout_actor.py:143`）。

### 3.5 Verifier / Env（白盒与黑盒）

- Env 接口 `BaseEnv`（`envs/base_env.py:8-14`）：`step(action) -> (Conversation, float, bool, RolloutTrace|None, Any)`。`MathEnv`（`envs/math_env.py:18-36`）注入 `score_provider`（verifier）、`metric_calculator`、`conv_tokenizer`。
- Verifier 实现（全部规则 / 执行 / 外部库，无模型 reward）：`MathVerifier`（`verifier/math_reward.py:13-34`，取最后一个 `\boxed`，`is_equiv` 归一化比较，文件头注明 copied from verl）；`DapoVerifier`（`dapo_verifier.py:10-17`，调 `dapo_utils.compute_score(strict_box_verify=True)`，先截末 300 字符再验，`dapo_utils.py:236-266`）；`GSM8KVerifier`（`gsm8k.py:13-80`，只匹配末 300 字符内的 `#### <数字>`）；`HfMathVerifier`（`hf_math_verifier.py:14-45`，外部库 `math_verify`，timeout 30s，超时 / 异常返回 0.0，期间 `gc.disable()`）；`LeetCodeVerifier`（`blackbox_rl/leetcode_verifier.py:67-93`，E2B 沙箱跑测试，`_validate_e2b_only_runner`，`:168`）；`SearchR1Verifier` 评分细节未读到（SearchEnv 内用 `structure_format_score=0.0, retrieval_score=0.0` 构造），**未验证**。
- `leetcode_executor.unsafe_execute`（`verifier/leetcode_executor.py:44-86`）：在临时目录 `exec(prompt+completion+test+check(entry))`，`time_limit`（SIGALRM，`:89-102`）+ `reliability_guard`（禁用 `os.system/fork/chdir` 等、限内存，`:137-188`），黑名单式沙箱，绕过面未验证；直接调非 E2B 路径是否被封堵**未验证**。

### 3.6 OPD（On-Policy Distillation，代码自证非猜测）

- `OPDConfig` docstring 原文（`configs.py:246-247`）：“On-policy distillation settings shared by rollout annotation and training loss.” 配置（`:246-277`）：`enabled=False` 默认；`backend: sglang|megatron`；`opd_alpha=0.1`；`reverse_kl_clip=10.0`；`student_logprob_source ∈ {rollout_logprobs, old_logprobs, cur_logprobs}` 默认 `cur_logprobs`；Megatron 侧 `teacher_weight_name="opd_teacher_weights"`。挂在 `GrpoTrainerConfig.opd`（`configs.py:561`）。
- Rollout 侧标注 `annotate_sglang_teacher_logprobs`（`opd/teacher_logprobs.py:80-117`）：对每条 turn sample，用 teacher SGLang 服务发一次 `GenerationInput(..., max_new_tokens=0, capture_input_logprobs=True, ...)`，即**只做 prefill 打分、不生成**；`align_input_logprobs_to_sample`（`:51-77`）逐 token 校验 id 对齐、要求窗口只覆盖可训练 label、要求有限值，再写入 `sample.teacher_logprobs`。
- 训练侧 `apply_opd_to_advantages`（`trainer/grpo_trainer.py:488-567`）：`reverse_kl = (student - teacher).detach()`（`:525`），clamp（`:527-528`），`adjusted_advantages = (1-alpha)*advantages - alpha*reverse_kl`（`:530`）。**OPD 不是独立 KL loss 项，而是 advantage shaping**，detach 无梯度。指标输出 15+ 项（`:549-565`）。Megatron backend 标注路径未读到，**未验证**；与 `turn_reward_alpha` 的交互顺序**未验证**；每 turn 各发一次 teacher prefill 请求（`teacher_logprobs.py:109`，`asyncio.gather`），300 轮即 300 次请求，batch / 缓存优化与 teacher 不可用时降级**未验证**（`assert result.trace is not None`，`:85`，失败即 crash）。

### 3.7 `openai_proxy/`：黑盒 harness 捕获

- 职责切分：server **不直接调 SGLang、不打包模型响应**（`server.py:273-305` 类 docstring）；每个 rollout 一个会话 URL `http://host:port/sessions/{session_id}/v1`；controller 起一个共享 server + registry，各 env 不自建。
- Registry：`_OpenAIProxySessionRegistryCore`（`server.py:71-163`），每 `session_id` 一个 `asyncio.Queue`，`close_session` 时把排队请求以 499 fail（`:84-93`），默认 `request_timeout_seconds=600.0`（`:72`）；Ray actor 版 `_RemoteOpenAIProxySessionRegistry`（`:164-172`）使跨进程 / 节点共享。
- FastAPI 只暴露 3 路由（`:385-459`）：`GET /health`、`GET /sessions/{id}/v1/models`（返回假 model id `openai-proxy-session-{id}`）、`POST /sessions/{id}/v1/chat/completions`（原始 body + headers 存入 `OpenAIPendingRequest` 推入队列，等 env 侧 `respond`）；无 session 的 `/v1/chat/completions` 直接 404。鉴权仅当配置 `auth_token` 时校验 `Bearer`（`secrets.compare_digest`，`:461-478`），未配置则无鉴权，`auth_token` 配置来源 / 默认值**未验证**。
- Adapter（`chat_adapter.py`）：用 chat template 把 OpenAI messages 渲染成 prompt token id；**采样仍走 GRPO controller 的全局 `SamplingConfig`，harness 侧采样参数被忽略**（`:81-89`）；`convert_request`（`:117-143`）+ `build_response`（`:144-151`），用 SGLang 的 `FunctionCallParser` 解析 tool call（`:478-532`）；**不支持的字段直接拒绝**（`:258-305`：`stream=true`、`n != 1`、`logprobs`、`response_format`、`logit_bias`、`seed` 等），而非静默忽略；多模态 content 明确不支持（`:354`）；请求校验复用 `sglang.srt.entrypoints.openai.protocol.ChatCompletionRequest`（`:253-256`）。
- 闭环：`BlackBoxEnv` 7 步流程（`blackbox_env.py:48-59`）：建 proxy session → 启动 runtime → 等下一个模型请求 → 转 `GenerationInput` → 生成 → 打包成 OpenAI 响应放行 runtime → 停 runtime 并收集 verifier 文本。**“捕获”不是旁路日志，而是 harness 的每次模型调用都被同步替换为 SGLang 生成并记入同一 `RolloutTrace`，最终直接成训练样本。** `model_io_events`（`:568-613`）只见追加内存列表，落盘 / 消费处**未查到**，长轨迹无界增长内存的风险**未验证**。

### 3.8 `processor/`、`data/`、`datasets/`、`recipe/`、`metrics/`、`utils/`、`runner/`、`example/`

- Processor：`BaseProcessor`（`processor/base_processor.py:6-13`）仅 `process(item)`。`ProcessorPool`（`processor_pool.py:54-106`）是 `InferWorker`，用 `multiprocessing.spawn` 起子进程（`:67`），经 `SimpleQueue` 分发，`timeout_seconds=60.0` 默认（`:60`），超时后**重启对应 worker**（`:91-100`），tokenizer / verifier / adapter 等 CPU 处理器被隔离，超时可恢复。
- Token 化：`ConversationTokenizer`（`conversation_tokenizer.py:11-41`），`AutoProcessor.from_pretrained(use_fast=True)`，要求最后一条消息 `role=="user"`（`:26`），`apply_chat_template(..., add_generation_prompt=True)` + `processor(text, add_special_tokens=False)`。`AppendedMessageTokenizer`（`appended_message_tokenizer.py:18-101`）增量 token 化：用含 sentinel assistant 内容的最小前缀对话渲染，取 sentinel 之后的 suffix，并断言 suffix 首 token 为 assistant boundary token 后去掉（`:91-101`）。`chat_template_utils.py:20-41` 用 sentinel 探测 assistant 后缀，**断言恰好 1 个 token**；注释承认 Qwen3 模板会在后续 user 轮重写 assistant 历史、Qwen3.5 原生 tool 模板拒绝裸 `assistant->tool` 序列，故用 fake turn 做前缀。非 Qwen 模板是否断言失败**未验证**。
- Datasets：`BaseDataset`（`datasets/base_dataset.py:15-23`），内存持 `Conversation` / label / `score_history` / `length_history`（`_history_capacity=128`）；`_load_from_hf`（`:74-93`）支持 parquet/csv/json/jsonl；每行拼成 `system + user + " Please reason step by step, and put your final answer within \boxed{}."`（`:95-123`）；`filter_by_max_prompt_length`（`:52-72`）、`sample`（`:148-`，支持 `sample_type`）、`concat`（`:32-50`，断言 `conversation_id` 不重复）。注册表 `datasets/__init__.py:14-43`，内置仅 6 个：AIME-2024、DAPO-Math-17k、GSM8K train/test、FlashRAG NQ train/test。**`tests/` 下无 `datasets` 目录**。
- Recipe：`BaseRecipe`（`recipe/base_recipe.py:18-26`）要求 recipe 只保存 experiment config，禁止持有不可序列化对象；钩子 `run_rollout(conversation, runtime)`（`:70-81`）、`initialize_local_processors`（`:83-96`，每 rollout actor 初始化一次）、`initialize_shared_workers`（`:98-110`，每 experiment 一次）、`start_services/stop_services`（`:49-60`，仅 driver 侧）。真正可跑的 recipe 不在 `axrl/recipe/`，而在顶层 `axis_recipe/`。
- Metrics：`ResponseMetric`（`metrics/response_metric.py:14-49`）：token 数、unique 比、n-gram 重复、三种推理行为关键词（backtracking / verification / causal，**子串匹配**，`_has_key_words`，`:80-84`）、rollout 侧（cached tokens、retry、e2e 耗时、finish_reason 计数）、`is_low_quality`（阈值硬编码，`:52-58`）、`score` 等；`aggregate_response_metrics`（`:154-180`）按 `score>0/<=0` 分组；`ConversationMetricsStore`（`conv_metrics_store.py:20-74`）按 `(dataset, index)` 存最近 64 条；`report_mismatch.py`（`:83-109,244-261,283-`）对 `rollout_logprobs` vs `ref_logprobs` 算 KL / top-K / 按位置分析，画 3×3 图。`BaseMetric`（`base_metric.py:4-7`）仅为 `compute()` 抽象，未被上述类使用，调用方**未验证**。
- Utils（未全读）：`config_utils`（配置加载校验）、`kl_utils`、`tensor_store`（`TensorHandle`）、`timer`（`SessionTimer`）、`zst_utils` 等单文件；子包 `logger/`（console / buffer / metric / tb / wandb）、`megatron/`（`prefix_tree`、`magi_forward`、`router_replay` R3、`seqlen_balancing`、`sglang_weight_updater` 等）、`sglang/`、`hf/`、`visualization/`。
- Runner（与训练流水线无关，是 agent 环境沙箱）：`BaseRunner`（`runner/base_runner.py:13-30`）；`CgroupRunner`（`cgroup_runner.py:21`），cgroup 管进程树清理 + `ulimit -v`，docstring 明确“不是安全沙箱”（`:30-34`）；`E2BRunner`（`e2b_runner.py:40`），受限出站网络，模板默认 `"axrl-openhands"`（`:26`）。与训练模块平级放在 `axrl/runner/`，初读易误认为是训练 runner。
- Example：`config_examples.py:6-42` 给一个 `MegatronWorkerConfig` 示例；其余 `download_data.py` / `prepare_*` / `estimate_bubble.py` / `qwen_vl/` 未细读。

### 3.9 Checkpoint / 恢复

- Trainer 本身无 checkpoint 逻辑，全在 worker + controller 层：`MegatronWorker.save_checkpoint`（`worker/megatron_worker.py:456-466`）→ Megatron-Bridge `save_checkpoint`；`load_checkpoint`（`:782-796`）返回 `state.train_state.step`。`initialize_checkpoint_configs`（`:321-364`）：`load_rng=False`、`load_optim=not inference_only`。
- `BaseController.save_checkpoint`（`controller/base_controller.py:153-182`）：Megatron checkpoint + 额外把 `train_dataset._score_history` 存为 `checkpoint_extra_info/{step}.zst`，失败重试 5 次；可选 HF 格式（`:184-218`，保留最近 k 个 `step_*` 目录）。`load_checkpoint`（`:220-244`）：重试 3 次，并**断言** `extra_info` 文件存在（`:234`）；外来 / 旧 checkpoint 无该文件时会重试 3 次同样失败后才抛错。
- 梯度 spike 快照是独立目录 `spike_snapshots/`（`megatron_worker.py:366-376`），见 §3.10。

### 3.10 评估 / Benchmark / Spike replay

- Spike replay（`GradSpikeDebugConfig`，`configs.py:286-306`，默认关闭，`warmup_steps=10`，`spike_ratio=2.0`，`history_window=100`，`max_snapshots=3`）：`MegatronWorker._detect_spike`（`megatron_worker.py:511-`）在 `train_step` 中 grad norm 超标时触发；`_snapshot_spike_debug_info`（`:536-`）把 checkpoint + 输入 batch（`torch.save(batch, ...)`，`:559`）+ per-param grad norms + trainer（`torch.save(self.trainer, ...)`，`:579`）+ metadata + 路由 payload 存入 `spike_snapshots/iter_XXXXXXX/`；`reproduce_spike`（`:659-`）加载快照（`load_optim=False, load_rng=False`，`:701-702`）、恢复 batch 与路由 payload、重跑 forward-backward + clip、比对 grad norms。Ray 侧 `RayMegatronWorker.reproduce_spike`（`ray_megatron_worker.py:432-437`）返回 rank 0 结果。`torch.load(..., weights_only=False)` 出现在恢复路径（`megatron_worker.py:603,717`），加载任意 pickle，仅用于本地调试快照时可接受，需注意来源。
- Benchmark（全部需 GPU，不适合 CPU 验证）：`benchmark/run_memory_benchmark.sh`（Qwen3-30B-A3B，8×H200，14 个实验，recompute / offload 对比，指标从日志 grep `peak_mem_gbs` 等）；`run_parallelism_benchmark.sh`（23 个配置，约束 `DP*TP*PP*CP=8` 等，VPP 要求 TP=1 且 VPP≥4 失败）；`benchmark/magi_forward/`（Magi merged vs `gptmodel_forward` 对比，flat / merged sweep，`run_smoke.sh` 4 case 单 8 卡节点，`pressure_test.py` 4 节点 × 8 GPU = 32 GPU）；`benchmark/weight_update/`（colocated 权重同步基准 + FP8 对比，`warmup=1, measured=3`）。**待核实**：`run_memory_benchmark.sh` 引用的 `axis_recipe/moe/grpo_config.yaml` 在 `axis_recipe/` 目录清单（5 个配方，无 `moe/`）中不存在，可能是陈旧路径，未逐行核验。
- Skills（4 个，各仅 1 个 `SKILL.md`）：`axrl-fix-linting`（修 lint，若改动影响 logits / 训练语义则停下问用户）、`axrl-review-staged-changes`（只 review `git diff --cached`）、`axrl-run-regression-tests`（先 lint 再到 GPU 节点跑全量回归）、`axrl-spike-replay`（复现 / 分析梯度 spike 快照 + 5 个已验证 mitigation 方法 md）。**已验证的不一致**：回归 / spike 两个 skill 引用 `skills/axrl-run-gpu-task/SKILL.md` 与 `skills/axrl-copy-log-to-tmp-and-analyze/SKILL.md`，但 `skills/` 目录只有上述 4 个目录，这两个路径均返回不存在（纯文档类问题，不列为候选）。

### 3.11 测试覆盖（`tests/`，约 88 个 `test_*.py`）

- 重点覆盖：`data`（13 个，conversation / tokenization / packing / routing，`test_rollout_trace_packing.py` 28427 B 最大）、`mcore`（11 个，rollout-trace / MagiAttention / prefix-tree / 训练曲线一致性，`test_rollout_trace.py` 53190 B）、`routing_replay`（11 个）、`rollout_worker`（12 个，权重同步 / partial rollout / tool call）、`utils`（15 个，verifier / kl / pack / seqlen / routing）、`pipeline`（7 个）。
- **核心模块无专属测试文件**（按文件名对照，未逐函数 grep 引用）：`controller/grpo_controller.py`（47805 B）、`pipeline/controller.py`（94319 B，1795 行）、`trainer/grpo_trainer.py`（40178 B）/ `sft_trainer.py` / `value_trainer.py`（`tests/trainer/` 仅 losses / GAE / OPD / packed-batch 等价性 4 个文件）、`worker/megatron_worker.py`（72463 B）/ `sglang_worker.py`（39589 B）/ `hf_worker.py`（`tests/worker/` 仅 3 个文件）、`datasets/` 全部 5 个文件（无 `tests/datasets/`）、`agent/`、`recipe/base_recipe.py`、`metrics/`（无对应测试目录）、`utils/megatron/` 18 个文件（仅间接覆盖）、`runner/e2b_runner.py`（`tests/runner/` 仅 cgroup）、`openai_proxy/chat_adapter.py`（仅 1 个测试文件，分支覆盖未验证）。
- `tests/tokenizer/` 仅 `__init__.py`（0 B），**0 个测试文件**；`tests/processor/` 仅 `test_processor_pool.py`。
- `tests/conftest.py`（138 行）：每个模块必须自己 reap ray / sglang / megatron worker，否则泄漏毒害下一个测试；模块 teardown 断言无泄漏 worker，失败时最多拖 300 秒（`_LEFTOVER_KILL_RETRY_WAIT_SEC=60.0` × 5 次）。`tests/test_configs.py` 是共享 fixture / 配置而非 pytest 用例，含 `VPP testing is disabled until Megatron Bridge stage checks handle it` 注释。
- `scripts/run-tests.sh`（1962 B）：先跑 `download_data` + `prepare_conv_data`（下载模型 / 数据），再全量 `pytest tests`，无 `-x`、无 `--lf`，`flock` 单实例锁。本地复现前需预留磁盘与时间。

---

## 4. 设计判断：与同类框架相比的取舍

> 对比部分为基于 axrl 代码 + 同类框架公开定位的判断，非逐行对照 verl / slime / OpenRLHF 源码；具体到对方实现的断言标“未逐行对照”。

| 维度 | axrl 的取舍 |
|---|---|
| 做重了 | **Rollout-trainer 一致性**：mismatch 分析、routing replay（R3）、spike replay、prefix-tree 合并、MagiAttention、打包 token 守恒断言、SFT 前缀断言，整套防御体系是全仓最重的部分。**可观测性 / 可复现调试**：spike 快照（权重 + 优化器 + batch + 逐参 grad + 路由）→ replay 比对闭环，`skills/axrl-spike-replay` 收录 5 个已验证 mitigation 方法。**黑盒捕获**：OpenAI 兼容代理 + 会话路由 + E2B / OpenHands 闭环，把外部 harness 的每次调用同步替换为 SGLang 生成并记入同一 `RolloutTrace`。**多 objective 并列**：PPO / GRPO / GRPO2 / GSPO / TOPR / TIS / Kimi2.5 共用同一 `calculate_pg_loss_and_clip_ratio`，新增 objective 只加分支。 |
| 刻意没做 | **不做自己的 serving / training 引擎**：SGLang 与 Megatron 是核心引擎，axrl 只做系统层契约（README / 博客自述）。**无 learned reward**：无 RM / LLM-as-judge，全部规则 / 执行式 verifier。**无 vLLM 路径**：`RolloutWorker.get_worker` 只支持 `engine_type=="sglang"`，其余抛错（`rollout_worker.py:23-29`）。**无自己的数据集生态**：内置仅 6 个数据集注册。**无分布式训练器抽象**：训练后端统一 Megatron，无 HF Trainer / FSDP2 等第二后端。**无持续维护的社区治理**：无 CONTRIBUTING / 模板 / DCO / CLA，CI 不跑测试。 |
| 与 verl 的差异（判断） | verl 是 Ray + HF / vLLM 的一体化单框架，数据流与训练器耦合更紧；axrl 是“Megatron 训练 + SGLang rollout + 轻 driver + handle 数据面”的分层架构，driver 只做调度 / 元数据，大 payload 走 handle 按需读。axrl 的一致性防御（R3 / mismatch / spike replay）比 verl 的常规测试更重；但 verl 的生态（多后端、数据集、社区）远大于 axrl。未逐行对照 verl 源码。 |
| 与 slime / OpenRLHF 的差异（判断） | slime 同样 SGLang + Megatron 路线，axrl 的差异化在黑盒代理捕获 + OPD advantage shaping + spike replay 闭环；OpenRLHF 是 Ray + vLLM / DeepSpeed 路线，axrl 与之引擎选型完全不同（SGLang + Megatron + MagiAttention）。未逐行对照对方源码。 |

### 最值得学的 2-3 个设计点

1. **Handle-based 数据面 + 前缀树合并 + token 守恒**：`TensorHandle`（`tensor_store.py:28-`）行切片、`rollout_actor.py:138-179` 打包 + `ray.put` + strip heavy payload、`prefix_tree.py` 合并、`rollout_trace.py:581` 打包前后可训练 token 总数不变的断言、`global_batch_iterator.py:1-22` 同一轨迹必进同一 global batch。长轨迹切分后梯度更新数与切分数无关，是“300+ turn 能跑”的真正机制。
2. **Spike snapshot → replay 可复现调试闭环**：`GradSpikeDebugConfig`（`configs.py:286-306`）+ `megatron_worker.py:536-` 快照 + `reproduce_spike`（`:659-`）重跑比对 + 5 个已验证 mitigation 方法收录进 skills。把“训练 spike”从玄学变成可复现实验。
3. **OPD 做 advantage shaping 而非另加 loss + 黑盒代理职责切分**：OPD `(1-α)·adv − α·reverse_KL`（`grpo_trainer.py:524-530`），detach 无梯度，行为可解释且易开关；标注侧用 `max_new_tokens=0` 的 prefill 打分（`teacher_logprobs.py:97-105`），不浪费生成算力。代理 server 只做会话路由 + 请求停车（`server.py:273-305`），env 拥有生成 / 打包 / 记 trace；不支持的 OpenAI 字段显式拒绝而非静默忽略（`chat_adapter.py:258-305`）。

---

## 5. 最可疑的设计 / 实现点（带证据与推理）

> 勘误（2026-10-28）：原疑点 1 经逐行核验为误报，已删除。错在把“mask 变量”和“过滤机制”混为一谈：`policy_loss_mask` 确实只进 `calculate_metrics`（`grpo_trainer.py:781`），但过滤走的是乘法因子 `policy_mismatch_tir = token_mismatch_tir * seq_mask * icepop_token_mask`（`:754`），它在 `:756-762` 传给全部 6 个 loss 分支，每个分支在聚合前都乘了它（grpo `:225`、tis `:256`、kimi2_5 `:286`、topr `:327`、gspo `:372`、grpo2 `:446`），被 mask 的 token PG 梯度为 0。上游自带测试也断言了这一点：`test_grpo_icepop_masks_policy_loss_and_logs_mask_ratios` 断言 icepop 下 loss 为 -1/3、denom 为 3；`test_grpo_sequence_mask_only_gates_policy_gradient_not_kl` 断言序列 mask 只 gate PG 不 gate KL。**不要以“mask 被忽略 / no-op”为由提 issue。** 真实但小得多的两点：① 6 个 mismatch/icepop 开关默认均为 None，5 个 shipped 配方也全部设 null，即默认关闭，需手动开启；② 被 mask 的 token 仍计入 PG 聚合分母与返回的 micro-batch denom（上游测试明确断言此语义），是分母稀释而非硬 mask，KL / entropy 按注释有意保留原始 mask。本地无 torch，数值探针未跑，结论基于逐行静态核验 + 上游测试断言。

1. **无权重版本号 / fence + abort 重试无上限 + 通用 rollout 循环无兜底，三处“靠调用时序保证正确”。** 权重同步只断言 `is_paused()`（`sglang_worker.py:778,788`），已读文件未见 weight version / step tag / fence；`SGLangWorker.generate` 的 abort 重试循环无显式最大重试 / 退避（`sglang_worker.py:339-405`）；`rollout_from_env` 是裸 `while True`，无 max-turn / 超时 / token 预算（`pipeline/utils.py:86-94`），任何新 env 忘记设 `done` 即无限循环。单点上各自有道理（driver 时序、预算、外层 env 兜底），但三处叠加意味着新 env / 新 placement 出错时失败模式是静默错位或无限循环，而非 loud failure。外层是否有兜底**未验证**。
2. **两套 controller 并存 + 阈值不一致 + checkpoint 强断言。** 旧 `controller/grpo_controller.py`（断言 `colocated=True`）与新 `pipeline/controller.py` 并存，各有独立的 group 归一化 / 过滤实现，易漂移。阈值不一致：`get_filter_type` 用 `reward_std > 1e-2` 判定 pass（`grpo_controller.py:352`），而 `normalize_group_rewards` 用 `std < 1e-8` 置零（`:391`），中间地带（`1e-8 < std < 1e-2`）会算出被 `/(std+1e-6)` 放大的 advantage（`clip ±5` 兜底）。`BaseController.load_checkpoint` 对 `extra_info` 文件强断言（`base_controller.py:234`），外来 / 旧 checkpoint 无该文件时会重试 3 次同样失败后才抛错。
3. （补充，次优先级）**OPD 配置校验与“controller 后填”矛盾**：`configs.py:256-259` 注释说 `sglang_host/port` 由 controller 在启动前填写，但 `model_post_init`（`:267-274`）在 `enabled + backend==sglang` 时立即断言 `sglang_port is not None`。若配置在 controller 填写前就被校验会直接失败；若 controller 先改字段后重建则无事。Controller 实际填写时序**未验证**。另有小瑕疵：`MathEnv` 留有 `random.random() < 0.0001` 的 `pprint` 调试（`math_env.py:48-51`）；`response_metric.py` 关键词子串匹配、`is_low_quality` 阈值硬编码、n-gram 调用 `n=20` 与函数默认 `n=5` 不一致（`:136` vs `:86`），只影响观测不影响 reward。

---

## 6. 潜在贡献切入点（0 open issue 下从代码找活，按价值 / 风险排序）

搜索口径：`TODO` 全仓仅 1 处（`pyproject.toml` ruff 注释，无代码 TODO）；`FIXME:` 0；`HACK` 0；bare `except:` 0；`NotImplementedError` 15 处中 14 处为基类抽象方法（已排除），仅 1 处为候选类。`except Exception` 49 处，多数带日志 / 重抛 / 降级，属正常模式。

| # | 候选 | 问题描述 | 证据 | 影响范围 | 验证方式 | 需 GPU | 需 maintainer 语义确认 |
|---|---|---|---|---|---|---|---|
| 1 | `MathVerifier.compute_score` 用 `print` 吞异常并静默返回 0 | `compute_score()` 在 `try` 失败时 `retval` 保持 `0.0` 并返回，异常只 `print(e)` 到 stdout，无结构化日志、无异常类型、无输入上下文；0 分与“答案错误”不可区分，可能掩盖解析 / 等价判断 bug | `axrl/verifier/math_reward.py:31-32`（结合 `:23-34`）；同文件 `:51-52`、`:108-109`、`:138-139` 也有异常回退，但 `:31-32` 是唯一决定 reward 为 0 的位置 | Reward 计算路径，大规模训练日志中 `print` 易被淹没 | CPU 即可。构造触发异常的 `solution_str` / `ground_truth`（如畸形 `\boxed`），断言改动后仍返回 0 且走 `logger.exception` / `logger.warning`；跑 `tests/utils/test_math_verifier.py` + `ruff check` | 否 | **是**——需确认“解析异常一律记 0 分”是既定语义（只改日志通道），还是希望区分“解析失败”与“答案错误”；未确认前不要改返回值。文件头注明改编自 verl 的 `math_reward.py`，改动时保持最小 diff |
| 2 | SGLang 健康检查轮询 `except httpx.HTTPError: pass` 无任何记录 | 循环内每次失败都静默 `pass`，只在最终超时才抛 `TimeoutError`；连接拒绝 vs 超时 vs TLS / 协议错误的早期信号在日志中完全缺失 | `axrl/utils/sglang_launch_utils.py:514-520`（最终超时 `:521`） | 调试 SGLang 启动失败，可观测性增强，不改行为 | CPU 即可（mock `httpx.AsyncClient.get` 抛 `httpx.ConnectError`，断言超时前有日志且仍抛 `TimeoutError`）；跑 `tests/utils/test_sglang_launch_utils.py` | 否（轮询逻辑与 GPU 无关） | 否（不改重试 / 超时语义），但需确认是否接受新增日志量（建议首次失败记一次 debug / 每 N 次记一次，避免刷屏） |
| 3 | `get_ray_status_metrics` 吞异常且无日志 | `except Exception` 后无任何 `logger` 调用，异常类型与堆栈丢失；调用方只看到 `ray_status_query_failed=1.0`，无法区分 Ray 未就绪 / 权限问题 / API 变化 | `axrl/utils/system_utils.py:271-276` | 监控 / 诊断 metrics，范围小 | CPU 即可。mock `ray` 为 `is_initialized()=True` 但 `cluster_resources()` 抛异常的 fake，断言返回值不变且记录一条日志（`caplog`）；跑 `tests/utils/test_system_utils.py` | 否 | 否（仅补日志，不改返回值），但需确认日志级别（`debug` vs `warning`，避免高频轮询刷屏） |
| 4 | 为无测试的纯 CPU 工具加测试（以 `processor` / `tokenizer` 为例） | `tests/tokenizer/` 仅 `__init__.py`（0 B），0 个测试文件；`tests/processor/` 仅 `test_processor_pool.py`；而 `axrl/processor/` 有 `text_tokenizer.py`、`chat_template_utils.py`、`conversation_tokenizer.py`、`appended_message_tokenizer.py`、`text_decoder.py` 等，正是 README 强调的“tokenization / chat template 差异会导致 rollout-trainer 不一致”的高风险区 | 目录 listing + `axrl/processor/` 文件清单 | 小到中。分词 / 模板 / 解码边界，纯 CPU 可测 | CPU 即可。先写最小复现：用 fake tokenizer / 固定字符串断言输出；跑 `pytest tests/processor tests/tokenizer -vv` + `ruff check`。若被测函数需真实 tokenizer 文件下载，则降级为“需 maintainer 确认 fixture 策略” | 否 | 部分——需确认接受为这些工具补单测的测试风格（fixture 用 fake 还是真实小 tokenizer），以及测试放在 `tests/processor/` 还是 `tests/tokenizer/` |

**明确排除：** `NotImplementedError` 其余 14 处均为基类抽象方法（`base_env` / `base_agent` / `base_metric` / `worker/*` / `base_trainer` / `base_verifier` / `base_processor` / `base_recipe` / `base_controller` / `base_dataset` / `weight_update` 等），属设计非缺失；`pipeline/controller.py:410` 的 `NotImplementedError` 与 `:192` 白名单一致，属显式拒绝非法模式。纯文档类（缺失 skill 引用、benchmark 脚本硬编码 `PROJECT_NAME`、benchmark 硬编码模型 / 硬件、`configs.py` 默认 `master_addr/port`）按任务要求排除。`print(` 全仓 40 处，多数在 `__main__` 示例 / 基准 / 脚本中，与候选 1 的库内 reward 路径性质不同，不重复列。另有次选小项（未进前 4）：`retrieval_server.py:59-74` 的 `pooling()` 非法 `pooling_method` 的 `NotImplementedError` 不带入参值，CPU 可验证，但需确认保留 `NotImplementedError` 还是改为 `ValueError`，价值低。

**贡献策略提醒：** 0 issue + 82 天零维护 + 无 CONTRIBUTING / 模板 / DCO / CLA，提 PR 前无法通过 issue 对齐语义。候选 1 / 2 / 3 是“只加日志、不改行为”的安全型，适合首个 PR。另需注意：`benchmark/` 与 `magi_forward/pressure_test` 均需 8–32 GPU，外部贡献者无法验证，不要碰。

---

## 7. 陷阱与注意事项

1. **Fork main 是干净镜像，无个人笔记 commit。** Fork 9 个 commit 全为上游作者（Junjie Li、Jinyu Wang），无个人笔记 commit；fork 根目录无 `notes/`、`paper/` 等个人笔记目录。**结论：后续 contrib 分支可直接从上游 main / fork main 切出，无需先剥离个人笔记。但个人笔记绝不能进上游 PR diff——本文只在 fork 的 `notes/learning` 分支，未来 contrib 分支必须从上游 main 起，提交前核对 diff 文件列表。**
2. **“72 天前更新”是误读，实际是 82 天零维护。** 最后提交 2026-07-24，之后零代码提交、零 issue、零 PR。`updated_at=2026-10-13` 是仓库元数据，不是提交。不要在文档中写“72 天”或“10-13 有更新”。
3. **CI 不跑测试，`ruff` 通过 ≠ 测试通过。** CI 仅 ruff + shellcheck，无 pytest / mypy。本地必须跑 `pytest` + pre-commit。`scripts/run-tests.sh` 会先下载数据 / 模型且跑全量测试，需预留磁盘与时间；`conftest.py` 要求每个模块自己 reap ray / sglang / megatron worker，否则 teardown 失败且最多拖 300 秒。
4. **路径陷阱：** `tests/magi-forward/` 目录名带连字符，写导入 / 路径时需用字符串路径而非点号导入。`benchmark/run_memory_benchmark.sh` 引用的 `axis_recipe/moe/grpo_config.yaml` 在 `axis_recipe/` 清单中不存在，可能是陈旧路径，引用前需核验。`search_r1/run_train.sh` 不透传 `"$@"`，CLI 覆盖在该配方单机脚本中无效（与 grpo/ppo 不同）。
5. **两套 controller 并存，读代码时先确认走哪套。** 旧 `controller/grpo_controller.py`（断言 `colocated=True`）vs 新 `pipeline/controller.py`；新配方均走 pipeline 侧。`axrl/recipe/` 只有 `BaseRecipe` 抽象基类，真正可跑的 recipe 在顶层 `axis_recipe/`。`axrl/runner/` 是 agent 环境沙箱，不是训练 runner。
6. **Blackbox 配方是 WIP，默认 `run_mode=eval_only`，checkpoint 默认关闭。** 不要把它当成已验证的训练路径写进文档。`OpenHandsEnv` 的 `max_model_calls` 默认 16，与“300+ turn”不在同一量级，黑盒长任务需显式调大。
7. **“300+ turn / 数百 B 参数”仅为 README / 博客主张，无可验证数据。** 文档中必须标注为“官方主张，未附可验证 benchmark”，不要当成实测结论。`SearchR1Verifier` 评分细节、`get_kl_loss` 内部 kl_type、各 loss 分支对 `token_mismatch_tir` 的消费、OPD Megatron backend 路径、权重同步外层兜底等均**未验证**，写文档时标注清楚，不要臆测填补。
8. **背景（`browser.search`，仅作背景，非一手证据）：** `xyz-lab.ai` 有《AI4AI at Scale》博客（XYZ Agentic Team，显示日期 2026-07-24），介绍有界探索、验证门控的完整 agentic 系统，并在 XYZ-Aquila（基于 Qwen3.6-35B-A3B 与 Qwen3.5-397B-A17B 的 Deep Search agent 家族）上实例化；README 亦链接 `xyz-lab.ai` 与该博客。第三方日报称 axrl 为 7-23 创建的新项目，定位长程 Agent RL 后训练框架。论文数字与星数未用一手 API 复核，仅为搜索摘要转述。
