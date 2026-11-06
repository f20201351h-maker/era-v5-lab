"""Tests for the RL loop: advantages and losses on hand-worked numbers, the frozen router, the model adapters, and
end-to-end smoke runs where the held-out reward must rise with the router checksum unchanged. CPU only, no network.

Run:  python -m pytest teams/10-post-training/rl-loop/test_rl_loop.py -q -p no:cacheprovider
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grpo  # noqa: E402
import router  # noqa: E402
import tasks  # noqa: E402
import train  # noqa: E402
from policy import completion_logprobs  # noqa: E402


# --- advantages (worked by hand) ---------------------------------------------------------------

def test_advantages_on_hand_worked_numbers():
    r = torch.tensor([1.0, 0.0, 0.0, 1.0])  # mean 0.5, population std 0.5
    g = grpo.advantages(r, "grpo")
    assert torch.allclose(g, torch.tensor([1.0, -1.0, -1.0, 1.0]) * 0.5 / (0.5 + 1e-4))
    assert torch.allclose(grpo.advantages(r, "dr_grpo"), torch.tensor([0.5, -0.5, -0.5, 0.5]))
    # rloo: r_i minus the mean of the other three. For r_1: 1 - (0+0+1)/3 = 2/3
    assert torch.allclose(grpo.advantages(r, "rloo"), torch.tensor([2 / 3, -2 / 3, -2 / 3, 2 / 3]))
    assert torch.allclose(grpo.advantages(r, "gspo"), g)


def test_a_group_with_equal_rewards_has_no_signal_and_is_dropped():
    for algo in grpo.ALGOS:
        assert grpo.advantages(torch.tensor([1.0, 1.0, 1.0]), algo) is None
        assert grpo.advantages(torch.tensor([0.0, 0.0]), algo) is None


def test_unknown_algo_is_an_error():
    with pytest.raises(ValueError):
        grpo.advantages(torch.tensor([1.0, 0.0]), "ppo2")


# --- losses (worked by hand) -------------------------------------------------------------------

def test_token_ratio_is_clipped_for_a_positive_advantage():
    # one sequence, one token, new/old probability ratio 2, advantage 1, clip 0.2: min(2*1, 1.2*1) = 1.2
    new, old = torch.tensor([[math.log(0.4)]]), torch.tensor([[math.log(0.2)]])
    loss, st = grpo.policy_loss(new, old, torch.tensor([1.0]), torch.ones(1, 1, dtype=torch.bool), "grpo", 0.2)
    assert loss.item() == pytest.approx(-1.2) and st["clip_frac"] == 1.0


def test_gspo_uses_one_sequence_ratio():
    # token log-ratios log 2 and 0 -> sequence ratio exp(log2 / 2) = 1.414, clipped to 1.2 for A = 1
    new = torch.tensor([[math.log(2.0), 0.0]])
    loss, _ = grpo.policy_loss(new, torch.zeros(1, 2), torch.tensor([1.0]), torch.ones(1, 2, dtype=torch.bool), "gspo", 0.2)
    assert loss.item() == pytest.approx(-1.2)
    loss, _ = grpo.policy_loss(new, torch.zeros(1, 2), torch.tensor([1.0]), torch.ones(1, 2, dtype=torch.bool), "gspo", 0.5)
    assert loss.item() == pytest.approx(-math.sqrt(2.0), rel=1e-5)


def test_masked_tokens_do_not_count():
    new = torch.tensor([[0.0, 5.0]])  # the second token is after EOS: its huge log-ratio must not matter
    m = torch.tensor([[True, False]])
    for algo in ("grpo", "gspo"):
        loss, _ = grpo.policy_loss(new, torch.zeros(1, 2), torch.tensor([1.0]), m, algo, 0.2)
        assert loss.item() == pytest.approx(-1.0)


def test_dr_grpo_has_no_length_normalisation():
    # same per-token surrogate, but a longer completion contributes more (sum / constant), unlike grpo
    new, old = torch.zeros(2, 4), torch.zeros(2, 4)
    adv = torch.tensor([1.0, 1.0])
    m = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
    loss_dr, _ = grpo.policy_loss(new, old, adv, m, "dr_grpo")
    loss_g, _ = grpo.policy_loss(new, old, adv, m, "grpo")
    assert loss_dr.item() == pytest.approx(-(4 + 1) / 8) and loss_g.item() == pytest.approx(-1.0)


def test_kl_is_zero_when_the_policy_equals_the_reference():
    lp = torch.tensor([[-1.0, -2.0]])
    _, st = grpo.policy_loss(lp, lp, torch.tensor([1.0]), torch.ones(1, 2, dtype=torch.bool), "grpo", 0.2, lp, 0.1)
    assert st["kl"] == pytest.approx(0.0)


# --- frozen router ----------------------------------------------------------------------------

def tiny_olmoe():
    from transformers import OlmoeConfig, OlmoeForCausalLM
    torch.manual_seed(0)
    return OlmoeForCausalLM(OlmoeConfig(vocab_size=16, hidden_size=32, intermediate_size=32, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=4, num_experts=4,
                                        num_experts_per_tok=2, max_position_embeddings=64, eos_token_id=None,
                                        bos_token_id=None, pad_token_id=None))


def _train_steps(model, steps=3):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    ids = torch.randint(0, 16, (4, 6))
    for _ in range(steps):
        opt.zero_grad()
        model(input_ids=ids).logits.pow(2).mean().backward()
        opt.step()


def test_frozen_router_keeps_its_checksum_and_the_rest_trains():
    m = tiny_olmoe()
    names = router.freeze_router(m)
    assert names == ["model.layers.0.mlp.gate.weight", "model.layers.1.mlp.gate.weight"]
    assert not any("experts" in n for n in names)  # experts.gate_up_proj must not match
    before = router.router_checksum(m)
    other_before = m.model.layers[0].self_attn.q_proj.weight.detach().clone()
    _train_steps(m)
    assert router.router_checksum(m) == before
    assert not torch.equal(m.model.layers[0].self_attn.q_proj.weight, other_before)


def test_a_trainable_router_is_caught():
    m = tiny_olmoe()
    before = router.router_checksum(m)
    with pytest.raises(AssertionError):
        router.assert_router_frozen(m)
    _train_steps(m)
    assert router.router_checksum(m) != before  # the check can see a change


def test_a_model_without_a_router_is_refused():
    with pytest.raises(ValueError):
        router.freeze_router(nn.Sequential(nn.Linear(4, 4)))


def test_attention_only_training_keeps_router_and_experts_frozen():
    m = tiny_olmoe()
    n = train.set_trainable(m, [r"\.self_attn\."])
    router.freeze_router(m)
    trainable = [k for k, p in m.named_parameters() if p.requires_grad]
    assert n > 0 and trainable and all(".self_attn." in k for k in trainable)


def test_checksum_works_for_bf16():
    m = tiny_olmoe().to(torch.bfloat16)
    assert len(router.router_checksum(m)) == 64


# --- adapters ---------------------------------------------------------------------------------

def test_completion_logprobs_match_a_direct_computation():
    policy, _ = train.build_model({"kind": "tiny_olmoe", "config": {"vocab_size": 16, "hidden_size": 32,
                                   "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 4,
                                   "num_key_value_heads": 4, "num_experts": 4, "num_experts_per_tok": 2,
                                   "max_position_embeddings": 64}}, "cpu")
    seq, cmask = policy.sample([1, 2, 3], k=3, max_new=4, temperature=1.0, generator=torch.Generator().manual_seed(0))
    assert seq.shape == (3, 7) and cmask.shape == (3, 4) and cmask.all()
    lp = completion_logprobs(policy, seq, 3)
    logits = policy.model(input_ids=seq).logits
    want = torch.log_softmax(logits[:, 2:-1].float(), -1).gather(-1, seq[:, 3:, None]).squeeze(-1)
    assert torch.allclose(lp, want, atol=1e-5)


def test_toy_rules():
    t = tasks.ToyTask(vocab=16, offset=3, n_train=4, n_heldout=2)
    p = [1, 2, 5]
    assert t.reward(p, [8]) == 1.0 and t.reward(p, [9]) == 0.0 and t.reward(p, []) == 0.0
    t = tasks.ToyTask(vocab=50304, rule="parity", n_train=4, n_heldout=2)
    assert t.reward(p, [7]) == 1.0 and t.reward(p, [8]) == 0.0


def test_reward_env_task_maps_each_prompt_to_its_env():
    class Tok:
        def apply_chat_template(self, msgs, add_generation_prompt, tokenize):
            return [ord(c) % 50 for c in msgs[0]["content"]]

        def decode(self, ids, skip_special_tokens):
            return "".join(chr(i + 60) for i in ids)

    seen = []
    t = tasks.RewardEnvTask(Tok(), [("code/a", "first"), ("code/b", "second")], [("code/c", "third")],
                            reward_fn=lambda env_id, text: (seen.append((env_id, text)) or (1.0, {})))
    assert t.reward(t.train[1], [1, 2]) == 1.0 and seen == [("code/b", "=>")]


# --- end to end ------------------------------------------------------------------------------

def _smoke(config: str, tmp_path, **train_overrides):
    cfg = yaml.safe_load((HERE / "config" / config).read_text())
    cfg.update(out_dir=str(tmp_path / "out"), verbose=False)
    cfg["train"].update(train_overrides)
    return train.train(cfg)


def test_smoke_tiny_olmoe_heldout_reward_rises_with_the_router_frozen(tmp_path):
    for algo in grpo.ALGOS:  # a loop, not parametrize: this repo's pr-guard flags "+@pytest..." lines as emails
        s = _smoke("smoke_tiny_olmoe.yaml", tmp_path / algo, algo=algo)
        assert s["router_unchanged"], algo
        # starts near 1/16. With the batched sampler (5 Nov) the gains over seeds 1234 and 7 were +0.39 to +0.53,
        # except dr_grpo on seed 7 at +0.107, so +0.1 is the bar but that case is close to it
        assert s["heldout_reward_end"] >= s["heldout_reward_start"] + 0.1, (algo, s["heldout_curve"])
        assert (tmp_path / algo / "out" / "metrics.jsonl").exists() and (tmp_path / algo / "out" / "summary.json").exists()


def test_smoke_team5_model_heldout_reward_rises_with_the_router_frozen(tmp_path):
    if not (train.TEAM5_HARNESS / "week2" / "model.py").exists():
        pytest.skip("Team 5 harness not in this checkout")
    s = _smoke("smoke_team5_tiny.yaml", tmp_path)
    assert s["router_params_frozen"] == ["blocks.0.moe.router.weight", "blocks.1.moe.router.weight"]
    assert s["router_unchanged"] and s["heldout_reward_end"] >= s["heldout_reward_start"] + 0.2


def test_hf_checkpoint_path_loads_and_trains(tmp_path):
    tiny_olmoe().save_pretrained(tmp_path / "ckpt")
    cfg = yaml.safe_load((HERE / "config" / "smoke_tiny_olmoe.yaml").read_text())
    cfg.update(out_dir=str(tmp_path / "out"), verbose=False, model={"kind": "hf", "path": str(tmp_path / "ckpt")})
    s = train.train(cfg)
    assert s["router_unchanged"] and s["heldout_reward_end"] > s["heldout_reward_start"]
