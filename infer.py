# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

import pandas as pd

SYSTEM_PROMPT_TASK1 = (
    "You are an expert translator specializing in Chinese idioms, proverbs, "
    "and cultural expressions. Translate the following Chinese expression "
    "into natural, idiomatic English. If there is an equivalent English "
    "proverb or idiom, use it. Otherwise, provide a clear paraphrase."
)

SYSTEM_PROMPT_TASK2_DISC = (
    "You are an expert in Chinese-English non-literal translation. Given a "
    "Chinese expression and several English candidate translations, classify "
    "EACH candidate independently as either `gold` or `silver`. `gold` = a "
    "commonly accepted equivalent (idiom/proverb-style or standard equivalent) "
    "that SHOULD be selected; `silver` = an explanatory paraphrase or a "
    "non-equivalent that should NOT be selected. For every option letter, "
    "output exactly one line in the format `LETTER: gold` or `LETTER: silver`, "
    "following the given order."
)


def build_task1_user(zh: str) -> str:
    return f"Chinese: {zh}\nEnglish translation:"


def build_task2_user(zh: str, options: List[Tuple[str, str]]) -> str:
    options_str = "\n".join(f"{letter}. {text}" for letter, text in options)
    return (
        f"Chinese expression: {zh}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Classify each option as gold or silver."
    )


# ============ 输出解析工具 ============
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# 逐行匹配 "LETTER: gold/silver"（LETTER 可能是 A..Z / AA / AB ...）
LABEL_LINE_RE = re.compile(r"([A-Za-z]+)\s*[:：]\s*(gold|silver)", re.IGNORECASE)


def strip_think(text: str) -> str:
    if not text:
        return ""
    text = THINK_BLOCK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def parse_task1_output(text: str) -> str:
    """task1：取第一非空行作为译文。"""
    text = strip_think(text)
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


def parse_task2_output(text: str, options: List[Tuple[str, str]]) -> str:
    """
    变体 B 解析：从模型输出抽取每个 LETTER 的 gold/silver 判定，
    再按 CSV 选项顺序聚合所有 gold 字母（逗号拼接）。
    只认属于当前题目的合法选项字母；解析失败则回退到第一个选项。
    """
    text = strip_think(text)
    valid = [l for l, _ in options]
    verdict: dict[str, str] = {}
    for m in LABEL_LINE_RE.finditer(text):
        letter = m.group(1).upper()
        label = m.group(2).lower()
        if letter in valid and letter not in verdict:
            verdict[letter] = label

    gold = [l for l in valid if verdict.get(l) == "gold"]

    if not gold:
        fallback = valid[0] if valid else "A"
        print(
            f"[infer][warn] task2 no gold parsed, raw={text!r}, "
            f"valid={valid}, fallback={fallback}"
        )
        return fallback
    return ",".join(gold)


def load_task2_options(test2_csv: str):
    """返回 (ordered_ids, id2zh, id2opts)；opts 按 CSV 列顺序（A,B,...,Z,AA,...）。"""
    df = pd.read_csv(test2_csv, dtype=str)
    letter_cols = [c for c in df.columns if c not in ("ID", "chinese")]
    ordered_ids: list[int] = []
    id2zh: dict[int, str] = {}
    id2opts: dict[int, list[tuple[str, str]]] = {}
    for _, row in df.iterrows():
        sid = int(row["ID"])
        ordered_ids.append(sid)
        id2zh[sid] = str(row["chinese"]).strip()
        opts = []
        for letter in letter_cols:
            cell = row.get(letter)
            if cell is None or (isinstance(cell, float) and pd.isna(cell)):
                continue
            t = str(cell).strip()
            if not t or t.lower() == "nan":
                continue
            opts.append((letter, t))
        id2opts[sid] = opts
    return ordered_ids, id2zh, id2opts


# ============ 模型封装 ============
class Generator:
    def __init__(self, model_path: str, backend: str, tp_size: int):
        self.backend = backend
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if backend == "sglang":
            import sglang as sgl
            print(f"[infer] loading {model_path} with sglang (tp={tp_size}) ...")
            self.llm = sgl.Engine(
                model_path=model_path,
                tp_size=tp_size,
                dtype="bfloat16",
                trust_remote_code=True,
                skip_tokenizer_init=False,
            )
        else:
            import torch
            from transformers import AutoModelForCausalLM
            print(f"[infer] loading {model_path} with transformers ...")
            self.torch = torch
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=torch.bfloat16, device_map="auto",
            ).eval()

    def _template(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    def generate_batch(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """贪心解码批量生成。"""
        if self.backend == "sglang":
            outs = self.llm.generate(
                prompts,
                {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": max_new_tokens},
            )
            res = []
            for o in outs:
                if isinstance(o, dict):
                    res.append(o.get("text", "") or "")
                else:
                    res.append(getattr(o, "text", "") or "")
            return res
        else:
            res = []
            for p in prompts:
                inputs = self.tokenizer(p, return_tensors="pt").to(self.llm.device)
                with self.torch.no_grad():
                    out = self.llm.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    )
                res.append(self.tokenizer.decode(
                    out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                ))
            return res

    def build_prompt_task1(self, zh: str) -> str:
        return self._template(SYSTEM_PROMPT_TASK1, build_task1_user(zh))

    def build_prompt_task2(self, zh: str, options) -> str:
        return self._template(SYSTEM_PROMPT_TASK2_DISC, build_task2_user(zh, options))

    def shutdown(self):
        if self.backend == "sglang":
            try:
                self.llm.shutdown()
            except Exception:
                pass


# ============ 主流程 ============
def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=" HF 的模型路径")
    ap.add_argument("--test1", default=os.path.join(here, "DatasetB", "DatasetB_test_task1_public.csv"))
    ap.add_argument("--test2", default=os.path.join(here, "DatasetB", "DatasetB_test_task2_public.csv"))
    ap.add_argument("--output", default=os.path.join(here, "d_submit.csv"))
    ap.add_argument("--backend", choices=["sglang", "hf"], default="sglang")
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--t1-max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    test1 = pd.read_csv(args.test1, dtype=str)
    t1_ids = [int(r["ID"]) for _, r in test1.iterrows()]
    t1_zh = {int(r["ID"]): str(r["chinese"]).strip() for _, r in test1.iterrows()}
    t2_ids, t2_zh, t2_opts = load_task2_options(args.test2)

    print(f"[infer] task1={len(t1_ids)} task2={len(t2_ids)}")

    gen = Generator(args.model, args.backend, args.tp_size)

    # ---- task1 批量 ----
    t1_prompts = [gen.build_prompt_task1(t1_zh[i]) for i in t1_ids]
    t1_raw = gen.generate_batch(t1_prompts, args.t1_max_new_tokens)
    t1_pred = {i: parse_task1_output(r) for i, r in zip(t1_ids, t1_raw)}

    # ---- task2 批量（逐项判别；按选项数给足输出长度）----
    t2_prompts = [gen.build_prompt_task2(t2_zh[i], t2_opts[i]) for i in t2_ids]
    max_opts = max((len(t2_opts[i]) for i in t2_ids), default=1)
    t2_max_new = min(1024, max_opts * 12 + 32)
    t2_raw = gen.generate_batch(t2_prompts, t2_max_new)
    t2_pred = {i: parse_task2_output(r, t2_opts[i]) for i, r in zip(t2_ids, t2_raw)}

    gen.shutdown()

    # ---- 汇总输出（ID, task1, task2）----
    rows = []
    for i in t1_ids:
        rows.append({"ID": i, "task1": t1_pred[i], "task2": ""})
    for i in t2_ids:
        rows.append({"ID": i, "task1": "", "task2": t2_pred[i]})
    out_df = pd.DataFrame(rows, columns=["ID", "task1", "task2"])

    if args.output.endswith(".xlsx"):
        out_df.to_excel(args.output, index=False)
    else:
        out_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[infer] wrote -> {args.output}")


if __name__ == "__main__":
    main()
