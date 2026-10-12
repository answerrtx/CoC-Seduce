# Seduced by the Narrative: Assessing Rule Adherence in Semi-Open Textual Sandboxes

## Abstract

As LLMs are increasingly deployed as autonomous adjudicators in games such as Call of Cthulhu (CoC), robust rule adherence becomes critical when user intent conflicts with system rules. However, as these models are trained to be helpful and compliant, they may be vulnerable to a class of manipulations we term Rhetorical Injection, where adversarial users exploit narrative framing techniques such as pseudo-logical reasoning and authoritative coercion to bypass adjudication logic. We present CoC-Seduce, a multi-agent adversarial benchmark built on CoC, a Tabletop Role-Playing Game (TRPG) in which rules are explicit about which risky actions require adjudication, yet interaction remains entirely in natural language. Three LLMs, i.e., GPT-5.4, Claude Sonnet 4.6, Gemini 3.5 Flash, serve as adversarial generators producing 5,376 samples across 4 world settings and 16 skill categories. We then benchmark 22 target adjudicators against this corpus. Evaluation across 22 models reveals that neither newer releases nor explicit reasoning reliably confer adjudication robustness, that Pseudo-Logic framing is the most effective rhetorical style, and that the world setting, including culturally distant ones, has only a modest effect.

Arxiv: https://arxiv.org/abs/2607.02802
