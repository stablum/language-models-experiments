# Models

This project currently contains small token-level autoregressive language models. Each model predicts a SentencePiece token, written w_i, from a finite vocabulary V. The beginning-of-sequence token is used only as context, so generated candidates live in

```math
\mathcal{V}_\star = \mathcal{V} \setminus \{\mathrm{BOS}\}.
```

For a token sequence from w_1 through w_T, the autoregressive likelihood is

```math
P(w_{1:T}) = \prod_{i=1}^{T} P(w_i \mid h_i),
```

where h_i is the available history. The bigram history is the previous token, and the trigram history is the previous two tokens. In training and evaluation, explicit BOS context tokens and an EOS target token are included when the tokenizer provides them.

## Notation

Counts are written as

```math
c(w), \qquad c(v,w), \qquad c(u,v,w).
```

Marginal counts are written as

```math
c(v,\cdot) = \sum_{w \in \mathcal{V}_\star} c(v,w),
```

and

```math
c(u,v,\cdot) = \sum_{w \in \mathcal{V}_\star} c(u,v,w).
```

The number of observed continuation types after a context is

```math
N_+(v,\cdot) = |\{w \in \mathcal{V}_\star : c(v,w) > 0\}|,
```

and

```math
N_+(u,v,\cdot) = |\{w \in \mathcal{V}_\star : c(u,v,w) > 0\}|.
```

The add-k smoothing constant is non-negative. The absolute discount is D, with the usual practical range

```math
0 \leq D \leq 1.
```

## Bigram

Registered name: `bigram`.

The bigram model is a first-order Markov model:

```math
P(w_i \mid w_{1:i-1}) = P(w_i \mid w_{i-1}).
```

It uses add-k smoothing over next-token counts:

```math
P_{\mathrm{bi}}(w \mid v)
=
\frac{c(v,w) + k}{c(v,\cdot) + k|\mathcal{V}_\star|}.
```

If the previous token was never observed and the smoothing constant is positive, the model becomes uniform:

```math
P_{\mathrm{bi}}(w \mid v)
=
\frac{k}{k|\mathcal{V}_\star|}
=
\frac{1}{|\mathcal{V}_\star|}.
```

This model is intentionally simple. It is useful as a sparse transition-table baseline:

```math
v \mapsto \{(w, c(v,w)) : c(v,w) > 0\}.
```

## Interpolated Trigram

Registered name: `trigram`.

The interpolated trigram model uses ordinary unigram, bigram, and trigram counts with add-k smoothing at each order.

The smoothed unigram distribution is

```math
P_1(w)
=
\frac{c(w) + k}{\sum_{x \in \mathcal{V}_\star} c(x) + k|\mathcal{V}_\star|}.
```

The smoothed bigram distribution is

```math
P_2(w \mid v)
=
\frac{c(v,w) + k}{c(v,\cdot) + k|\mathcal{V}_\star|}.
```

The smoothed trigram distribution is

```math
P_3(w \mid u,v)
=
\frac{c(u,v,w) + k}{c(u,v,\cdot) + k|\mathcal{V}_\star|}.
```

The final model is a linear interpolation:

```math
P_{\lambda}(w \mid u,v)
=
\lambda_1 P_1(w)
+ \lambda_2 P_2(w \mid v)
+ \lambda_3 P_3(w \mid u,v).
```

The weights are non-negative and normalized:

```math
\lambda_1,\lambda_2,\lambda_3 \geq 0,
\qquad
\lambda_1 + \lambda_2 + \lambda_3 = 1.
```

The default weights are

```math
\lambda_1 = 0.1,
\qquad
\lambda_2 = 0.3,
\qquad
\lambda_3 = 0.6.
```

This model is smooth everywhere when the smoothing constant is positive, but the lower-order models are ordinary lower-order frequency models. A word is probable as a unigram if it appears often, not necessarily if it appears in many different contexts.

## Absolute-Discount Trigram

Registered name: `trigram-absolute-discount`.

This model applies absolute discounting only to the trigram row. At each observed trigram context, a fixed discount is subtracted from every positive trigram count. The removed probability mass is assigned to an ordinary add-k smoothed bigram model.

The lower-order add-k bigram distribution is

```math
P_{+2}(w \mid v)
=
\frac{c(v,w) + k}{c(v,\cdot) + k|\mathcal{V}_\star|}.
```

For observed trigram contexts,

```math
P_{\mathrm{AD},3}(w \mid u,v)
=
\frac{\max(c(u,v,w)-D,0)}{c(u,v,\cdot)}
+ \gamma(u,v)P_{+2}(w \mid v),
```

where

```math
\gamma(u,v)
=
\frac{D N_+(u,v,\cdot)}{c(u,v,\cdot)}.
```

For unseen trigram contexts, the model backs off to the ordinary add-k bigram:

```math
P_{\mathrm{AD},3}(w \mid u,v)
=
P_{+2}(w \mid v).
```

This keeps the model normalized because the discounted mass is

```math
\sum_{w:c(h,w)>0} \frac{D}{c(h,\cdot)}
=
\frac{D N_+(h,\cdot)}{c(h,\cdot)}.
```

The important distinction from recursive models is that the lower-order model is not itself discounted:

```math
P_{+2}(w \mid v)
\neq
P_{\mathrm{AD},2}(w \mid v).
```

## Interpolated Kneser-Ney Trigram

Registered name: `trigram-kneser-ney`.

This is the recursive discounted/interpolated model usually called interpolated Kneser-Ney smoothing. Like absolute discounting, it subtracts a fixed discount from positive counts and interpolates with a lower-order distribution. Unlike ordinary absolute discounting, the lower-order distributions are continuation distributions.

For trigram counts, the highest-order model still starts from ordinary counts:

```math
c(u,v,w).
```

The Kneser-Ney bigram continuation count is the number of distinct left contexts in which a token pair appears:

```math
c_{\mathrm{KN}}(v,w)
=
N_+(\cdot,v,w)
=
|\{u : c(u,v,w) > 0\}|.
```

Its marginal is

```math
c_{\mathrm{KN}}(v,\cdot)
=
\sum_{w \in \mathcal{V}_\star} c_{\mathrm{KN}}(v,w).
```

The Kneser-Ney unigram continuation count is the number of distinct previous tokens from which a token appears as a continuation:

```math
c_{\mathrm{KN}}(w)
=
N_+(\cdot,w)
=
|\{v : c_{\mathrm{KN}}(v,w) > 0\}|.
```

Its total is

```math
c_{\mathrm{KN}}(\cdot)
=
\sum_{w \in \mathcal{V}_\star} c_{\mathrm{KN}}(w).
```

The base distribution is uniform:

```math
P_{\mathrm{KN},0}(w)
=
\frac{1}{|\mathcal{V}_\star|}.
```

The unigram continuation model is recursively discounted and interpolated with the uniform base:

```math
P_{\mathrm{KN},1}(w)
=
\frac{\max(c_{\mathrm{KN}}(w)-D,0)}{c_{\mathrm{KN}}(\cdot)}
+ \gamma_{\mathrm{KN}}(\cdot)P_{\mathrm{KN},0}(w),
```

where

```math
\gamma_{\mathrm{KN}}(\cdot)
=
\frac{D|\{x : c_{\mathrm{KN}}(x)>0\}|}{c_{\mathrm{KN}}(\cdot)}.
```

If the continuation unigram total is zero, the model uses the uniform base distribution.

The Kneser-Ney bigram model is

```math
P_{\mathrm{KN},2}(w \mid v)
=
\frac{\max(c_{\mathrm{KN}}(v,w)-D,0)}{c_{\mathrm{KN}}(v,\cdot)}
+ \gamma_{\mathrm{KN}}(v)P_{\mathrm{KN},1}(w),
```

where

```math
\gamma_{\mathrm{KN}}(v)
=
\frac{D|\{x : c_{\mathrm{KN}}(v,x)>0\}|}{c_{\mathrm{KN}}(v,\cdot)}.
```

If the Kneser-Ney bigram context is unseen, the model uses the Kneser-Ney unigram distribution.

The final trigram model is

```math
P_{\mathrm{KN},3}(w \mid u,v)
=
\frac{\max(c(u,v,w)-D,0)}{c(u,v,\cdot)}
+ \gamma_{\mathrm{KN}}(u,v)P_{\mathrm{KN},2}(w \mid v),
```

where

```math
\gamma_{\mathrm{KN}}(u,v)
=
\frac{D N_+(u,v,\cdot)}{c(u,v,\cdot)}.
```

If the trigram context is unseen, the model uses the Kneser-Ney bigram distribution.

The intuition is that lower-order probability should reward tokens that appear in many different contexts, not merely tokens that are frequent:

```math
\text{ordinary unigram strength} \propto c(w),
```

but

```math
\text{Kneser-Ney unigram strength} \propto |\{v : c_{\mathrm{KN}}(v,w)>0\}|.
```

That is why Kneser-Ney is often a strong n-gram smoothing method: it treats lower-order distributions as distributions over plausible continuations.

## Prediction

For all models, most-probable decoding chooses

```math
\hat{w}_{i}
=
\arg\max_{w \in \mathcal{V}_\star} P(w \mid h_i).
```

Sampling uses the full categorical distribution

```math
w_i \sim \mathrm{Categorical}(P(\cdot \mid h_i)).
```

With positive temperature, the sampling weights are proportional to

```math
P(w \mid h_i)^{1/\tau}.
```

When the temperature is one, sampling uses the model probabilities directly. As the temperature approaches zero, sampling approaches greedy decoding.

## Evaluation

Next-token accuracy is

```math
\mathrm{accuracy}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
\left[
w_i
=
\arg\max_{w \in \mathcal{V}_\star} P(w \mid h_i)
\right].
```

Top-K accuracy is

```math
\mathrm{topK}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
\left[
w_i \in \mathrm{TopK}(P(\cdot \mid h_i))
\right].
```

Negative log-likelihood is

```math
\mathrm{NLL}
=
-
\sum_{i=1}^{N}
\log P(w_i \mid h_i).
```

Average negative log-likelihood is

```math
\overline{\mathrm{NLL}}
=
\frac{\mathrm{NLL}}{N}.
```

Cross entropy in bits per token is

```math
H_2
=
\frac{\overline{\mathrm{NLL}}}{\log 2}.
```

Perplexity is

```math
\mathrm{PPL}
=
\exp(\overline{\mathrm{NLL}}).
```
