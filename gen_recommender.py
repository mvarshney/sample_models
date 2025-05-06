from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                 is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    """ Simple implementation of scaled dot-product attention.
    https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

    Or use torch.functional.scaled_dot_product_attention if the compiler supports it.
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CustomMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, common_prefix_len: int):
        batch_size = query.size(0)

        # Linear projections
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # compute prefix attn using usual causal mask
        # Use torch.functional.scaled_dot_product_attention if the compiler supports it.
        prefix_context = scaled_dot_product_attention(
            Q[:, :, :common_prefix_len, :],
            K[:, :, :common_prefix_len, :],
            V[:, :, :common_prefix_len, :],
            dropout_p=0.0,
            is_causal=True,
            scale=None,
        )

        # items after prefix are candidates to rank, so we need to create a special mask for them
        # such that they all attend to the common prefix and themselves but not to other candidates
        candidate_Q = Q[:, :, common_prefix_len:, :]

        num_candidates = candidate_Q.size(2)

        # final mask has shape [num_candidates, common_prefix_len + num_candidates]
        candidate_attn_mask = torch.cat(
            (
                torch.ones(num_candidates, common_prefix_len, dtype=torch.bool, device=Q.device),
                torch.eye(num_candidates, dtype=torch.bool, device=Q.device),
            ),
            dim=-1,
        )

        # Or use torch.functional.scaled_dot_product_attention if the compiler supports it.
        candidate_context = scaled_dot_product_attention(
            candidate_Q, K, V, dropout_p=0.0, scale=None, attn_mask=candidate_attn_mask
        )
        context = torch.cat([prefix_context, candidate_context], dim=2)

        # Concatenate heads and apply final linear transformation
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        return output, None


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        num_heads
    ):
        super().__init__()
        self.mha = CustomMultiHeadAttention(d_model=d_model, num_heads=num_heads)

        self.ffwd = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.prenorm_1 = nn.LayerNorm(normalized_shape=d_model)
        self.prenorm_2 = nn.LayerNorm(normalized_shape=d_model)

        self.skip_connection_norm_1 = nn.LayerNorm(normalized_shape=d_model)
        self.skip_connection_norm_2 = nn.LayerNorm(normalized_shape=d_model)

    def attention(self, x, common_prefix_len: int | None = None) -> torch.Tensor:
        attn_output, _ = self.mha(x, x, x, common_prefix_len=common_prefix_len)
        return attn_output

    def forward(self, x, common_prefix_len: int | None = None) -> torch.Tensor:
        attn = self.attention(self.prenorm_1(x), common_prefix_len=common_prefix_len)
        x = self.skip_connection_norm_1(x + attn)

        ffwd = self.ffwd(self.prenorm_2(x))
        output = self.skip_connection_norm_2(x + ffwd)
        return output


class GenerativeRecommender(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        d_model: int,
        dff_multiplier,
        d_labels,
        num_heads,
        num_layers,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model

        self.register_buffer("sequence_indices", torch.arange(1, self.sequence_length + 1))

        transformer_layers = []
        for i in range(num_layers):
            module = TransformerLayer(
                d_model=self.d_model,
                num_heads=num_heads,
                d_ff=int(self.d_model * dff_multiplier),
            )
            transformer_layers.append(module)

        self.transformer_layers = nn.ModuleList(transformer_layers)
        self.head = nn.Linear(self.d_model, d_labels)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        common_prefix_len: int | None = None,
    ):
        """
        Return the logits for each position in the input embeddings sequence.

        input_embeddings is a tensor of shape (1, <context-history-len> + <num_candidates>, d_model),
        where the last <num_candidate> items in the sequence are the candidates to score / return action logits for.
        In this case, the attn_mask is created based on the common_prefix_len, such that, the candidates do not
        attend to each other but all of them attend to the common prefix and themselves.

        context-history is an interleaved sequence of items and actions embeddings.

        :return: logits for every position of the sequence.
        """

        # good practice to clone this since we keep directly work with this embedding tensor (update it)
        computed_embeddings = input_embeddings.clone()

        for layer in self.transformer_layers:
            computed_embeddings = layer(computed_embeddings, common_prefix_len=common_prefix_len)

        logits = self.head(computed_embeddings)
        return logits

    def score(
        self,
        member_embeddings: torch.Tensor,
        candidate_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score the candidate items in the context of the given member embeddings.
        This will do the "prefill" computation on member_embeddings only once and use that to score all candidate
        items.

        :param member_embeddings: A tensor of shape (2 * sequence_length, emb_dim) representing the
        member (context) embeddings.
        :param candidate_items: A tensor of shape (num_candidates, emb_dim) representing the candidate items.
        :param should_score_candidates_individually: whether to score candidates individually or use multi-item
        scoring (MIS). MIS creates a single input of shape <context-history-len> + <num-candidates> and a special attn mask to
        score all candidates in a single forward pass, reusing computation for the common (history) prefix.

        :return: A tensor of shape (num_candidates), representing the score for the candidate items.
        items.
        """

        common_prefix_len = member_embeddings.size(1)
        all_embeddings = torch.cat((member_embeddings, candidate_items), dim=1)
        with torch.no_grad():
            logits = self.forward(input_embeddings=all_embeddings, common_prefix_len=common_prefix_len)

        logits = logits[0, -candidate_items.size(1):, :]
        label_scores = torch.sigmoid(logits)
        return label_scores


@dataclass
class ModelSpec:
    sequence_length: int
    d_model: int
    num_layers: int
    num_heads: int
    dff_multiplier: int = 4
    d_labels: int = 2


if __name__ == "__main__":
    # 12M params
    small_model_spec = ModelSpec(
        sequence_length=1024,
        d_model=256,
        num_layers=16,
        num_heads=4,
    )

    # 100M params
    large_model_spec = ModelSpec(
        sequence_length=1024,
        d_model=512,
        num_layers=32,
        num_heads=8,
    )

    model_spec = large_model_spec  # switch to `small_model_spec` for smaller model

    model = GenerativeRecommender(**asdict(model_spec))

    num_candidates = 512

    member_embeddings = torch.randn(1,
                                    2*model_spec.sequence_length,
                                    model_spec.d_model)  # (batch_size, sequence_length, d_model)
    candidate_items = torch.randn(1, num_candidates, model_spec.d_model)  # (batch_size, num_candidates, d_model)
    common_prefix_len = 2*model_spec.sequence_length

    params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {params:,}")

    output = model.score(member_embeddings, candidate_items)
    print(output.shape)  # Should be (1, num_candidates, d_labels)
    print(output)
