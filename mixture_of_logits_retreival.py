import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity


class MixtureOfLogits(nn.Module):
    """
    Implementation of mixture of logits model. https://arxiv.org/pdf/2306.04039.pdf
    """

    def __init__(self,
                 num_items: int,
                 emb_dim: int,
                 num_item_emb: int,
                 num_query_emb: int,
                 weight_mlp_dims: list[int],
                 topk: int,
                 device):
        super(MixtureOfLogits, self).__init__()
        self.num_query_emb = num_query_emb
        self.num_item_emb = num_item_emb
        self.emb_dim = emb_dim
        self.topk = topk

        # item embeddings (random weights for testing)
        self.item_embeddings = torch.randn(num_items, num_item_emb, emb_dim)
        if device is not None:
            self.item_embeddings = self.item_embeddings.to(device)

        # gating layer
        num_logit = num_query_emb * num_item_emb
        self.query_weight_fn = nn.Sequential()
        self.item_weight_fn = nn.Sequential()
        self.logit_weight_fn = nn.Sequential()

        # Three MLPs. Defining them here, rather than separate MLP class, for code brevity.
        self.query_weight_fn.append(nn.Linear(num_query_emb * emb_dim, weight_mlp_dims[0]))
        self.item_weight_fn.append(nn.Linear(num_item_emb * emb_dim, weight_mlp_dims[0]))
        self.logit_weight_fn.append(nn.Linear(num_logit, weight_mlp_dims[0]))

        self.query_weight_fn.append(nn.SiLU())
        self.item_weight_fn.append(nn.SiLU())
        self.logit_weight_fn.append(nn.SiLU())

        for i, dim in enumerate(weight_mlp_dims[1:]):
            self.query_weight_fn.append(nn.Linear(weight_mlp_dims[i], dim))
            self.item_weight_fn.append(nn.Linear(weight_mlp_dims[i], dim))
            self.logit_weight_fn.append(nn.Linear(weight_mlp_dims[i], dim))
            self.query_weight_fn.append(nn.SiLU())
            self.item_weight_fn.append(nn.SiLU())
            self.logit_weight_fn.append(nn.SiLU())

        self.query_weight_fn.append(nn.Linear(weight_mlp_dims[-1], num_logit))
        self.item_weight_fn.append(nn.Linear(weight_mlp_dims[-1], num_logit))
        self.logit_weight_fn.append(nn.Linear(weight_mlp_dims[-1], num_logit))

        self.gate_layer = nn.Sequential(
            nn.Linear(num_logit, num_logit, bias=True),
            nn.Softmax(dim=-1),
        )

    def forward(self, query_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_embeddings = query_embeddings.to_dense()  # [query_batch_size, num_query_emb, emb_dim]
        item_embeddings = self.item_embeddings  # [item_batch_size, num_item_emb, emb_dim]
        query_batch_size = query_embeddings.size(0)
        item_batch_size = item_embeddings.size(0)

        # Mixture of logits
        # query_embeddings = query_embeddings.unsqueeze(1)  # [query_batch_size, 1, num_query_emb, emb_dim]
        # item_embeddings = item_embeddings.transpose(1, 2).unsqueeze(0)  # [1, item_batch_size, emb_dim, num_item_emb]

        # shape: [query_batch_size, item_batch_size, num_query_emb, num_item_emb]
        logits = torch.matmul(
            query_embeddings.unsqueeze(1),  # [query_batch_size, 1, num_query_emb, emb_dim]
            item_embeddings.transpose(1, 2).unsqueeze(0),  # [1, item_batch_size, emb_dim, num_item_emb]
        )  # shape: [query_batch_size, item_batch_size, num_query_emb, num_item_emb]
        logits = logits.view(query_batch_size, item_batch_size, self.num_query_emb * self.num_item_emb)

        # Decomposed gating
        query_embeddings = query_embeddings.view(query_batch_size, self.num_query_emb * self.emb_dim)
        item_embeddings = item_embeddings.view(item_batch_size, self.num_item_emb * self.emb_dim)

        query_weights = self.query_weight_fn(query_embeddings)  # [query_batch_size, num_logit]
        item_weights = self.item_weight_fn(item_embeddings)  # [item_batch_size, num_logit]
        logit_weights = self.logit_weight_fn(logits)  # [query_batch_size, item_batch_size, num_logit]

        query_weights = torch.unsqueeze(query_weights, dim=1)  # [query_batch_size, 1, num_logit]
        item_weights = torch.unsqueeze(item_weights, dim=0)  # [1, item_batch_size, num_logit]

        # [query_batch_size, item_batch_size, num_logit]
        gating_weights = self.gate_layer(query_weights * item_weights + logit_weights)

        weighted_logits = logits * gating_weights  # [query_batch_size, item_batch_size, num_logit]

        similarities = torch.sum(weighted_logits, dim=-1)
        similarity_scores, indices = torch.topk(similarities, k=self.topk)
        return (similarity_scores, indices)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_QUERY_EMBEDDING = 2
    N_ITEM_EMBEDDING = 3
    EMB_DIM = 64
    BATCH_SIZE = 1
    NUM_ITEMS = 10_000_000
    TOPK = 500

    model = MixtureOfLogits(
        num_items=NUM_ITEMS,
        emb_dim=EMB_DIM,
        num_item_emb=N_ITEM_EMBEDDING,
        num_query_emb=N_QUERY_EMBEDDING,
        weight_mlp_dims=[32, 16],
        topk=TOPK,
        device=device)

    print(model)
    query = torch.randn(BATCH_SIZE, N_QUERY_EMBEDDING, EMB_DIM)  # [batch_size, num_query_emb, emb_dim]
    model = model.to(device)
    query = query.to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(100):
            out = model(query)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            out = model(query)

    print(out[0].shape)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")

