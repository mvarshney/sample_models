import torch
import torch.nn as nn

# Please see https://microsoft.sharepoint.com/:w:/t/BrainSmith/EfrPr1T3zTFEvrjVL5ll4k8B0xeDXeAaI-71FqtEpUb6lA?e=oiqZEg&nav=eyJoIjoiMTgzMzkwMjMyIn0

class SimilarityEBR(nn.Module):
    def __init__(self,
                 item_embeddings: torch.Tensor,
                 item_attributes: torch.Tensor,
                 item_ids: torch.Tensor,
                 topk: int):
        """Initializes the SimilarityEBR class.

        Args:
            item_embeddings (torch.Tensor): A tensor containing the embeddings of the items.
            item_attributes (torch.Tensor): A tensor containing the attributes of the items.
            item_ids (torch.Tensor): A tensor containing the unique identifiers of the items.
            topk (int): The number of top similar items to consider.
        """

        super(SimilarityEBR, self).__init__()
        self.item_embeddings = item_embeddings
        self.item_attributes = item_attributes
        self.item_ids = item_ids
        self.topk = topk

    def forward(self,
                query: torch.Tensor,  # [query_batch_size, emb_dim]
                query_filters: list[torch.Tensor],  # [query_batch_size, num_filters]
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the top-k similar items based on the query and filters.
        Args:
            query (torch.Tensor): A tensor containing the query embeddings.
            query_filters (torch.Tensor): A tensor containing the filters to apply.
        Returns:
            tuple: A tuple containing the scores and the top-k item IDs.
        """

        similarities = torch.mm(self.item_embeddings, query.T)

        """
        creation_time > $t-30d 
        AND is_spam = false 
        AND ( 
            (author_urn IN [$followed_and_not_blocked_authors])  
        OR    
            (author_urn IN [$connected_and_not_blocked_authors] )) 
        AND language in [“EN”] 
        """
        creation_time_filter = self.item_attributes[:, 0] > query_filters[0]
        is_spam_filter = self.item_attributes[:, 1] > 0
        author_urn_followed_filter = torch.isin(self.item_attributes[:, 2], query_filters[1])
        author_urn_connected_filter = torch.isin(self.item_attributes[:, 3], query_filters[2])
        language_filter = torch.isin(self.item_attributes[:, 4], query_filters[3])

        author_filter = torch.logical_or(author_urn_followed_filter, author_urn_connected_filter)
        match = torch.logical_and(author_filter, language_filter)
        match = torch.logical_and(match, creation_time_filter)
        match = torch.logical_and(match, is_spam_filter)
        match = match.unsqueeze(dim=0).T

        filtered = torch.where(match, similarities, float("-inf"))

        scores, indices = torch.topk(filtered, self.topk, dim=0)
        topk_ids = torch.gather(self.item_ids, 0, indices.squeeze())

        return scores, topk_ids


if __name__ == "__main__":
    # Example usage
    num_items = 1_000_000
    emb_dim = 64
    num_attrs = 5
    query_max_size = 100
    topk = 500
    batch_size = 1

    float_type = torch.float16

    items = torch.randn(num_items, emb_dim, dtype=float_type)
    attrs = torch.randint(0, 100, (num_items, num_attrs), dtype=torch.int64)
    ids = torch.randint(0, 10000000, (num_items,), dtype=torch.int64)

    user = torch.randn(batch_size, emb_dim, dtype=float_type)
    conds = [torch.randint(0, 100, (batch_size,), dtype=torch.int64),
             torch.randint(0, 100, (batch_size, query_max_size,), dtype=torch.int64),
             torch.randint(0, 100, (batch_size, query_max_size,), dtype=torch.int64),
             torch.randint(0, 100, (batch_size, query_max_size,), dtype=torch.int64)]

    model = SimilarityEBR(items, attrs, ids, topk)

    scores, topk_ids = model(user, conds)
    print("Scores:", scores.T)
    print("Top-k IDs:", topk_ids)
