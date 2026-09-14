"""
Unified DDI prediction model (V5).

Composes: encoder(s) + MLP classifier.
For v5 single-source cells (01-10):
  - Single-source: encoder(drug_ids) -> pair concat -> classifier
  - Pair-wise REL (Cells 09, 10): forward_pair(d1, d2) for contextualized embeddings

Classifier input: 256d (128d per drug, concatenated).

Source: v2_single_modality/models/ddi_model.py (simplified for v5 single-source)
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

from models.encoders import GATEncoder


class DDIModel(nn.Module):

    def __init__(
        self,
        encoders: Dict[str, nn.Module],
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        rel_is_pairwise: bool = False,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.source_names = sorted(encoders.keys())
        self.rel_is_pairwise = rel_is_pairwise

        # Single-source: hidden_dim per drug -> pair concat
        classifier_dim = hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch):
        """
        Main forward. Accepts DDIBatch (from collate_fn).
        Returns: logits [B, num_classes]
        """
        drug1_ids = batch.drug1_ids
        drug2_ids = batch.drug2_ids

        # Collect per-drug embeddings
        embs_d1 = {}
        embs_d2 = {}

        for source in self.source_names:
            encoder = self.encoders[source]

            if source == 'mol' and isinstance(encoder, GATEncoder):
                # GAT: use pre-batched mol graphs from collator
                if hasattr(batch, 'mol_batch_d1') and batch.mol_batch_d1 is not None:
                    embs_d1['mol'] = encoder(mol_batch=batch.mol_batch_d1)
                    embs_d2['mol'] = encoder(mol_batch=batch.mol_batch_d2)
                else:
                    embs_d1['mol'] = encoder(drug1_ids)
                    embs_d2['mol'] = encoder(drug2_ids)

            elif source == 'rel' and self.rel_is_pairwise:
                # R-GCN / GT-KG: use pre-batched subgraphs from collator
                rel_batch = getattr(batch, 'rel_subgraph_batch', None)
                h_d1, h_d2 = encoder.forward_pair(
                    drug1_ids, drug2_ids,
                    rel_subgraph_batch=rel_batch)
                embs_d1['rel'] = h_d1
                embs_d2['rel'] = h_d2

            else:
                # MorganFP, ChemBERTa, SMILES-CNN, TransE
                embs_d1[source] = encoder(drug1_ids)
                embs_d2[source] = encoder(drug2_ids)

        # Single source: get the one embedding
        h1 = self._get_single_source_embedding(embs_d1)
        h2 = self._get_single_source_embedding(embs_d2)
        pair = torch.cat([h1, h2], dim=1)

        return self.classifier(pair)

    def _get_single_source_embedding(self, embs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract single source embedding (for single-source configs only)."""
        source_list = sorted(embs.keys())
        assert len(source_list) == 1, f"Expected 1 source, got {len(source_list)}"
        return embs[source_list[0]]

    def pair_embedding(self, batch):
        """Return classifier-input pair embedding without running classifier.
        For analysis / t-SNE."""
        drug1_ids = batch.drug1_ids
        drug2_ids = batch.drug2_ids
        embs_d1, embs_d2 = {}, {}
        for source in self.source_names:
            encoder = self.encoders[source]
            if source == 'mol' and isinstance(encoder, GATEncoder):
                if hasattr(batch, 'mol_batch_d1') and batch.mol_batch_d1 is not None:
                    embs_d1['mol'] = encoder(mol_batch=batch.mol_batch_d1)
                    embs_d2['mol'] = encoder(mol_batch=batch.mol_batch_d2)
                else:
                    embs_d1['mol'] = encoder(drug1_ids)
                    embs_d2['mol'] = encoder(drug2_ids)
            elif source == 'rel' and self.rel_is_pairwise:
                rel_batch = getattr(batch, 'rel_subgraph_batch', None)
                h_d1, h_d2 = encoder.forward_pair(
                    drug1_ids, drug2_ids,
                    rel_subgraph_batch=rel_batch)
                embs_d1['rel'] = h_d1
                embs_d2['rel'] = h_d2
            else:
                embs_d1[source] = encoder(drug1_ids)
                embs_d2[source] = encoder(drug2_ids)
        h1 = self._get_single_source_embedding(embs_d1)
        h2 = self._get_single_source_embedding(embs_d2)
        return torch.cat([h1, h2], dim=1)
