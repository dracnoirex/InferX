import torch
import torch.nn.functional as F
from typing import List, Optional
from app.models.model_loader import model_loader
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class LayerExtractor:
    def __init__(self):
        self.loader = model_loader

    def extract(
        self,
        texts: List[str],
        layer: Optional[int] = None,
        normalize: bool = True
    ) -> dict:
        """
        Extract embeddings from input texts.

        Args:
            texts: List of input sentences
            layer: Which hidden layer to extract (None = last layer)
            normalize: Whether to normalize embeddings to unit length

        Returns:
            dict with embeddings, shape, and layer_used
        """

        if not self.loader.is_loaded:
            raise RuntimeError("Model not loaded!")

        # Step 1: Tokenize inputs
        inputs = self.loader.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.max_length,
            return_tensors="pt"
        )

        # Step 2: Move to device (GPU/CPU)
        inputs = {k: v.to(self.loader.device) for k, v in inputs.items()}

        # Step 3: Run model forward pass
        with torch.no_grad():
            outputs = self.loader.model(**inputs)

        # Step 4: Extract hidden states
        # outputs.hidden_states = tuple of (num_layers + 1) tensors
        # each tensor shape: [batch_size, seq_len, hidden_size]
        hidden_states = outputs.hidden_states

        # Determine which layer to use
        total_layers = len(hidden_states) - 1
        if layer is None:
            layer = total_layers
        elif layer > total_layers:
            logger.warning(f"Layer {layer} doesn't exist, using last layer {total_layers}")
            layer = total_layers

        # Get hidden state from selected layer
        selected = hidden_states[layer]

        # Step 5: Mean pooling over all tokens
        attention_mask = inputs["attention_mask"]
        embeddings = self._mean_pooling(selected, attention_mask)

        # Step 6: Normalize if requested
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        embeddings_list = embeddings.cpu().tolist()

        return {
            "embeddings": embeddings_list,
            "shape": list(embeddings.shape),
            "layer_used": layer,
            "total_layers": total_layers
        }

    def _mean_pooling(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mean pooling over token embeddings,
        ignoring padding tokens.
        """
        mask = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


# Global extractor object
layer_extractor = LayerExtractor()