import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class BertEmbedder:
    """\
    Extracts fixed-length contextual embeddings (CLS pooled representation)
    for use with traditional ML models.

    This supports the paper's baseline idea: train classical models on
    BERT embeddings.
    """

    def __init__(self, model_name: str = "bert-base-multilingual-cased", max_length: int = 128, batch_size: int = 16):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts):
        """Return a (n, hidden) numpy array of CLS embeddings."""
        if isinstance(texts, (str, bytes)):
            texts = [texts]

        embeddings = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc=f"Encoding with {self.model_name}"):
            batch = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            outputs = self.model(**enc)
            # Prefer pooler_output when available; otherwise use CLS from last_hidden_state
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                cls = outputs.pooler_output
            else:
                cls = outputs.last_hidden_state[:, 0, :]

            embeddings.append(cls.detach().cpu().numpy())

        return np.vstack(embeddings)
