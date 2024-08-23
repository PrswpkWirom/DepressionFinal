from transformers import BertTokenizer, BertModel
import torch

class BertEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        """
        Initialize the BERT Embedder with a specified BERT model.
        
        Args:
            model_name (str): The name of the pre-trained BERT model.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def embed_text(self, text):
        """
        Embed the input text using the BERT model.

        Args:
            text (str): The input text to embed.

        Returns:
            torch.Tensor: The embedding of the input text.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)

        # Get the outputs from BERT model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # The last hidden state is the embedding of the text
        embeddings = outputs.last_hidden_state

        return embeddings

    def embed_text_as_numpy(self, text):
        """
        Embed the input text using the BERT model and return as a NumPy array.

        Args:
            text (str): The input text to embed.

        Returns:
            numpy.ndarray: The embedding of the input text.
        """
        embeddings = self.embed_text(text)
        return embeddings.cpu().numpy()
