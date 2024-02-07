# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
from FlagEmbedding import BGEM3FlagModel

MODEL_NAME = "BAAI/bge-m3"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Enable faster download speed
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        self.model = BGEM3FlagModel(
            MODEL_NAME,  
            use_fp16=True
        )

    def predict(
        self,
        sentences_1: str = Input(description="Input Sentence list 1 - Each sentence should be split by a newline"),
        sentences_2: str = Input(description="Input Sentence list 2 - Each sentence should be split by a newline"),
        embedding_type: str = Input(description="Type of embedding to use", default="dense", choices=["dense", "sparse", "colbert"]),
        max_length: int = Input(description="Maximum length of the input for dense embeddings, use a smaller value to speed up the encoding process", default=8192)
    ) -> str:
        """Run a single prediction on the model"""
        #split sentences by newline
        sentences_1 = sentences_1.strip().splitlines()
        sentences_2 = sentences_2.strip().splitlines()
        print("Sentences_1 split out:")
        print(sentences_1)
        print("Sentences_2 split out:")
        print(sentences_2)

        result = ""
        if(embedding_type == "dense"):
            embeddings_1 = self.model.encode(
                sentences_1, 
                batch_size=12, 
                max_length=max_length, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
            )['dense_vecs']
            embeddings_2 = self.model.encode(sentences_2)['dense_vecs']
            result = str(embeddings_1 @ embeddings_2.T)
        elif(embedding_type == "sparse"):
            output_1 = self.model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=False)
            output_2 = self.model.encode(sentences_2, return_dense=True, return_sparse=True, return_colbert_vecs=False)
            print("The weights for each token:")
            print(self.model.convert_id_to_token(output_1['lexical_weights']))
            lexical_scores = self.model.compute_lexical_matching_score(output_1['lexical_weights'][0], output_2['lexical_weights'][0])
            result = str(lexical_scores) + "\n"
            result += str(self.model.compute_lexical_matching_score(output_1['lexical_weights'][0], output_1['lexical_weights'][1]))
        else:
            output_1 = self.model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=True)
            output_2 = self.model.encode(sentences_2, return_dense=True, return_sparse=True, return_colbert_vecs=True)
            result = str(self.model.colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][0])) + "\n"
            result += str(self.model.colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][1]))

        return result