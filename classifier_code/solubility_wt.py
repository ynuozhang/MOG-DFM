import sys
import os
import xgboost as xgb
import torch
import numpy as np
import warnings
import numpy as np
from rdkit import Chem, rdBase, DataStructs
from transformers import AutoTokenizer, EsmModel

rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class Solubility:
    def __init__(self):
        # change model path
        self.predictor = xgb.Booster(model_file='../classifier_ckpt/best_model_solubility.json')
        
        # Load ESM model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.model.eval()
    
    def generate_embeddings(self, sequences):
        """Generate ESM embeddings for protein sequences"""
        embeddings = []
        
        # Process sequences in batches to avoid memory issues
        batch_size = 8
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_sequences, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                self.model = self.model.cuda()
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Get last hidden states
                last_hidden_states = outputs.last_hidden_state
                
                # Compute mean pooling (excluding padding tokens)
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_hidden_states = last_hidden_states * attention_mask
                sum_hidden_states = masked_hidden_states.sum(dim=1)
                seq_lengths = attention_mask.sum(dim=1)
                batch_embeddings = sum_hidden_states / seq_lengths
                
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.append(batch_embeddings)
        
        if embeddings:
            return np.vstack(embeddings)
        else:
            return np.array([])
    
    def get_scores(self, input_seqs: list):
        scores = np.zeros(len(input_seqs))
        features = self.generate_embeddings(input_seqs)
        
        if len(features) == 0:
            return scores
        
        features = np.nan_to_num(features, nan=0.)
        features = np.clip(features, np.finfo(np.float32).min, np.finfo(np.float32).max)
        
        features = xgb.DMatrix(features)
        
        scores = self.predictor.predict(features)
        return scores
    
    def __call__(self, input_seqs: list):
        scores = self.get_scores(input_seqs)
        return scores
    
def unittest():
    solubility = Solubility()
    sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MSEGIRQAFVLAKSIWPARVARFTVDNRIRSLVKTYEAIKVDPYNPAFLEVLD"
    ]    
    
    scores = solubility(input_seqs=sequences)
    print(scores)
    
if __name__ == '__main__':
    unittest()