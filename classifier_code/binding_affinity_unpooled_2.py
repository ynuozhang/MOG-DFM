import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr
from collections import defaultdict
import pandas as pd
import logging
import os
import torch.optim as optim
from datetime import datetime
from transformers import AutoModel, AutoConfig, AutoTokenizer

import os

# point HF_ENDPOINT at your mirror
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class UnpooledBindingPredictor(nn.Module):
    def __init__(self, 
                 esm_model_name="facebook/esm2_t33_650M_UR50D",
                 hidden_dim=512,
                 kernel_sizes=[3, 5, 7],
                 n_heads=8,
                 n_layers=3,
                 dropout=0.1,
                 freeze_esm=True):
        super().__init__()
        
        # Define binding thresholds
        self.tight_threshold = 7.5    # Kd/Ki/IC50 ≤ ~30nM
        self.weak_threshold = 6.0     # Kd/Ki/IC50 > 1μM
        
        # Load ESM model for computing embeddings on the fly
        self.esm_model = AutoModel.from_pretrained(esm_model_name)
        self.config = AutoConfig.from_pretrained(esm_model_name)
        
        # Freeze ESM parameters if needed
        if freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False
        
        # Get ESM hidden size
        esm_dim = self.config.hidden_size
        
        # Output channels for CNN layers
        output_channels_per_kernel = 64
        
        # CNN layers for handling variable length sequences
        self.protein_conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=esm_dim,
                out_channels=output_channels_per_kernel,
                kernel_size=k,
                padding='same'
            ) for k in kernel_sizes
        ])
        
        self.binder_conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=esm_dim,
                out_channels=output_channels_per_kernel,
                kernel_size=k,
                padding='same'
            ) for k in kernel_sizes
        ])
        
        # Calculate total features after convolution and pooling
        total_features_per_seq = output_channels_per_kernel * len(kernel_sizes) * 2
        
        # Project to same dimension after CNN processing
        self.protein_projection = nn.Linear(total_features_per_seq, hidden_dim)
        self.binder_projection = nn.Linear(total_features_per_seq, hidden_dim)
        
        self.protein_norm = nn.LayerNorm(hidden_dim)
        self.binder_norm = nn.LayerNorm(hidden_dim)
        
        # Cross attention blocks with layer norm
        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout),
                'norm1': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                ),
                'norm2': nn.LayerNorm(hidden_dim)
            }) for _ in range(n_layers)
        ])
        
        # Prediction heads
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Regression head
        self.regression_head = nn.Linear(hidden_dim, 1)
        
        # Classification head (3 classes: tight, medium, loose binding)
        self.classification_head = nn.Linear(hidden_dim, 3)
        
    def get_binding_class(self, affinity):
        """Convert affinity values to class indices
        0: tight binding (>= 7.5)
        1: medium binding (6.0-7.5)
        2: weak binding (< 6.0)
        """
        if isinstance(affinity, torch.Tensor):
            tight_mask = affinity >= self.tight_threshold
            weak_mask = affinity < self.weak_threshold
            medium_mask = ~(tight_mask | weak_mask)
            
            classes = torch.zeros_like(affinity, dtype=torch.long)
            classes[medium_mask] = 1
            classes[weak_mask] = 2
            return classes
        else:
            if affinity >= self.tight_threshold:
                return 0  # tight binding
            elif affinity < self.weak_threshold:
                return 2  # weak binding
            else:
                return 1  # medium binding
    
    def compute_embeddings(self, input_ids, attention_mask=None):
        """Compute ESM embeddings on the fly"""
        esm_outputs = self.esm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get the unpooled last hidden states (batch_size x seq_length x hidden_size)
        return esm_outputs.last_hidden_state
    
    def process_sequence(self, unpooled_emb, conv_layers, attention_mask=None):
        """Process a sequence through CNN layers and pooling"""
        # Transpose for CNN: [batch_size, hidden_size, seq_length]
        x = unpooled_emb.transpose(1, 2)
        
        # Apply CNN layers and collect outputs
        conv_outputs = []
        for conv in conv_layers:
            conv_out = F.relu(conv(x))
            conv_outputs.append(conv_out)
        
        # Concatenate along channel dimension
        conv_output = torch.cat(conv_outputs, dim=1)
        
        # Global pooling (both max and average)
        # If attention mask is provided, use it to create a proper mask for pooling
        if attention_mask is not None:
            # Create a mask for pooling (1 for valid positions, 0 for padding)
            # Expand mask to match conv_output channels
            expanded_mask = attention_mask.unsqueeze(1).expand(-1, conv_output.size(1), -1)
            
            # Apply mask (set padding to large negative value for max pooling)
            masked_output = conv_output.clone()
            masked_output = masked_output.masked_fill(expanded_mask == 0, float('-inf'))
            
            # Max pooling along sequence dimension
            max_pooled = torch.max(masked_output, dim=2)[0]
            
            # Average pooling (sum divided by number of valid positions)
            sum_pooled = torch.sum(conv_output * expanded_mask, dim=2)
            valid_positions = torch.sum(expanded_mask, dim=2)
            valid_positions = torch.clamp(valid_positions, min=1.0)  # Avoid division by zero
            avg_pooled = sum_pooled / valid_positions
        else:
            # If no mask, use standard pooling
            max_pooled = torch.max(conv_output, dim=2)[0]
            avg_pooled = torch.mean(conv_output, dim=2)
        
        # Concatenate the pooled features
        pooled = torch.cat([max_pooled, avg_pooled], dim=1)
        
        return pooled
        
    def forward(self, protein_input_ids, binder_input_ids, protein_mask=None, binder_mask=None):
        # Compute embeddings on the fly using the ESM model
        protein_unpooled = self.compute_embeddings(protein_input_ids, protein_mask)
        binder_unpooled = self.compute_embeddings(binder_input_ids, binder_mask)
        
        # Process protein and binder sequences through CNN layers
        protein_features = self.process_sequence(protein_unpooled, self.protein_conv_layers, protein_mask)
        binder_features = self.process_sequence(binder_unpooled, self.binder_conv_layers, binder_mask)
        
        # Project to same dimension
        protein = self.protein_norm(self.protein_projection(protein_features))
        binder = self.binder_norm(self.binder_projection(binder_features))
        
        # Reshape for attention: from [batch_size, hidden_dim] to [1, batch_size, hidden_dim]
        protein = protein.unsqueeze(0)
        binder = binder.unsqueeze(0)
        
        # Cross attention layers
        for layer in self.cross_attention_layers:
            # Protein attending to binder
            attended_protein = layer['attention'](
                protein, binder, binder
            )[0]
            protein = layer['norm1'](protein + attended_protein)
            protein = layer['norm2'](protein + layer['ffn'](protein))
            
            # Binder attending to protein
            attended_binder = layer['attention'](
                binder, protein, protein
            )[0]
            binder = layer['norm1'](binder + attended_binder)
            binder = layer['norm2'](binder + layer['ffn'](binder))
        
        # Remove sequence dimension
        protein_pool = protein.squeeze(0)
        binder_pool = binder.squeeze(0)
        
        # Concatenate both representations
        combined = torch.cat([protein_pool, binder_pool], dim=-1)
        
        # Shared features
        shared_features = self.shared_head(combined)
        
        regression_output = self.regression_head(shared_features)
        classification_logits = self.classification_head(shared_features)
        
        return regression_output, classification_logits

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Import the model class from your module or redefine it here

    # Initialize model with the same parameters used during training
    model = UnpooledBindingPredictor(
        esm_model_name="facebook/esm2_t33_650M_UR50D",
        hidden_dim=384,
        kernel_sizes=[3, 5, 7],
        n_heads=8,
        n_layers=4,
        dropout=0.14561457009902096,
        freeze_esm=True
    ).to(device)
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    return model


def prepare_inputs(protein_sequence, binder_sequence, tokenizer, max_length=1024, device='cuda'):
    """Tokenize protein and binder sequences."""
    protein_tokens = tokenizer(
        protein_sequence,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    )
    
    binder_tokens = tokenizer(
        binder_sequence,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    )
    
    return {
        'protein_input_ids': protein_tokens['input_ids'].to(device),
        'protein_attention_mask': protein_tokens['attention_mask'].to(device),
        'binder_input_ids': binder_tokens['input_ids'].to(device),
        'binder_attention_mask': binder_tokens['attention_mask'].to(device)
    }

# Perform prediction
def predict_binding(model, protein_sequence, binder_sequence, device='cuda'):
    """Predict binding affinity between protein and binder sequences."""
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    inputs = prepare_inputs(protein_sequence, binder_sequence, tokenizer, device=device)
    
    with torch.no_grad():
        regression_output, classification_logits = model(
            inputs['protein_input_ids'],
            inputs['binder_input_ids'],
            inputs['protein_attention_mask'],
            inputs['binder_attention_mask']
        )
    
    # Get numerical prediction (pKd/pKi)
    predicted_affinity = regression_output.item()
    
    # Get classification prediction (tight, medium, weak)
    predicted_class_idx = torch.argmax(classification_logits, dim=1).item()
    class_names = ['Tight binding', 'Medium binding', 'Weak binding']
    predicted_class = class_names[predicted_class_idx]
    
    # Get class probabilities
    class_probs = F.softmax(classification_logits, dim=1).cpu().numpy()[0]
    
    return {
        'predicted_affinity': predicted_affinity,
        'binding_class': predicted_class,
        'class_probabilities': {name: prob for name, prob in zip(class_names, class_probs)},
        'tight_threshold': model.tight_threshold,  # 7.5 (≤ ~30nM)
        'weak_threshold': model.weak_threshold     # 6.0 (> 1μM)
    }

# Example usage
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = load_model(r'F:\pbg-h100\flow_matching\classifier_ckpt\binding_affinity_unpooled.pt', device)
    
    # Example protein sequences (replace with actual sequences)
    binders = ['DFVAAGV', 'ELDAWAS']
    protein_sequence = "RITLKESGPPLVKPTQTLTLTCSFSGFSLSDFGVGVGWIRQPPGKALEWLAIIYSDDDKRYSPSLNTRLTITKDTSKNQVVLVMTRVSPVDTATYFCAHRRGPTTLFGVPIARGPVNAMDVWGQGITVTISSTSTKGPSVFPLAPSGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYTCNVNHKPSNTKVDKRVEPKSC"
    
    # name = "CLIC1_10_moppit"
    # print(name)
    # with open(f'/home/tc415/flow_matching/samples/unconditional_samples/12.txt', 'r') as f:
    #     binders = f.readlines()
    # binders = [binder.strip() for binder in binders]
    # binders = binders[:100]

    # # Make prediction
    affinities = []
    for binder in binders:
        result = predict_binding(model, protein_sequence, binder, device)
        print(result['predicted_affinity'])
        affinities.append(result['predicted_affinity'])

    # with open('/home/tc415/flow_matching/scores/affinity/EWSFLI1_12_unconditional.txt', 'w') as f:
    #     for score in affinities:
    #         f.write(str(score) + '\n')    
    
    # print(sum(affinities) / len(affinities))

    # with open(f'/home/tc415/flow_matching/scores/affinity/{name}.txt', 'w') as f:
    #     for score in affinities:
    #         f.write(str(round(score, 4)) + '\n')
    
    # Display results
    # print(f"Predicted binding affinity (pKd/pKi): {result['predicted_affinity']:.2f}")
    # print(f"Binding class: {result['binding_class']}")
    # print("Class probabilities:")
    # for class_name, prob in result['class_probabilities'].items():
    #     print(f"  {class_name}: {prob:.2f}")