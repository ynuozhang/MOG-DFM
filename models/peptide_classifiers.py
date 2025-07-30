import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from transformers import EsmModel
import xgboost as xgb
import numpy as np

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL

from models.peptide_models import CNNModel

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
        # classification_logits = self.classification_head(shared_features)
        
        # return regression_output, classification_logits
        return regression_output

class ImprovedBindingPredictor(nn.Module):
    def __init__(self, 
                 esm_dim=1280,
                 smiles_dim=1280,
                 hidden_dim=512,
                 n_heads=8,
                 n_layers=5,
                 dropout=0.1):
        super().__init__()
        
        # Define binding thresholds
        self.tight_threshold = 7.5    # Kd/Ki/IC50 ≤ ~30nM
        self.weak_threshold = 6.0     # Kd/Ki/IC50 > 1μM
        
        # Project to same dimension
        self.smiles_projection = nn.Linear(smiles_dim, hidden_dim)
        self.protein_projection = nn.Linear(esm_dim, hidden_dim)
        self.protein_norm = nn.LayerNorm(hidden_dim)
        self.smiles_norm = nn.LayerNorm(hidden_dim)
        
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
        
    def forward(self, protein_emb, binder_emb):
        
        protein = self.protein_norm(self.protein_projection(protein_emb))
        smiles = self.smiles_norm(self.smiles_projection(binder_emb))
        
        protein = protein.transpose(0, 1)
        smiles = smiles.transpose(0, 1)
        
        # Cross attention layers
        for layer in self.cross_attention_layers:
            # Protein attending to SMILES
            attended_protein = layer['attention'](
                protein, smiles, smiles
            )[0]
            protein = layer['norm1'](protein + attended_protein)
            protein = layer['norm2'](protein + layer['ffn'](protein))
            
            # SMILES attending to protein
            attended_smiles = layer['attention'](
                smiles, protein, protein
            )[0]
            smiles = layer['norm1'](smiles + attended_smiles)
            smiles = layer['norm2'](smiles + layer['ffn'](smiles))
        
        # Get sequence-level representations
        protein_pool = torch.mean(protein, dim=0)
        smiles_pool = torch.mean(smiles, dim=0)
        
        # Concatenate both representations
        combined = torch.cat([protein_pool, smiles_pool], dim=-1)
        
        # Shared features
        shared_features = self.shared_head(combined)
        
        regression_output = self.regression_head(shared_features)
        
        return regression_output

class PooledAffinityModel(nn.Module):
    def __init__(self, affinity_predictor, target_sequence):
        super(PooledAffinityModel, self).__init__()
        self.affinity_predictor = affinity_predictor
        self.target_sequence = target_sequence
        self.esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(self.target_sequence.device)
        for param in self.esm_model.parameters():
            param.requires_grad = False
    
    def compute_embeddings(self, input_ids, attention_mask=None):
        """Compute ESM embeddings on the fly"""
        esm_outputs = self.esm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get the unpooled last hidden states (batch_size x seq_length x hidden_size)
        return esm_outputs.last_hidden_state
    
    def forward(self, x):
        target_sequence = self.target_sequence.repeat(x.shape[0], 1)

        protein_emb = self.compute_embeddings(input_ids=target_sequence)
        binder_emb = self.compute_embeddings(input_ids=x)
        return self.affinity_predictor(protein_emb=protein_emb, binder_emb=binder_emb).squeeze(-1)  

class AffinityModel(nn.Module):
    def __init__(self, affinity_predictor, target_sequence):
        super(AffinityModel, self).__init__()
        self.affinity_predictor = affinity_predictor
        self.target_sequence = target_sequence
    
    def forward(self, x):
        target_sequence = self.target_sequence.repeat(x.shape[0], 1)
        return self.affinity_predictor(protein_input_ids=target_sequence, binder_input_ids=x).squeeze(-1) 

class HemolysisModel:
    def __init__(self, device):
        self.predictor = xgb.Booster(model_file='./classifier_ckpt/best_model_hemolysis.json')
        
        self.model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
        self.model.eval()

        self.device = device
    
    def generate_embeddings(self, sequences):
        """Generate ESM embeddings for protein sequences"""
        with torch.no_grad():
            embeddings = self.model(input_ids=sequences).last_hidden_state.mean(dim=1)
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def get_scores(self, input_seqs):
        scores = np.ones(len(input_seqs))
        features = self.generate_embeddings(input_seqs)
        
        if len(features) == 0:
            return scores
        
        features = np.nan_to_num(features, nan=0.)
        features = np.clip(features, np.finfo(np.float32).min, np.finfo(np.float32).max)
        
        features = xgb.DMatrix(features)
        
        probs = self.predictor.predict(features)
        # return the probability of it being not hemolytic
        return torch.from_numpy(scores - probs).to(self.device)
    
    def __call__(self, input_seqs: list):
        scores = self.get_scores(input_seqs)
        return scores

class NonfoulingModel:
    def __init__(self, device):
        # change model path
        self.predictor = xgb.Booster(model_file='./classifier_ckpt/best_model_nonfouling.json')
        
        self.model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
        self.model.eval()

        self.device = device
    
    def generate_embeddings(self, sequences):
        """Generate ESM embeddings for protein sequences"""
        with torch.no_grad():
            embeddings = self.model(input_ids=sequences).last_hidden_state.mean(dim=1)
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def get_scores(self, input_seqs):
        scores = np.zeros(len(input_seqs))
        features = self.generate_embeddings(input_seqs)
        
        if len(features) == 0:
            return scores
        
        features = np.nan_to_num(features, nan=0.)
        features = np.clip(features, np.finfo(np.float32).min, np.finfo(np.float32).max)
        
        features = xgb.DMatrix(features)
        
        scores = self.predictor.predict(features)
        return torch.from_numpy(scores).to(self.device)
    
    def __call__(self, input_seqs: list):
        scores = self.get_scores(input_seqs)
        return scores

class SolubilityModel:
    def __init__(self, device):
        # change model path
        self.predictor = xgb.Booster(model_file='./classifier_ckpt/best_model_solubility.json')
        
        self.model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
        self.model.eval()

        self.device = device
    
    def generate_embeddings(self, sequences):
        """Generate ESM embeddings for protein sequences"""
        with torch.no_grad():
            embeddings = self.model(input_ids=sequences).last_hidden_state.mean(dim=1)
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def get_scores(self, input_seqs: list):
        scores = np.zeros(len(input_seqs))
        features = self.generate_embeddings(input_seqs)
        
        if len(features) == 0:
            return scores
        
        features = np.nan_to_num(features, nan=0.)
        features = np.clip(features, np.finfo(np.float32).min, np.finfo(np.float32).max)
        
        features = xgb.DMatrix(features)
        
        scores = self.predictor.predict(features)
        return torch.from_numpy(scores).to(self.device)
    
    def __call__(self, input_seqs: list):
        scores = self.get_scores(input_seqs)
        return scores

class PeptideCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dims[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=5, padding=1)
        self.fc = nn.Linear(hidden_dims[1], output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.predictor = nn.Linear(output_dim, 1)  # For regression/classification

        self.esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.esm_model.eval()

    def forward(self, input_ids, attention_mask=None, return_features=False):
        with torch.no_grad():
            x = self.esm_model(input_ids, attention_mask).last_hidden_state
        # x shape: (B, L, input_dim)
        x = x.permute(0, 2, 1)  # Reshape to (B, input_dim, L) for Conv1d
        x = nn.functional.relu(self.conv1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # Reshape back to (B, L, hidden_dims[1])
        
        # Global average pooling over the sequence dimension (L)
        x = x.mean(dim=1)  # Shape: (B, hidden_dims[1])
        
        features = self.fc(x)  # features shape: (B, output_dim)
        if return_features:
            return features
        return self.predictor(features)  # Output shape: (B, 1)

class HalfLifeModel:
    def __init__(self, device):
        input_dim = 1280
        hidden_dims = [input_dim // 2, input_dim // 4]
        output_dim = input_dim // 8
        dropout_rate = 0.3
        self.model = PeptideCNN(input_dim, hidden_dims, output_dim, dropout_rate).to(device)
        self.model.load_state_dict(torch.load('./classifier_ckpt/best_model_half_life.pth'))
        self.model.eval()

    def __call__(self, x):
        prediction = self.model(x, return_features=False)
        return torch.clamp(prediction.squeeze(-1), max=2.0)


def load_solver(checkpoint_path, vocab_size, device):
    lr = 1e-4
    epochs = 200
    embed_dim = 512
    hidden_dim = 256
    epsilon = 1e-3
    batch_size = 256
    warmup_epochs = epochs // 10
    device = 'cuda:0'
    

    probability_denoiser = CNNModel(alphabet_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    probability_denoiser.load_state_dict(torch.load(checkpoint_path, map_location=device))
    probability_denoiser.eval()
    for param in probability_denoiser.parameters():
        param.requires_grad = False

    # instantiate a convex path object
    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)

    class WrappedModel(ModelWrapper):
        def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
            return torch.softmax(self.model(x, t), dim=-1)

    wrapped_probability_denoiser = WrappedModel(probability_denoiser)
    solver = MixtureDiscreteEulerSolver(model=wrapped_probability_denoiser, path=path, vocabulary_size=vocab_size)

    return solver


def load_pooled_affinity_predictor(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = ImprovedBindingPredictor().to(device)
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    return model

def load_affinity_predictor(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = UnpooledBindingPredictor(
        esm_model_name="facebook/esm2_t33_650M_UR50D",
        hidden_dim=384,
        kernel_sizes=[3, 5, 7],
        n_heads=8,
        n_layers=4,
        dropout=0.14561457009902096,
        freeze_esm=True
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() 
    
    return model
