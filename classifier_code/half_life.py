import numpy as np
import torch
import xgboost as xgb
from transformers import EsmModel, EsmTokenizer
import torch.nn as nn
import pdb

class PeptideCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dims[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=5, padding=1)
        self.fc = nn.Linear(hidden_dims[1], output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.predictor = nn.Linear(output_dim, 1)  # For regression/classification

        self.esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
        self.esm_model.eval()

    def forward(self, input_ids, attention_mask=None, return_features=False):
        with torch.no_grad():
            x = self.esm_model(input_ids, attention_mask).last_hidden_state
        # pdb.set_trace()
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 1280
hidden_dims = [input_dim // 2, input_dim // 4]
output_dim = input_dim // 8
dropout_rate = 0.3

nn_model = PeptideCNN(input_dim, hidden_dims, output_dim, dropout_rate).to(device)
nn_model.load_state_dict(torch.load('../classifier_ckpt/half_life.pth'))
nn_model.eval()

def predict(inputs):
    with torch.no_grad():
        prediction = nn_model(**inputs, return_features=False)

    return prediction.item()

if __name__ == '__main__':
    sequence = 'TATVVAFKDK' 
    
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

    prediction = predict(inputs)
    print(prediction)
    print(f"Predicted half life of {sequence} is {(10**prediction):.4f} h")
