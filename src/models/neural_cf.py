import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class RatingsDataset(Dataset):
    def __init__(self, user_indices, item_indices, ratings):
        self.user_indices = torch.tensor(user_indices, dtype=torch.long)
        self.item_indices = torch.tensor(item_indices, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_indices[idx], self.item_indices[idx], self.ratings[idx]

class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=64, hidden_layers=[128,64,32], dropout=0.2):
        super(NeuMF, self).__init__()
        
        # GMF embeddings
        self.user_embed_gmf = nn.Embedding(n_users, embed_dim)
        self.item_embed_gmf = nn.Embedding(n_items, embed_dim)
        
        # MLP embeddings
        self.user_embed_mlp = nn.Embedding(n_users, embed_dim)
        self.item_embed_mlp = nn.Embedding(n_items, embed_dim)
        
        # MLP layers — built dynamically so any hidden_layers list works
        mlp_seq = []
        in_dim = 2 * embed_dim
        for h in hidden_layers:
            mlp_seq += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.mlp_layers = nn.Sequential(*mlp_seq)
        
        # Final prediction layer
        self.final_layer = nn.Linear(embed_dim + hidden_layers[-1], 1)
        
    def forward(self, user, item):
        # GMF branch
        user_gmf = self.user_embed_gmf(user)
        item_gmf = self.item_embed_gmf(item)
        gmf_out = user_gmf * item_gmf
        
        # MLP branch
        user_mlp = self.user_embed_mlp(user)
        item_mlp = self.item_embed_mlp(item)
        mlp_in = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_out = self.mlp_layers(mlp_in)
        
        # Concat and output
        concat_out = torch.cat([gmf_out, mlp_out], dim=-1)
        pred = self.final_layer(concat_out)
        
        return pred.squeeze(-1)

class NeuralCFRecommender:
    def __init__(self, n_users, n_items, embed_dim=64, hidden_layers=[128,64,32], dropout=0.2, lr=0.001, batch_size=1024):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuMF(n_users, n_items, embed_dim, hidden_layers, dropout).to(self.device)
        self.lr = lr
        self.batch_size = batch_size
        self.user_encoder = None
        self.item_encoder = None
        self.train_losses = []
        
    def fit(self, train_df, val_df=None, epochs=20, patience=3):
        train_dataset = RatingsDataset(train_df['user_idx'].values, train_df['item_idx'].values, train_df['rating'].values)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        best_val_rmse = float('inf')
        patience_counter = 0
        
        os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
        save_path = os.path.join('data', 'processed', 'best_ncf.pt')
        
        print("Training Neural CF Model...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for u, i, r in pbar:
                u, i, r = u.to(self.device), i.to(self.device), r.to(self.device)
                
                optimizer.zero_grad()
                preds = self.model(u, i)
                loss = criterion(preds, r)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            avg_loss = total_loss / len(train_loader)
            self.train_losses.append(avg_loss)
            
            if val_df is not None:
                # Bug fix: mini-batch validation to avoid OOM on large test sets
                self.model.eval()
                val_preds_all = []
                val_r = val_df['rating'].values
                VAL_BATCH = 4096
                with torch.no_grad():
                    for start in range(0, len(val_df), VAL_BATCH):
                        end = start + VAL_BATCH
                        val_u = torch.tensor(val_df['user_idx'].values[start:end], dtype=torch.long).to(self.device)
                        val_i = torch.tensor(val_df['item_idx'].values[start:end], dtype=torch.long).to(self.device)
                        batch_preds = self.model(val_u, val_i).cpu().numpy()
                        val_preds_all.extend(batch_preds)
                val_preds = np.array(val_preds_all)
                val_rmse = np.sqrt(((val_preds - val_r) ** 2).mean())

                print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val RMSE: {val_rmse:.4f}")

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    patience_counter = 0
                    torch.save(self.model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")
                torch.save(self.model.state_dict(), save_path)
                
        if val_df is not None and os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path))

    def predict(self, user_id, item_id):
        if not self.user_encoder or not self.item_encoder:
            raise ValueError("user_encoder and item_encoder must be set to predict.")
            
        try:
            u_idx = self.user_encoder.transform([user_id])[0]
            i_idx = self.item_encoder.transform([item_id])[0]
        except ValueError:
            return 3.0  # Cold start
            
        self.model.eval()
        with torch.no_grad():
            u_tens = torch.tensor([u_idx], dtype=torch.long).to(self.device)
            i_tens = torch.tensor([i_idx], dtype=torch.long).to(self.device)
            pred = self.model(u_tens, i_tens).item()
            
        return max(1.0, min(5.0, pred))

    def get_top_n(self, user_id, all_item_ids, n=10, exclude_rated=True, user_rated_items=None):
        if not self.user_encoder or not self.item_encoder:
            raise ValueError("user_encoder and item_encoder must be set to predict.")
            
        try:
            u_idx = self.user_encoder.transform([user_id])[0]
        except ValueError:
            return []
            
        valid_items = []
        for iid in all_item_ids:
            if exclude_rated and user_rated_items and iid in user_rated_items:
                continue
            valid_items.append(iid)
            
        if not valid_items:
            return []
            
        # Safely encode items
        encoded_items = []
        filtered_items = []
        for iid in valid_items:
            try:
                idx = self.item_encoder.transform([iid])[0]
                encoded_items.append(idx)
                filtered_items.append(iid)
            except ValueError:
                pass
                
        if not filtered_items:
            return []
            
        self.model.eval()
        with torch.no_grad():
            u_tens = torch.tensor([u_idx]*len(filtered_items), dtype=torch.long).to(self.device)
            i_tens = torch.tensor(encoded_items, dtype=torch.long).to(self.device)
            preds = self.model(u_tens, i_tens).cpu().numpy()
            
        predictions = list(zip(filtered_items, preds))
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'user_encoder': self.user_encoder,
                'item_encoder': self.item_encoder,
                'lr': self.lr,
                'batch_size': self.batch_size,
                'train_losses': self.train_losses
            }, f)
        
        model_path = path.replace('.pkl', '.pt')
        torch.save(self.model.state_dict(), model_path)
        
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.user_encoder = data['user_encoder']
            self.item_encoder = data['item_encoder']
            self.lr = data['lr']
            self.batch_size = data['batch_size']
            self.train_losses = data['train_losses']
            
        model_path = path.replace('.pkl', '.pt')
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
