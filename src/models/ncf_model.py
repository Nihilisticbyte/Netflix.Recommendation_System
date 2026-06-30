"""
src/models/ncf_model.py
-----------------------
Neural Collaborative Filtering (NCF) — He et al., 2017.

Architecture:
    - GMF branch : element-wise product of user and item embeddings
    - MLP branch : concatenated embeddings through dense layers
    - NeuMF      : concat(GMF output, MLP output) → sigmoid → rating

We train with MSE loss on explicit ratings (1–5 scale).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# PyTorch model definition
# ---------------------------------------------------------------------------

class NeuMF(nn.Module):
    """Neural Matrix Factorization (GMF + MLP fusion)."""

    def __init__(
        self,
        n_users: int,
        n_movies: int,
        gmf_dim: int = 32,
        mlp_dims: list = None,
        dropout: float = 0.4,
    ):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [64, 32, 16]

        # GMF embeddings
        self.gmf_user_emb = nn.Embedding(n_users, gmf_dim)
        self.gmf_item_emb = nn.Embedding(n_movies, gmf_dim)

        # MLP embeddings
        mlp_in = mlp_dims[0]
        self.mlp_user_emb = nn.Embedding(n_users, mlp_in // 2)
        self.mlp_item_emb = nn.Embedding(n_movies, mlp_in // 2)

        # MLP layers
        layers = []
        in_dim = mlp_in
        for out_dim in mlp_dims[1:]:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # Output layer: GMF output (gmf_dim) + MLP output (mlp_dims[-1]) → 1
        self.output = nn.Linear(gmf_dim + mlp_dims[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user_emb, self.gmf_item_emb,
                    self.mlp_user_emb, self.mlp_item_emb]:
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # GMF branch
        gmf_u = self.gmf_user_emb(user_ids)
        gmf_i = self.gmf_item_emb(item_ids)
        gmf_out = gmf_u * gmf_i  # element-wise product

        # MLP branch
        mlp_u = self.mlp_user_emb(user_ids)
        mlp_i = self.mlp_item_emb(item_ids)
        mlp_out = self.mlp(torch.cat([mlp_u, mlp_i], dim=-1))

        # Fusion
        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        out = self.output(fused).squeeze(-1)

        # Scale sigmoid output from [0,1] → [1,5]
        return 1.0 + 4.0 * torch.sigmoid(out)


# ---------------------------------------------------------------------------
# Wrapper class matching our model interface
# ---------------------------------------------------------------------------

class NCFModel:
    def __init__(
        self,
        gmf_dim: int = 32,
        mlp_dims: list = None,
        lr: float = 1e-3,
        batch_size: int = 1024,
        epochs: int = 10,
        dropout: float = 0.2,
        device: str = None,
        random_state: int = 42,
    ):
        if mlp_dims is None:
            mlp_dims = [64, 32, 16]
        self.gmf_dim = gmf_dim
        self.mlp_dims = mlp_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.random_state = random_state

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.rating_matrix_csr = None
        self.n_users = None
        self.n_movies = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame):
        """Train NeuMF on explicit ratings."""
        torch.manual_seed(self.random_state)
        self.n_users = train_df["user_idx"].max() + 1
        self.n_movies = train_df["movie_idx"].max() + 1

        self.model = NeuMF(
            n_users=self.n_users,
            n_movies=self.n_movies,
            gmf_dim=self.gmf_dim,
            mlp_dims=self.mlp_dims,
            dropout=self.dropout,
        ).to(self.device)

        # Add weight_decay to your Adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        users_t = torch.LongTensor(train_df["user_idx"].values)
        items_t = torch.LongTensor(train_df["movie_idx"].values)
        ratings_t = torch.FloatTensor(train_df["rating"].values)

        dataset = TensorDataset(users_t, items_t, ratings_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Store rating matrix for recommendation generation
        from scipy.sparse import csr_matrix
        sparse = csr_matrix(
            (train_df["rating"].values,
             (train_df["user_idx"].values, train_df["movie_idx"].values)),
            shape=(self.n_users, self.n_movies),
        )
        self.rating_matrix_csr = sparse

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for u_batch, i_batch, r_batch in loader:
                u_batch = u_batch.to(self.device)
                i_batch = i_batch.to(self.device)
                r_batch = r_batch.to(self.device)

                optimizer.zero_grad()
                preds = self.model(u_batch, i_batch)
                loss = criterion(preds, r_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(r_batch)

            rmse = (total_loss / len(train_df)) ** 0.5
            print(f"  [NCF] Epoch {epoch:2d}/{self.epochs} | Train RMSE: {rmse:.4f}")

        print(f"[NCF] Fit complete: {self.n_users} users × {self.n_movies} movies")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_rating(self, user_idx: int, movie_idx: int) -> float:
        self.model.eval()
        with torch.no_grad():
            u = torch.LongTensor([user_idx]).to(self.device)
            m = torch.LongTensor([movie_idx]).to(self.device)
            pred = self.model(u, m).item()
        return float(np.clip(pred, 1.0, 5.0))

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Batch prediction for all rows in test_df."""
        self.model.eval()
        users_t = torch.LongTensor(test_df["user_idx"].values).to(self.device)
        items_t = torch.LongTensor(test_df["movie_idx"].values).to(self.device)

        preds = []
        with torch.no_grad():
            for i in range(0, len(users_t), self.batch_size):
                u_b = users_t[i:i + self.batch_size]
                m_b = items_t[i:i + self.batch_size]
                p = self.model(u_b, m_b).cpu().numpy()
                preds.append(p)

        return np.clip(np.concatenate(preds), 1.0, 5.0)

    # ------------------------------------------------------------------
    # Recommendation generation
    # ------------------------------------------------------------------

    def recommend(self, user_idx: int, top_k: int = 10, seen_movies: set = None) -> list:
        """Return Top-K recommended movie indices for a user."""
        if seen_movies is None:
            seen_movies = set(self.rating_matrix_csr[user_idx].nonzero()[1])

        candidates = [m for m in range(self.n_movies) if m not in seen_movies]
        users_t = torch.LongTensor([user_idx] * len(candidates)).to(self.device)
        items_t = torch.LongTensor(candidates).to(self.device)

        self.model.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(candidates), self.batch_size):
                u_b = users_t[i:i + self.batch_size]
                m_b = items_t[i:i + self.batch_size]
                s = self.model(u_b, m_b).cpu().numpy()
                all_scores.append(s)

        scores = np.clip(np.concatenate(all_scores), 1.0, 5.0)
        top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        return [(candidates[i], float(scores[i])) for i in top_k_idx]

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"[NCF] Model saved to {path}")

    def load(self, path: str, n_users: int, n_movies: int):
        self.n_users = n_users
        self.n_movies = n_movies
        self.model = NeuMF(
            n_users=n_users, n_movies=n_movies,
            gmf_dim=self.gmf_dim, mlp_dims=self.mlp_dims, dropout=self.dropout
        ).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[NCF] Model loaded from {path}")
