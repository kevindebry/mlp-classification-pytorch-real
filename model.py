import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_filters,
        kernel_sizes,
        hidden_dim,
        num_classes,
        dropout,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=kernel_size,
            )
            for kernel_size in kernel_sizes
        ])
        feature_dim = num_filters * len(kernel_sizes)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            conv_x = torch.relu(conv(x))
            pooled_x = torch.max(conv_x, dim=2).values
            conv_outputs.append(pooled_x)

        features = torch.cat(conv_outputs, dim=1)
        return self.classifier(features)
