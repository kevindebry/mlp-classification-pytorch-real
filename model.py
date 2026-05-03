import torch
import torch.nn as nn


class CNNBranch(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes, dropout):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        self.output_dim = num_filters * len(kernel_sizes)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=kernel_size,
            )
            for kernel_size in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        # input_ids: [B, L]
        x = self.embedding(input_ids)  # [B, L, E]
        x = x.transpose(1, 2)  # [B, E, L]

        conv_outputs = []
        for conv in self.convs:
            conv_x = torch.relu(conv(x))  # [B, F, L-K+1]
            pooled_x = torch.max(conv_x, dim=2).values  # [B, F]
            conv_outputs.append(pooled_x)

        feature = torch.cat(conv_outputs, dim=1)  # [B, F * len(kernel_sizes)]
        feature = self.dropout(feature)
        return feature


class MultiViewEmotionCNN(nn.Module):
    def __init__(
            self,
            char_vocab_size,
            word_vocab_size,
            phrase_vocab_size,
            char_embed_dim,
            word_embed_dim,
            phrase_embed_dim,
            char_num_filters,
            word_num_filters,
            phrase_num_filters,
            char_kernel_sizes,
            word_kernel_sizes,
            phrase_kernel_sizes,
            hidden_dim,
            num_classes,
            branch_dropout,
            classifier_dropout,
            gate_min_value,
    ):
        super().__init__()
        self.gate_min_value = gate_min_value

        self.char_branch = CNNBranch(
            vocab_size=char_vocab_size,
            embed_dim=char_embed_dim,
            num_filters=char_num_filters,
            kernel_sizes=char_kernel_sizes,
            dropout=branch_dropout,
        )
        self.word_branch = CNNBranch(
            vocab_size=word_vocab_size,
            embed_dim=word_embed_dim,
            num_filters=word_num_filters,
            kernel_sizes=word_kernel_sizes,
            dropout=branch_dropout,
        )
        self.phrase_branch = CNNBranch(
            vocab_size=phrase_vocab_size,
            embed_dim=phrase_embed_dim,
            num_filters=phrase_num_filters,
            kernel_sizes=phrase_kernel_sizes,
            dropout=branch_dropout,
        )

        total_feature_dim = (
            self.char_branch.output_dim
            + self.word_branch.output_dim
            + self.phrase_branch.output_dim
        )
        self.gate = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, char_input_ids, word_input_ids, phrase_input_ids, return_gates=False):
        char_feature = self.char_branch(char_input_ids)
        word_feature = self.word_branch(word_input_ids)
        phrase_feature = self.phrase_branch(phrase_input_ids)

        raw_fused_feature = torch.cat([char_feature, word_feature, phrase_feature], dim=1)
        gate_logits = self.gate(raw_fused_feature)
        gate_weights = torch.sigmoid(gate_logits)
        gate_weights = self.gate_min_value + (1 - self.gate_min_value) * gate_weights

        char_weight = gate_weights[:, 0].unsqueeze(1)
        word_weight = gate_weights[:, 1].unsqueeze(1)
        phrase_weight = gate_weights[:, 2].unsqueeze(1)

        gated_char_feature = char_feature * char_weight
        gated_word_feature = word_feature * word_weight
        gated_phrase_feature = phrase_feature * phrase_weight

        fused_feature = torch.cat([gated_char_feature, gated_word_feature, gated_phrase_feature], dim=1)
        logits = self.classifier(fused_feature)

        if return_gates:
            return logits, gate_weights

        return logits
