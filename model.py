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
        x = self.embedding(input_ids)
        x = x.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            conv_x = torch.relu(conv(x))
            pooled_x = torch.max(conv_x, dim=2).values
            conv_outputs.append(pooled_x)

        feature = torch.cat(conv_outputs, dim=1)
        return self.dropout(feature)


class AblationEmotionCNN(nn.Module):
    VALID_BRANCHES = ("char", "word", "phrase")

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
        enabled_branches=None,
        use_gate=True,
    ):
        super().__init__()
        if enabled_branches is None:
            enabled_branches = list(self.VALID_BRANCHES)

        invalid_branches = [branch for branch in enabled_branches if branch not in self.VALID_BRANCHES]
        if invalid_branches:
            raise ValueError(f"Unsupported branches: {invalid_branches}")
        if len(enabled_branches) == 0:
            raise ValueError("enabled_branches must contain at least one branch")
        if use_gate and set(enabled_branches) != set(self.VALID_BRANCHES):
            raise ValueError("use_gate=True is only supported when char, word, and phrase are all enabled")

        self.enabled_branches = list(enabled_branches)
        self.use_gate = bool(use_gate)
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

        self.branch_modules = nn.ModuleDict({
            "char": self.char_branch,
            "word": self.word_branch,
            "phrase": self.phrase_branch,
        })
        self.branch_dims = {
            "char": self.char_branch.output_dim,
            "word": self.word_branch.output_dim,
            "phrase": self.phrase_branch.output_dim,
        }
        total_feature_dim = sum(self.branch_dims[branch] for branch in self.enabled_branches)
        self.total_feature_dim = total_feature_dim

        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Linear(total_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(hidden_dim, len(self.enabled_branches)),
            )
        else:
            self.gate = None

        self.classifier = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, char_input_ids, word_input_ids, phrase_input_ids, return_gates=False):
        inputs = {
            "char": char_input_ids,
            "word": word_input_ids,
            "phrase": phrase_input_ids,
        }
        features = [
            self.branch_modules[branch](inputs[branch])
            for branch in self.enabled_branches
        ]

        if self.use_gate:
            raw_fused_feature = torch.cat(features, dim=1)
            gate_logits = self.gate(raw_fused_feature)
            gate_weights = torch.sigmoid(gate_logits)
            gate_weights = self.gate_min_value + (1 - self.gate_min_value) * gate_weights
            gated_features = [
                feature * gate_weights[:, idx].unsqueeze(1)
                for idx, feature in enumerate(features)
            ]
            fused_feature = torch.cat(gated_features, dim=1)
        else:
            fused_feature = features[0] if len(features) == 1 else torch.cat(features, dim=1)
            gate_weights = None

        logits = self.classifier(fused_feature)
        if return_gates:
            return logits, gate_weights
        return logits


class MultiViewEmotionCNN(AblationEmotionCNN):
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
        super().__init__(
            char_vocab_size=char_vocab_size,
            word_vocab_size=word_vocab_size,
            phrase_vocab_size=phrase_vocab_size,
            char_embed_dim=char_embed_dim,
            word_embed_dim=word_embed_dim,
            phrase_embed_dim=phrase_embed_dim,
            char_num_filters=char_num_filters,
            word_num_filters=word_num_filters,
            phrase_num_filters=phrase_num_filters,
            char_kernel_sizes=char_kernel_sizes,
            word_kernel_sizes=word_kernel_sizes,
            phrase_kernel_sizes=phrase_kernel_sizes,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            branch_dropout=branch_dropout,
            classifier_dropout=classifier_dropout,
            gate_min_value=gate_min_value,
            enabled_branches=["char", "word", "phrase"],
            use_gate=True,
        )
