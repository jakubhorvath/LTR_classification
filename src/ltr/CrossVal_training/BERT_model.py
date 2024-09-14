import torch 
from torch import nn
import transformers
class LTRBERT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = transformers.BertForSequenceClassification.from_pretrained('zhihan1996/DNA_bert_6', num_labels=num_classes)

        self.long_layers = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * (768 - 3 + 1) // 2, 128),  # Adjust input size to match the output of Conv1D + MaxPool1D
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ]
        )

    def predict_short(self, input_ids):
        outputs = self.bert(input_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def predict_long(self, input_ids):
        window_size = 350
        stride = 116 # ~ 1/3 of window size

        outputs = []
        sequences = []
        for seq in X_train:
            seq_windows = []
            for i in range(0, len(seq), stride):
                start = i
                end = i + window_size

                if end > len(seq):
                    end = len(seq)
                seq_windows.append(seq[start:end])
            sequences.append(seq_windows)

    # TODO find a way to do this without 
        counter = 0
        for s in sequences:
            if counter % 500 == 0 and counter != 0:
                print(f"processing sequence {counter}")
            X_test_tokenized = tokenizer([tok_func(x) for x in s], padding=True, truncation=True, max_length=350) # Create torch dataset
            test_dataset = Dataset(X_test_tokenized) # Load trained model
            test_trainer = Trainer(model) # Make prediction
            output, _, _ = test_trainer.predict(test_dataset) # Preprocess raw predictions
            outputs.append(output)

        return outputs
     
    def forward(self, input_ids, attention_mask):

        # for short sequences
        if input_ids.shape[1] <= 512:
            return self.predict_short(input_ids)

        # for long sequences
        if input_ids.shape[1] > 512:
            pooled_logits = self.predict_long(input_ids)
            for layer in self.long_layers:
                pooled_logits = layer(pooled_logits)
            return pooled_logits