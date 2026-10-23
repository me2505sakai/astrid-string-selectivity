import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import misc_utils


#We choose a simple DL model.
#More complex models could give better results
class SelectivityEstimator(nn.Module):
    def __init__(self, string_helper, configs):
        super(SelectivityEstimator, self).__init__()
        self.embedding_dimension = configs.embedding_dimension
        self.string_helper = string_helper
        self.max_string_length = self.string_helper.max_string_length
        self.alphabet_size = self.string_helper.alphabet_size

        layer_sizes = [128, 64, 32, 16, 8]
        self.model = nn.Sequential(
            nn.Linear(self.embedding_dimension, layer_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(layer_sizes[0]),
            nn.Dropout(0.001),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(layer_sizes[1]),
            nn.Dropout(0.01),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
            nn.BatchNorm1d(layer_sizes[2]),
            nn.Dropout(0.01),
            nn.Linear(layer_sizes[2], layer_sizes[3]),
            nn.ReLU(),
            nn.BatchNorm1d(layer_sizes[3]),
            nn.Dropout(0.01),
            nn.Linear(layer_sizes[3], layer_sizes[4]),
            nn.ReLU(),
            nn.BatchNorm1d(layer_sizes[4]),
            nn.Dropout(0.01),

            nn.Linear(layer_sizes[4], 1),
        )

    def forward(self, x):
        x = torch.sigmoid(self.model(x))
        return x



def train_selEst_model(configs, train_loader,  string_helper, log_output_file_path):
    model = SelectivityEstimator(string_helper, configs)
    #Comment this line during experimentation.
    #model = torch.jit.script(model)
    model = model.to(configs.device)

    optimizer = optim.Adam(model.parameters(), lr=configs.lr)
    device = configs.device

    # for logging
    history = []

    for epoch in tqdm(range(configs.num_epochs), desc="Epochs", position=0, leave=False):
        # learning phase
        model.train()
        running_train_loss = []

        for batch_idx, (string_queries, true_selectivities) in enumerate(tqdm(train_loader, desc="Training", leave=False, disable=not configs.log_by_batch)):
        #for batch_idx, output in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            string_queries = string_queries.to(device)

            optimizer.zero_grad()
            predicted_selectivities = model(string_queries)
            loss =  misc_utils.qerror_loss(predicted_selectivities, true_selectivities.float(), configs.min_val, configs.max_val)

            loss.backward()
            optimizer.step()
            running_train_loss.append(loss.item())

        mean_train_loss = np.mean(running_train_loss)

        print(f"Epoch: {epoch+1}/{configs.num_epochs} - Train Loss: {mean_train_loss:.4f}")
        print("Summary stats of Loss: Percentile: [0.75, 0.9, 0.95, 0.99] ", [np.quantile(running_train_loss, q) for q in [0.75, 0.9, 0.95, 0.99]])

        history.append({
            "epoch": epoch + 1,
            "train_loss": mean_train_loss
        })

    log_df = pd.DataFrame(history)
    log_df.to_csv(log_output_file_path, index=False)
    print(f"Training log successfully saved to {log_output_file_path}")

    return model
