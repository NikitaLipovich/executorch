import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SelfAttentiveEncoder(nn.Module):

    def __init__(self):
        super(SelfAttentiveEncoder, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.ws1 = nn.Linear(128, 20, bias=False)  # Keeping 128 to avoid increasing params significantly
        self.ws2 = nn.Linear(20, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.attention_hops = 1

    def forward(self, outp):
        size = outp.size()  # [bsz, len, nhid] e.g., bsz, seq_len, 128
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, 20]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, 1]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, 1, len]
        alphas = self.softmax(alphas.view(-1, size[1]))  # [bsz*1, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, 1, len]
        return torch.bmm(alphas, outp)  # [bsz, 1, nhid]

class A_LSTM(nn.Module):  # Renamed to A_Transformer for clarity, but keeping name
    def __init__(self, sequence_size=12):
        super(A_LSTM, self).__init__()
        self.sequence_size = sequence_size
        d_model = 128  # Matching the unidirectional hidden size to keep params low
        num_layers = 2  # To approximate stacked GRU
        nhead = 4  # Reasonable number of heads
        dim_feedforward = 128  # Small FF dim to avoid param increase (approx 200k total for transformer part)

        self.embed = nn.Linear(39, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,  # No additional dropout here
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.bn1 = nn.BatchNorm1d(sequence_size)
        self.selfattention = SelfAttentiveEncoder()
        self.fc1 = nn.Linear(d_model, 320)  # Using d_model=128 to keep params similar
        self.bn2 = nn.BatchNorm1d(320)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(320, 320)
        self.bn3 = nn.BatchNorm1d(320)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(320, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(39)

    def forward(self, x):  # input: length, bsz, 39
        length, bsz, feature = x.shape
        x = x.contiguous().view(-1, feature)
        x = self.bn5(x)
        x = x.contiguous().view(length, bsz, feature)

        # Transformer expects (bsz, seq, features)
        x = x.permute(1, 0, 2)  # bsz, length, 39
        x = self.embed(x)  # bsz, length, d_model

        # Optional: Add positional encoding if needed (not added here to minimize params)
        # For short sequences like 12, it might not be critical, but you can add if desired

        x = self.transformer(x)  # bsz, length, d_model=128

        x = self.bn1(x)
        x = self.selfattention(x).squeeze(1)  # bsz, 128
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.bn4(x)
        return x

    def gen_embedding(self, x):
        # Similar to forward, adapted for gen_embedding
        length, bsz, feature = x.shape
        x = x.contiguous().view(-1, feature)
        x = self.bn5(x)
        x = x.contiguous().view(length, bsz, feature)
        x = x.permute(1, 0, 2)  # bsz, length, 39
        x = self.embed(x)
        x = self.transformer(x)
        x = self.bn1(x)
        x = self.selfattention(x).squeeze(1)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.bn4(x)
        return x

class MMD_NCA_Net(nn.Module):
    def __init__(self, sequence_size):
        super(MMD_NCA_Net, self).__init__()
        self.A_LSTM = A_LSTM(sequence_size)
        self.seq_emb_size = sequence_size

    def forward(self, x):
        return self.A_LSTM(x)

    def load_weights(self, weights_path, device):
        self.load_state_dict(torch.load(weights_path))
        self.to(device)
        self.eval()
        self.device = device

    def gen_embedding(self, x):
        tensor = Variable(torch.Tensor(x)).float() \
            .to(self.device).squeeze() \
            .view(-1, self.seq_emb_size, 39).permute(1, 0, 2)
        return self.A_LSTM.gen_embedding(tensor)