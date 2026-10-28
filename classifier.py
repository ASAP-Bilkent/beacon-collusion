import torch.nn as nn
import torch

class OnlineDefender:
    def __init__(self, classifier, device="cuda", threshold=0.05):
        self.classifier = classifier.to(device)
        # self.threshold = threshold
        self.device = device

        # keep a buffer of past queries
        self.history = []

    def reset(self):
        """Reset history (e.g., new attacker session)."""
        self.history = []
        self.classifier.eval()

    def process_query(self, new_query):
        """
        new_query: tensor of shape (input_dim,)
        Returns: decision (1 = honest, 0 = lie)
        """
        self.eval_mode()

        # (1, 1, input_dim) for single step
        new_query = new_query.unsqueeze(0).unsqueeze(0).to(self.device)

        if len(self.history) == 0:
            # initialize hidden state once
            self.hidden = None
            prob_before = torch.tensor([0.0], device=self.device)
        else:
            # prob_before = last prob_after from previous step
            prob_before = self.last_prob

        # run classifier only on the new query, passing hidden state forward
        prob_after, self.hidden = self.classifier(new_query, self.hidden)

        # update history (if you still need it for later inspection)
        self.history.append(new_query.squeeze(0))

        # cache prob for next step
        self.last_prob = prob_after

        # return as Python float
        return prob_after.squeeze().cpu().item()

    def eval_mode(self):
        self.classifier.eval()

    def train_mode(self):
        self.classifier.train()

    def load_model(self, path):
        self.classifier.load_state_dict(torch.load(path, map_location=self.device))
        self.classifier.to(self.device)

class IdentificationRNN(nn.Module):
    def __init__(self, query_dim, hidden_dim=64, num_layers=3):
        super(IdentificationRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # RNN backbone (GRU is simpler & stable, LSTM also possible)
        self.rnn = nn.GRU(
            input_size=query_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Classifier head
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None):
        # x: [batch=1, seq=1, query_dim]
        out, hidden = self.rnn(x, hidden)  # out: [1,1,hidden_dim]
        out = out[:, -1, :]                # take last timestep [1, hidden_dim]

        logits = self.fc(out)              # [1,1]
        prob = self.sigmoid(logits)        # probability in [0,1]

        return prob.squeeze(1), hidden     # prob: [1], hidden: for next step
