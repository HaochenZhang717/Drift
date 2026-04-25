"""PyTorch implementation of the TimeGAN predictive score metric."""

import numpy as np
import torch
import torch.nn as nn

def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len


class PostHocGRUPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.shape[1]
        )
        return torch.sigmoid(self.proj(out))


def _build_prediction_data(data, dim, max_seq_len):
    x_list, y_list, lengths = [], [], []
    for seq in data:
        seq = np.asarray(seq)
        seq_len = len(seq)
        cur_x = np.zeros((max_seq_len - 1, dim - 1), dtype=np.float32)
        cur_y = np.zeros((max_seq_len - 1, 1), dtype=np.float32)
        valid_len = max(seq_len - 1, 0)
        if valid_len > 0:
            cur_x[:valid_len] = seq[:-1, :(dim - 1)]
            cur_y[:valid_len, 0] = seq[1:, dim - 1]
        x_list.append(cur_x)
        y_list.append(cur_y)
        lengths.append(valid_len)
    return (
        torch.from_numpy(np.stack(x_list)),
        torch.from_numpy(np.stack(y_list)),
        torch.tensor(lengths, dtype=torch.long),
    )


def predictive_score_metrics(ori_data, generated_data, device=None):
    """Report the performance of Post-hoc RNN one-step ahead prediction."""

    no, _, dim = np.asarray(ori_data).shape
    if dim < 2:
        # The original TimeGAN predictive metric uses the first dim-1 channels
        # to predict the last channel. That definition is not valid for
        # univariate series, so skip cleanly instead of constructing GRU(0, ...).
        return float("nan")
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    hidden_dim = max(1, int(dim / 2))
    iterations = 5000
    batch_size = 128

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = PostHocGRUPredictor(dim - 1, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.L1Loss(reduction="none")

    gen_x, gen_y, gen_t = _build_prediction_data(generated_data, dim, max_seq_len)
    ori_x, ori_y, ori_t = _build_prediction_data(ori_data, dim, max_seq_len)
    gen_x = gen_x.to(device)
    gen_y = gen_y.to(device)
    gen_t = gen_t.to(device)
    ori_x = ori_x.to(device)
    ori_y = ori_y.to(device)
    ori_t = ori_t.to(device)

    for _ in range(iterations):
        idx = np.random.permutation(len(generated_data))[:batch_size]
        idx = torch.as_tensor(idx, device=device, dtype=torch.long)
        x_mb = gen_x[idx]
        y_mb = gen_y[idx]
        t_mb = gen_t[idx]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(x_mb, t_mb)

        mask = (
            torch.arange(max_seq_len - 1, device=device)[None, :] < t_mb[:, None]
        ).unsqueeze(-1)
        loss = criterion(y_pred, y_mb)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_y = model(ori_x, ori_t)

    eval_mask = (
        torch.arange(max_seq_len - 1, device=device)[None, :] < ori_t[:, None]
    ).unsqueeze(-1)
    abs_err = (pred_y - ori_y).abs() * eval_mask
    per_seq_mae = abs_err.sum(dim=(1, 2)) / eval_mask.sum(dim=(1, 2)).clamp_min(1)
    return per_seq_mae.mean().item()
