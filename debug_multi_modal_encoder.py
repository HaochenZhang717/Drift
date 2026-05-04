import torch

from models.modality_imputation.multi_modal_encoder import MultiModalEncoder


def main() -> None:
    torch.manual_seed(42)

    # Small test config
    batch_size = 4
    seq_len = 32
    channels = 1
    n_tokens = 3
    dim_in = 128
    dim_out = 64

    model = MultiModalEncoder(
        n_tokens=n_tokens,
        dim_in=dim_in,
        dim_out=dim_out,
        ae_input_dim=channels,
        ae_d_model=128,
        ae_nhead=4,
        ae_num_layers=2,
        ae_max_len=512,
        nhead_cross=4,
        ckpt_paths=None,  # Set checkpoint dict here if needed.
    )
    model.train()

    # Random modality inputs
    heart_rate = torch.randn(batch_size, seq_len, channels)
    calorie = torch.randn(batch_size, seq_len, channels)
    physical_activity = torch.randn(batch_size, seq_len, channels)
    respiratory_rate = torch.randn(batch_size, seq_len, channels)

    # Observed masks with random missing points
    def random_mask() -> torch.Tensor:
        m = (torch.rand(batch_size, seq_len, channels) > 0.2).float()
        # Ensure at least one observed point per sample to avoid degenerate all-missing sequences.
        m[:, 0, :] = 1.0
        return m

    heart_rate_observed_mask = random_mask()
    calorie_observed_mask = random_mask()
    physical_activity_observed_mask = random_mask()
    respiratory_rate_observed_mask = random_mask()

    out = model(
        heart_rate=heart_rate,
        calorie=calorie,
        physical_activity=physical_activity,
        respiratory_rate=respiratory_rate,
        heart_rate_observed_mask=heart_rate_observed_mask,
        calorie_observed_mask=calorie_observed_mask,
        physical_activity_observed_mask=physical_activity_observed_mask,
        respiratory_rate_observed_mask=respiratory_rate_observed_mask,
    )

    expected_shape = (batch_size, 4 * n_tokens, dim_out)
    print(f"Output shape: {tuple(out.shape)}")
    assert tuple(out.shape) == expected_shape, (
        f"Unexpected output shape: got {tuple(out.shape)}, expected {expected_shape}"
    )

    # Backward pass to verify trainable parts get gradients.
    loss = out.pow(2).mean()
    loss.backward()
    print(f"Loss: {loss.item():.6f}")

    # Check AEs are frozen
    ae_param_trainable = []
    for ae_name in [
        "heart_rate_ae",
        "calorie_ae",
        "physical_activity_ae",
        "respiratory_rate_ae",
    ]:
        ae = getattr(model, ae_name)
        n_trainable = sum(int(p.requires_grad) for p in ae.parameters())
        ae_param_trainable.append((ae_name, n_trainable))
        assert n_trainable == 0, f"{ae_name} has trainable params but should be frozen"

    print("AE trainable parameter checks:")
    for ae_name, n_trainable in ae_param_trainable:
        print(f"  {ae_name}: trainable tensors = {n_trainable}")

    # Check that modality special tokens and cross-attn params receive gradients
    for modality in model.MODALITIES:
        token_grad = model.modality_special_tokens[modality].grad
        assert token_grad is not None, f"special token grad missing for modality={modality}"

        ca = model.cross_attentions[modality]
        ca_grads = [p.grad for p in ca.parameters() if p.requires_grad]
        assert any(g is not None for g in ca_grads), f"cross-attention grad missing for modality={modality}"

    print("Gradient checks passed for special tokens and modality-specific cross attentions.")
    print("All MultiModalEncoder debug tests passed.")


if __name__ == "__main__":
    main()
