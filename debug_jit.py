import torch

# import your model
from models.multimodal_jit.model_jit import JiT_B_16, JiT_B_2_Tiny


def main():
    # --------------------------------------------------
    # Create model
    # --------------------------------------------------
    model = JiT_B_2_Tiny(
        input_size=16,
        in_channels=1,
        num_classes=4,
    )

    model.eval()

    # --------------------------------------------------
    # Dummy input
    # --------------------------------------------------
    B = 2
    C = 1
    H = 16
    W = 16

    x = torch.randn(B, C, H, W)

    # diffusion timesteps
    t = torch.randint(
        low=0,
        high=4,
        size=(B,)
    )

    # class labels
    y = torch.randn(B, 12, 128)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    with torch.no_grad():
        out = model(x, t, y)

    # --------------------------------------------------
    # Print shapes
    # --------------------------------------------------
    print("=" * 50)

    print(f"Input image shape:      {x.shape}")
    print(f"Timestep shape:         {t.shape}")
    print(f"Label shape:            {y.shape}")

    print("-" * 50)

    print(f"Output image shape:     {out.shape}")

    print("=" * 50)


if __name__ == "__main__":
    main()