import torch

from src.config import DUMMY_INPUT_SHAPE
from src.models.lkcnet_stub import LKCNetStub


def run_pipeline():
    x = torch.randn(*DUMMY_INPUT_SHAPE)
    print("Input shape:", x.shape)

    model = LKCNetStub()
    model.eval()

    with torch.no_grad():
        out_train = model(x)
    print("Train graph output shape:", out_train.shape)

    for m in model.modules():
        if hasattr(m, "switch_to_deploy"):
            m.switch_to_deploy()

    with torch.no_grad():
        out_deploy = model(x)
    print("Deploy graph output shape:", out_deploy.shape)

    diff = (out_train - out_deploy).abs().max()
    print("Max diff after reparam:", diff.item())


if __name__ == "__main__":
    run_pipeline()
