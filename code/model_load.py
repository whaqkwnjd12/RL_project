import os
import json
import argparse
import torch
from utils import get_device
from models import TransformerPolicy

def get_params():
    parser = argparse.ArgumentParser(description="Data Path parameter")
    parser.add_argument(
        "--model_direc",
        type=str,
        help="Path to your model directory",
        required=True
    )
    return parser



if __name__ == "__main__":
    args = get_params().parse_args()

    assert os.path.exists(args.model_direc), f"There is no model direc: {args.model_direc}"

    model_path = os.path.join(args.model_direc, 'best_policy.pth')
    with open(os.path.join(args.model_direc, 'training_info.json'), 'r') as f:
        model_info = json.load(f)

    policy = TransformerPolicy(**model_info['params'])
    device = get_device()
    policy = policy.to(device)

    # weight 로드할 때 map_location 필요 (GPU ↔ CPU 환경 호환)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    print("Model loaded successfully!")