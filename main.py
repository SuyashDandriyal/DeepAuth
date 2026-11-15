"""
DeepGuardAI - Model Training and Evaluation Pipeline Entrypoint
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPFAKE_DIR = os.path.join(ROOT_DIR, "Deepfake")

for path in [ROOT_DIR, DEEPFAKE_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import Deepfake.main as deepfake_main

if __name__ == "__main__":
    transform = deepfake_main.transforms.Compose([
        deepfake_main.transforms.Resize((224, 224)),
        deepfake_main.transforms.ToTensor(),
    ])

    real_dir = os.path.join(DEEPFAKE_DIR, "dataset", "real")
    fake_dir = os.path.join(DEEPFAKE_DIR, "dataset", "fake")

    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"Error: Dataset directory not found at {real_dir} or {fake_dir}")
        sys.exit(1)

    print("Loading datasets from Deepfake/dataset...")
    real_dataset = deepfake_main.FaceDataset(real_dir, label=0, transform=transform)
    fake_dataset = deepfake_main.FaceDataset(fake_dir, label=1, transform=transform)
    full_dataset = real_dataset + fake_dataset

    print(f"Dataset loaded: {len(full_dataset)} total images ({len(real_dataset)} REAL, {len(fake_dataset)} FAKE)")

    model = deepfake_main.SimpleCNN()
    model_save_path = os.path.join(DEEPFAKE_DIR, "model.pth")

    print("Starting model training...")
    deepfake_main.train_model(model, full_dataset, num_epochs=5)

    deepfake_main.torch.save(model.state_dict(), model_save_path)
    # Also save to root model.pth for convenience
    root_save_path = os.path.join(ROOT_DIR, "model.pth")
    deepfake_main.torch.save(model.state_dict(), root_save_path)
    print(f"Model saved to {model_save_path} and {root_save_path}")

    print("Evaluating trained model...")
    deepfake_main.evaluate_model(model, full_dataset)
