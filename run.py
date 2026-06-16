import torch

import config
import dataset
import evaluation
import model as models
import training
import visualization


def main():
    print("=" * 70)
    print("ADENOMYOSIS U-NET SEGMENTATION PIPELINE")
    print("=" * 70)

    config.setup_reproducibility()
    print(f"Using device: {config.DEVICE}")

    # --- Data ---
    print("\n--- Loading dataset...")
    train_gen, val_loader, test_loader, test_files, steps_per_epoch = dataset.get_dataloaders()

    # --- Training ---
    print("\n--- Starting training...")
    net, train_losses, val_losses, val_dices = training.train(train_gen, val_loader, steps_per_epoch)

    # --- Training curves ---
    print("\n--- Plotting training curves...")
    visualization.plot_training_curves(train_losses, val_losses, val_dices)

    # --- Evaluation ---
    print("\n--- Evaluating on test set...")
    best_model = models.new_model()
    best_model.load_state_dict(torch.load(f"{config.MODELS_DIR}/best_model.pth", map_location=config.DEVICE))
    all_metrics = evaluation.evaluate(best_model, test_loader)

    # --- Visualization ---
    print("\n--- Generating segmentation visualizations...")
    visualization.plot_predictions(best_model, test_loader, num_samples=15)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
