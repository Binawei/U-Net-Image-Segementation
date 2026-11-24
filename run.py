import torch

import config
import dataset
import evaluation
import model as models
import training
import visualization


def main():
    print("=" * 70)
    print(f"U-NET SEGMENTATION PIPELINE  [{config.MODE.upper()}]")
    print("=" * 70)

    config.setup_reproducibility()
    print(f"Using device: {config.DEVICE}")

    print("\n--- Loading dataset...")
    train_gen, val_loader, test_loader, test_files, steps_per_epoch = dataset.get_dataloaders()

    print("\n--- Starting training...")
    if config.MODE == "multibinary":
        net, train_losses, val_losses, val_dices = training.train_multibinary(train_gen, val_loader, steps_per_epoch)
    else:
        net, train_losses, val_losses, val_dices = training.train(train_gen, val_loader, steps_per_epoch)

    print("\n--- Plotting training curves...")
    visualization.plot_training_curves(train_losses, val_losses, val_dices)

    print("\n--- Evaluating on test set...")
    best_model = models.new_model()
    best_model.load_state_dict(torch.load(f"{config.MODELS_DIR}/best_model.pth", map_location=config.DEVICE))
    evaluation.evaluate(best_model, test_loader)

    print("\n--- Generating segmentation visualizations...")
    visualization.plot_predictions(best_model, test_loader, num_samples=15)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
