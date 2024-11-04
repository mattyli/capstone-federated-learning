import numpy as np
import torch
from matplotlib import pyplot as plt
from model import UNet
from dataset import SegmentationDataset  # Make sure this is the correct path to your dataset class
from torch.utils.data import DataLoader
from torchvision import transforms

def dice_score(preds, targets, threshold=0.5):
    """
    Compute DICE score between predictions and targets.
    """
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    dice = (2.0 * intersection) / (preds.sum() + targets.sum())
    return dice.item()

def test_model(model, test_loader, device):
    """
    Test the U-Net model and visualize predictions with DICE scores.
    """
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)

            # Get model predictions
            outputs = model(images)
            preds = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities

            # Calculate DICE score for each image in the batch
            for i in range(images.size(0)):
                dice = dice_score(preds[i], masks[i])
                dice_scores.append(dice)

                # Convert tensors to numpy arrays for visualization
                image_np = images[i].cpu().squeeze().numpy()
                mask_np = masks[i].cpu().squeeze().numpy()
                pred_np = preds[i].cpu().squeeze().numpy() > 0.5  # Threshold for binary mask

                # Plot the image, ground truth, and prediction
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(image_np, cmap='gray')
                axes[0].set_title("Input Image")
                axes[1].imshow(mask_np, cmap='gray')
                axes[1].set_title("Ground Truth")
                axes[2].imshow(pred_np, cmap='gray')
                axes[2].set_title(f"Prediction (DICE: {dice:.3f})")

                for ax in axes:
                    ax.axis('off')

                plt.show()

            # Limit visualization to a few examples
            if idx == 4:  # Adjust as needed
                break

    # Calculate average DICE score across the dataset
    avg_dice = np.mean(dice_scores)
    max_dice = np.max(dice_scores)
    min_dice = np.min(dice_scores)
    std_dice = np.std(dice_scores)
    print(f"Average DICE Score: {avg_dice:.4f}")
    print(f"Maximum DICE Score: {max_dice:.4f}")
    print(f"Minimum DICE Score: {min_dice:.4f}")
    print(f"Standard DICE Score: {std_dice:.4f}")
    return avg_dice

if __name__ == "__main__":

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and move it to the appropriate device
    model = UNet(in_channels=1, out_channels=1).to(device)

    # Load the state dict
    model.load_state_dict(torch.load("model.pt", map_location=device))

    # Define data transformations for test dataset
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.Resize((256, 256)),  # Resize to match model input size
        transforms.ToTensor()           # Convert to tensor
    ])

    # Set paths for test images and masks
    test_img_dir = r"C:\Users\David Hoernke\PycharmProjects\capstone-federated-learning\datasets\anasmohammedtahir\covidqu\versions\7\Infection Segmentation Data\Infection Segmentation Data\Test\COVID-19\images"
    test_mask_dir = r"C:\Users\David Hoernke\PycharmProjects\capstone-federated-learning\datasets\anasmohammedtahir\covidqu\versions\7\Infection Segmentation Data\Infection Segmentation Data\Test\COVID-19\infection masks"

    # Initialize test dataset and DataLoader
    test_dataset = SegmentationDataset(test_img_dir, test_mask_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Run the test model function
    test_model(model, test_loader, device)
