import torch
from torch.utils.data import random_split, DataLoader
from torchsummary import summary
from model import UNet
from dataset import SegmentationDataset
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    counter = 0
    for images, masks in train_loader:
        print("batch number {}",counter)
        counter = counter + 1
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(val_loader.dataset)
    return epoch_loss


if __name__ == "__main__":

    num_epochs = 50
    save_file = "model.pt"
    plot_file = "plot.png"
    batch_size = 8

    masks_dir = r'C:\Users\David Hoernke\PycharmProjects\capstone-federated-learning\datasets\anasmohammedtahir\covidqu\versions\7\Infection Segmentation Data\Infection Segmentation Data\Train\COVID-19\infection masks'
    img_dir = r'C:\Users\David Hoernke\PycharmProjects\capstone-federated-learning\datasets\anasmohammedtahir\covidqu\versions\7\Infection Segmentation Data\Infection Segmentation Data\Train\COVID-19\images'

    torch.cuda.device_count()
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tUsing device:', device)

    model = UNet(in_channels=1, out_channels=1).to(device)
    summary(model, input_size=(1, 256, 256))

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.Resize((256, 256)),  # Ensure 256x256 size
        transforms.ToTensor()  # Convert to tensor
    ])

    dataset = SegmentationDataset(img_dir, masks_dir, transform)
    dataset_size = len(dataset)
    print("Original dataset size:", dataset_size)

    train_size = int(0.93 * dataset_size)
    val_size = dataset_size - train_size

    # Randomly split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # For binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Lists to track losses
    train_losses = []
    val_losses = []

    for images, masks in train_loader:
        print("Image shape:", images.shape)  # Should print torch.Size([batch_size, 1, 256, 256])
        print("Mask shape:", masks.shape)  # will print torch.Size([batch_size, 1, 256, 256])
        break

    # Training loop
    print("Training...")
    for epoch in range(num_epochs):
        print(epoch)
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(num_epochs)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the model if validation loss decreases
        if epoch == 0 or val_loss < min(val_losses[:-1]):
            torch.save(model.state_dict(), save_file)
            print(f"Model saved with validation loss {val_loss:.4f}")

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(plot_file)
    plt.show()


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
    Test the U-Net model and visualize predictions with DICE scores
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
    print(f"Average DICE Score: {avg_dice:.4f}")
    return avg_dice