import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import transforms

class PencilSketchDataset(Dataset):
    """
    A dataset loader for pencil sketch images.
    Assumes all images are located in a single folder.
    """
    def __init__(self, root_dir, image_size=256):
        """
        Args:
            root_dir (str): Directory with all the pencil sketch images.
            image_size (int): The resolution to which images will be resized.
        """
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transforms.Compose([
            transforms.Resize((image_size + 30, image_size + 30)),  # Larger resize for cropping
            transforms.RandomCrop(image_size),                      # Random cropping
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),                        # Added vertical flip
            transforms.RandomRotation(15),                          # Random rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomGrayscale(p=0.1),                     # Occasional grayscale
            transforms.ToTensor(),
            # Normalize images to [-1, 1]. For RGB images, we use 3 channels.
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Convert to RGB to ensure we have three channels even if the image is originally grayscale.
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image

# Example usage
if __name__ == "__main__":
    dataset = PencilSketchDataset(root_dir="downloaded_images", image_size=256)
    print("Total images in dataset:", len(dataset))
    # Display information about the first image
    sample = dataset[0]
    print("Shape of the first image tensor:", sample.shape)