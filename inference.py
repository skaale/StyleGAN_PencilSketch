import os
import torch
from torchvision.utils import save_image
from models import Generator

class GANInference:
    def __init__(self, 
                 checkpoint_path="generator_checkpoint.pth", 
                 output_dir="inference_output",
                 z_dim=100, 
                 features_g=64, 
                 image_channels=3, 
                 device=None):
        """
        Initializes the inference engine for the GAN model.
        
        Args:
            checkpoint_path (str): Path to the generator checkpoint.
            output_dir (str): Directory to save generated images.
            z_dim (int): Dimension of the latent vector.
            features_g (int): Base feature size for the generator.
            image_channels (int): Number of image channels (e.g., 3 for RGB).
            device: Torch device. If None, it selects GPU if available.
        """
        self.z_dim = z_dim
        self.features_g = features_g
        self.image_channels = image_channels
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize the generator model
        self.generator = Generator(z_dim=self.z_dim, image_channels=self.image_channels, features_g=self.features_g).to(self.device)
        
        # Check if checkpoint exists
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found at {self.checkpoint_path}. "
                "Please make sure you have trained the model and saved the checkpoint, "
                "or provide the correct path to an existing checkpoint."
            )
            
        # Load the checkpoint for the generator
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint["generator_state_dict"])
            self.generator.eval()
            print(f"Loaded generator from {checkpoint_path} on {self.device}")
        except Exception as e:
            raise Exception(f"Error loading checkpoint: {str(e)}")

    def generate_samples(self, num_samples=10, save_filename="generated_samples.png", nrow=5):
        """
        Generates samples using the trained generator and saves them as an image file.
        
        Args:
            num_samples (int): Number of samples to generate.
            save_filename (str): Filename for the output image.
            nrow (int): Number of images in each row of the grid.
        """
        with torch.no_grad():
            # Use multiple noise vectors and average results
            num_noise_samples = 5
            all_images = []
            
            for _ in range(num_noise_samples):
                noise = torch.randn(num_samples, self.z_dim, 1, 1, device=self.device)
                fake_images = self.generator(noise)
                all_images.append(fake_images)
            
            # Average the results
            fake_images = torch.stack(all_images).mean(0)
            fake_images = (fake_images + 1) / 2  # Denormalize
            fake_images = torch.clamp(fake_images, 0, 1)  # Ensure valid range
            
            save_path = os.path.join(self.output_dir, save_filename)
            save_image(fake_images, save_path, nrow=nrow, normalize=False)
            print(f"Inference complete! Check the '{self.output_dir}' folder for new generated images.")

if __name__ == "__main__":
    # Create an instance of the inference engine
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "generator_checkpoint.pth")
    inference_engine = GANInference(
        checkpoint_path=checkpoint_path,  # Updated checkpoint path
        output_dir="inference_output",
        z_dim=100,
        features_g=64,
        image_channels=3
    )
    
    # Generate and save samples
    inference_engine.generate_samples(num_samples=10, save_filename="generated_samples.png", nrow=5)