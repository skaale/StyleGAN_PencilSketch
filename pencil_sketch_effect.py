import os
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from models import Generator

class PencilSketchEffect:
    """
    This class applies a pencil sketch effect to an input image by inverting the GAN's latent space.
    It optimizes a latent vector z so that the generator produces an image close to the input.
    NOTE: Since the generator wasn’t explicitly trained as an encoder, the inversion is approximate.
    """
    def __init__(self, 
                 project_dir="C:\\codes\\StyleGAN_PencilSketch\\data", 
                 checkpoint_dir="C:\\codes\\StyleGAN_PencilSketch\\checkpoints",
                 checkpoint_filename="generator_checkpoint.pth", 
                 z_dim=100, 
                 features_g=64, 
                 image_channels=3, 
                 device=None):
        """
        Args:
            project_dir (str): The base directory for input and output images.
            checkpoint_dir (str): The directory where the checkpoint is stored.
            checkpoint_filename (str): Checkpoint filename in checkpoint_dir.
            z_dim (int): Dimension of the latent vector.
            features_g (int): Base feature size for the generator.
            image_channels (int): The number of image channels.
            device: Torch device (CPU or GPU). If None, selects GPU if available.
        """
        self.project_dir = project_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
        self.z_dim = z_dim
        self.features_g = features_g
        self.image_channels = image_channels

        # Select device (GPU if available)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize the generator and load the checkpoint from the checkpoint directory
        self.generator = Generator(z_dim=self.z_dim, image_channels=self.image_channels, features_g=self.features_g).to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.generator.eval()

        # Transformation for resizing and normalizing the input image to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),        # match training dimensions
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        print(f"Initialized PencilSketchEffect with checkpoint {self.checkpoint_path} on {self.device}")

    def apply_effect(self, input_image_filename, output_image_filename, num_iterations=500, lr=0.01):
        """
        Applies the pencil sketch effect by optimizing the latent vector so that the generator output
        approximates the provided input image.
        
        Args:
            input_image_filename (str): The filename of the input image (located in project_dir).
            output_image_filename (str): The filename for the output image (saved in project_dir).
            num_iterations (int): Number of iterations for the optimization.
            lr (float): Learning rate for the optimization.
        """
        # Construct full paths for the input and output images
        input_image_path = os.path.join(self.project_dir, input_image_filename)
        output_image_path = os.path.join(self.project_dir, output_image_filename)

        # Load and transform the input image
        img = Image.open(input_image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)  # shape: [1, 3, 256, 256]

        # Initialize a latent vector z (with gradients enabled)
        z = torch.randn(1, self.z_dim, 1, 1, device=self.device, requires_grad=True)

        optimizer = optim.Adam([z], lr=lr)
        loss_fn = torch.nn.MSELoss()

        print("Starting inversion to apply pencil sketch effect...")
        for i in range(num_iterations):
            optimizer.zero_grad()
            generated = self.generator(z)
            loss = loss_fn(generated, img_tensor)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.4f}")

        # Generate final image from the optimized latent vector
        with torch.no_grad():
            final_generated = self.generator(z)
            # Denormalize from [-1, 1] to [0, 1]
            output_img = (final_generated + 1) / 2
            save_image(output_img, output_image_path)
            print(f"Pencil sketch effect applied. Output saved to {output_image_path}")

if __name__ == "__main__":
    # Example usage:
    effect_applicator = PencilSketchEffect(
        project_dir="C:\\codes\\StyleGAN_PencilSketch\\data",
        checkpoint_dir="C:\\codes\\StyleGAN_PencilSketch\\checkpoints",
        checkpoint_filename="generator_checkpoint.pth",
        z_dim=100,
        features_g=64,
        image_channels=3
    )
    
    # Set the input and output filenames (these files reside in the project directory)
    input_filename = "input.jpg"         # Replace with your actual input image file in project_dir
    output_filename = "stylized_output.jpg"
    
    effect_applicator.apply_effect(input_filename, output_filename, num_iterations=10000, lr=0.005)