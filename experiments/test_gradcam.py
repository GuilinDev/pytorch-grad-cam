import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import os

def main():
    # 1. downgrade numpy version to solve compatibility issues
    # pip install numpy==1.24.3
    
    # 2. load pre-trained model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # use the new weights parameter instead of pretrained
    model.eval()

    # 3. select target layer
    target_layers = [model.layer4[-1]]

    # 4. build CAM object
    cam = GradCAM(model=model, target_layers=target_layers)

    # 5. load and preprocess image
    # use absolute path or ensure image exists
    image_path = "examples/both.png"  # use the example image in the repository
    
    # check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    rgb_img = cv2.imread(image_path, 1)
    if rgb_img is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # correct color conversion
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                  mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

    # 6. generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # 7. visualize the result
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # 8. create the results directory and save the result
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "gradcam_result.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    main() 
