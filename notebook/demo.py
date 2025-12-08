import sys
import os
import cv2
import matplotlib.pyplot as plt

parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)

from utils import (
    setup_sam_3d_body,
    setup_visualizer,
    visualize_2d_results,
    visualize_3d_mesh,
    save_mesh_results,
    display_results_grid,
    process_image_with_mask,
)

# Set up SAM 3D Body estimator
estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")
# Set up visualizer
visualizer = setup_visualizer()


# Load and process the image
image_path = "images/dancing.jpg"  # Relative to notebook folder
img_cv2 = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

# Process the image with SAM 3D Body
print("Processing image with SAM 3D Body...")
outputs = estimator.process_one_image(image_path)

print(f"Number of people detected: {len(outputs)}")
print(
    f"Output keys for first person: {list(outputs[0].keys()) if outputs else 'No people detected'}"
)

# Display the original image
plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Original Image")
plt.show()

# Visualize 2D results using utils
if outputs:
    vis_results = visualize_2d_results(img_cv2, outputs, visualizer)

    # Display results using grid function
    titles = [f"Person {i} - 2D Keypoints & BBox" for i in range(len(vis_results))]
    display_results_grid(vis_results, titles, figsize_per_image=(6, 6))
else:
    print("No people detected in the image")

if outputs:
    mesh_results = visualize_3d_mesh(img_cv2, outputs, estimator.faces)

    # Display results
    for i, combined_img in enumerate(mesh_results):
        combined_rgb = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(20, 5))
        plt.imshow(combined_rgb)
        plt.title(f"Person {i}: Original | Mesh Overlay | Front View | Side View")
        plt.axis("off")
        plt.show()
else:
    print("No people detected for 3D mesh visualization")


if outputs:
    # Get image name without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create output directory
    output_dir = f"output/{image_name}"

    # Save all results (PLY meshes, overlay images, bbox images)
    ply_files = save_mesh_results(
        img_cv2, outputs, estimator.faces, output_dir, image_name
    )

    print(f"\n=== Saved Results for {image_name} ===")
    print(f"Output directory: {output_dir}")
    print(f"Number of PLY files created: {len(ply_files)}")

else:
    print("No results to save - no people detected")


# Load mask and run inference
mask_path = "images/dancing_mask.png"

if os.path.exists(mask_path):
    # Load and display the mask
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(8, 6))
    plt.imshow(mask_img, cmap="gray")
    plt.title("External Mask")
    plt.axis("off")
    plt.show()

    # Process with external mask
    mask_outputs = process_image_with_mask(estimator, image_path, mask_path)

    # Visualize and save results
    if mask_outputs:
        mask_mesh_results = visualize_3d_mesh(img_cv2, mask_outputs, estimator.faces)

        for i, combined_img in enumerate(mask_mesh_results):
            combined_rgb = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(20, 5))
            plt.imshow(combined_rgb)
            plt.title(
                f"Mask-Based Person {i}: Original | Mesh Overlay | Front View | Side View"
            )
            plt.axis("off")
            plt.show()

        # Save results
        mask_output_dir = f"output/mask_based_{image_name}"
        mask_ply_files = save_mesh_results(
            img_cv2,
            mask_outputs,
            estimator.faces,
            mask_output_dir,
            f"mask_{image_name}",
        )
        print(f"Saved mask-based results to: {mask_output_dir}")
    else:
        print("No people detected with mask-based approach")

else:
    print(f"Mask file not found: {mask_path}")
