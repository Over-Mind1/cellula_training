
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

def create_overlay_visualization(image, true_mask, pred_mask, alpha=0.5):
    """Create overlay visualization of predictions"""
    # Create RGB composite
    rgb = np.stack([image[:,:,3], image[:,:,2], image[:,:,1]], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    
    # Create colored masks
    true_colored = np.zeros((*true_mask.shape[:2], 3))
    true_colored[:,:,2] = true_mask[:,:,0]  # Blue for true
    
    pred_colored = np.zeros((*pred_mask.shape[:2], 3))
    pred_colored[:,:,0] = pred_mask[:,:,0]  # Red for predicted
    
    # Create overlays
    overlay_true = rgb.copy()
    mask_indices = true_mask[:,:,0] > 0
    overlay_true[mask_indices] = alpha * true_colored[mask_indices] + (1-alpha) * rgb[mask_indices]
    
    overlay_pred = rgb.copy()
    mask_indices = pred_mask[:,:,0] > 0
    overlay_pred[mask_indices] = alpha * pred_colored[mask_indices] + (1-alpha) * rgb[mask_indices]
    
    return rgb, overlay_true, overlay_pred

def visualize_results(X_sample, y_sample):
    # Get model prediction (replace with your actual model prediction)
    pred_sample = model.predict(X_sample[:1], verbose=0)
    pred_binary = (pred_sample[0] > 0.5).astype(np.float32)
    
    # Create visualizations
    rgb, overlay_true, overlay_pred = create_overlay_visualization(
        X_sample[0], y_sample[0], pred_binary
    )
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(rgb)
    axes[0].set_title('Original RGB Composite')
    axes[0].axis('off')
    
    axes[1].imshow(overlay_true)
    axes[1].set_title('True Mask Overlay (Blue)')
    axes[1].axis('off')
    
    axes[2].imshow(overlay_pred)
    axes[2].set_title('Predicted Mask Overlay (Red)')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

# Create Gradio interface
def process_sample(sample_idx):
    # Get sample from your test generator
    X_sample, y_sample = test_generator[sample_idx]
    
    # Generate visualization
    fig = visualize_results(X_sample, y_sample)
    return fig

# Create interface
iface = gr.Interface(
    fn=process_sample,
    inputs=gr.Slider(0, len(test_generator)-1, step=1, label="Sample Index"),
    outputs="plot",
    title="Segmentation Results Visualization",
    description="Visualize original image, true mask, and predicted mask overlays"
)

iface.launch()
