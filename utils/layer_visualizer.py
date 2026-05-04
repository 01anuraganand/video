import torch
import matplotlib.pyplot as plt
from torchviz import make_dot
import os

class LayerVisualizer:
    def __init__(self, model, output_dir='output/layer_visuals'):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.activations = {}
        self.hooks = []
        
    def hook_fn(self, name):
        def fn(module, input, output):
            self.activations[name] = output.detach()
        return fn

    def register_hooks(self, layer_names):
        # Attach hooks to specific layers to extract feature maps
        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self.hook_fn(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def visualize_feature_maps(self, image_tensor, model_name="model"):
        """Passes image_tensor through model and saves feature maps."""
        self.activations.clear()
        
        # Ensure model is in eval mode
        self.model.eval()
        with torch.no_grad():
            try:
                self.model(image_tensor)
            except Exception as e:
                print(f"Inference failed during hook extraction: {e}")

        for name, act in self.activations.items():
            if act.dim() == 4:
                # Average across channels
                act_map = act[0].mean(dim=0).cpu().numpy()
                plt.figure(figsize=(6,6))
                plt.imshow(act_map, cmap='viridis')
                plt.title(f"{model_name} - {name} Feature Map")
                plt.axis('off')
                
                safe_name = name.replace('.', '_')
                plt.savefig(os.path.join(self.output_dir, f"{model_name}_{safe_name}.png"))
                plt.close()

    def plot_architecture(self, dummy_input, filename="model_architecture"):
        """Uses torchviz to plot the computational graph of the model."""
        try:
            out = self.model(dummy_input)
            # Depending on output format, grab a tensor to trace
            if isinstance(out, dict):
                loss_tensor = list(out.values())[0]
            elif isinstance(out, tuple) or isinstance(out, list):
                loss_tensor = out[0]
            else:
                loss_tensor = out
                
            dot = make_dot(loss_tensor, params=dict(self.model.named_parameters()))
            dot.render(os.path.join(self.output_dir, filename), format="png")
            print(f"Architecture saved to {self.output_dir}/{filename}.png")
        except Exception as e:
            print(f"Could not plot architecture: {e}")

    def print_layer_values(self):
        """Prints the weights/parameters of each CNN layer in the model."""
        import torch.nn as nn
        print(f"--- Printing CNN Layer Values for {self.model.__class__.__name__} ---")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight.detach()
                print(f"Layer: {name}")
                print(f"  Shape: {weight.shape}")
                print(f"  Mean: {weight.mean().item():.4f}, Std: {weight.std().item():.4f}")
                print(f"  Min: {weight.min().item():.4f}, Max: {weight.max().item():.4f}")
                print("-" * 40)
