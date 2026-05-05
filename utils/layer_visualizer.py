"""
layer_visualizer.py
===================
Provides Keras plot_model-style architecture diagrams for PyTorch models
using torchview → graphviz, plus forward-hook feature-map extraction and
per-layer weight statistics.
"""

import os
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

OUTPUT_DIR = 'output/layer_visuals'


class LayerVisualizer:
    def __init__(self, model, output_dir=OUTPUT_DIR):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.activations = {}
        self.hooks = []

    # ── forward hooks ─────────────────────────────────────────────────────────

    def hook_fn(self, name):
        def fn(module, input, output):
            self.activations[name] = output.detach()
        return fn

    def register_hooks(self, layer_names):
        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self.hook_fn(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    # ── feature map visualisation ──────────────────────────────────────────────

    def visualize_feature_maps(self, image_tensor, model_name='model'):
        self.activations.clear()
        self.model.eval()
        with torch.no_grad():
            try:
                self.model(image_tensor)
            except Exception as e:
                print(f'Inference failed during hook extraction: {e}')

        for name, act in self.activations.items():
            if act.dim() == 4:
                act_map = act[0].mean(dim=0).cpu().numpy()
                plt.figure(figsize=(6, 6))
                plt.imshow(act_map, cmap='viridis')
                plt.title(f'{model_name} — {name}')
                plt.axis('off')
                safe = name.replace('.', '_')
                plt.savefig(os.path.join(self.output_dir, f'{model_name}_{safe}.png'),
                            bbox_inches='tight', dpi=120)
                plt.close()

    # ── weight statistics ──────────────────────────────────────────────────────

    def print_layer_values(self):
        print(f'--- CNN Layer Stats: {self.model.__class__.__name__} ---')
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                w = module.weight.detach()
                print(f'  {name:<50s}  shape={tuple(w.shape)}  '
                      f'mean={w.mean():.4f}  std={w.std():.4f}  '
                      f'min={w.min():.4f}  max={w.max():.4f}')

    # ── Keras-style plot_model architecture diagram ────────────────────────────

    def plot_architecture(self, filename='model_architecture',
                          input_size=(1, 3, 640, 640),
                          max_layers=None):
        """
        Generate a Keras plot_model-style architecture diagram using torchview.

        Each node shows: layer name, type, and output shape.
        Edges show data flow (forward pass).
        Saves <output_dir>/<filename>.png and returns the path.

        Args:
            filename:   output filename stem (no extension)
            input_size: tuple passed to torchview as the dummy input shape
            max_layers: unused (kept for API compat); torchview handles depth
        """
        try:
            from torchview import draw_graph
        except ImportError:
            print("torchview not installed. Run: pip install torchview")
            return None

        model_name = self.model.__class__.__name__
        print(f'  Drawing architecture for {model_name} ...')

        try:
            self.model.eval()
            # torchview traces the model and builds a graphviz Digraph
            graph = draw_graph(
                self.model,
                input_size=input_size,
                graph_name=model_name,
                expand_nested=True,      # show sub-modules like Keras
                hide_inner_tensors=True, # only show layer nodes, not intermediate tensors
                hide_module_functions=False,
                depth=10,
                device='cpu',
            )

            # Save as PNG via graphviz
            out_stem = os.path.join(self.output_dir, filename)
            # visual_graph is a graphviz.Digraph object
            graph.visual_graph.format = 'png'
            graph.visual_graph.render(out_stem, cleanup=True)
            out_path = out_stem + '.png'
            print(f'  ✓ Saved → {out_path}')
            return out_path

        except Exception as e:
            print(f'  torchview failed for {model_name}: {e}')
            print('  Falling back to text summary...')
            self._save_text_summary(filename)
            return None

    def _save_text_summary(self, filename):
        """Fallback: save a text-based layer summary as PNG using matplotlib."""
        try:
            from torchinfo import summary as ti_summary
            import io, sys
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            try:
                ti_summary(self.model, input_size=(1, 3, 640, 640), depth=5,
                           col_names=["input_size", "output_size", "num_params"])
            except Exception:
                pass
            sys.stdout = old_stdout
            text = buf.getvalue()
        except ImportError:
            lines = []
            for name, m in self.model.named_modules():
                lines.append(f'{name or "(root)":50s}  {type(m).__name__}')
            text = '\n'.join(lines)

        lines = text.split('\n')
        fig, ax = plt.subplots(figsize=(12, max(6, len(lines) * 0.22)))
        ax.axis('off')
        ax.text(0.01, 0.99, text, transform=ax.transAxes,
                fontsize=6.5, va='top', fontfamily='monospace')
        out_path = os.path.join(self.output_dir, filename + '.png')
        plt.savefig(out_path, bbox_inches='tight', dpi=130)
        plt.close()
        print(f'  ✓ Text summary saved → {out_path}')
        return out_path
