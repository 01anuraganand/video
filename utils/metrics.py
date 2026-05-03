import matplotlib.pyplot as plt

def plot_fps_comparison(results_dict, save_path="output/fps_comparison.png"):
    """
    Plots a bar chart comparing the FPS of different models.
    results_dict: dict of {'model_name': fps_value}
    """
    models = list(results_dict.keys())
    fps_vals = list(results_dict.values())

    plt.figure(figsize=(8, 5))
    plt.bar(models, fps_vals, color=['blue', 'orange', 'green'][:len(models)])
    plt.ylabel('Frames Per Second (FPS)')
    plt.title('Inference Speed Comparison')
    plt.savefig(save_path)
    plt.close()
    print(f"FPS comparison plot saved to {save_path}")
