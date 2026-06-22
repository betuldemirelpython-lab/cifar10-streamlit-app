import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history_dict):
    """Plot training/validation accuracy and loss.

    Parameters
    ----------
    history_dict : dict
        Dictionary with keys 'accuracy', 'val_accuracy', 'loss', 'val_loss'.
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    epochs = range(1, len(history_dict.get('accuracy', [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy subplot
    ax = axes[0]
    ax.plot(epochs, history_dict.get('accuracy', []), label='Training Accuracy')
    ax.plot(epochs, history_dict.get('val_accuracy', []), label='Validation Accuracy')
    ax.set_title('Model Doğruluğu')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid(alpha=0.3)
    ax.legend()

    # Loss subplot
    ax = axes[1]
    ax.plot(epochs, history_dict.get('loss', []), label='Training Loss')
    ax.plot(epochs, history_dict.get('val_loss', []), label='Validation Loss')
    ax.set_title('Model Kaybı')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_prediction_bar(predictions_array, class_names):
    """Plot a horizontal bar chart of prediction probabilities.

    Parameters
    ----------
    predictions_array : np.ndarray
        1‑D array of probabilities for each of the 10 classes.
    class_names : list[str]
        List of class names (Turkish).
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    predictions = predictions_array.squeeze()
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(class_names))
    colors = ['orange' if i == np.argmax(predictions) else 'steelblue' for i in range(len(class_names))]
    ax.barh(y_pos, predictions, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()  # highest probability on top
    ax.set_xlabel('Olasılık')
    ax.set_xlim(0, 1)
    # Add percentage labels
    for i, v in enumerate(predictions):
        ax.text(v + 0.02, i, f"{v*100:.1f}%", va='center')
    fig.tight_layout()
    return fig
