import matplotlib.pyplot as plt
import numpy as np
import os

def plot_pred_vs_true(y_true, y_pred, stock_names=None, save_path=None):
    """
    Plot true vs. predicted returns over time for each stock.

    Args:
        y_true (np.ndarray): Ground truth returns (T, N).
        y_pred (np.ndarray): Predicted returns (T, N).
        stock_names (list, optional): List of stock names for subplot titles.
        save_path (str, optional): If provided, saves plot to disk; otherwise shows interactively.

    Returns:
        None
    """
    n_samples, n_stocks = y_true.shape

    n_cols = 4
    n_rows = int(np.ceil(n_stocks / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), squeeze=False)
    axs = axs.flatten()

    all_values = np.concatenate((y_true.flatten(), y_pred.flatten()))
    y_min, y_max = all_values.min(), all_values.max()
    y_range = y_max - y_min
    padding = y_range * 0.1
    global_ymin = y_min - padding
    global_ymax = y_max + padding

    true_color = '#096191' 
    pred_color = '#e46027' 

    for i in range(n_stocks):
        ax = axs[i]
        ax.plot(y_true[:, i], label='True', color=true_color, linestyle='-', linewidth=1.5)
        ax.plot(y_pred[:, i], label='Predicted', color=pred_color, linestyle='-', linewidth=1.5)

        #Axis labels 
        ax.set_ylabel("Excess Return", fontsize=10)
        ax.set_xlabel("Sample Index", fontsize=10) 

        title = stock_names[i] if stock_names else f"Stock {i+1}"
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, loc='upper left', frameon=False)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=9)

        ax.set_ylim(global_ymin, global_ymax)

    for j in range(n_stocks, len(axs)):
        axs[j].axis('off')

    plt.suptitle("Predicted vs. True Stock Price for Individual Stocks", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_loss_curve(train_losses, val_losses, save_path=None, title="Training vs. Validation Loss Curve"):
    """
    Plot the training and validation loss curves over epochs.

    Args:
        train_losses (list or np.ndarray): Training loss per epoch.
        val_losses (list or np.ndarray): Validation loss per epoch.
        save_path (str, optional): If provided, saves plot to disk; otherwise shows interactively.
        title (str): Title of the plot.

    Returns:
        None
    """
    plt.figure(figsize=(9, 6))

    epochs = range(1, len(train_losses) + 1)

    train_color = '#096191' 
    val_color = '#e46027'  

    plt.plot(epochs, train_losses,
             label="Training Loss",
             color=train_color,
             linestyle='-',
             linewidth=2.0)

    plt.plot(epochs, val_losses,
             label="Validation Loss",
             color=val_color,
             linestyle='-',
             linewidth=2.0)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10, frameon=False)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

#Dummy data tests 
if __name__ == "__main__":
    output_dir = "dummy_plots_color"
    os.makedirs(output_dir, exist_ok=True)

    n_samples_pred = 100 #dummy vars 
    n_stocks_pred = 8
    np.random.seed(42)

    y_true_dummy = np.cumsum(np.random.randn(n_samples_pred, n_stocks_pred) * 0.005, axis=0)
    noise = np.random.randn(n_samples_pred, n_stocks_pred) * 0.01
    y_pred_dummy = y_true_dummy + noise + (np.sin(np.linspace(0, 5, n_samples_pred)) * 0.02)[:, np.newaxis]

    stock_names_dummy = [f"Stock {i+1}" for i in range(n_stocks_pred)]

    print(f"Calling plot_pred_vs_true for {n_stocks_pred} stocks (color version)...")
    plot_pred_vs_true(y_true_dummy, y_pred_dummy,
                      stock_names=stock_names_dummy,
                      save_path=os.path.join(output_dir, "pred_vs_true_dummy_color.png"))
    print(f"Plot saved to {os.path.join(output_dir, 'pred_vs_true_dummy_color.png')}")

    n_epochs_loss = 200

    train_losses_dummy = np.linspace(1.5, 0.2, n_epochs_loss) + np.random.rand(n_epochs_loss) * 0.1
    val_losses_dummy = np.linspace(1.2, 0.25, n_epochs_loss) + np.random.rand(n_epochs_loss) * 0.12
    val_losses_dummy[int(n_epochs_loss * 0.8):] += np.linspace(0, 0.1, n_epochs_loss - int(n_epochs_loss * 0.8))

    print(f"\nCalling plot_loss_curve for {n_epochs_loss} epochs (color version)...")
    plot_loss_curve(train_losses_dummy.tolist(), val_losses_dummy.tolist(),
                    save_path=os.path.join(output_dir, "loss_curve_dummy_color.png"))
    print(f"Plot saved')")