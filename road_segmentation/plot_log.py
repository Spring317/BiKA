import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_log(csv_path, output_dir):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['loss'], label='Training Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, 'loss_plot_v2.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved loss plot to {loss_plot_path}")

    # Plot IoU
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['iou'], label='Training IoU')
    plt.plot(df['epoch'], df['val_iou'], label='Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Training and Validation IoU')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    iou_plot_path = os.path.join(output_dir, 'iou_plot_v2.png')
    plt.savefig(iou_plot_path)
    plt.close()
    print(f"Saved IoU plot to {iou_plot_path}")

    # Plot Learning Rate
    # plt.figure(figsize=(10, 5))
    # plt.plot(df['epoch'], df['lr'], label='Learning Rate', color='green')
    # plt.xlabel('Epoch')
    # plt.ylabel('Learning Rate')
    # plt.title('Learning Rate over Epochs')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # lr_plot_path = os.path.join(output_dir, 'lr_plot.png')
    # plt.savefig(lr_plot_path)
    # plt.close()
    # print(f"Saved Learning Rate plot to {lr_plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training log from CSV.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to log.csv')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save plots')
    args = parser.parse_args()

    plot_log(args.csv_path, args.output_dir)
