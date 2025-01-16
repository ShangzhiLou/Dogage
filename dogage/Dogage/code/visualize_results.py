import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def visualize_results(log_dir='results'):
    """可视化训练结果并保存图像"""

    for model_name in os.listdir(log_dir):
        model_path = os.path.join(log_dir, model_name)
        if os.path.isdir(model_path):
          print(f'Visualizing results for: {model_name}')
          try:
            # Load training log data
            log_df = pd.read_csv(os.path.join(model_path,'training_log.csv'))
            # Create a plot for accuracy and loss.
            fig, axes = plt.subplots(1,2,figsize=(12, 4))
            sns.lineplot(data=log_df, x='epoch', y='train_loss', label = 'train_loss', ax=axes[0])
            sns.lineplot(data=log_df, x='epoch', y='val_loss', label = 'val_loss', ax=axes[0])
            axes[0].set_title(f"Loss vs Epoch for {model_name}")
            axes[0].set_xlabel('Epochs')
            axes[0].set_ylabel('Loss')
            sns.lineplot(data=log_df, x='epoch', y='train_acc', label = 'train_acc', ax=axes[1])
            sns.lineplot(data=log_df, x='epoch', y='val_acc', label = 'val_acc', ax=axes[1])
            axes[1].set_title(f"Accuracy vs Epoch for {model_name}")
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('Accuracy')
            plt.tight_layout()
            plt.savefig(os.path.join(model_path, f"{model_name}_training.png"))
            plt.close()

            # Load classification report
            report_df = pd.read_csv(os.path.join(model_path, 'classification_report.csv'), index_col=0)
            report_df = report_df.drop(['accuracy'],axis=0)
            plt.figure(figsize=(10, 6))
            sns.heatmap(report_df.iloc[:, :-1].T, annot=True, fmt=".2f", cmap="YlGnBu")
            plt.title(f"Classification Report Heatmap - {model_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(model_path, f"{model_name}_classification_report.png"))
            plt.close()

            # Load confusion matrix
            cm_df = pd.read_csv(os.path.join(model_path, 'confusion_matrix.csv'), index_col=0)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", xticklabels=['Young', 'Adult', 'Senior'], yticklabels=['Young', 'Adult', 'Senior'])
            plt.title(f"Confusion Matrix - {model_name}")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.tight_layout()
            plt.savefig(os.path.join(model_path, f"{model_name}_confusion_matrix.png"))
            plt.close()
          except Exception as e:
              print(f"Error during processing for {model_name}: {e}")

if __name__ == '__main__':
    visualize_results()