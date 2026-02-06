from pathlib import Path
from joblib import Parallel, delayed
import pandas as pd
import matplotlib.pyplot as plt
from src.model import train_save_model


def process(csv_path):
    print(f'Training {csv_path}...')
    return train_save_model(csv_path)

if __name__ == '__main__':
    p = Path('data')
    files = sorted(p.glob('data_*.csv'))
    print(f"Found {len(files)} CSV files to train")
    results = Parallel(n_jobs=4)(delayed(process)(str(f)) for f in files)
    print("\nTraining complete!")
    
    # Aggregate feature importances
    feature_importance_dict = {}
    for r in results:
        if r:
            print(f"  {r['ticker']}: {r['accuracy']:.2%}")
            # Average importance across all tickers for each feature
            for feat_name, importance in zip(r['feature_names'], r['feature_importances']):
                if feat_name not in feature_importance_dict:
                    feature_importance_dict[feat_name] = []
                feature_importance_dict[feat_name].append(importance)
    
    # Calculate mean importance per feature
    mean_importance = {feat: sum(imps) / len(imps) for feat, imps in feature_importance_dict.items()}
    
    # Create combined feature importance plot
    sorted_features = sorted(mean_importance.keys(), key=lambda x: mean_importance[x])
    sorted_importances = [mean_importance[f] for f in sorted_features]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_features, sorted_importances, color='steelblue')
    ax.set_xlabel('Mean Feature Importance (Across All Tickers)')
    ax.set_title('Combined Feature Importance - All 30 Stocks')
    plt.tight_layout()
    
    plot_dir = Path('results')
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_file = plot_dir / 'feature_importance_combined.png'
    plt.savefig(plot_file, dpi=100)
    plt.close()
    print(f"\nCombined feature importance plot saved: {plot_file}")

