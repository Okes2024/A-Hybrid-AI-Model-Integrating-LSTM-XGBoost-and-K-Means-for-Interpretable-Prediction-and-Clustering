"""
Visualization utilities
"""

import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime
import os
from src.config import Config


class Visualizer:
    """Create visualization dashboard and individual README charts"""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = Config.RESULTS_DIR
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_dashboard(self, df, ensemble, X_test, y_test, preprocessor):
        """Create 6-panel dashboard (your existing method, preserved)"""

        predictions = ensemble.predict(X_test, return_individual=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hybrid AI Model - Water Quality Analysis', fontsize=16, fontweight='bold')

        # 1. WQI Distribution
        wqi_counts = df['WQI_Class'].value_counts()
        colors = ['#2ecc71', '#3498db', '#f1c40f', '#e67e22', '#e74c3c']
        axes[0, 0].bar(wqi_counts.index, wqi_counts.values, color=colors[:len(wqi_counts)])
        axes[0, 0].set_title('WQI Class Distribution')
        axes[0, 0].set_ylabel('Count')

        # 2. Prediction vs Actual
        axes[0, 1].scatter(y_test, predictions['ensemble'], alpha=0.6, c='red', s=100, label='Ensemble')
        axes[0, 1].scatter(y_test, predictions['xgboost'], alpha=0.4, c='green', s=50, label='XGBoost')
        axes[0, 1].scatter(y_test, predictions['lstm'], alpha=0.4, c='blue', s=50, label='LSTM')
        min_val, max_val = y_test.min(), y_test.max()
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        axes[0, 1].set_xlabel('Actual WQI')
        axes[0, 1].set_ylabel('Predicted WQI')
        axes[0, 1].set_title('Prediction Accuracy')
        axes[0, 1].legend()

        # 3. Feature Importance
        importance = ensemble.xgb.feature_importances_
        features = list(Config.STANDARDS.keys()) + ['Cluster']
        feat_imp = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values('Importance')
        axes[0, 2].barh(feat_imp['Feature'], feat_imp['Importance'], color='green')
        axes[0, 2].set_title('Feature Importance')

        # 4. Clusters (PCA)
        features_list = list(Config.STANDARDS.keys())
        X_full = df[features_list].values
        X_full_scaled = preprocessor.scaler.transform(X_full)
        all_clusters = preprocessor.kmeans.predict(X_full_scaled)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_full_scaled)

        scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                     c=all_clusters.astype(int),
                                     cmap='viridis', 
                                     s=100,
                                     edgecolors='k',
                                     linewidth=0.5)
        axes[1, 0].set_title(f'Clusters (k={preprocessor.optimal_k})')
        plt.colorbar(scatter, ax=axes[1, 0])

        # 5. Residuals
        residuals = y_test - predictions['ensemble']
        axes[1, 1].scatter(predictions['ensemble'], residuals, alpha=0.6, c='purple')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residual Plot')

        # 6. Model Comparison
        models = ['XGBoost', 'LSTM', 'Ensemble']
        r2_scores = [
            r2_score(y_test, predictions['xgboost']),
            r2_score(y_test, predictions['lstm']),
            r2_score(y_test, predictions['ensemble'])
        ]
        bars = axes[1, 2].bar(models, r2_scores, color=['green', 'blue', 'red'])
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_title('R² Comparison')
        for bar, score in zip(bars, r2_scores):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'{score:.3f}', ha='center')

        plt.tight_layout()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{self.output_dir}/dashboard_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Dashboard saved: {filename}")
        plt.close()
        return fig

    # =====================================================================
    # README CHARTS - Individual exports for documentation
    # =====================================================================

    def save_readme_charts(self, df, ensemble, X_test, y_test, preprocessor,
                          output_dir='results/readme_charts'):
        """
        Generate and save individual charts for README documentation.
        Creates 6-7 professional charts in results/readme_charts/ folder.
        """
        os.makedirs(output_dir, exist_ok=True)

        predictions = ensemble.predict(X_test, return_individual=True)

        print(f"\n{'='*60}")
        print("GENERATING README CHARTS")
        print(f"{'='*60}")

        # Chart 1: Model Architecture Diagram (Code-generated)
        self._save_architecture_chart(output_dir)

        # Chart 2: Prediction vs Actual Scatter
        self._save_prediction_accuracy_chart(predictions, y_test, output_dir)

        # Chart 3: Feature Importance
        self._save_feature_importance_chart(ensemble, output_dir)

        # Chart 4: WQI Class Distribution
        self._save_wqi_distribution_chart(df, output_dir)

        # Chart 5: Residual Analysis
        self._save_residual_chart(predictions, y_test, output_dir)

        # Chart 6: Model Performance Comparison
        self._save_model_comparison_chart(predictions, y_test, output_dir)

        # Chart 7: LSTM Training History (if available)
        if hasattr(ensemble.lstm, 'history') and ensemble.lstm.history is not None:
            self._save_lstm_training_chart(ensemble, output_dir)

        print(f"\n✅ All README charts saved to: {output_dir}/")
        print(f"{'='*60}\n")

        return output_dir

    def _save_architecture_chart(self, output_dir):
        """Chart 1: Hybrid AI Architecture Diagram (matplotlib-based)"""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('Hybrid AI Architecture: LSTM + XGBoost + K-Means', 
                     fontsize=16, fontweight='bold', pad=20)

        # Helper to draw boxes
        def draw_box(x, y, w, h, text, color, text_color='black', fontsize=10):
            rect = plt.Rectangle((x, y), w, h, facecolor=color, 
                                edgecolor='black', linewidth=2, zorder=2)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                   fontsize=fontsize, fontweight='bold', color=text_color, zorder=3)

        # Helper to draw arrows
        def draw_arrow(x1, y1, x2, y2):
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='#34495E'))

        # Input Layer
        draw_box(5, 8.5, 4, 1, 'Water Parameters\n(pH, EC, TDS, NO3, Cl, SO4, Ca, Mg, Na, Iron)', 
                '#E3F2FD', fontsize=9)

        # Preprocessing
        draw_box(5, 6.8, 4, 1, 'Preprocessing\nStandardScaler + K-Means (k=3)', 
                '#FFF3E0', fontsize=10)
        draw_arrow(7, 8.5, 7, 7.8)

        # Branch to models
        draw_arrow(7, 6.8, 3.5, 6)
        draw_arrow(7, 6.8, 10.5, 6)

        # XGBoost
        draw_box(1, 4.8, 3, 1.2, 'XGBoost Regressor\n200 trees, lr=0.05', 
                '#E8F5E9', fontsize=10)
        draw_arrow(3.5, 6, 2.5, 6)

        # LSTM
        draw_box(10, 4.8, 3, 1.2, 'LSTM Network\n64→32→16 units\nBatchNorm + Dropout', 
                '#E8F5E9', fontsize=9)
        draw_arrow(10.5, 6, 11.5, 6)

        # Meta-learner
        draw_box(5, 3, 4, 1.2, 'Ridge Meta-Learner\n(Stacking Ensemble)', 
                '#F3E5F5', fontsize=10)
        draw_arrow(2.5, 4.8, 5.5, 4.2)
        draw_arrow(11.5, 4.8, 8.5, 4.2)

        # Output
        draw_box(5, 1.2, 4, 1, 'Final WQI Prediction\n(Regression + Classification)', 
                '#FFEBEE', fontsize=10)
        draw_arrow(7, 3, 7, 2.2)

        # Legend/info box
        info_text = ('Ensemble Strategy:\n'
                    '• XGBoost: Gradient boosting for structured data\n'
                    '• LSTM: Sequential pattern learning\n'
                    '• Ridge: Optimal weight combination')
        ax.text(0.5, 0.5, info_text, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f'{output_dir}/01_architecture.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✅ 01_architecture.png saved")

    def _save_prediction_accuracy_chart(self, predictions, y_test, output_dir):
        """Chart 2: Prediction vs Actual Scatter (THE key credibility chart)"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Calculate metrics
        r2_ensemble = r2_score(y_test, predictions['ensemble'])
        mae_ensemble = mean_absolute_error(y_test, predictions['ensemble'])

        # Plot predictions
        ax.scatter(y_test, predictions['ensemble'], alpha=0.7, c='#E74C3C', s=100, 
                   edgecolors='black', linewidth=1, label=f'Ensemble (R²={r2_ensemble:.3f})')
        ax.scatter(y_test, predictions['xgboost'], alpha=0.5, c='#27AE60', s=60, 
                   edgecolors='black', linewidth=0.5, label='XGBoost')
        ax.scatter(y_test, predictions['lstm'], alpha=0.5, c='#3498DB', s=60, 
                   edgecolors='black', linewidth=0.5, label='LSTM')

        # Perfect prediction line
        min_val, max_val = y_test.min(), y_test.max()
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2.5, 
                label='Perfect Prediction', alpha=0.8)

        # Styling
        ax.set_xlabel('Actual WQI', fontsize=14, fontweight='bold')
        ax.set_ylabel('Predicted WQI', fontsize=14, fontweight='bold')
        ax.set_title(f'Model Prediction Accuracy\nEnsemble R² = {r2_ensemble:.3f}, MAE = {mae_ensemble:.2f}', 
                     fontsize=16, fontweight='bold', pad=15)
        ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')

        # Add confidence band
        std_err = np.std(predictions['ensemble'] - y_test)
        ax.fill_between([min_val, max_val], 
                        [min_val - std_err, max_val - std_err],
                        [min_val + std_err, max_val + std_err],
                        alpha=0.1, color='gray', label=f'±1σ ({std_err:.2f})')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/02_prediction_accuracy.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')  # Best practice from [^9^]
        plt.close()
        print("✅ 02_prediction_accuracy.png saved")

    def _save_feature_importance_chart(self, ensemble, output_dir):
        """Chart 3: Feature Importance Bar Chart"""
        fig, ax = plt.subplots(figsize=(12, 8))

        importance = ensemble.xgb.feature_importances_
        features = list(Config.STANDARDS.keys()) + ['Cluster']

        feat_imp = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=True)

        # Color gradient based on importance
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(feat_imp)))

        bars = ax.barh(feat_imp['Feature'], feat_imp['Importance'], 
                       color=colors, edgecolor='black', linewidth=1)

        # Add value labels
        for bar, val in zip(bars, feat_imp['Importance']):
            ax.text(val + 0.008, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

        ax.set_xlabel('Importance Score (XGBoost)', fontsize=14, fontweight='bold')
        ax.set_title('Feature Importance: What Drives Water Quality Predictions\n'
                     '(Higher = More Impact on WQI)', 
                     fontsize=16, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, max(importance) * 1.2)

        # Add interpretation note
        top_feature = feat_imp.iloc[-1]['Feature']
        note_text = f"Key Insight: {top_feature} is the strongest predictor of water quality"
        ax.text(0.02, 0.02, note_text, transform=ax.transAxes, fontsize=10,
                style='italic', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f'{output_dir}/03_feature_importance.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✅ 03_feature_importance.png saved")

    def _save_wqi_distribution_chart(self, df, output_dir):
        """Chart 4: WQI Class Distribution"""
        fig, ax = plt.subplots(figsize=(10, 7))

        wqi_counts = df['WQI_Class'].value_counts()

        # Professional color scheme matching water quality intuition
        color_map = {
            'Excellent': '#27AE60',  # Green
            'Good': '#3498DB',       # Blue
            'Fair': '#F1C40F',       # Yellow
            'Poor': '#E67E22',       # Orange
            'Unsuitable': '#E74C3C'  # Red
        }
        colors = [color_map.get(c, '#95A5A6') for c in wqi_counts.index]

        bars = ax.bar(wqi_counts.index, wqi_counts.values, 
                      color=colors, edgecolor='black', linewidth=2)

        # Add count and percentage labels
        total = len(df)
        for bar in bars:
            height = bar.get_height()
            pct = (height / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.8,
                    f'{int(height)}\n({pct:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax.set_ylabel('Number of Water Samples', fontsize=14, fontweight='bold')
        ax.set_xlabel('Water Quality Index Class', fontsize=14, fontweight='bold')
        ax.set_title(f'Water Quality Distribution in Dataset\n'
                     f'n = {total} samples from Bayelsa State, Nigeria\n'
                     f'(Symmetric pH Weighting: 5.5↔7.0↔8.5)', 
                     fontsize=16, fontweight='bold', pad=15)
        ax.set_ylim(0, max(wqi_counts.values) * 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add legend for pH symmetry note
        ax.text(0.98, 0.95, 'Symmetric pH:\nAcidic & Alkaline\ndeviations from 7.0\nweighted equally',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        plt.tight_layout()
        plt.savefig(f'{output_dir}/04_wqi_distribution.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✅ 04_wqi_distribution.png saved")

    def _save_residual_chart(self, predictions, y_test, output_dir):
        """Chart 5: Residual Analysis"""
        fig, ax = plt.subplots(figsize=(10, 7))

        residuals = y_test - predictions['ensemble']

        # Main scatter
        ax.scatter(predictions['ensemble'], residuals, alpha=0.6, c='#9B59B6', s=80,
                   edgecolors='black', linewidth=0.5, label='Residuals')

        # Zero line
        ax.axhline(y=0, color='red', linestyle='--', lw=2.5, label='Zero Error')

        # Confidence bands
        std_residuals = np.std(residuals)
        ax.axhline(y=2*std_residuals, color='orange', linestyle=':', lw=2, 
                   alpha=0.7, label=f'±2σ ({2*std_residuals:.2f})')
        ax.axhline(y=-2*std_residuals, color='orange', linestyle=':', lw=2, alpha=0.7)

        # Fill between bands
        ax.fill_between([predictions['ensemble'].min(), predictions['ensemble'].max()],
                        -2*std_residuals, 2*std_residuals, alpha=0.1, color='orange')

        ax.set_xlabel('Predicted WQI', fontsize=14, fontweight='bold')
        ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=14, fontweight='bold')
        ax.set_title('Residual Analysis: Model Validation\n'
                   '(Random scatter = good fit; Pattern = systematic bias)', 
                   fontsize=16, fontweight='bold', pad=15)
        ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add statistics box
        stats_text = (f'Mean: {np.mean(residuals):.3f}\n'
                      f'Std: {np.std(residuals):.3f}\n'
                      f'Max|Error|: {np.max(np.abs(residuals)):.2f}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f'{output_dir}/05_residual_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✅ 05_residual_analysis.png saved")

    def _save_model_comparison_chart(self, predictions, y_test, output_dir):
        """Chart 6: Model Performance Comparison"""
        fig, ax = plt.subplots(figsize=(10, 7))

        models = ['XGBoost', 'LSTM', 'Ensemble']

        # Calculate all metrics
        r2_scores = [
            r2_score(y_test, predictions['xgboost']),
            r2_score(y_test, predictions['lstm']),
            r2_score(y_test, predictions['ensemble'])
        ]
        mae_scores = [
            mean_absolute_error(y_test, predictions['xgboost']),
            mean_absolute_error(y_test, predictions['lstm']),
            mean_absolute_error(y_test, predictions['ensemble'])
        ]
        rmse_scores = [
            np.sqrt(mean_squared_error(y_test, predictions['xgboost'])),
            np.sqrt(mean_squared_error(y_test, predictions['lstm'])),
            np.sqrt(mean_squared_error(y_test, predictions['ensemble']))
        ]

        x = np.arange(len(models))
        width = 0.25

        # R² bars (primary metric)
        bars1 = ax.bar(x - width, r2_scores, width, label='R² Score', 
                       color=['#27AE60', '#3498DB', '#E74C3C'], edgecolor='black')

        # MAE bars (secondary, scaled)
        mae_scaled = [m/10 for m in mae_scores]  # Scale to 0-1 range for visibility
        bars2 = ax.bar(x, mae_scaled, width, label='MAE (÷10)', 
                       color=['#2ECC71', '#5DADE2', '#EC7063'], edgecolor='black', alpha=0.7)

        # RMSE bars (tertiary, scaled)
        rmse_scaled = [r/10 for r in rmse_scores]
        bars3 = ax.bar(x + width, rmse_scaled, width, label='RMSE (÷10)', 
                       color=['#82E0AA', '#85C1E9', '#F1948A'], edgecolor='black', alpha=0.5)

        # Add value labels on bars
        for bars, scores in [(bars1, r2_scores), (bars2, mae_scores), (bars3, rmse_scores)]:
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                label = f'{score:.3f}' if score < 1 else f'{score:.2f}'
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        label, ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel('Score (R²) / Scaled Error (MAE, RMSE ÷10)', fontsize=13, fontweight='bold')
        ax.set_title('Model Performance Comparison\n(Higher R² = Better, Lower Error = Better)', 
                     fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
        ax.set_ylim(0, 1.2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add ensemble highlight
        ax.text(2, max(r2_scores) + 0.15, '★ BEST', ha='center', fontsize=12, 
                fontweight='bold', color='#E74C3C')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/06_model_comparison.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✅ 06_model_comparison.png saved")

    def _save_lstm_training_chart(self, ensemble, output_dir):
        """Chart 7: LSTM Training History (optional, if available)"""
        if not hasattr(ensemble.lstm, 'history') or ensemble.lstm.history is None:
            print("⚠️ LSTM training history not available, skipping training chart")
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        history = ensemble.lstm.history.history

        epochs = range(1, len(history['loss']) + 1)

        ax.plot(epochs, history['loss'], 'b-', lw=2.5, label='Training Loss (MSE)')
        if 'val_loss' in history:
            ax.plot(epochs, history['val_loss'], 'r--', lw=2.5, label='Validation Loss (MSE)')

        # Mark early stopping point
        stop_epoch = len(history['loss'])
        ax.axvline(x=stop_epoch, color='green', linestyle=':', lw=2, alpha=0.7,
                   label=f'Early Stop (epoch {stop_epoch})')

        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean Squared Error', fontsize=14, fontweight='bold')
        ax.set_title('LSTM Training Convergence\n(BatchNorm + Dropout + Early Stopping)', 
                     fontsize=16, fontweight='bold', pad=15)
        ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_yscale('log')  # Log scale often better for loss curves

        # Add final loss annotation
        final_loss = history['loss'][-1]
        ax.text(0.98, 0.95, f'Final Loss: {final_loss:.4f}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f'{output_dir}/07_lstm_training.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✅ 07_lstm_training.png saved")