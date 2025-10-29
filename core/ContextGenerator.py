"""
Unified Context Generator for Production
Single class that handles both XGBoost context generation and partner separation
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class ContextGenerator:
    """
    Single class for production context generation
    Combines XGBoost context generation with partner separation
    """

    def __init__(self,
                 artifact_dir: str = "model_artifacts",
                 n_estimators: int = 50,
                 max_depth: int = 5,
                 min_child_weight: int = 50,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 random_state: int = 42,
                 tree_subsets: Dict = None):
        """
        Initialize the unified context generator

        Args:
            artifact_dir: Directory to save/load artifacts
            n_estimators: Number of XGBoost trees
            max_depth: Maximum tree depth
            min_child_weight: Minimum samples in leaf
            learning_rate: Learning rate for XGBoost
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing trees
            random_state: Random seed for reproducibility
            tree_subsets: Dict mapping context levels to number of trees (optional)
        """
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(exist_ok=True)

        # XGBoost parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        # Model and encodings
        self.model = None
        self.partner_encoding = {}
        self.feature_names = []

        # Hierarchy configuration - use provided or default
        if tree_subsets is not None:
            self.tree_subsets = tree_subsets
        else:
            # Default tree subsets based on n_estimators
            self.tree_subsets = {
                'L1': min(5, n_estimators),
                'L2': min(10, n_estimators),
                'L3': min(15, n_estimators),
                'L4': min(20, n_estimators)
            }

        # Metadata
        self.metadata = {
            'train_date': None,
            'model_performance': {},
            'partner_encoding': {},
            'context_stats': {}
        }

    def fit_transform(self,
                      df: pd.DataFrame,
                      target_col: str = 'won',
                      affiliate_col: str = 'affiliate_integration_id',
                      verbose: bool = True) -> pd.DataFrame:
        """
        Fit the model and transform data with contexts (pipeline compatibility)

        Args:
            df: Feature matrix with target and affiliate columns
            target_col: Name of target column
            affiliate_col: Name of affiliate column
            verbose: Print progress

        Returns:
            DataFrame with added context columns
        """
        # Extract components
        X = df.drop([target_col, affiliate_col], axis=1)
        y = df[target_col]
        partner_series = df[affiliate_col]

        # Train model
        self.train(X, y, partner_series, verbose)

        # Generate contexts
        contexts = self.generate_contexts(X, partner_series)

        # Combine with original data
        result = pd.concat([df, contexts], axis=1)

        return result

    def train(self,
              X: pd.DataFrame,
              y: pd.Series,
              partner_series: pd.Series,
              verbose: bool = True) -> 'UnifiedContextGenerator':
        """
        Train the context generator

        Args:
            X: Feature matrix from FeatureEncoder
            y: Target (won)
            partner_series: Series containing affiliate_integration_id for each row
            verbose: Print progress

        Returns:
            Self for chaining
        """
        if verbose:
            print("="*60)
            print("TRAINING UNIFIED CONTEXT GENERATOR")
            print("="*60)

        # Create partner encoding
        self._create_partner_encoding(partner_series, verbose)

        # Train XGBoost
        self._train_xgboost(X, y, verbose)

        # Analyze context distribution
        if verbose:
            test_contexts = self.generate_contexts(X, partner_series)
            self._analyze_context_distribution(test_contexts)

        # Save artifacts
        self.save_artifacts()

        if verbose:
            print("\n✅ Unified context generator trained and saved!")

        return self

    def generate_contexts(self,
                         X: pd.DataFrame,
                         partner_series: pd.Series) -> pd.DataFrame:
        """
        Generate partner-separated contexts for production
        """
        if self.model is None:
            raise ValueError("Model not trained! Call train() or load_artifacts() first")

        # Get XGBoost leaf indices
        leaf_indices = self.model.apply(X)

        # Use affiliate_integration_ids directly (no mapping needed)
        partner_ids = partner_series.fillna(99999).astype(int)  # 99999 for unknown

        # Generate contexts for each level
        contexts = pd.DataFrame(index=X.index)

        for level, n_trees in self.tree_subsets.items():
            # Get base context from XGBoost trees
            level_indices = leaf_indices[:, :n_trees]
            base_contexts = []

            for row in level_indices:
                # Hash leaf indices to create base context
                leaf_path = '_'.join(map(str, row))
                context_hash = hashlib.md5(leaf_path.encode()).hexdigest()[:8]
                base_contexts.append(int(context_hash, 16) % 10_000_000)

            # Create partner-separated context
            # Format: [5-digit Partner_ID][7-digit Base_Context]
            # Result: 12-digit context ID
            contexts[f'context_{level}'] = (
                partner_ids * 10_000_000 +  # 5-digit ID becomes high-order digits
                np.array(base_contexts)      # 7-digit context in lower digits
            )

        return contexts

    def generate_single_context(self,
                               X_single: pd.DataFrame,
                               partner_id: int) -> Dict[str, int]:
        """
        Generate context for a single row (real-time prediction)

        Args:
            X_single: Single row of features (1xN DataFrame)
            partner_id: Affiliate integration ID (5-digit integer)
        """
        if self.model is None:
            raise ValueError("Model not trained!")

        # Use partner_id directly (no mapping lookup needed!)
        # Convert to string for lookup since keys are stored as strings
        partner_id_str = str(partner_id)
        if partner_id_str in self.partner_encoding:
            partner_id = int(self.partner_encoding[partner_id_str])
        else:
            partner_id = 99999  # Default for unknown

        # Get leaf indices
        leaf_indices = self.model.apply(X_single)[0]

        # Generate contexts
        contexts = {}
        for level, n_trees in self.tree_subsets.items():
            level_indices = leaf_indices[:n_trees]
            leaf_path = '_'.join(map(str, level_indices))
            context_hash = hashlib.md5(leaf_path.encode()).hexdigest()[:8]
            base_context = int(context_hash, 16) % 10_000_000

            # Direct usage of partner_id
            contexts[level] = partner_id * 10_000_000 + base_context

        return contexts

    def _create_partner_encoding(self, partner_series: pd.Series, verbose: bool):
        """Store affiliate_integration_ids directly (no sequential mapping needed)"""

        # Get unique partner IDs sorted by frequency
        partner_counts = partner_series.value_counts()

        if verbose:
            print(f"\n📊 Partner Distribution (affiliate_integration_id):")

        # Direct mapping: affiliate_integration_id -> itself
        # No sequential IDs needed since they're already 5-digit integers
        for affiliate_id, count in partner_counts.items():
            self.partner_encoding[int(affiliate_id)] = int(affiliate_id)  # Map to itself!

            if verbose and len(self.partner_encoding) <= 10:  # Show first 10
                pct = count / len(partner_series) * 100
                print(f"  Affiliate ID {affiliate_id:5d}: {count:8,} ({pct:5.1f}%)")

        self.metadata['partner_encoding'] = {int(k): int(v) for k, v in self.partner_encoding.items()}

    def _train_xgboost(self, X: pd.DataFrame, y: pd.Series, verbose: bool):
        """Train XGBoost model"""

        # Store feature names
        self.feature_names = list(X.columns)

        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        if verbose:
            print(f"\n📊 Training Data:")
            print(f"  - Training samples: {len(X_train):,}")
            print(f"  - Validation samples: {len(X_val):,}")
            print(f"  - Features: {X.shape[1]}")
            print(f"  - Win rate: {y.mean():.4%}")

        # Calculate scale_pos_weight
        scale_pos_weight = (1 - y_train.mean()) / y_train.mean()

        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'max_depth': self.max_depth,
            'n_estimators': self.n_estimators,
            'min_child_weight': self.min_child_weight,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'logloss',
            'random_state': self.random_state,
            'verbosity': 0
        }

        # Train model
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate
        val_pred = self.model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred)

        self.metadata['model_performance'] = {
            'val_auc': float(val_auc),
            'win_rate': float(y.mean()),
            'n_samples': len(X)
        }

        if verbose:
            print(f"\n📈 Model Performance:")
            print(f"  - Validation AUC: {val_auc:.4f}")

            # Feature importance
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\n🎯 Top 5 Features:")
            for _, row in importance.head(5).iterrows():
                print(f"  - {row['feature']:30s}: {row['importance']:.4f}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.model is None:
            return {}

        importance = {}
        for i, feature_name in enumerate(self.feature_names):
            importance[feature_name] = float(self.model.feature_importances_[i])

        return importance

    def _analyze_context_distribution(self, contexts: pd.DataFrame):
        """Analyze context distribution"""

        print(f"\n📊 Context Distribution:")
        print(f"{'Level':<6} {'Unique':<10} {'Min':<8} {'Median':<8} {'Max':<8}")
        print("-" * 40)

        for level in self.tree_subsets.keys():
            col = f'context_{level}'
            if col in contexts.columns:
                counts = contexts[col].value_counts()

                self.metadata['context_stats'][level] = {
                    'unique': int(contexts[col].nunique()),
                    'min_samples': int(counts.min()),
                    'median_samples': float(counts.median()),
                    'max_samples': int(counts.max())
                }

                print(f"{level:<6} {contexts[col].nunique():<10,} {counts.min():<8} "
                      f"{counts.median():<8.0f} {counts.max():<8,}")

    def save_artifacts(self, artifact_dir: Optional[str] = None):
        """Save all artifacts for production"""

        if artifact_dir:
            self.artifact_dir = Path(artifact_dir)
            self.artifact_dir.mkdir(exist_ok=True)

        # Save model using XGBoost native format (secure)
        model_path = self.artifact_dir / "xgboost_model.json"
        self.model.save_model(str(model_path))

        # Save metadata
        self.metadata['train_date'] = datetime.now().isoformat()
        self.metadata['tree_subsets'] = self.tree_subsets
        self.metadata['feature_names'] = self.feature_names
        self.metadata['xgboost_params'] = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_child_weight': self.min_child_weight,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state
        }

        metadata_path = self.artifact_dir / "context_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

        print(f"\n💾 Artifacts saved to {self.artifact_dir}")

    def load_artifacts(self, artifact_dir: Optional[str] = None):
        """Load from saved artifacts"""

        if artifact_dir:
            self.artifact_dir = Path(artifact_dir)

        # Load model (try secure format first)
        model_path_json = self.artifact_dir / "xgboost_model.json"
        model_path_old = self.artifact_dir / "XGBOOST_CONTEXT_GENERATOR.json"
        model_path_pkl = self.artifact_dir / "XGBOOST_CONTEXT_GENERATOR.pkl"

        if model_path_json.exists():
            # Load using secure XGBoost native format
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(model_path_json))
        elif model_path_old.exists():
            # Try old naming convention
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(model_path_old))
        elif model_path_pkl.exists():
            # Fallback to old format with warning
            print("  ⚠️ Loading legacy .pkl format. Consider re-training for security.")
            import joblib
            self.model = joblib.load(model_path_pkl)
        else:
            raise FileNotFoundError(f"XGBoost model not found in {self.artifact_dir}")

        # Load metadata (try new name first, fallback to old)
        metadata_path_new = self.artifact_dir / "context_metadata.json"
        metadata_path_old = self.artifact_dir / "CONTEXT_TABLE.json"

        if metadata_path_new.exists():
            metadata_path = metadata_path_new
        elif metadata_path_old.exists():
            metadata_path = metadata_path_old
            print("  ⚠️ Using legacy metadata filename")
        else:
            raise FileNotFoundError(f"Context metadata not found in {self.artifact_dir}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.partner_encoding = self.metadata.get('partner_encoding', {})
        self.tree_subsets = self.metadata.get('tree_subsets', {})
        self.feature_names = self.metadata.get('feature_names', [])

        # Restore XGBoost params if available
        if 'xgboost_params' in self.metadata:
            params = self.metadata['xgboost_params']
            self.n_estimators = params.get('n_estimators', 50)
            self.max_depth = params.get('max_depth', 5)
            self.min_child_weight = params.get('min_child_weight', 50)
            self.learning_rate = params.get('learning_rate', 0.1)
            self.subsample = params.get('subsample', 0.8)
            self.colsample_bytree = params.get('colsample_bytree', 0.8)
            self.random_state = params.get('random_state', 42)

        print(f"✅ Loaded unified context generator")
        print(f"   - Partners: {len(self.partner_encoding)}")
        print(f"   - Features: {len(self.feature_names)}")
        if 'model_performance' in self.metadata:
            print(f"   - Val AUC: {self.metadata['model_performance'].get('val_auc', 0):.4f}")