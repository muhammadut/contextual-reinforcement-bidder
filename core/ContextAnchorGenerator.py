"""
Flexible Context Level Anchor Analyzer
Can generate anchors starting from any context level (L4, L3, L2, L1)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
from pathlib import Path


class FlexibleContextAnalyzer:
    """
    Find anchor bids using configuration wins with flexible context level selection
    """
    
    def __init__(self,
                 min_wins_for_anchor: int = 10,
                 recency_days: int = None,
                 primary_context_level: str = 'context_L3'):
        """
        Args:
            min_wins_for_anchor: Minimum WINS (not samples) to trust an anchor
            recency_days: Consider only recent data for anchor selection (None = use all data)
            primary_context_level: Starting context level ('context_L4', 'context_L3', 'context_L2', 'context_L1')
        """
        self.min_wins_for_anchor = min_wins_for_anchor
        self.recency_days = recency_days
        self.primary_context_level = primary_context_level
        
        # Define hierarchy based on primary level
        all_levels = ['context_L4', 'context_L3', 'context_L2', 'context_L1']
        if primary_context_level not in all_levels:
            raise ValueError(f"Invalid context level. Must be one of {all_levels}")
        
        # Create fallback hierarchy from primary level
        start_idx = all_levels.index(primary_context_level)
        self.context_hierarchy = all_levels[start_idx:]
        
        # Storage for analysis
        self.context_anchors = {}
        self.context_stats = {}
        self.fallback_stats = defaultdict(int)
        self.affiliate_mapping = {}
        self.affiliate_id_to_name = {}  # Auto-generated in find_anchors()
        
        # Add storage for global fallback
        self.global_fallback_anchor = None
        self.global_fallback_stats = {}
        
    def find_anchors(self, 
                     df: pd.DataFrame) -> Dict[int, Dict]:
        """
        Find anchor bids using configuration wins only
        
        Args:
            df: Historical data with context columns, won, final_modified, variant_id
            
        Returns:
            Dictionary mapping context_id to anchor information
        """
        print("="*60)
        print(f"ANCHOR DISCOVERY USING {self.primary_context_level.upper()}")
        print("="*60)
        
        # AUTO-GENERATE ID TO NAME MAPPING
        if 'affiliate_integration_id' in df.columns and 'affiliate_name' in df.columns:
            # Create mapping from the data itself
            id_name_pairs = df[['affiliate_integration_id', 'affiliate_name']].drop_duplicates()
            self.affiliate_id_to_name = dict(zip(
                id_name_pairs['affiliate_integration_id'].astype(int),
                id_name_pairs['affiliate_name']
            ))
            print(f"\n📊 Auto-generated mapping for {len(self.affiliate_id_to_name)} affiliates")
        elif 'affiliate_integration_id' in df.columns:
            # No name column, create default display names
            unique_ids = df['affiliate_integration_id'].unique()
            self.affiliate_id_to_name = {
                int(aid): f"Affiliate_{aid}" for aid in unique_ids
            }
            print(f"\n⚠️ No affiliate_name column found, using default names")
        else:
            self.affiliate_id_to_name = {}
        
        # IMPORTANT: Get ALL unique contexts from full dataset first
        # This ensures we preserve contexts even if they haven't appeared recently
        all_unique_contexts = df[self.primary_context_level].unique()
        print(f"\n📊 Total unique contexts in dataset: {len(all_unique_contexts):,}")

        # BUILD AFFILIATE MAPPING from FULL dataset (context -> affiliate_id)
        if 'affiliate_integration_id' in df.columns:
            for context_id in all_unique_contexts:
                affiliate_ids = df[df[self.primary_context_level] == context_id]['affiliate_integration_id'].unique()
                if len(affiliate_ids) == 1:
                    self.affiliate_mapping[context_id] = int(affiliate_ids[0])
                elif len(affiliate_ids) > 1:
                    most_common = df[df[self.primary_context_level] == context_id]['affiliate_integration_id'].mode()[0]
                    self.affiliate_mapping[context_id] = int(most_common)

        # NOW filter to recent data for anchor discovery
        recent_df = df.copy()
        if self.recency_days is not None and 'ping_date' in df.columns:
            cutoff_date = df['ping_date'].max() - timedelta(days=self.recency_days)
            recent_df = df[df['ping_date'] >= cutoff_date].copy()
            print(f"📅 Using last {self.recency_days} days for anchor discovery")
            print(f"   Total samples: {len(recent_df):,}")
            print(f"   Contexts in recent data: {recent_df[self.primary_context_level].nunique():,}")
        else:
            print(f"📅 Using ALL data for anchor discovery (no recency filter)")
            print(f"   Total samples: {len(recent_df):,}")
            print(f"   Contexts in data: {recent_df[self.primary_context_level].nunique():,}")

        # Filter to non-RL data (configuration or None/NULL - before experiments or never in experiments)
        # Include: bid_modification_variant_id == 'configuration' OR isna() OR anything != 'rl'
        config_df = recent_df[
            (recent_df['bid_modification_variant_id'] == 'configuration') |
            (recent_df['bid_modification_variant_id'].isna()) |
            (~recent_df['bid_modification_variant_id'].str.lower().str.contains('rl', na=False))
        ].copy()
        print(f"   Non-RL baseline samples: {len(config_df):,} ({len(config_df)/len(recent_df)*100:.1f}%)")

        # Filter to wins only
        config_wins = config_df[config_df['won'] == 1].copy()
        config_win_pct = (len(config_wins)/len(config_df)*100) if len(config_df) > 0 else 0
        print(f"   Configuration wins: {len(config_wins):,} ({config_win_pct:.3f}%)")

        # Handle case with NO configuration data - use ALL wins as fallback
        if len(config_wins) == 0:
            print("\n⚠️  WARNING: No configuration wins found!")
            print("   Falling back to using ALL winning bids for anchor discovery")
            config_wins = recent_df[recent_df['won'] == 1].copy()
            print(f"   Using {len(config_wins):,} total wins from recent data")

        # Ensure action is integer
        if len(config_wins) > 0:
            config_wins['action'] = config_wins['final_modified'].astype(int)

        # Calculate global fallback anchor from ALL configuration wins in recent data
        if len(config_wins) > 0:
            all_winning_bids = config_wins['action'].value_counts()
            self.global_fallback_anchor = int(all_winning_bids.index[0])
            self.global_fallback_stats = {
                'anchor_bid': self.global_fallback_anchor,
                'total_wins': int(all_winning_bids.iloc[0]),
                'total_config_wins': len(config_wins),
                'percentage': float(all_winning_bids.iloc[0] / len(config_wins))
            }
            print(f"\n🌐 Global fallback anchor: {self.global_fallback_anchor}% "
                  f"({self.global_fallback_stats['total_wins']:,}/{len(config_wins):,} wins)")

        # Get unique contexts at primary level - USE ALL CONTEXTS
        unique_contexts = all_unique_contexts
        print(f"\n📊 Unique contexts at {self.primary_context_level}: {len(unique_contexts):,}")
        print(f"   Fallback hierarchy: {' → '.join(self.context_hierarchy)} → affiliate → global")

        # Find anchors for each context - PRESERVE ALL CONTEXTS
        for context_id in unique_contexts:
            anchor_info = self._find_anchor_for_context(
                config_wins, recent_df, df, context_id
            )
            # ALWAYS add the context, even if anchor_info is None
            # The ContextTable will handle None anchors with exploration
            self.context_anchors[context_id] = anchor_info

        # Count how many contexts have anchors vs None
        contexts_with_anchors = sum(1 for v in self.context_anchors.values() if v is not None)
        contexts_without_anchors = len(self.context_anchors) - contexts_with_anchors

        print(f"\n✅ Preserved all {len(self.context_anchors):,} contexts")
        print(f"   With anchors: {contexts_with_anchors:,}")
        print(f"   Without anchors (will explore): {contexts_without_anchors:,}")
        
        # Analyze results
        self._analyze_anchors(df, config_wins)
        
        return self.context_anchors
    
    def _find_anchor_for_context(self,
                                 config_wins: pd.DataFrame,
                                 recent_df: pd.DataFrame,
                                 full_df: pd.DataFrame,
                                 context_id: int) -> Optional[Dict]:
        """
        Find anchor for a single context with hierarchical fallback

        Args:
            config_wins: Configuration wins from recent data (for finding anchors)
            recent_df: Recent data for looking up context relationships
            full_df: Full historical data for calculating stats
            context_id: The context ID to find anchor for
        """
        # Try each level in hierarchy
        for level in self.context_hierarchy:
            # Get wins for this context at this level
            if level == self.primary_context_level:
                context_wins = config_wins[config_wins[level] == context_id]
            else:
                # For fallback, find the corresponding context at lower level
                # Use recent_df first, fallback to full_df if not in recent data
                sample = recent_df[recent_df[self.primary_context_level] == context_id].head(1)
                if sample.empty:
                    sample = full_df[full_df[self.primary_context_level] == context_id].head(1)
                if sample.empty or level not in sample.columns:
                    continue
                fallback_context = sample[level].iloc[0]
                context_wins = config_wins[config_wins[level] == fallback_context]
            
            # Check if we have enough wins
            if len(context_wins) >= self.min_wins_for_anchor:
                # Find most common winning bid
                action_counts = context_wins['action'].value_counts()
                most_common_action = action_counts.index[0]
                
                # Calculate statistics
                if level == self.primary_context_level:
                    total_samples_at_level = len(full_df[full_df[level] == context_id])
                else:
                    total_samples_at_level = len(full_df[full_df[level] == fallback_context])
                
                self.fallback_stats[level] += 1
                
                # Get affiliate ID for this context
                affiliate_id = self.affiliate_mapping.get(context_id, 99999)
                
                return {
                    'anchor_bid': int(most_common_action),
                    'wins_with_anchor': int(action_counts.iloc[0]),
                    'total_wins': len(context_wins),
                    'total_samples': total_samples_at_level,
                    'win_rate': len(context_wins) / total_samples_at_level if total_samples_at_level > 0 else 0,
                    'level': level,
                    'affiliate_id': affiliate_id,  # Store the ID
                    'affiliate': self.affiliate_id_to_name.get(affiliate_id, f"Affiliate_{affiliate_id}")  # Display name
                }
        
        # Try affiliate fallback - UPDATED FOR affiliate_integration_id
        if context_id in self.affiliate_mapping:
            affiliate_id = self.affiliate_mapping[context_id]
            
            # Filter wins by affiliate_integration_id
            affiliate_wins = config_wins[
                config_wins['affiliate_integration_id'] == affiliate_id
            ] if 'affiliate_integration_id' in config_wins.columns else pd.DataFrame()
            
            if len(affiliate_wins) >= self.min_wins_for_anchor:
                action_counts = affiliate_wins['action'].value_counts()
                most_common_action = action_counts.index[0]
                
                total_affiliate_samples = len(
                    full_df[full_df['affiliate_integration_id'] == affiliate_id]
                ) if 'affiliate_integration_id' in full_df.columns else 0
                
                self.fallback_stats['affiliate'] += 1
                
                return {
                    'anchor_bid': int(most_common_action),
                    'wins_with_anchor': int(action_counts.iloc[0]),
                    'total_wins': len(affiliate_wins),
                    'total_samples': total_affiliate_samples,
                    'win_rate': len(affiliate_wins) / total_affiliate_samples if total_affiliate_samples > 0 else 0,
                    'level': 'affiliate',
                    'affiliate_id': affiliate_id,
                    'affiliate': self.affiliate_id_to_name.get(affiliate_id, f"Affiliate_{affiliate_id}")
                }
        
        # Global fallback - use most common bid across ALL config wins
        # This ensures ALL contexts get an anchor
        if self.global_fallback_anchor is not None:
            self.fallback_stats['global'] += 1

            # Get affiliate info if available
            affiliate_id = self.affiliate_mapping.get(context_id, 99999)

            # Get context-specific stats for proper tracking
            context_samples = len(full_df[full_df[self.primary_context_level] == context_id])
            context_wins = len(config_wins[config_wins[self.primary_context_level] == context_id])

            return {
                'anchor_bid': self.global_fallback_anchor,
                'wins_with_anchor': self.global_fallback_stats['total_wins'],
                'total_wins': context_wins,  # Context-specific wins (may be 0)
                'total_samples': context_samples,  # Context-specific samples
                'win_rate': context_wins / context_samples if context_samples > 0 else 0,
                'level': 'global',
                'affiliate_id': affiliate_id,
                'affiliate': self.affiliate_id_to_name.get(affiliate_id, f"Affiliate_{affiliate_id}")
            }

        # This should never happen if we have any config wins at all
        # But keep as safety net
        self.fallback_stats['none'] += 1

        # Get affiliate info even for None case
        affiliate_id = self.affiliate_mapping.get(context_id, 99999)

        # Return a minimal anchor structure even if no global anchor
        # This allows the context to exist but will rely on exploration
        return {
            'anchor_bid': 60,  # Default exploratory bid
            'wins_with_anchor': 0,
            'total_wins': 0,
            'total_samples': len(full_df[full_df[self.primary_context_level] == context_id]),
            'win_rate': 0.0,
            'level': 'default',
            'affiliate_id': affiliate_id,
            'affiliate': self.affiliate_id_to_name.get(affiliate_id, f"Affiliate_{affiliate_id}")
        }
    
    def _analyze_anchors(self, df: pd.DataFrame, config_wins: pd.DataFrame):
        """
        Analyze the discovered anchors
        """
        print("\n" + "="*60)
        print("ANCHOR ANALYSIS")
        print("="*60)
        
        # Fallback level distribution
        print("\n📈 Fallback Levels Used:")
        total_contexts = df[self.primary_context_level].nunique()
        total_anchors = sum(v for k, v in self.fallback_stats.items())

        for level in self.context_hierarchy + ['affiliate', 'global', 'default', 'none']:
            if level in self.fallback_stats:
                count = self.fallback_stats[level]
                pct_of_all = count / total_contexts * 100
                pct_of_anchored = count / total_anchors * 100 if total_anchors > 0 else 0
                print(f"   - {level:15s}: {count:6,} ({pct_of_all:5.1f}% of all, {pct_of_anchored:5.1f}% of anchored)")
        
        # Calculate how many used primary level
        primary_success_rate = self.fallback_stats.get(self.primary_context_level, 0) / total_anchors * 100 if total_anchors > 0 else 0
        print(f"\n🎯 Primary Level Success: {primary_success_rate:.1f}% found at {self.primary_context_level}")
        
        # Anchor bid distribution
        if self.context_anchors:
            anchor_bids = [info['anchor_bid'] for info in self.context_anchors.values()]
            print("\n📊 Anchor Bid Distribution:")
            bid_counts = Counter(anchor_bids)
            for bid in sorted(bid_counts.keys()):
                count = bid_counts[bid]
                pct = count / len(anchor_bids) * 100
                print(f"   - {bid:3d}%: {count:6,} contexts ({pct:5.1f}%)")
            
            # Win statistics
            total_wins_with_anchors = sum(info['wins_with_anchor'] for info in self.context_anchors.values())
            total_wins_all = sum(info['total_wins'] for info in self.context_anchors.values())
            
            print(f"\n📊 Win Statistics:")
            print(f"   - Total configuration wins: {len(config_wins):,}")
            print(f"   - Wins in anchored contexts: {total_wins_all:,}")
            print(f"   - Wins with anchor bid: {total_wins_with_anchors:,}")
            if total_wins_all > 0:
                print(f"   - Anchor bid accuracy: {total_wins_with_anchors/total_wins_all:.1%}")
        
        # Context coverage
        print(f"\n📈 Context Coverage:")
        covered = len(self.context_anchors)
        print(f"   - Total contexts:      {total_contexts:,}")
        print(f"   - With anchors:        {covered:,} ({covered/total_contexts*100:.1f}%)")
        print(f"   - Without anchors:   {total_contexts - covered:,} ({(total_contexts-covered)/total_contexts*100:.1f}%)")
    
    def analyze_context_density(self, df: pd.DataFrame):
        """
        Analyze how many samples and wins each context level has
        """
        print("\n" + "="*60)
        print("CONTEXT DENSITY ANALYSIS")
        print("="*60)
        
        config_df = df[df['bid_modification_variant_id'] == 'configuration']
        config_wins = config_df[config_df['won'] == 1]
        
        print("\n📊 Context Density by Level:")
        print(f"{'Level':<15} {'Unique':<10} {'Avg Samples':<12} {'Avg Wins':<10} {'% w/ 10+ wins':<15}")
        print("-" * 65)
        
        for level in ['context_L1', 'context_L2', 'context_L3', 'context_L4']:
            if level in df.columns:
                unique_count = df[level].nunique()
                
                # Calculate average samples per context
                samples_per_context = df.groupby(level).size()
                avg_samples = samples_per_context.mean()
                
                # Calculate average wins per context
                wins_per_context = config_wins.groupby(level).size()
                avg_wins = wins_per_context.mean() if len(wins_per_context) > 0 else 0
                
                # Calculate percentage with 10+ wins
                contexts_with_10_wins = (wins_per_context >= 10).sum() if len(wins_per_context) > 0 else 0
                pct_with_10_wins = contexts_with_10_wins / unique_count * 100
                
                print(f"{level:<15} {unique_count:<10,} {avg_samples:<12.1f} {avg_wins:<10.2f} {pct_with_10_wins:<14.1f}%")
        
        # Show impact of using each level
        print("\n🎯 Recommended Level Analysis:")
        for level in ['context_L2', 'context_L3', 'context_L4']:
            if level in df.columns:
                analyzer_temp = FlexibleContextAnalyzer(
                    min_wins_for_anchor=10,
                    recency_days=self.recency_days,
                    primary_context_level=level
                )
                # Quick check without full analysis
                unique = df[level].nunique()
                wins_per_context = config_wins.groupby(level).size() if len(config_wins) > 0 else pd.Series()
                viable = (wins_per_context >= 10).sum() if len(wins_per_context) > 0 else 0
                
                print(f"\n  Using {level}:")
                print(f"     - Contexts: {unique:,}")
                print(f"     - Viable contexts (10+ wins): {viable:,} ({viable/unique*100:.1f}%)")


# ============================================================================
# USAGE FUNCTIONS
# ============================================================================

def analyze_with_level(df: pd.DataFrame, 
                       context_level: str = 'context_L3',
                       min_wins: int = 10,
                       recency_days: int = 60):
    """
    Analyze anchors using specified context level
    
    Args:
        df: Historical data
        context_level: Primary context level to use ('context_L4', 'context_L3', 'context_L2', 'context_L1')
        min_wins: Minimum wins for anchor
        recency_days: Days of recent data to use
    """
    
    print(f"\n{'='*60}")
    print(f"ANALYZING WITH {context_level.upper()} AS PRIMARY LEVEL")
    print(f"{'='*60}")
    
    # Initialize analyzer with specified level
    analyzer = FlexibleContextAnalyzer(
        min_wins_for_anchor=min_wins,
        recency_days=recency_days,
        primary_context_level=context_level
    )
    
    # Find anchors
    anchors = analyzer.find_anchors(df)
    
    # Analyze context density
    analyzer.analyze_context_density(df)
    
    return analyzer, anchors


def compare_context_levels(df: pd.DataFrame, 
                           levels_to_test: List[str] = ['context_L3', 'context_L2'],
                           min_wins: int = 10,
                           recency_days: int = 60):
    """
    Compare different context levels to find optimal granularity
    
    Args:
        df: Historical data
        levels_to_test: List of context levels to compare
        min_wins: Minimum wins for anchor
        recency_days: Days of recent data
    """
    
    print("\n" + "="*60)
    print("CONTEXT LEVEL COMPARISON")
    print("="*60)
    
    results = {}
    
    for level in levels_to_test:
        print(f"\n{'='*40}")
        print(f"Testing {level}")
        print(f"{'='*40}")
        
        analyzer = FlexibleContextAnalyzer(
            min_wins_for_anchor=min_wins,
            recency_days=recency_days,
            primary_context_level=level
        )
        
        anchors = analyzer.find_anchors(df)
        
        # Collect metrics
        total_contexts = df[level].nunique()
        anchored_contexts = len(anchors)
        primary_level_anchors = analyzer.fallback_stats.get(level, 0)
        
        results[level] = {
            'total_contexts': total_contexts,
            'anchored_contexts': anchored_contexts,
            'coverage': anchored_contexts / total_contexts * 100,
            'primary_level_success': primary_level_anchors / anchored_contexts * 100 if anchored_contexts > 0 else 0,
            'fallback_stats': dict(analyzer.fallback_stats)
        }
    
    # Summary comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\n{'Level':<15} {'Contexts':<10} {'Anchored':<10} {'Coverage':<10} {'Primary Success':<15}")
    print("-" * 60)
    
    for level, metrics in results.items():
        print(f"{level:<15} {metrics['total_contexts']:<10,} {metrics['anchored_contexts']:<10,} "
              f"{metrics['coverage']:<9.1f}% {metrics['primary_level_success']:<14.1f}%")
    
    return results


# Main execution function
def run_analysis(df: pd.DataFrame, primary_level: str = 'context_L3'):
    """
    Main function to run the analysis
    """
    # First, analyze context density to inform decision
    print("\n" + "="*60)
    print("INITIAL CONTEXT DENSITY CHECK")
    print("="*60)
    
    temp_analyzer = FlexibleContextAnalyzer(primary_context_level=primary_level)
    temp_analyzer.analyze_context_density(df)
    
    # Analyze with the specified primary level
    print("\n" + "="*60)
    print(f"DETAILED ANALYSIS WITH {primary_level.upper()}")
    print("="*60)
    
    analyzer, anchors = analyze_with_level(df, primary_level)
    
    # Check if we need to fall back to a lower level (e.g., L3 if primary is L4)
    coverage = len(anchors) / df[primary_level].nunique() * 100
    primary_success = analyzer.fallback_stats.get(primary_level, 0) / len(anchors) * 100 if anchors else 0
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    if primary_success < 30:
        print(f"\n⚠️ {primary_level.upper()} primary success rate is low ({primary_success:.1f}%)")
        print("   Recommendation: Consider using a less granular level (e.g., L3)")
    else:
        print(f"\n✅ {primary_level.upper()} appears suitable:")
        print(f"   - Primary success: {primary_success:.1f}%")
        print(f"   - Coverage: {coverage:.1f}%")
        
    return {primary_level: (analyzer, anchors)}


if __name__ == "__main__":
    print("Ready to analyze with flexible context levels!")
    print("\nUsage:")
    print("------")
    print("# Run complete analysis:")
    print("results = run_analysis(df_with_contexts)")
    print("")
    print("# Or analyze specific level:")
    print("analyzer, anchors = analyze_with_level(df_with_contexts, 'context_L3')")
    print("")
    print("# Or compare levels:")
    print("comparison = compare_context_levels(df_with_contexts, ['context_L3', 'context_L2'])")