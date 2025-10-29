"""
Context Table Generator with Bid Lower Strategy
Focused purely on initialization from training data
No unknown context handling - that's the updater's job
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import defaultdict, deque, Counter
import json
from datetime import datetime


class ContextTableBidLowerStrategy:
    """
    Context Table Generator using Bid Lower Strategy
    - Uses ACTUAL anchors from L3 analysis
    - Bids BELOW the anchor based on partner win rate
    - Creates initial table for known contexts only
    - Unknown context handling is done at runtime by updater
    """

    def __init__(self,
                 window_days: int = 15,
                 exploration_rate: float = 0.15,
                 prior_weight_ratio: float = 0.1,
                 score_scale: float = 1000.0,
                 min_action: int = 40,
                 max_action: int = 97,
                 strategy_tiers: dict = None):
        """
        Initialize the Context Table Generator

        Args:
            window_days: Rolling window for score calculation
            exploration_rate: Epsilon for epsilon-greedy exploration
            prior_weight_ratio: Prior weight as ratio of recent data
            score_scale: Multiplier for scores (1000 = show 0.000979 as 0.979)
            min_action: Lower bound for actions (default: 40)
            max_action: Upper bound for actions (default: 97)
            strategy_tiers: Dict defining win rate thresholds and action offsets
                           Format: {"LOW": {"threshold": 0.01, "offsets": [-4, -3, -2]}, ...}
        """
        self.window_days = window_days
        self.exploration_rate = exploration_rate
        self.prior_weight_ratio = prior_weight_ratio
        self.score_scale = score_scale
        self.MIN_ACTION = min_action
        self.MAX_ACTION = max_action

        # Default strategy tiers
        self.strategy_tiers = strategy_tiers or {
            "LOW": {"threshold": 0.01, "offsets": [-4, -3, -2]},
            "MEDIUM": {"threshold": 0.50, "offsets": [-6, -4, -3]},
            "HIGH": {"threshold": float('inf'), "offsets": [-8, -6, -4]}
        }
        
        # Main data structures
        self.contexts = {}  # context_id -> context info
        self.table = defaultdict(lambda: defaultdict(dict))  # context -> action -> metrics
        self.partner_stats = {}  # partner -> global stats
        self.partner_id_to_stats = {}  # affiliate_integration_id -> stats
        self.context_to_affiliate = {}  # context_id -> affiliate_id mapping
        self.affiliate_anchors = {}  # affiliate_id -> anchor_bid (for unknown contexts)
        self.min_wins_for_affiliate_anchor = 10  # Minimum wins to create affiliate anchor
        
    def initialize_with_anchors(self, anchors_l3: Dict, df_with_contexts: pd.DataFrame):
        """
        Initialize with L3 anchors - USE THE ACTUAL ANCHOR VALUES
        
        Args:
            anchors_l3: Dictionary of context_id -> anchor_info from L3 analysis
            df_with_contexts: Historical data with affiliate information
        """
        if df_with_contexts is None:
            raise ValueError("df_with_contexts is required for initialization!")
            
        print("="*60)
        print("CONTEXT TABLE GENERATOR - BID LOWER STRATEGY")
        print("="*60)
        
        # Process contexts from anchors_l3 - KEEP ACTUAL ANCHORS
        for context_id, anchor_info in anchors_l3.items():
            affiliate_id = anchor_info.get('affiliate_id', 99999)
            
            self.contexts[context_id] = {
                'anchor': anchor_info['anchor_bid'],  # USE ACTUAL ANCHOR FROM L3
                'affiliate': anchor_info['affiliate'],
                'affiliate_id': affiliate_id,
                'fallback_level': anchor_info['level'],
                'win_rate': anchor_info.get('win_rate', 0),
                'total_wins': anchor_info.get('total_wins', 0),
                'total_samples': anchor_info.get('total_samples', 0),
                'actions': None,
                'strategy_type': None
            }
            
            self.context_to_affiliate[context_id] = affiliate_id
        
        print(f"✅ Initialized {len(self.contexts)} contexts with L3 anchors")
        
        # Calculate partner stats for strategy application
        self._calculate_partner_stats(df_with_contexts)
        print(f"📊 Calculated stats for {len(self.partner_stats)} partners")

        # Build affiliate-level anchors for unknown context handling
        self._build_affiliate_anchors(df_with_contexts)
        print(f"🎯 Built affiliate anchors for {len(self.affiliate_anchors)} affiliates")

        return self
    
    def _calculate_partner_stats(self, df: pd.DataFrame):
        """
        Calculate partner-level statistics for strategy tier determination
        """
        # Build mapping of ID to display name from contexts
        id_to_name = {}
        for ctx_info in self.contexts.values():
            if 'affiliate_id' in ctx_info and ctx_info['affiliate_id'] != 99999:
                id_to_name[ctx_info['affiliate_id']] = ctx_info['affiliate']
        
        # Calculate stats for each partner
        for partner_id in df['affiliate_integration_id'].unique():
            partner_id = int(partner_id)
            partner_data = df[df['affiliate_integration_id'] == partner_id]
            
            total_samples = len(partner_data)
            total_wins = partner_data['won'].sum()
            win_rate = total_wins / total_samples if total_samples > 0 else 0
            avg_revenue = partner_data['expected_revenue'].mean() if 'expected_revenue' in partner_data else 50
            
            display_name = id_to_name.get(partner_id, f"Affiliate_{partner_id}")
            
            self.partner_stats[display_name] = {
                'partner_id': partner_id,
                'total_samples': int(total_samples),
                'total_wins': int(total_wins),
                'win_rate': float(win_rate),
                'avg_revenue': float(avg_revenue)
            }
            
            self.partner_id_to_stats[partner_id] = self.partner_stats[display_name]

    def _build_affiliate_anchors(self, df: pd.DataFrame):
        """
        Build affiliate-level anchors from configuration wins.
        These are used as fallback for unknown contexts with known affiliates.

        Fallback hierarchy:
        1. Known context -> use context anchor
        2. Unknown context, known affiliate -> use affiliate anchor
        3. Unknown affiliate -> use global default (70%)
        """
        # Filter to non-RL data (configuration or None/NULL - before experiments or never in experiments)
        # Include: bid_modification_variant_id == 'configuration' OR isna() OR anything != 'rl'
        config_df = df[
            (df['bid_modification_variant_id'] == 'configuration') |
            (df['bid_modification_variant_id'].isna()) |
            (~df['bid_modification_variant_id'].str.lower().str.contains('rl', na=False))
        ].copy()
        config_wins = config_df[config_df['won'] == 1].copy()

        if len(config_wins) == 0:
            print("⚠️ No non-RL baseline wins found for affiliate anchor calculation")
            return

        # Ensure action is integer
        config_wins['action'] = config_wins['final_modified'].astype(int)

        # Group by affiliate_integration_id and find most common winning bid
        for affiliate_id in df['affiliate_integration_id'].unique():
            affiliate_id = int(affiliate_id)

            # Get wins for this affiliate
            affiliate_wins = config_wins[config_wins['affiliate_integration_id'] == affiliate_id]

            if len(affiliate_wins) >= self.min_wins_for_affiliate_anchor:
                # Find most common winning bid (mode)
                action_counts = affiliate_wins['action'].value_counts()
                most_common_action = int(action_counts.index[0])

                self.affiliate_anchors[affiliate_id] = most_common_action

                # Print details for verification
                if len(self.affiliate_anchors) <= 5:
                    affiliate_name = self.partner_id_to_stats.get(affiliate_id, {}).get('partner_id', affiliate_id)
                    print(f"   Affiliate {affiliate_id}: anchor={most_common_action}% "
                          f"({action_counts.iloc[0]}/{len(affiliate_wins)} wins)")

    def apply_strategy(self, strategy_type: str = 'bid_lower'):
        """
        Apply bid lower strategy - bid BELOW the actual anchor based on win rate tier
        Creates the scoring table structure for all known contexts
        """
        print("\n" + "="*60)
        print("APPLYING BID LOWER STRATEGY & INITIALIZING TABLE")
        print("="*60)
        
        # Group contexts by partner
        contexts_by_partner = defaultdict(list)
        for context_id, context_info in self.contexts.items():
            contexts_by_partner[context_info['affiliate']].append(context_id)
        
        total_scores_initialized = 0
        total_actions_created = 0
        
        # Apply strategy per partner
        for partner, context_ids in contexts_by_partner.items():
            partner_win_rate = self.partner_stats.get(partner, {}).get('win_rate', 0.007)

            # Determine strategy tier based on win rate using configurable tiers
            tier = None
            action_offsets = None
            tier_names = ["LOW", "MEDIUM", "HIGH"]
            for i, tier_name in enumerate(tier_names):
                if tier_name in self.strategy_tiers:
                    tier_config = self.strategy_tiers[tier_name]
                    # Use <= for the last tier to ensure it always matches
                    is_last_tier = (i == len(tier_names) - 1)
                    threshold_check = (partner_win_rate <= tier_config["threshold"]) if is_last_tier else (partner_win_rate < tier_config["threshold"])

                    if threshold_check:
                        tier = tier_name
                        action_offsets = tier_config["offsets"]
                        break

            # Fallback: if no tier matched (shouldn't happen with proper config), use HIGH tier
            if action_offsets is None:
                tier = "HIGH"
                if "HIGH" in self.strategy_tiers:
                    action_offsets = self.strategy_tiers["HIGH"]["offsets"]
                else:
                    action_offsets = [-8, -6, -4]  # Default HIGH tier offsets
                print(f"⚠️  Warning: Partner {partner} win rate {partner_win_rate:.3%} didn't match any tier, using {tier}")

            if len(context_ids) <= 10 or partner in list(contexts_by_partner.keys())[:3]:
                print(f"\n📊 {partner[:30]:30s}: WR={partner_win_rate:.3%} → {tier} tier")
                print(f"   Contexts: {len(context_ids)}, Offsets from anchor: {action_offsets}")

            contexts_with_actions = 0
            actions_below_min_count = 0

            for context_id in context_ids:
                context = self.contexts[context_id]

                if context.get('actions') is not None:
                    continue

                # USE THE ACTUAL ANCHOR FROM L3
                anchor = context['anchor']

                # Generate actions BELOW the anchor
                actions = []

                for offset in action_offsets:
                    action = anchor + offset
                    if action < self.MIN_ACTION:
                        actions_below_min_count += 1
                        actions.append(self.MIN_ACTION)  # Cap at minimum
                    elif action <= self.MAX_ACTION:
                        actions.append(action)
                
                # Remove duplicates and sort
                actions = sorted(list(set(actions)))
                
                # Store actions and strategy type
                context['actions'] = actions
                context['strategy_type'] = tier
                contexts_with_actions += 1
                total_actions_created += len(actions)
                
                # Initialize the scoring table
                self._initialize_action_scores(context_id, actions, anchor, partner_win_rate)
                total_scores_initialized += len(actions)
            
            if contexts_with_actions > 0 and (len(context_ids) <= 10 or partner in list(contexts_by_partner.keys())[:3]):
                print(f"   ✓ Applied to {contexts_with_actions} contexts")
                if actions_below_min_count > 0:
                    print(f"   ⚠️  {actions_below_min_count} actions capped at {self.MIN_ACTION}%")
        
        print(f"\n✅ Strategy applied to all {len(self.contexts)} contexts")
        print(f"✅ Total actions created: {total_actions_created}")
        print(f"✅ Total scores initialized: {total_scores_initialized}")
        
        self._display_comprehensive_summary()
    
    def _initialize_action_scores(self, 
                                  context_id: int, 
                                  actions: List[int], 
                                  anchor: int,
                                  partner_win_rate: float):
        """
        Initialize scores for each action in the scoring table
        Uses optimistic initialization where lower bids start with higher scores
        """
        sorted_actions = sorted(actions)
        
        for action in actions:
            # Create the table structure
            self.table[context_id][action] = {
                'daily_buckets': deque(maxlen=self.window_days),
                'score': 0,
                'total_opportunity': 0,
                'total_profit': 0,
                'total_wins': 0,
                'total_attempts': 0
            }
            
            # Optimistic initialization - lower bids get higher initial scores
            position = sorted_actions.index(action)
            n_actions = len(sorted_actions)
            
            # Linear decay from most aggressive to most conservative
            if n_actions > 1:
                decay_factor = 2.0 - (1.5 * position / (n_actions - 1))
            else:
                decay_factor = 1.0
            
            # Base score = expected profit rate
            margin = (100 - action) / 100
            base_score = partner_win_rate * margin
            
            # Apply decay and scale
            initial_score = base_score * decay_factor * self.score_scale
            
            self.table[context_id][action]['score'] = initial_score
    
    def _display_comprehensive_summary(self):
        """
        Display comprehensive summary of the generated context table
        """
        # Anchor distribution
        anchor_values = [c['anchor'] for c in self.contexts.values()]
        
        print("\n" + "="*60)
        print("GENERATION SUMMARY")
        print("="*60)
        
        print(f"\n📊 Anchor Distribution (from L3 analysis):")
        print(f"  Range: {min(anchor_values)}% - {max(anchor_values)}%")
        print(f"  Mean:  {np.mean(anchor_values):.1f}%")
        print(f"  Median: {np.median(anchor_values):.1f}%")
        
        # Most common anchors
        anchor_counts = Counter(anchor_values)
        print(f"  Most common anchors:")
        for anchor, count in sorted(anchor_counts.most_common(5)):
            print(f"    {anchor}%: {count} contexts ({count/len(anchor_values)*100:.1f}%)")
        
        # Action distribution
        all_actions = []
        for context in self.contexts.values():
            if context['actions']:
                all_actions.extend(context['actions'])
        
        if all_actions:
            print(f"\n📊 Action Distribution (bidding below anchors):")
            print(f"  Total actions: {len(all_actions)}")
            print(f"  Unique actions: {len(set(all_actions))}")
            print(f"  Range: {min(all_actions)}% - {max(all_actions)}%")
            print(f"  Mean:  {np.mean(all_actions):.1f}%")
            print(f"  Median: {np.median(all_actions):.1f}%")
        
        # Strategy distribution
        strategy_counts = defaultdict(int)
        for context in self.contexts.values():
            if context.get('strategy_type'):
                strategy_counts[context['strategy_type']] += 1
        
        print("\n📊 Strategy Tier Distribution:")
        total = sum(strategy_counts.values())
        for tier in ['LOW', 'MEDIUM', 'HIGH']:
            count = strategy_counts[tier]
            pct = count / total * 100 if total > 0 else 0
            print(f"  {tier:8s}: {count:4d} contexts ({pct:5.1f}%)")
        
        # Top partners
        print("\n📊 Top Partners by Win Rate:")
        sorted_partners = sorted(self.partner_stats.items(), 
                               key=lambda x: x[1]['win_rate'], 
                               reverse=True)[:5]
        for partner, stats in sorted_partners:
            print(f"  {partner[:30]:30s}: WR={stats['win_rate']:.3%}, "
                  f"Samples={stats['total_samples']:,}")
        
        # Table statistics
        total_scores = sum(len(actions) for actions in self.table.values())
        print("\n📊 Scoring Table Statistics:")
        print(f"  Total Contexts: {len(self.contexts)}")
        print(f"  Total Action Scores: {total_scores}")
        print(f"  Avg Actions per Context: {total_scores/len(self.contexts):.1f}")
        print(f"  Action Bounds: {self.MIN_ACTION}% - {self.MAX_ACTION}%")
    
    def save_state(self, filepath: str = "context_table_bid_lower.json"):
        """
        Save the generated context table to JSON
        Compatible with ContextTableUpdater for runtime loading
        """
        # Convert context IDs to strings for JSON
        contexts_serializable = {str(k): v for k, v in self.contexts.items()}
        
        # Save the scoring table
        table_serializable = {}
        for context_id, actions in self.table.items():
            table_serializable[str(context_id)] = {}
            for action, metrics in actions.items():
                buckets_list = list(metrics['daily_buckets'])
                # Convert any dates to strings (shouldn't be any at init)
                for bucket in buckets_list:
                    if 'date' in bucket and hasattr(bucket['date'], 'isoformat'):
                        bucket['date'] = bucket['date'].isoformat()
                
                table_serializable[str(context_id)][str(action)] = {
                    'daily_buckets': buckets_list,
                    'score': float(metrics['score']),
                    'total_opportunity': float(metrics['total_opportunity']),
                    'total_profit': float(metrics['total_profit']),
                    'total_wins': int(metrics['total_wins']),
                    'total_attempts': int(metrics['total_attempts'])
                }
        
        # Serialize partner stats
        partner_stats_serializable = {}
        for partner, stats in self.partner_stats.items():
            partner_stats_serializable[partner] = {
                'partner_id': int(stats.get('partner_id', 99999)),
                'total_samples': int(stats['total_samples']),
                'total_wins': int(stats['total_wins']),
                'win_rate': float(stats['win_rate']),
                'avg_revenue': float(stats.get('avg_revenue', 50))
            }
        
        # Serialize affiliate anchors
        affiliate_anchors_serializable = {
            str(k): int(v) for k, v in self.affiliate_anchors.items()
        }

        # Create state with all required fields for updater compatibility
        state = {
            'contexts': contexts_serializable,
            'partner_stats': partner_stats_serializable,
            'affiliate_anchors': affiliate_anchors_serializable,
            'table': table_serializable,
            'unknown_contexts_handled': 0,  # Updater will track this
            'new_contexts_seen': [],  # Updater will populate this
            'config': {
                'window_days': self.window_days,
                'exploration_rate': self.exploration_rate,
                'prior_weight_ratio': self.prior_weight_ratio,
                'score_scale': self.score_scale,
                'min_wins_for_affiliate_anchor': 10  # For compatibility
            },
            'generation_metadata': {
                'strategy': 'bid_lower',
                'min_action': self.MIN_ACTION,
                'max_action': self.MAX_ACTION,
                'generated_at': datetime.now().isoformat(),
                'total_contexts': len(self.contexts),
                'total_scores': sum(len(actions) for actions in table_serializable.values())
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        total_scores = sum(len(actions) for actions in table_serializable.values())
        print(f"\n✅ Context table saved to {filepath}")
        print(f"   - Contexts: {len(self.contexts)}")
        print(f"   - Action scores: {total_scores}")
        print(f"   - Partners: {len(self.partner_stats)}")
        print(f"   - File size: ~{len(json.dumps(state)) / 1024:.1f} KB")
    
    def get_context_performance(self, context_id: int) -> Dict:
        """
        Get initial performance metrics for a context (all zeros at generation)
        """
        if context_id not in self.contexts:
            return {}
        
        context = self.contexts[context_id]
        performance = {
            'anchor': context['anchor'],
            'affiliate': context['affiliate'],
            'affiliate_id': context.get('affiliate_id', 99999),
            'strategy_type': context['strategy_type'],
            'actions': {}
        }
        
        for action in context['actions']:
            if action in self.table[context_id]:
                metrics = self.table[context_id][action]
                performance['actions'][action] = {
                    'score': metrics['score'],
                    'attempts': metrics['total_attempts'],
                    'wins': metrics['total_wins'],
                    'total_profit': metrics['total_profit'],
                    'total_opportunity': metrics['total_opportunity']
                }
        
        if performance['actions']:
            best_action = max(performance['actions'], 
                            key=lambda a: performance['actions'][a]['score'])
            performance['best_action'] = best_action
        
        return performance
    
    def get_stats_summary(self) -> Dict:
        """
        Get summary statistics of the generated table
        """
        total_scores = sum(len(actions) for actions in self.table.values())
        
        strategy_counts = defaultdict(int)
        for context in self.contexts.values():
            if context.get('strategy_type'):
                strategy_counts[context['strategy_type']] += 1
        
        return {
            'total_contexts': len(self.contexts),
            'total_partners': len(self.partner_stats),
            'total_action_scores': total_scores,
            'strategy_distribution': dict(strategy_counts),
            'avg_actions_per_context': total_scores / len(self.contexts) if self.contexts else 0
        }


def initialize_context_table_bid_lower(anchors_l3: Dict,
                                       df_with_contexts: pd.DataFrame,
                                       output_path: str = "context_table_bid_lower.json",
                                       min_action: int = 40,
                                       max_action: int = 97,
                                       strategy_tiers: dict = None) -> ContextTableBidLowerStrategy:
    """
    Complete initialization pipeline for Context Table with Bid Lower Strategy

    Args:
        anchors_l3: Anchors from L3 analysis - uses actual anchor values
        df_with_contexts: Historical data with context columns
        output_path: Where to save the generated table
        min_action: Lower bound for actions (default: 40)
        max_action: Upper bound for actions (default: 97)
        strategy_tiers: Dict defining win rate thresholds and action offsets

    Returns:
        Initialized ContextTableBidLowerStrategy instance
    """
    print("\n" + "="*60)
    print("CONTEXT TABLE GENERATION PIPELINE")
    print("="*60)

    # Create generator
    generator = ContextTableBidLowerStrategy(
        window_days=15,
        exploration_rate=0.15,
        prior_weight_ratio=0.1,
        score_scale=1000.0,
        min_action=min_action,
        max_action=max_action,
        strategy_tiers=strategy_tiers
    )
    
    # Initialize with L3 anchors
    generator.initialize_with_anchors(anchors_l3, df_with_contexts)
    
    # Apply bid lower strategy
    generator.apply_strategy(strategy_type='bid_lower')
    
    # Save to file
    generator.save_state(output_path)
    
    # Verify generation
    stats = generator.get_stats_summary()
    print(f"\n✅ Context table ready for deployment!")
    print(f"   Output: {output_path}")
    print(f"   Contexts: {stats['total_contexts']}")
    print(f"   Action scores: {stats['total_action_scores']}")
    
    if stats['total_action_scores'] == 0:
        print("\n❌ WARNING: No scores initialized! Check the strategy application.")
    
    return generator


