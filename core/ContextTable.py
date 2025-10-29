"""
Complete Production-Ready ContextTable Implementation
CRITICAL: Properly saves and loads scores for production use
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import json
from datetime import datetime, timedelta, date


class ContextTable:
    """
    Production-ready ContextTable with proper score initialization, saving, and loading
    """
    
    def __init__(self, 
                 window_days: int = 15,
                 exploration_rate: float = 0.15,
                 prior_weight_ratio: float = 0.1,
                 score_scale: float = 1000.0,
                 min_wins_for_affiliate_anchor: int = 10):
        """
        Initialize the ContextTable
        
        Args:
            window_days: Rolling window for score calculation
            exploration_rate: Epsilon for epsilon-greedy exploration
            prior_weight_ratio: Prior weight as ratio of recent data (0.1 = 10%)
            score_scale: Multiplier for scores (1000 = show 0.000979 as 0.979)
            min_wins_for_affiliate_anchor: Minimum wins to create affiliate-level anchor
        """
        self.window_days = window_days
        self.exploration_rate = exploration_rate
        self.prior_weight_ratio = prior_weight_ratio
        self.score_scale = score_scale
        self.min_wins_for_affiliate_anchor = min_wins_for_affiliate_anchor
        
        # Main data structures
        self.contexts = {}  # context_id -> context info
        self.table = defaultdict(lambda: defaultdict(dict))  # context -> action -> metrics WITH SCORES
        self.partner_stats = {}  # partner -> global stats
        self.partner_id_to_stats = {}  # affiliate_integration_id -> stats
        
        # Affiliate anchor mapping for unknown contexts
        self.affiliate_anchors = {}  # affiliate_id -> anchor_bid
        self.context_to_affiliate = {}  # context_id -> affiliate_id mapping
        
        # Track initialization state
        self.initialized = False
        self.strategy_applied = False
        self.unknown_contexts_handled = 0
        self.new_affiliates_seen = set()
        
    def initialize_with_anchors(self, anchors_l3: Dict, df_with_contexts: pd.DataFrame):
        """
        Initialize the table with L3 anchors and build affiliate anchor mapping
        
        Args:
            anchors_l3: Dictionary from anchor analysis {context_id: anchor_info}
            df_with_contexts: DataFrame with context columns and historical data (REQUIRED)
        """
        if df_with_contexts is None:
            raise ValueError("df_with_contexts is required for initialization!")
            
        print("="*60)
        print("INITIALIZING CONTEXT TABLE WITH ANCHORS")
        print("="*60)
        
        # Store anchors with both affiliate name and ID
        for context_id, anchor_info in anchors_l3.items():
            affiliate_id = anchor_info.get('affiliate_id', 99999)
            
            self.contexts[context_id] = {
                'anchor': anchor_info['anchor_bid'],
                'affiliate': anchor_info['affiliate'],
                'affiliate_id': affiliate_id,
                'fallback_level': anchor_info['level'],
                'win_rate': anchor_info.get('win_rate', 0),
                'total_wins': anchor_info.get('total_wins', 0),
                'total_samples': anchor_info.get('total_samples', 0),
                'actions': None,
                'strategy_type': None
            }
            
            # Build context to affiliate mapping
            self.context_to_affiliate[context_id] = affiliate_id
        
        print(f"\n✅ Initialized {len(self.contexts)} contexts")
        
        # Calculate partner-level stats
        self._calculate_partner_stats(df_with_contexts)
        print(f"📊 Calculated stats for {len(self.partner_stats)} partners")
        
        # Build affiliate anchor mapping from configuration wins
        self._build_affiliate_anchors(df_with_contexts)
        print(f"🎯 Built affiliate anchors for {len(self.affiliate_anchors)} affiliates")
        
        # Print summary
        self._print_initialization_summary()
        
        self.initialized = True
        
    def _build_affiliate_anchors(self, df: pd.DataFrame):
        """
        Build affiliate-level anchors using configuration wins
        """
        print("\n🔨 Building Affiliate-Level Anchors...")
        
        # Filter to configuration wins only
        config_df = df[df['bid_modification_variant_id'] == 'configuration'].copy()
        config_wins = config_df[config_df['won'] == 1].copy()
        
        if len(config_wins) == 0:
            print("⚠️ No configuration wins found for affiliate anchor calculation")
            return
        
        # Ensure action is integer
        config_wins['action'] = config_wins['final_modified'].astype(int)
        
        # Group by affiliate_integration_id
        for affiliate_id in df['affiliate_integration_id'].unique():
            affiliate_id = int(affiliate_id)
            
            # Get wins for this affiliate
            affiliate_wins = config_wins[config_wins['affiliate_integration_id'] == affiliate_id]
            
            if len(affiliate_wins) >= self.min_wins_for_affiliate_anchor:
                # Find most common winning bid (mode)
                action_counts = affiliate_wins['action'].value_counts()
                most_common_action = int(action_counts.index[0])
                
                self.affiliate_anchors[affiliate_id] = most_common_action
                
                # Debug info for top affiliates
                if len(self.affiliate_anchors) <= 5:
                    wins_count = len(affiliate_wins)
                    total_samples = len(df[df['affiliate_integration_id'] == affiliate_id])
                    win_rate = wins_count / total_samples if total_samples > 0 else 0
                    print(f"  Affiliate {affiliate_id}: anchor={most_common_action}%, "
                          f"wins={wins_count}, WR={win_rate:.3%}")
            else:
                # Not enough wins - calculate from win rate
                affiliate_data = df[df['affiliate_integration_id'] == affiliate_id]
                if len(affiliate_data) > 0:
                    win_rate = affiliate_data['won'].mean()
                    
                    # Simple heuristic based on win rate
                    if win_rate < 0.005:
                        anchor = 80
                    elif win_rate < 0.01:
                        anchor = 75
                    elif win_rate < 0.02:
                        anchor = 70
                    elif win_rate < 0.05:
                        anchor = 65
                    else:
                        anchor = 70
                    
                    self.affiliate_anchors[affiliate_id] = anchor
        
        print(f"  ✓ Created anchors for {len(self.affiliate_anchors)} affiliates")
        
        # Show distribution of anchors
        if self.affiliate_anchors:
            anchor_values = list(self.affiliate_anchors.values())
            print(f"  ✓ Anchor distribution: min={min(anchor_values)}%, "
                  f"median={int(np.median(anchor_values))}%, "
                  f"max={max(anchor_values)}%")
    
    def _calculate_partner_stats(self, df: pd.DataFrame):
        """
        Calculate partner-level win rates for strategy application
        """
        # Build a mapping of ID to display name from contexts
        id_to_name = {}
        for ctx_info in self.contexts.values():
            if 'affiliate_id' in ctx_info and ctx_info['affiliate_id'] != 99999:
                id_to_name[ctx_info['affiliate_id']] = ctx_info['affiliate']
        
        # Use affiliate_integration_id 
        for partner_id in df['affiliate_integration_id'].unique():
            partner_id = int(partner_id)
            partner_data = df[df['affiliate_integration_id'] == partner_id]
            
            total_samples = len(partner_data)
            total_wins = partner_data['won'].sum()
            win_rate = total_wins / total_samples if total_samples > 0 else 0
            
            # Calculate average revenue if available
            avg_revenue = partner_data['expected_revenue'].mean() if 'expected_revenue' in partner_data else 50
            
            # Get display name from mapping or create default
            display_name = id_to_name.get(partner_id, f"Affiliate_{partner_id}")
            
            # Store by display name for consistency with anchors
            self.partner_stats[display_name] = {
                'partner_id': partner_id,
                'total_samples': int(total_samples),
                'total_wins': int(total_wins),
                'win_rate': float(win_rate),
                'avg_revenue': float(avg_revenue)
            }
            
            # Also store by ID for quick lookup
            self.partner_id_to_stats[partner_id] = self.partner_stats[display_name]
    
    def apply_strategy(self, strategy_type: str = 'simple'):
        """
        Apply bidding strategy to generate actions for each context
        CRITICAL: This initializes all action scores!
        """
        if not self.initialized:
            raise ValueError("Must initialize with anchors first!")
        
        print("\n" + "="*60)
        print("APPLYING BIDDING STRATEGY & INITIALIZING SCORES")
        print("="*60)
        
        # Group contexts by partner
        contexts_by_partner = defaultdict(list)
        for context_id, context_info in self.contexts.items():
            contexts_by_partner[context_info['affiliate']].append(context_id)
        
        total_scores_initialized = 0
        
        # Apply strategy per partner
        for partner, context_ids in contexts_by_partner.items():
            partner_win_rate = self.partner_stats.get(partner, {}).get('win_rate', 0.007)
            
            # Determine strategy tier based on win rate
            if partner_win_rate < 0.01:
                tier = "LOW"
                base_actions_template = [-5, -3, -2]  # Updated
            elif partner_win_rate < 0.50:
                tier = "MEDIUM"
                base_actions_template = [-6, -4, -3]  # Updated
            else:
                tier = "HIGH"
                base_actions_template = [-8, -6, -4]  # Updated
            
            print(f"\n📊 {partner[:30]:30s}: WR={partner_win_rate:.3%} → {tier} tier")
            print(f"   Contexts: {len(context_ids)}")
            
            # Apply to each context
            contexts_with_actions = 0
            for context_id in context_ids:
                context = self.contexts[context_id]
                
                # Skip if already has actions
                if context.get('actions') is not None:
                    continue
                    
                anchor = context['anchor']
                
                # Generate actions from template
                actions = []
                for offset in base_actions_template:
                    action = anchor + offset
                    if 50 <= action <= 97:
                        actions.append(action)
                
                # Ensure we have at least 3 actions
                if len(actions) < 3:
                    actions = [max(50, anchor-5), anchor, min(97, anchor+5)]
                
                actions = sorted(list(set(actions)))
                
                # Store actions
                context['actions'] = actions
                context['strategy_type'] = tier
                contexts_with_actions += 1
                
                # CRITICAL: Initialize the scoring table for each action
                self._initialize_action_scores(context_id, actions, anchor, partner_win_rate)
                total_scores_initialized += len(actions)
            
            if contexts_with_actions > 0:
                print(f"   ✓ Applied to {contexts_with_actions} contexts, {total_scores_initialized} scores initialized")
        
        print(f"\n✅ Strategy applied to all contexts")
        print(f"✅ Total scores initialized: {total_scores_initialized}")
        
        self.strategy_applied = True
        self._print_strategy_summary()
    
    def _initialize_action_scores(self, 
                                  context_id: int, 
                                  actions: List[int], 
                                  anchor: int,
                                  partner_win_rate: float):
        """
        Initialize scores for each action in a context
        CRITICAL: This creates the initial scores that get saved!
        """
        sorted_actions = sorted(actions)
        
        for action in actions:
            # Initialize the data structure
            self.table[context_id][action] = {
                'daily_buckets': deque(maxlen=self.window_days),
                'score': 0,
                'total_opportunity': 0,
                'total_profit': 0,
                'total_wins': 0,
                'total_attempts': 0
            }
            
            # OPTIMISTIC INITIALIZATION: Lower bids start higher
            position = sorted_actions.index(action)
            n_actions = len(sorted_actions)
            
            # Linear decay from aggressive to conservative
            if n_actions > 1:
                decay_factor = 2.0 - (1.5 * position / (n_actions - 1))
            else:
                decay_factor = 1.0
            
            # Base score = expected profit rate
            margin = (100 - action) / 100
            base_score = partner_win_rate * margin
            
            # Apply decay and scale for readability
            initial_score = base_score * decay_factor * self.score_scale
            
            self.table[context_id][action]['score'] = initial_score
    
    def _extract_affiliate_from_context(self, context_id: int) -> int:
        """
        Extract affiliate_id from context_id
        Context format: [affiliate_id][7-digit base_context]
        """
        return context_id // 10_000_000
    
    def _initialize_unknown_context(self, context_id: int) -> bool:
        """
        Initialize a new context on-the-fly using affiliate anchors
        """
        # Extract affiliate ID from context
        affiliate_id = self._extract_affiliate_from_context(context_id)
        
        # Validate extraction
        if affiliate_id > 99999:
            print(f"⚠️ Invalid affiliate_id {affiliate_id} extracted from context {context_id}")
            return False
        
        # Get anchor for this affiliate
        if affiliate_id in self.affiliate_anchors:
            anchor = self.affiliate_anchors[affiliate_id]
            source = "affiliate_anchor"
        elif affiliate_id in self.partner_id_to_stats:
            # Fallback: calculate from win rate
            win_rate = self.partner_id_to_stats[affiliate_id]['win_rate']
            if win_rate < 0.01:
                anchor = 70
            elif win_rate < 0.50:
                anchor = 65
            else:
                anchor = 70
            source = "win_rate_based"
        else:
            # Unknown affiliate - track it
            if affiliate_id not in self.new_affiliates_seen:
                self.new_affiliates_seen.add(affiliate_id)
                print(f"🆕 NEW AFFILIATE {affiliate_id} detected (not in training data)")
            
            anchor = 70
            source = "new_affiliate_default"
        
        # Get affiliate name if available
        affiliate_name = f"Affiliate_{affiliate_id}"
        for partner_name, stats in self.partner_stats.items():
            if stats.get('partner_id') == affiliate_id:
                affiliate_name = partner_name
                break
        
        # Create context entry
        self.contexts[context_id] = {
            'anchor': anchor,
            'affiliate': affiliate_name,
            'affiliate_id': affiliate_id,
            'fallback_level': 'dynamic',
            'initialized_at': datetime.now().isoformat(),
            'source': source,
            'actions': None,
            'strategy_type': None
        }
        
        # Store mapping
        self.context_to_affiliate[context_id] = affiliate_id
        
        # Generate actions based on affiliate win rate
        partner_win_rate = self.partner_id_to_stats.get(affiliate_id, {}).get('win_rate', 0.007)
        
        # Determine strategy tier
        if partner_win_rate < 0.01:
            strategy_type = "LOW"
            action_offsets = [-4, -3, -2]
        elif partner_win_rate < 0.50:
            strategy_type = "MEDIUM"
            action_offsets = [-6, -4, -3]
        else:
            strategy_type = "HIGH"
            action_offsets = [-8, -6, -4]
        
        # Generate actions
        actions = []
        for offset in action_offsets:
            action = anchor + offset
            if 50 <= action <= 97:
                actions.append(action)
        
        if len(actions) < 3:
            actions = [max(50, anchor-5), anchor, min(97, anchor+5)]
        
        actions = sorted(list(set(actions)))
        
        # Store in context
        self.contexts[context_id]['actions'] = actions
        self.contexts[context_id]['strategy_type'] = strategy_type
        
        # CRITICAL: Initialize scoring table for each action
        self._initialize_action_scores(context_id, actions, anchor, partner_win_rate)
        
        # Track this initialization
        self.unknown_contexts_handled += 1
        
        if self.unknown_contexts_handled <= 10 or self.unknown_contexts_handled % 100 == 0:
            print(f"🆕 Initialized unknown context {context_id}: "
                  f"Affiliate {affiliate_id} ({affiliate_name}), "
                  f"Anchor={anchor}% ({source}), "
                  f"Strategy={strategy_type}, "
                  f"Actions={actions}")
        
        return True
    
    def select_action(self, context_id: int) -> int:
        """
        Select an action for a context using epsilon-greedy
        Handles unknown contexts by initializing them on-the-fly
        """
        # Handle unknown context
        if context_id not in self.contexts:
            # Try to initialize it
            if self._initialize_unknown_context(context_id):
                # Successfully initialized
                pass
            else:
                # Failed to initialize
                print(f"⚠️ Failed to initialize context {context_id}, using default 85%")
                return 85
        
        # Get available actions
        actions = self.contexts[context_id]['actions']
        
        if not actions:
            return self.contexts[context_id]['anchor']
        
        # Epsilon-greedy selection
        if np.random.random() < self.exploration_rate:
            # Explore: random action
            return np.random.choice(actions)
        else:
            # Exploit: best scoring action
            scores = {action: self.table[context_id][action]['score'] 
                     for action in actions}
            return max(scores, key=scores.get)
    
    def update(self, 
               context_id: int, 
               action: int, 
               expected_revenue: float,
               won: bool,
               timestamp: Optional[datetime] = None):
        """
        Update the table with a new result using 15-day rolling window
        CRITICAL: This is how the model learns!
        """
        if context_id not in self.contexts:
            return  # Skip unknown contexts
        
        if action not in self.table[context_id]:
            return  # Skip unknown actions
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate profit
        profit = expected_revenue * (100 - action) / 100 if won else 0
        
        # Get today's date
        today = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        
        # Find or create today's bucket
        buckets = self.table[context_id][action]['daily_buckets']
        
        # Check if we need to create a new bucket
        if not buckets or buckets[-1]['date'] != today:
            # Create new daily bucket
            new_bucket = {
                'date': today,
                'opportunity': 0,
                'profit': 0,
                'wins': 0,
                'attempts': 0
            }
            buckets.append(new_bucket)
        
        # Update today's bucket
        buckets[-1]['opportunity'] += expected_revenue
        buckets[-1]['profit'] += profit
        buckets[-1]['wins'] += 1 if won else 0
        buckets[-1]['attempts'] += 1
        
        # Recalculate score
        self._recalculate_score(context_id, action)
    
    def _recalculate_score(self, context_id: int, action: int):
        """
        Recalculate score using 15-day rolling window
        """
        buckets = self.table[context_id][action]['daily_buckets']
        
        # Sum across all buckets
        total_opportunity = sum(b['opportunity'] for b in buckets)
        total_profit = sum(b['profit'] for b in buckets)
        
        # Store totals
        self.table[context_id][action]['total_opportunity'] = total_opportunity
        self.table[context_id][action]['total_profit'] = total_profit
        self.table[context_id][action]['total_wins'] = sum(b['wins'] for b in buckets)
        self.table[context_id][action]['total_attempts'] = sum(b['attempts'] for b in buckets)
        
        # Calculate adaptive prior
        prior_opportunity = max(1000, total_opportunity * self.prior_weight_ratio)
        
        # Prior assumes baseline win rate
        affiliate_id = self.contexts[context_id].get('affiliate_id', 99999)
        baseline_win_rate = self.partner_id_to_stats.get(affiliate_id, {}).get('win_rate', 0.007)
        prior_profit = prior_opportunity * baseline_win_rate * (100 - action) / 100
        
        # Calculate score (with scaling)
        if total_opportunity + prior_opportunity > 0:
            score = (total_profit + prior_profit) / (total_opportunity + prior_opportunity)
            score = score * self.score_scale
        else:
            margin = (100 - action) / 100
            score = baseline_win_rate * margin * self.score_scale
        
        self.table[context_id][action]['score'] = score
    
    def get_context_performance(self, context_id: int) -> Dict:
        """
        Get current performance metrics for a context
        """
        if context_id not in self.contexts:
            return {}
        
        context = self.contexts[context_id]
        performance = {
            'anchor': context['anchor'],
            'affiliate': context['affiliate'],
            'affiliate_id': context.get('affiliate_id', 99999),
            'strategy_type': context['strategy_type'],
            'source': context.get('source', 'initial'),
            'actions': {}
        }
        
        for action in context['actions']:
            if action in self.table[context_id]:
                metrics = self.table[context_id][action]
                
                win_rate = metrics['total_wins'] / metrics['total_attempts'] if metrics['total_attempts'] > 0 else 0
                
                performance['actions'][action] = {
                    'score': metrics['score'],
                    'attempts': metrics['total_attempts'],
                    'wins': metrics['total_wins'],
                    'win_rate': win_rate,
                    'total_profit': metrics['total_profit'],
                    'total_opportunity': metrics['total_opportunity']
                }
        
        # Identify best action
        if performance['actions']:
            best_action = max(performance['actions'], 
                            key=lambda a: performance['actions'][a]['score'])
            performance['best_action'] = best_action
        
        return performance
    
    def save_state(self, filepath: str = "context_table_state.json"):
        """
        Save the current state to file INCLUDING ALL SCORES
        CRITICAL: This must save the scoring table!
        """
        # Convert context IDs to strings for JSON
        contexts_serializable = {str(k): v for k, v in self.contexts.items()}
        
        # CRITICAL: Save the scoring table!
        table_serializable = {}
        for context_id, actions in self.table.items():
            table_serializable[str(context_id)] = {}
            for action, metrics in actions.items():
                # Convert deque to list for JSON
                buckets_list = list(metrics['daily_buckets'])
                # Convert dates to strings
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
        
        # Convert partner stats
        partner_stats_serializable = {}
        for partner, stats in self.partner_stats.items():
            partner_stats_serializable[partner] = {
                'partner_id': int(stats.get('partner_id', 99999)) if 'partner_id' in stats else None,
                'total_samples': int(stats['total_samples']),
                'total_wins': int(stats['total_wins']),
                'win_rate': float(stats['win_rate']),
                'avg_revenue': float(stats.get('avg_revenue', 50))
            }
        
        # Convert affiliate anchors
        affiliate_anchors_serializable = {str(k): int(v) for k, v in self.affiliate_anchors.items()}
        
        # Create full state
        state = {
            'contexts': contexts_serializable,
            'partner_stats': partner_stats_serializable,
            'affiliate_anchors': affiliate_anchors_serializable,
            'table': table_serializable,  # CRITICAL: Save the scores!
            'unknown_contexts_handled': self.unknown_contexts_handled,
            'new_affiliates_seen': list(self.new_affiliates_seen),
            'config': {
                'window_days': self.window_days,
                'exploration_rate': self.exploration_rate,
                'prior_weight_ratio': self.prior_weight_ratio,
                'score_scale': self.score_scale,
                'min_wins_for_affiliate_anchor': self.min_wins_for_affiliate_anchor
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        total_scores = sum(len(actions) for actions in table_serializable.values())
        print(f"✅ State saved to {filepath}")
        print(f"   - Contexts: {len(self.contexts)}")
        print(f"   - Scores saved: {total_scores} action scores")
        print(f"   - Affiliate anchors: {len(self.affiliate_anchors)}")
    
    def load_state(self, filepath: str = "context_table_state.json"):
        """
        Load previously saved state from file INCLUDING ALL SCORES
        CRITICAL: This must load the scoring table!
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore contexts
        self.contexts = {int(k): v for k, v in state['contexts'].items()}
        
        # Restore partner stats
        self.partner_stats = state['partner_stats']
        
        # Restore affiliate anchors
        self.affiliate_anchors = {int(k): v for k, v in state['affiliate_anchors'].items()}
        
        # Restore counters
        self.unknown_contexts_handled = state.get('unknown_contexts_handled', 0)
        self.new_affiliates_seen = set(state.get('new_affiliates_seen', []))
        
        # Rebuild helper mappings
        self.partner_id_to_stats = {}
        for partner_name, stats in self.partner_stats.items():
            if stats.get('partner_id') is not None:
                self.partner_id_to_stats[stats['partner_id']] = stats
        
        self.context_to_affiliate = {}
        for context_id, context_info in self.contexts.items():
            if 'affiliate_id' in context_info:
                self.context_to_affiliate[context_id] = context_info['affiliate_id']
        
        # CRITICAL: Load the scoring table!
        if 'table' in state:
            total_scores_loaded = 0
            
            for context_id_str, actions in state['table'].items():
                context_id = int(context_id_str)
                self.table[context_id] = {}
                
                for action_str, metrics in actions.items():
                    action = int(action_str)
                    
                    # Rebuild deque from list
                    buckets_list = metrics['daily_buckets']
                    # Convert date strings back to date objects
                    for bucket in buckets_list:
                        if 'date' in bucket and isinstance(bucket['date'], str):
                            # Handle both date and datetime strings
                            try:
                                bucket['date'] = datetime.fromisoformat(bucket['date']).date()
                            except:
                                bucket['date'] = datetime.strptime(bucket['date'], '%Y-%m-%d').date()
                    
                    # Create deque with proper maxlen
                    daily_buckets = deque(buckets_list, maxlen=self.window_days)
                    
                    self.table[context_id][action] = {
                        'daily_buckets': daily_buckets,
                        'score': metrics['score'],
                        'total_opportunity': metrics['total_opportunity'],
                        'total_profit': metrics['total_profit'],
                        'total_wins': metrics['total_wins'],
                        'total_attempts': metrics['total_attempts']
                    }
                    total_scores_loaded += 1
            
            print(f"   - Loaded {total_scores_loaded} action scores")
        else:
            print("⚠️ WARNING: No scoring table found in file!")
            print("   Reinitializing all scores...")
            self._reinitialize_all_scores()
        
        # Restore config
        config = state['config']
        self.window_days = config['window_days']
        self.exploration_rate = config['exploration_rate']
        self.prior_weight_ratio = config['prior_weight_ratio']
        self.score_scale = config.get('score_scale', 1000.0)
        self.min_wins_for_affiliate_anchor = config.get('min_wins_for_affiliate_anchor', 10)
        
        self.initialized = True
        self.strategy_applied = True
        
        print(f"✅ State loaded from {filepath}")
        print(f"   - Total contexts: {len(self.contexts)}")
        print(f"   - Affiliate anchors: {len(self.affiliate_anchors)}")
        print(f"   - Timestamp: {state.get('timestamp', 'unknown')}")
    
    def _reinitialize_all_scores(self):
        """
        Reinitialize all action scores if not loaded from file
        """
        print("🔄 Reinitializing all action scores...")
        
        total_reinitialized = 0
        for context_id, context_info in self.contexts.items():
            if 'actions' in context_info and context_info['actions']:
                affiliate_id = context_info.get('affiliate_id', 99999)
                partner_win_rate = self.partner_id_to_stats.get(affiliate_id, {}).get('win_rate', 0.007)
                
                self._initialize_action_scores(
                    context_id,
                    context_info['actions'],
                    context_info['anchor'],
                    partner_win_rate
                )
                total_reinitialized += len(context_info['actions'])
        
        print(f"✅ Reinitialized {total_reinitialized} action scores")
    
    def get_stats_summary(self) -> Dict:
        """
        Get summary statistics about the context table
        """
        initial_contexts = sum(1 for c in self.contexts.values() 
                              if c.get('source', 'initial') == 'initial')
        dynamic_contexts = sum(1 for c in self.contexts.values() 
                              if c.get('source', 'initial') != 'initial')
        
        total_scores = sum(len(actions) for actions in self.table.values())
        
        return {
            'total_contexts': len(self.contexts),
            'initial_contexts': initial_contexts,
            'dynamic_contexts': dynamic_contexts,
            'unknown_handled': self.unknown_contexts_handled,
            'new_affiliates_seen': len(self.new_affiliates_seen),
            'affiliates_with_anchors': len(self.affiliate_anchors),
            'total_partners': len(self.partner_stats),
            'total_action_scores': total_scores
        }
    
    def _print_initialization_summary(self):
        """
        Print summary after initialization
        """
        # Count by affiliate
        affiliate_counts = defaultdict(int)
        for context in self.contexts.values():
            affiliate_counts[context['affiliate']] += 1
        
        print("\n📊 Contexts by Partner:")
        for affiliate, count in sorted(affiliate_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            win_rate = self.partner_stats.get(affiliate, {}).get('win_rate', 0)
            partner_id = self.partner_stats.get(affiliate, {}).get('partner_id', 'N/A')
            anchor = self.affiliate_anchors.get(partner_id, 'N/A')
            print(f"  {affiliate[:30]:30s} (ID: {partner_id:>5}): "
                  f"{count:4d} contexts, WR={win_rate:.3%}, Anchor={anchor}")
    
    def _print_strategy_summary(self):
        """
        Print summary after strategy application
        """
        # Count by strategy type
        strategy_counts = defaultdict(int)
        for context in self.contexts.values():
            strategy_counts[context['strategy_type']] += 1
        
        print("\n📊 Strategy Distribution:")
        for strategy, count in strategy_counts.items():
            print(f"  {strategy:10s}: {count:4d} contexts")
        
        print(f"\n📊 Affiliate Anchor Coverage:")
        print(f"  Total affiliates: {len(self.partner_stats)}")
        print(f"  With anchors: {len(self.affiliate_anchors)}")
        print(f"  Coverage: {len(self.affiliate_anchors)/len(self.partner_stats)*100:.1f}%")
        
        # Count total scores
        total_scores = sum(len(actions) for actions in self.table.values())
        print(f"\n📊 Scoring Table:")
        print(f"  Total action scores initialized: {total_scores}")


# ============================================================================
# CRITICAL INITIALIZATION FUNCTION
# ============================================================================

def initialize_context_table_for_production(anchors_l3: Dict, 
                                           df_with_contexts: pd.DataFrame) -> ContextTable:
    """
    Complete initialization pipeline for production use
    CRITICAL: This saves everything including scores!
    
    Args:
        anchors_l3: Anchors from L3 analysis
        df_with_contexts: Historical data with context columns (REQUIRED)
        
    Returns:
        Fully initialized and ready ContextTable with scores
    """
    print("\n" + "="*60)
    print("PRODUCTION CONTEXT TABLE INITIALIZATION")
    print("="*60)
    
    # 1. Create context table
    context_table = ContextTable(
        window_days=15,
        exploration_rate=0.15,
        prior_weight_ratio=0.1,
        score_scale=1000.0,
        min_wins_for_affiliate_anchor=10
    )
    
    # 2. Initialize with anchors (builds affiliate anchors)
    context_table.initialize_with_anchors(anchors_l3, df_with_contexts)
    
    # 3. Apply strategy (THIS INITIALIZES ALL SCORES!)
    context_table.apply_strategy(strategy_type='simple')
    
    # 4. CRITICAL: Save initial state WITH SCORES
    context_table.save_state("CONTEXT_TABLE.json")
    
    # 5. Verify scores were saved
    stats = context_table.get_stats_summary()
    print("\n✅ Context table ready for production!")
    print(f"   - Initial contexts: {stats['initial_contexts']}")
    print(f"   - Action scores: {stats['total_action_scores']}")
    print(f"   - Affiliate anchors: {stats['affiliates_with_anchors']}")
    print(f"   - Exploration rate: {context_table.exploration_rate:.0%}")
    print(f"   - Window: {context_table.window_days} days")
    
    if stats['total_action_scores'] == 0:
        print("\n❌ WARNING: No scores initialized! Check the strategy application.")
    
    return context_table


# ============================================================================
# PRODUCTION USAGE EXAMPLE
# ============================================================================

def verify_saved_file(filepath: str = "CONTEXT_TABLE.json"):
    """
    Verify the saved file has everything needed for production
    """
    print("\n" + "="*60)
    print("VERIFYING SAVED CONTEXT TABLE")
    print("="*60)
    
    with open(filepath, 'r') as f:
        state = json.load(f)
    
    print(f"\n📁 File: {filepath}")
    print(f"📅 Saved: {state.get('timestamp', 'unknown')}")
    
    print(f"\n✓ Contexts: {len(state.get('contexts', {}))}")
    print(f"✓ Partner stats: {len(state.get('partner_stats', {}))}")
    print(f"✓ Affiliate anchors: {len(state.get('affiliate_anchors', {}))}")
    
    # CRITICAL: Check if scores are saved
    if 'table' in state:
        total_scores = sum(len(actions) for actions in state['table'].values())
        print(f"✓ Action scores: {total_scores}")
        
        # Show sample score
        if state['table']:
            first_context = list(state['table'].keys())[0]
            first_action = list(state['table'][first_context].keys())[0]
            sample_score = state['table'][first_context][first_action]['score']
            print(f"✓ Sample score: {sample_score:.3f}")
    else:
        print("❌ WARNING: No scoring table found!")
    
    return 'table' in state


if __name__ == "__main__":
    print("COMPLETE PRODUCTION-READY CONTEXT TABLE")
    print("=" * 60)
    print("\nKey Features:")
    print("✓ Properly initializes all action scores during apply_strategy()")
    print("✓ Saves scores in save_state() for production use")
    print("✓ Loads scores in load_state() for production")
    print("✓ Handles unknown contexts with affiliate anchors")
    print("✓ Updates scores with 15-day rolling window")
    print("\n" + "=" * 60)
    print("\nUsage:")
    print("------")
    print("# Initialize and save WITH SCORES:")
    print("context_table = initialize_context_table_for_production(anchors_l3, df_with_contexts)")
    print("")
    print("# Verify the saved file:")
    print("verify_saved_file('CONTEXT_TABLE.json')")
    print("")
    print("# Load in production:")
    print("context_table = ContextTable()")
    print("context_table.load_state('CONTEXT_TABLE.json')")
    print("")
    print("# Use for bidding:")
    print("action = context_table.select_action(context_id)")
    print("context_table.update(context_id, action, revenue, won)")