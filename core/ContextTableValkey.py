"""
Complete Production-Ready ContextTable Implementation
CRITICAL: Properly saves and loads scores for production use
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import json
import asyncio
from datetime import datetime, timedelta, date
from ValkeyGlidClusterClient import ValkeyClusterHelper
from glide import Logger,LogLevel

class ContextTableValkey:
    """
    Production-ready ContextTable with proper score initialization, saving, and loading
    """
    
    def __init__(self, valkey_helper: ValkeyClusterHelper):
        """
        Initialize the ContextTable
        
        Args:
            window_days: Rolling window for score calculation
            exploration_rate: Epsilon for epsilon-greedy exploration
            prior_weight_ratio: Prior weight as ratio of recent data (0.1 = 10%)
            score_scale: Multiplier for scores (1000 = show 0.000979 as 0.979)
            min_wins_for_affiliate_anchor: Minimum wins to create affiliate-level anchor
        """

        self.window_days = None
        self.exploration_rate = None
        self.prior_weight_ratio = None
        self.score_scale = None
        self.min_wins_for_affiliate_anchor = None
        
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

        # Redis client placeholder
        self.redis = valkey_helper  # To be set with ValkeyClusterHelper instance   

    # OPTIMIZATION 6: Date serialization helper methods
    def _serialize_date(self, date_obj):
        """Convert date object to ISO string"""
        return date_obj.isoformat() if hasattr(date_obj, 'isoformat') else str(date_obj)

    def _deserialize_date(self, date_str):
        """Convert ISO string back to date object"""
        try:
            return datetime.fromisoformat(date_str).date()
        except:
            return datetime.strptime(date_str, '%Y-%m-%d').date()
    
    async def _retry_redis_operation(self, operation, *args, default=None, max_retries=3):
        """OPTIMIZATION 3: Retry logic with exponential backoff for Redis operations"""
        for attempt in range(max_retries):
            try:
                return await operation(*args)
            except Exception as e:
                if attempt == max_retries - 1:
                    Logger.log(LogLevel.ERROR, "context_table", f"Redis operation failed after {max_retries} attempts: {e}")
                    if default is not None:
                        return default
                    raise e
                await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
       
    async def load_redis_state(self):
        """
        Load previously saved state from redis INCLUDING ALL SCORES
        CRITICAL: This must load the scoring table!
        """
       
        # Restore contexts
        self.contexts = await self.redis.get_json("contexts")
        
        # Restore partner stats
        self.partner_stats = await self.redis.get_json("partner_stats")
        
        # Restore affiliate anchors
        self.affiliate_anchors = await self.redis.get_json("affiliate_anchors")
        
        # Restore counters with proper type conversion
        counter = await self.redis.get_key("unknown_contexts_handled")
        self.unknown_contexts_handled = int(counter) if counter else 0
        
        affiliates_data = await self.redis.get_key("new_affiliates_seen")
        try:
            self.new_affiliates_seen = set(json.loads(affiliates_data)) if affiliates_data else set()
        except (json.JSONDecodeError, TypeError):
            self.new_affiliates_seen = set()
        
        # Rebuild helper mappings
        self.partner_id_to_stats = {}
        for partner_name, stats in self.partner_stats.items():
            if stats.get('partner_id') is not None:
                self.partner_id_to_stats[stats['partner_id']] = stats
        
        self.context_to_affiliate = {}
        for context_id, context_info in self.contexts.items():
            if 'affiliate_id' in context_info:
                self.context_to_affiliate[context_id] = context_info['affiliate_id']
        
        config = await self._retry_redis_operation(
            self.redis.get_json, 'config', default={}
        )
        if isinstance(config, list) and config:
            config = config[0]
            
        self.window_days = config.get('window_days', 15)
        self.exploration_rate = config.get('exploration_rate', 0.15)
        self.prior_weight_ratio = config.get('prior_weight_ratio', 0.1)
        self.score_scale = config.get('score_scale', 1000.0)
        self.min_wins_for_affiliate_anchor = config.get('min_wins_for_affiliate_anchor', 10)

        # CRITICAL: Load the scoring table!
        table = await self.redis.get_json("table")
        if table:
            total_scores_loaded = 0
            
            for context_id_str, actions in table.items():
                context_id = int(context_id_str)
                self.table[context_id] = {}
                
                for action_str, metrics in actions.items():
                    action = int(action_str)
                    
                    # Rebuild deque from list
                    buckets_list = metrics['daily_buckets']
                    
                    # OPTIMIZATION 5: Bucket trimming on load with helper method
                    if len(buckets_list) > self.window_days:
                        buckets_list = buckets_list[-self.window_days:]
                    
                    # Use date helper method for consistency
                    for bucket in buckets_list:
                        if 'date' in bucket and isinstance(bucket['date'], str):
                            bucket['date'] = self._deserialize_date(bucket['date'])
                    
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
            # print(f"   - Loaded {total_scores_loaded} action scores")
            Logger.log(LogLevel.INFO, "context_table" , f"✅ State loaded from Redis - Loaded {total_scores_loaded} action scores")
        else:
            Logger.log(LogLevel.INFO, "context_table" , f"⚠️ WARNING: No scoring table found in Redis!")
            Logger.log(LogLevel.INFO, "context_table" , f"   Reinitializing all scores...")
    
    
    async def _initialize_action_scores(self, 
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
            
            # Initialize the data structure
            metrics = {
                'daily_buckets': deque(maxlen=self.window_days),
                'score': initial_score,
                'total_opportunity': 0,
                'total_profit': 0,
                'total_wins': 0,
                'total_attempts': 0
            }
            buckets_list = list(metrics['daily_buckets'])
            for bucket in buckets_list:
                if 'date' in bucket and hasattr(bucket['date'], 'isoformat'):
                    bucket['date'] = bucket['date'].isoformat()

            action_data = {
                'daily_buckets': buckets_list,
                'score': float(metrics['score']),
                'total_opportunity': float(metrics['total_opportunity']),
                'total_profit': float(metrics['total_profit']),
                'total_wins': int(metrics['total_wins']),
                'total_attempts': int(metrics['total_attempts'])
            }

            # OPTIMIZATION 2: Direct path update instead of reading/writing entire table
            await self._retry_redis_operation(
                self.redis.set_json, "table", action_data, f"$.{context_id}.{action}"
            )
            Logger.log(LogLevel.INFO, "context_table" , f"Initialized score for context {context_id}, action {action}: {action_data}")

    
    def _extract_affiliate_from_context(self, context_id: int) -> int:
        """
        Extract affiliate_id from context_id
        Context format: [affiliate_id][7-digit base_context]
        """
        return context_id // 10_000_000
    
    async def _initialize_unknown_context(self, context_id: int) -> bool:
        """
        Initialize unknown context with affiliate-aware strategy

        Fallback hierarchy:
        1. Known affiliate → use affiliate anchor from training
        2. Unknown affiliate → default to 70%

        Actions are [anchor-5, anchor, anchor+5]
        """
        # Extract affiliate ID first
        affiliate_id = self._extract_affiliate_from_context(context_id)

        # Determine anchor based on affiliate_anchors from training
        if str(affiliate_id) in self.affiliate_anchors:
            # Known affiliate → use affiliate-specific anchor with multiple actions
            anchor_bid = self.affiliate_anchors[str(affiliate_id)]
            anchor_source = "affiliate_anchor"
            # Create actions: [anchor-6, anchor-4, anchor-2]
            # Bound by [40, 97] range
            UNKNOWN_CONTEXT_ACTIONS = [
                max(40, anchor_bid - 6),
                max(40, anchor_bid - 4),
                max(40, anchor_bid - 2)
            ]
            Logger.log(LogLevel.INFO, "context_table",
                      f"Unknown context {context_id}: Using affiliate {affiliate_id} anchor = {anchor_bid}% with actions {UNKNOWN_CONTEXT_ACTIONS}")
        else:
            # Unknown affiliate → use single default action of 70
            anchor_bid = 70
            anchor_source = "global_default"
            UNKNOWN_CONTEXT_ACTIONS = [70]  # Single action only
            Logger.log(LogLevel.INFO, "context_table",
                      f"Unknown context {context_id}: Affiliate {affiliate_id} unknown, using single default action = 70%")

        UNKNOWN_CONTEXT_ANCHOR = anchor_bid

        # Track new context with error handling
        try:
            contexts_data = await self.redis.get_key("new_contexts_seen")
            new_contexts_seen = set(json.loads(contexts_data)) if contexts_data else set()
        except (json.JSONDecodeError, TypeError):
            new_contexts_seen = set()
            
        if context_id not in new_contexts_seen:
            new_contexts_seen.add(context_id)
        await self.redis.set_key("new_contexts_seen", json.dumps(list(new_contexts_seen)))
        
        # Track by hour for monitoring (implementation placeholder)
        # current_hour = datetime.now().strftime("%Y-%m-%d %H:00")
        # You may want to implement hourly tracking in Redis if needed
        
        # Log periodically
        if len(new_contexts_seen) % 100 == 0:
            Logger.log(LogLevel.INFO, "context_table", 
                      f"⚠️ Total new contexts seen: {len(new_contexts_seen)}")
        # Extract affiliate ID from context
        affiliate_id = self._extract_affiliate_from_context(context_id)
        
        # Validate extraction
        if affiliate_id > 99999:
            Logger.log(LogLevel.INFO, "context_table" , f"⚠️ Invalid affiliate_id {affiliate_id} extracted from context {context_id}")  
            return False
        
        # Create context entry with affiliate-aware strategy
        data = {
            'anchor': UNKNOWN_CONTEXT_ANCHOR,
            'affiliate': f"Unknown_Affiliate_{affiliate_id}",
            'affiliate_id': affiliate_id,
            'fallback_level': anchor_source,  # 'affiliate_anchor' or 'global_default'
            'initialized_at': datetime.now().isoformat(),
            'source': 'runtime_unknown',
            'actions': UNKNOWN_CONTEXT_ACTIONS.copy(),
            'strategy_type': 'UNKNOWN_DYNAMIC'  # Changed from UNKNOWN_FIXED
        }
        
        Logger.log(LogLevel.INFO, "context_table", 
                f"Initializing unknown context {context_id}: {data}")
        await self.redis.set_json("contexts", data, f"$.{context_id}")

        # Store mapping
        await self.redis.set_json("context_to_affiliate", affiliate_id, f"$.{context_id}")
        
        # Initialize scores for the actions
        for action in UNKNOWN_CONTEXT_ACTIONS:
            # For single action (unknown affiliate), use baseline score
            if len(UNKNOWN_CONTEXT_ACTIONS) == 1:
                initial_score = 0.5 * self.score_scale  # Single action baseline
            else:
                # For multiple actions (known affiliate), use optimistic initialization
                # Lower bids (more aggressive) start with higher scores
                if action == UNKNOWN_CONTEXT_ACTIONS[0]:  # Most aggressive
                    initial_score = 0.7 * self.score_scale
                elif action == UNKNOWN_CONTEXT_ACTIONS[1]:  # Middle
                    initial_score = 0.5 * self.score_scale
                else:  # Most conservative
                    initial_score = 0.3 * self.score_scale
            
            # Initialize the scoring data
            action_data = {
                'daily_buckets': [],
                'score': float(initial_score),
                'total_opportunity': 0.0,
                'total_profit': 0.0,
                'total_wins': 0,
                'total_attempts': 0
            }
            
            # 1. Get the whole document
            doc_json = await self.redis.get_json("table")

            # 2. Add the new branch if missing
            doc_json.setdefault(context_id, {})[action] = action_data
            
            # 3. Write it back
            await self.redis.set_json("table", doc_json)
            
            Logger.log(LogLevel.INFO, "context_table" , f"Initialized score for context {context_id}, action {action}: {action_data}")
                            

        # Track this initialization
        unknown_contexts_handled = await self.redis.get_key("unknown_contexts_handled")
        unknown_contexts_handled = int(unknown_contexts_handled) + 1
        await self.redis.set_key("unknown_contexts_handled", str(unknown_contexts_handled))
        
        if unknown_contexts_handled <= 10 or unknown_contexts_handled % 100 == 0:
            Logger.log(LogLevel.INFO, "context_table", 
                  f"🆕 Unknown context {context_id}: "
                  f"Fixed actions={UNKNOWN_CONTEXT_ACTIONS}, "
                  f"Total unknown handled={unknown_contexts_handled}")
        
        return True
    
    async def select_action(self, context_id: int) -> tuple[int, str] :
        """
        Select an action for a context using epsilon-greedy
        Handles unknown contexts by initializing them on-the-fly
        """

        # OPTIMIZATION 1: Efficient context fetching - use JSONPath directly to avoid fetching 100KB document
        context = await self._retry_redis_operation(
            self.redis.get_json, "contexts", f"$.{context_id}", 
            default=None
        )
        # Handle JSONPath array response
        if isinstance(context, list) and len(context) > 0:
            context = context[0]
        
        Logger.log(LogLevel.INFO, "context_table" , f"Selecting action for context {context_id} : {context}")
        
        # Handle unknown context
        if not context:
            # Try to initialize it
            if await self._initialize_unknown_context(context_id):
                # Successfully initialized
                Logger.log(LogLevel.INFO, "context_table" , f"Initialized unknown context {context_id} on-the-fly")
                context = await self._retry_redis_operation(
                    self.redis.get_json, "contexts", f"$.{context_id}", default={}
                )
                if isinstance(context, list) and context:
                    context = context[0]
                actions = context.get('actions', []) if context else []
                Logger.log(LogLevel.INFO, "context_table" , f"Actions for context {context_id}: {actions}")
            else:
                # Failed to initialize
                Logger.log(LogLevel.INFO, "context_table" , f"⚠️ Failed to initialize context {context_id}, using default 70")
                return (70, "default")
        else:
            actions = context.get('actions', [])
        
    
        # Epsilon-greedy selection
        if np.random.random() < self.exploration_rate:
            # Explore: random action
            # self.logger.info(f"Exploring context {context_id}")
            Logger.log(LogLevel.INFO, "context_table" , f"Exploring context {context_id} with actions {actions}")
            return int(np.random.choice(actions)), "rl_explorer" 
        else:
            # Exploit: best scoring action  
            Logger.log(LogLevel.INFO, "context_table" , f"Exploiting context {context_id}")
            actions = await self._retry_redis_operation(
                self.redis.get_json, "table", f"$.{context_id}",
                default={}
            )
            if actions:
                # OPTIMIZATION 4: Fix type inconsistency - convert to int immediately
                scores = {int(action): data['score'] for action, data in actions.items()}
                best_action = max(scores, key=scores.get)  # Already an int
                return best_action, "rl_model"
            else:
                # Fallback to anchor if no scores
                anchor_data = await self._retry_redis_operation(
                    self.redis.get_json, "contexts", f"$.{context_id}.anchor", default=70
                )
                if isinstance(anchor_data, list) and anchor_data:
                    anchor_data = anchor_data[0]
                return anchor_data if anchor_data else 70, "default"
    
    async def update(self, 
               context_id: int, 
               action: int, 
               expected_revenue: float,
               won: bool,
               timestamp: Optional[datetime] = None):
        """
        Update the table with a new result using 15-day rolling window
        CRITICAL: This is how the model learns!
        """

        Logger.log(LogLevel.INFO, "context_table" , f"Updating context {context_id}, action {action}, expected_revenue {expected_revenue}, won {won}, timestamp {timestamp}")
        is_context_available = await self._retry_redis_operation(
            self.redis.get_json, "contexts", f"$.{context_id}", default=None
        )
        if isinstance(is_context_available, list) and is_context_available:
            is_context_available = is_context_available[0]
            
        is_action_available = await self._retry_redis_operation(
            self.redis.get_json, "table", f"$.{context_id}.{action}", default=None
        )
        Logger.log(LogLevel.INFO, "context_table" , f"Action data available: {is_action_available is not None}")

        if not is_context_available:
            # return  # Skip unknown contexts
            Logger.log(LogLevel.INFO, "context_table" , f"⚠️ Attempted to update unknown context {context_id}, skipping")   
            return
        
        if not is_action_available:
            # return  # Skip unknown actions
            Logger.log(LogLevel.INFO, "context_table" , f"⚠️ Attempted to update unknown action {action} for context {context_id}, skipping")
            return
        
        if timestamp is None:
            timestamp = datetime.now()

        # Calculate profit
        profit = expected_revenue * (100 - action) / 100 if won else 0
        
        # Get today's date
        today = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        today = today.isoformat()
        # Find or create today's bucket
        buckets = await self.redis.get_json("table", f"$.{context_id}.{action}.daily_buckets") 
        
        # CRITICAL FIX: If we're at the limit and need a new day, remove oldest first
        if buckets and len(buckets) >= self.window_days and buckets[-1]['date'] != today:
            buckets = buckets[-(self.window_days-1):]  # Make room for new bucket
        
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

        # Sum across all buckets
        total_opportunity = sum(b['opportunity'] for b in buckets)
        total_profit = sum(b['profit'] for b in buckets)

        # Calculate adaptive prior
        prior_opportunity = max(1000, total_opportunity * self.prior_weight_ratio)
        
        # Prior assumes baseline win rate
        affiliate_id = await self._retry_redis_operation(
            self.redis.get_json, "contexts", f"$.{context_id}.affiliate_id", default=99999
        )
        if isinstance(affiliate_id, list) and affiliate_id:
            affiliate_id = affiliate_id[0]
        if not affiliate_id:
            affiliate_id = 99999
        
        strategy_type = await self._retry_redis_operation(
            self.redis.get_json, "contexts", f"$.{context_id}.strategy_type", default=None
        )
        if isinstance(strategy_type, list) and strategy_type:
            strategy_type = strategy_type[0]
            
        if strategy_type == 'UNKNOWN_FIXED':
            baseline_win_rate = 0.01  # Special rate for unknowns
        else:
            baseline_win_rate = await self._retry_redis_operation(
                self.redis.get_json, "partner_id_to_stats", f"$.{affiliate_id}.win_rate", default=0.007
            )
            if isinstance(baseline_win_rate, list) and baseline_win_rate:
                baseline_win_rate = baseline_win_rate[0]
            if not baseline_win_rate:
                baseline_win_rate = 0.007
        
        prior_profit = prior_opportunity * baseline_win_rate * (100 - action) / 100
        
        # Calculate score (with scaling)
        if total_opportunity + prior_opportunity > 0:
            score = (total_profit + prior_profit) / (total_opportunity + prior_opportunity)
            score = score * self.score_scale
        else:
            margin = (100 - action) / 100
            score = baseline_win_rate * margin * self.score_scale

        # Convert deque to list for JSON
        buckets_list = list(buckets)
        # Use date helper method for consistent serialization
        for bucket in buckets_list:
            if 'date' in bucket:
                bucket['date'] = self._serialize_date(bucket['date'])
        
        # Calculate totals directly from buckets (which now includes the new update)
        final_total_wins = sum(b['wins'] for b in buckets)
        final_total_attempts = sum(b['attempts'] for b in buckets)
        
        data = {'daily_buckets' : buckets_list,
                'score' : score,
                'total_opportunity' : total_opportunity,
                'total_profit': total_profit,
                'total_wins' : final_total_wins,
                'total_attempts' : final_total_attempts}
        
        #update action
        Logger.log(LogLevel.INFO, "context_table" , f"Updated action data for context {context_id}, action {action}: {data}")
        await self.redis.set_json("table", data , f"$.{context_id}.{action}")
    
    
    def save_redis_state(self, filepath: str = "context_table_state.json"):
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
