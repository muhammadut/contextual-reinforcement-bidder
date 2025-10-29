

# 🚀 Contextual Reinforcement Learning Bid Optimization Pipeline

## 🎯 The Business Problem: Bidding at E-local

As the lead data scientist at E-local, I designed this system to solve a core challenge in our lead-bidding marketplace. E-local's business model is a real-time marketplace:

* **Upstream Partners:** These partners (lead generators) send us "pings", opportunities to buy a lead (like a phone call) for a specific service.
* **Downstream Buyers:** These are our customers (contractors like plumbers, electricians) who are willing to pay us for these leads.

### The Bidding Flow

1.  **Ping:** An upstream partner pings us: "Want to buy a plumber call from zip code 90120?"
2.  **Internal Search:** Our system instantly finds a downstream buyer willing to pay, for example, **$100** for that call.
3.  **The Bid:** We must now decide how much to bid against our competitors for that ping.

**The Traditional Method (Problem):**
Our original method was deterministic: bid a static **70%** of our buyer's price (e.g., $70). This "one-size-fits-all" rule is simple but highly inefficient. It overpays for low-quality pings and, more importantly, underbids on high-value ones, causing us to lose valuable opportunities.

**The Goal (Solution):**
This pipeline replaces that static rule with an intelligent, adaptive system. The goal is to automatically learn context-specific bidding strategies to maximize conversions and revenue while controlling costs.

---

## 🧠 The Solution: From Supervised Learning to RL

This approach combines **supervised learning (XGBoost)** for pattern recognition with **reinforcement learning (RL)** strategies for bid optimization.

Here's the high-level methodology:

1.  **Context Generation:** We use XGBoost not just to *predict* a win, but as a **context generator**. It analyzes all features (geo, time, affiliate, etc.) and groups similar pings into "contexts." For example, `Context_123` might represent 'weekday-morning-plumber-pings-from-Boston'.
2.  **Anchor Discovery:** For each context, we find a historically successful **anchor bid** (e.g., the most common *winning* bid for `Context_123`).
3.  **RL Strategy:** We then use RL-inspired strategies (like contextual bandits) to refine this anchor. This allows the system to balance **exploitation** (using the $70 bid we know works) with **exploration** (testing $72 or $68 to find a *better* price).
4.  **Real-Time Serving:** The final "context table" (mapping each context to its optimized bid) is loaded into a high-speed Valkey/Redis database. When a new ping arrives, the system instantly finds its context, looks up the optimal bid, and places it in milliseconds.



---

## 🧩 Pipeline Components

### 1. ⚙️ `core/ContextGenerator.py`: Context Discovery

**Concept:**
The `ContextGenerator` module leverages XGBoost not just for prediction, but as a tool for **context discovery and assignment**.

#### How It Works

1.  **Model Training:**
    * The XGBoost model is trained on historical data to predict the probability of a win.
    * Input features are engineered to capture all relevant signals (geo, time, affiliate, etc.).

2.  **Context Assignment:**
    * Instead of just using the prediction score, we extract information from the trained trees:
    * **Leaf Indices:** Each sample is assigned to a leaf node in each tree, which can be combined to form a unique context ID.
    * **Probability Bins:** Predicted probabilities are discretized into bins (e.g., quartiles) to group samples with similar likelihoods.
    * **Tree Subsets:** The config may specify custom groupings (context\_L1, context\_L2, etc.) based on tree structure.
    * Each sample is thus assigned one or more context IDs, representing its “state” in the bidding environment.

3.  **Context Levels:**
    * Multiple context levels (L1-L4) can be generated, representing different granularities (coarse to fine).
    * These context IDs are used downstream to aggregate statistics and optimize bids.

#### Theory: Representation Learning & Contextual Bandits

> * **Representation Learning:** By mapping raw features to context IDs, we create a lower-dimensional, interpretable representation of the data. Contexts group together samples that behave similarly, making it easier to learn effective strategies.
>
> * **Contextual Bandits / RL:** In reinforcement learning, a “state” summarizes the environment. Here, each context ID acts as a **state**: the system learns which bid (action) works best for each context.
>
> * **Benefits:** Instead of learning a separate strategy for every possible feature combination, we learn per-context, reducing complexity. This helps the model generalize to new samples that share similar characteristics.

**Summary:**
The `ContextGenerator` transforms raw data into actionable contexts using XGBoost’s tree structure. These contexts are the foundation for downstream bid optimization, enabling the system to make smarter, context-aware decisions.

---

### 2. ⚓ `core/ContextAnchorGenerator.py`: Anchor Bid Discovery

**Concept:**
The `ContextAnchorGenerator` module is responsible for discovering **anchor bids** for each context. An anchor bid is the most representative or optimal bid value for a given context, based on *historical winning bids*.

#### How It Works

1.  **Anchor Discovery:**
    * For each context (e.g., `context_L1`), the module examines historical data to find the bid value that most frequently resulted in a win (`won == 1`).
    * This value is designated as the **anchor bid** for that context.

2.  **Fallback Logic:**
    * If a context has too few winning samples (data sparsity), the module falls back to broader groupings:
        * **Affiliate-level anchor:** Uses the most common winning bid for that affiliate.
        * **Global anchor:** Uses the most common winning bid across the entire dataset.

3.  **Recency and Minimum Wins:**
    * The module can be configured to use only recent data (e.g., last N days) and to require a minimum number of wins for an anchor to be valid.

4.  **Anchor Statistics:**
    * The module tracks how many anchors were found at each level (context, affiliate, global) and reports these statistics for transparency.

#### Theory: Empirical Policy Initialization

> * **Empirical Anchors:** By using the most common *winning* bid as the anchor, the system starts with a bid value that has historically performed well. This reduces the risk of poor initial performance and speeds up learning.
>
> * **Fallbacks for Robustness:** Sparse data is a common challenge. By falling back to affiliate or global anchors, the module ensures that every context has a sensible starting bid.
>
> * **Foundation for RL Strategy:** These anchors serve as the crucial starting point for the RL-inspired bid optimization strategy. The system will then explore and adjust bids *around* these anchors.

**Summary:**
`ContextAnchorGenerator.py` finds the most representative bid for each context using historical win data, with robust fallbacks for sparse contexts. This empirical approach provides a strong foundation for adaptive, context-aware bid optimization.

---

### 3. 📈 `core/ContextTableBidLowerStrategy.py`: Adaptive Bid Strategy

**Concept:**
This module generates the final **context table**—a JSON mapping from each context ID to an optimized bid value. It uses reinforcement learning-inspired strategies to determine the best bid, balancing historical performance (the anchor) with ongoing exploration.

#### How It Works

1.  **Initialization with Anchors:**
    * The context table is first built using the anchor bids discovered by the `ContextAnchorGenerator`.

2.  **Strategy Application:**
    * The module applies a configurable RL-inspired strategy to adjust bids. Common strategies include:
        * **Exploration/Exploitation:** Uses an exploration rate (epsilon) to occasionally try new bids, preventing the system from getting stuck.
        * **Score Scaling:** Adjusts bids based on historical win rates, conversion rates, or other performance metrics for that context.
        * **Prior Weighting:** Blends anchor values with recent performance to avoid overfitting to short-term trends.
        * **Windowing:** Uses a rolling window of recent data to keep bid recommendations fresh.

3.  **Tiered Strategies:**
    * The system can apply different strategies or bid bounds for different tiers of contexts (e.g., more aggressive bidding for high-value contexts).

4.  **Output:**
    * The final context table is saved as a **`context_table.json`** artifact, mapping each context ID to its recommended bid.

#### Theory: Contextual Multi-Armed Bandits

> * **Contextual Bandits:** Each context is treated as a “state,” and the bid is the “action.” The system learns which bid (action) yields the best reward (win rate, revenue) for each context.
>
> * **Exploration vs. Exploitation:** To maximize long-term rewards, the system must balance **exploiting** known successful bids (the anchors) with **exploring** new possibilities to find even better bids.
>
> * **Adaptive Optimization:** By using rolling windows and prior weighting, the system adapts to recent trends while maintaining stability, ensuring it responds to changing market conditions.

**Summary:**
`ContextTableBidLowerStrategy.py` operationalizes the RL-inspired logic. It takes context IDs and anchor bids, applies adaptive strategies, and produces the final context table that can be deployed for real-time, data-driven, and continually improving bid decisions.

---

### 4. ⚡ `core/ContextTableValkey.py`: Real-Time Bid Serving

**Concept:**
This module provides a **production-ready interface** for serving the context table using Redis (specifically Valkey, a Redis-compatible database). It enables fast, real-time lookups of optimized bid values for each context, supporting scalable deployment in our live bidding environment.

#### How It Works

1.  **Loading the Context Table:**
    * The module loads the `context_table.json` artifact generated by the pipeline, which maps context IDs to bid values and statistics.

2.  **Redis/Valkey Integration:**
    * The context table is loaded into the Valkey database, allowing bid values to be accessed via simple key-value queries.
    * Each context ID acts as a key, and its optimized bid (plus metadata) is the value.

3.  **API for Bid Lookup:**
    * The module exposes functions or endpoints to query the bid for a given context ID.
    * This enables real-time, low-latency bid retrieval for our production bidding systems.

4.  **Updates and Monitoring:**
    * The system supports periodic updates to the context table as new data and strategies are deployed (e.g., daily or hourly refreshes).
    * Monitoring and logging ensure reliability and traceability.

#### Theory: Scalable Production Deployment

> * **Scalability:** Using an in-memory database like Valkey/Redis allows the context table to be served at high throughput and with sub-millisecond latency, which is essential for real-time bidding.
>
> * **Separation of Concerns:** The pipeline generates and optimizes the context table **offline**; `ContextTableValkey.py` handles the **online serving** and integration with production systems.
>
> * **Flexibility:** The approach supports dynamic updates, making it easy to refresh bid strategies without system downtime as market conditions change.

**Summary:**
`ContextTableValkey.py` operationalizes our bid optimization strategy. It bridges the gap between offline ML/RL optimization and real-time production deployment by serving the context table via a high-speed database for fast, reliable, and scalable bid lookups.
