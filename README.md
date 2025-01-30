# thompson-stm
Contextual Thompson Sampler With Short Term Memory


### 📌 Overview  
This repository implements **Contextual Thompson Sampling** with **Softmax Scaling** and **Short-Temr Memory** for **dynamic bid optimization**.  
The algorithm adapts bid values based on past performance, balancing **exploration and exploitation** to maximize expected wins while minimizing bid costs.

### ⚡ Features  
✅ **Context-Aware Learning:** Bids are optimized per `(campaign_id, revenue_bucket)`  
✅ **Thompson Sampling:** Bayesian optimization ensures dynamic bid adjustments  
✅ **Softmax Scaling:** Prevents over-exploitation and maintains exploration flexibility  
✅ **Adaptive History Window:** Keeps a short-term memory of bid performance  

### 🎯 Use Cases  
- **Ad Auctions & RTB (Real-Time Bidding)**
- **Online Marketplace Price Optimization**
- **Dynamic Pricing Strategies**
- **Bid Adjustment for Call Marketplace (e.g., eLocal)**

---

## 🏗️ Installation  
```bash
git clone https://github.com/your_username/ContextualTS-Bidding.git
cd ContextualTS-Bidding
pip install -r requirements.txt  # if any dependencies are added later
```

---

## 🚀 Usage

### 1️⃣ Initialize the Model  
```python
from contextual_ts import ContextualThompsonSampler

sampler = ContextualThompsonSampler(min_bid=80, max_bid=100, tau=2.0, history_window=10)
```

### 2️⃣ Sample a Bid  
```python
campaign_id = "solar_ads"
revenue_bucket = "high"
bid = sampler.sample_bid(campaign_id, revenue_bucket)
print(f"Suggested Bid: {bid}")
```

### 3️⃣ Update Model After Auction Outcome  
```python
sampler.update(campaign_id, revenue_bucket, bid=90, win=1)  # 1 if bid won, 0 if lost
```

### 4️⃣ Get Current Bid Distributions  
```python
print(sampler.get_bid_distribution(campaign_id, revenue_bucket))
```

---

## 🛠️ How It Works  

1. **Sampling**  
   - Uses **Beta distributions** per bid level (80-100)  
   - Chooses a bid via **softmax-scaled Thompson Sampling**  

2. **Updating**  
   - Stores `(bid, win)` history  
   - Updates **Beta distributions** dynamically  

3. **Exploration vs. Exploitation**  
   - Softmax scaling with `τ` controls randomness  
   - Higher `τ` = more exploration, Lower `τ` = more exploitation  

---

## 🏆 Why This is Powerful  
✅ **Better than ε-Greedy:** Adjusts probability dynamically instead of fixed random exploration  
✅ **More Efficient than UCB:** Thompson Sampling naturally balances exploration  
✅ **Practical for Dynamic Markets:** Works with **short-term bid history** rather than relying on long past data  

---

## 📜 License  
This project is licensed under the MIT License.  

---
