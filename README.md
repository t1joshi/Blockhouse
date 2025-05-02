# Order Flow Imbalance (OFI) Feature Extraction

This project implements the methodology from the paper  
**"Cross-Impact of Order Flow Imbalance in Equity Markets"**  
to compute key order flow imbalance (OFI) features from limit order book (LOB) data.

## Features Extracted

- **Best-Level OFI**: Order flow imbalance at the top of the book (best bid/ask).
- **Multi-Level OFI**: OFI aggregated across the top 10 LOB levels, normalized by average depth.
- **Integrated OFI**: Principal component (PCA-based) aggregation of multi-level OFIs.
- **Cross-Asset OFI**: Cross-impact of OFI from other assets (if multi-asset data is present).

## Data

- Input: `first_25000_rows.csv`  
  Must contain columns for timestamps, bid/ask prices and sizes at each level, and (optionally) a `symbol` column for multi-asset analysis.

## Usage

1. **Install dependencies**:
    ```
    pip install pandas numpy scikit-learn
    ```

2. **Run the feature extraction script**:
    ```
    python ofi_feature_extraction.py
    ```

3. **Output**:  
    - A CSV file with computed OFI features, one row per time interval.

## Methodology

- Follows the definitions and formulas in Cont et al. (2014) and the 2023 Quantitative Finance paper (see [paper PDF](Cross-impact-of-order-flow-imbalance-in-equity-markets-1.pdf)).
- See the paper for mathematical details and economic motivation.

## References

- [Cross-Impact of Order Flow Imbalance in Equity Markets](Cross-impact-of-order-flow-imbalance-in-equity-markets-1.pdf), Quantitative Finance, 2023.
- Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events.

---

**Author:** Tanvi Ganesh Joshi  
