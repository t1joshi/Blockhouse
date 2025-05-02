import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA

def compute_best_level_ofi(df):
    df['bid_of'] = np.where(df['bid_price'] > df['bid_price'].shift(1), df['bid_size'],
                          np.where(df['bid_price'] == df['bid_price'].shift(1), 
                                   df['bid_size'] - df['bid_size'].shift(1), 
                                   -df['bid_size']))
    df['ask_of'] = np.where(df['ask_price'] < df['ask_price'].shift(1), df['ask_size'],
                          np.where(df['ask_price'] == df['ask_price'].shift(1), 
                                   df['ask_size'] - df['ask_size'].shift(1), 
                                   -df['ask_size']))
    df['best_ofi'] = df['bid_of'] - df['ask_of']
    return df.groupby(pd.Grouper(freq='1min'))['best_ofi'].sum()

def compute_multi_level_ofi(df, levels=10):
    for m in range(1, levels+1):
        df[f'bid_of_m{m}'] = np.where(df[f'bid_price_{m}'] > df[f'bid_price_{m}'].shift(1), df[f'bid_size_{m}'],
                                    np.where(df[f'bid_price_{m}'] == df[f'bid_price_{m}'].shift(1),
                                             df[f'bid_size_{m}'] - df[f'bid_size_{m}'].shift(1),
                                             -df[f'bid_size_{m}']))
        df[f'ask_of_m{m}'] = np.where(df[f'ask_price_{m}'] < df[f'ask_price_{m}'].shift(1), df[f'ask_size_{m}'],
                                    np.where(df[f'ask_price_{m}'] == df[f'ask_price_{m}'].shift(1),
                                             df[f'ask_size_{m}'] - df[f'ask_size_{m}'].shift(1),
                                             -df[f'ask_size_{m}']))
    
    # Compute average depth Q
    depth_columns = [f'bid_size_{m}' for m in range(1,11)] + [f'ask_size_{m}' for m in range(1,11)]
    df['Q'] = df[depth_columns].mean(axis=1) / (2 * df.groupby(pd.Grouper(freq='1min')).size())
    
    # Sum normalized OFI across levels
    df['multi_ofi'] = (df[[f'bid_of_m{m}' for m in range(1,11)]].sum(axis=1) - 
                       df[[f'ask_of_m{m}' for m in range(1,11)]].sum(axis=1)) / df['Q']
    return df.groupby(pd.Grouper(freq='1min'))['multi_ofi'].sum()



def compute_integrated_ofi(df):
    pca = PCA(n_components=1)
    pca.fit(df[['OFI_1', ..., 'OFI_10']])
    weights = pca.components_[0]
    df['integrated_ofi'] = df[['OFI_1', ..., 'OFI_10']].dot(weights) / np.sum(np.abs(weights))
    return df['integrated_ofi']


def compute_cross_asset_ofi(ofi_matrix):
    model = Lasso(alpha=0.01)
    cross_ofi = pd.DataFrame()
    for asset in ofi_matrix.columns:
        X = ofi_matrix.drop(columns=asset)
        y = ofi_matrix[asset]
        model.fit(X, y)
        cross_ofi[asset] = model.predict(X)
    return cross_ofi


def compute_ofi_features(file_path, output_file='ofi_features.csv', freq='1min'):
    """Compute all OFI features from order book data"""
    print(f"Loading data from {file_path}...")
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime index
    df['timestamp'] = pd.to_datetime(df['ts_event'], unit='ns')
    df = df.set_index('timestamp')
    
    # Initialize output DataFrame
    results = pd.DataFrame(index=df.resample(freq).last().index)
    
    # 1. Best-Level OFI
    print("Computing Best-Level OFI...")
    results['best_ofi'] = compute_best_level_ofi(df, freq)
    
    # 2. Multi-Level OFI
    print("Computing Multi-Level OFI...")
    multi_ofi, multi_level_ofi_df = compute_multi_level_ofi(df, freq)
    results['multi_ofi'] = multi_ofi
    
    # 3. Integrated OFI
    print("Computing Integrated OFI...")
    results['integrated_ofi'] = compute_integrated_ofi(multi_level_ofi_df)
    
    # 4. Cross-Asset OFI
    print("Computing Cross-Asset OFI...")
    if 'symbol' in df.columns and df['symbol'].nunique() > 1:
        results['cross_asset_ofi'] = compute_cross_asset_ofi(df, freq)
    else:
        print("Warning: Cross-Asset OFI requires multiple assets. Using placeholder values.")
        results['cross_asset_ofi'] = np.nan
    
    # Save results
    results.to_csv(output_file)
    print(f"Results saved to {output_file}")
    return results

def compute_best_level_ofi(df, freq='1min'):
    """Compute Best-Level OFI (top of book only)"""
    df_copy = df.copy()
    
    # Calculate order flow for best bid
    df_copy['bid_of'] = np.where(
        df_copy['bid_px_00'] > df_copy['bid_px_00'].shift(1), 
        df_copy['bid_sz_00'],
        np.where(
            df_copy['bid_px_00'] == df_copy['bid_px_00'].shift(1),
            df_copy['bid_sz_00'] - df_copy['bid_sz_00'].shift(1),
            -df_copy['bid_sz_00']
        )
    )
    
    # Calculate order flow for best ask
    df_copy['ask_of'] = np.where(
        df_copy['ask_px_00'] < df_copy['ask_px_00'].shift(1),
        df_copy['ask_sz_00'],
        np.where(
            df_copy['ask_px_00'] == df_copy['ask_px_00'].shift(1),
            df_copy['ask_sz_00'] - df_copy['ask_sz_00'].shift(1),
            -df_copy['ask_sz_00']
        )
    )
    
    # Fill NaN values
    df_copy['bid_of'] = df_copy['bid_of'].fillna(0)
    df_copy['ask_of'] = df_copy['ask_of'].fillna(0)
    
    # Compute best level OFI and resample
    df_copy['best_level_ofi'] = df_copy['bid_of'] - df_copy['ask_of']
    return df_copy['best_level_ofi'].resample(freq).sum()

def compute_multi_level_ofi(df, freq='1min', levels=10):
    """Compute Multi-Level OFI (aggregated across depth levels)"""
    df_copy = df.copy()
    multi_level_of = pd.DataFrame(index=df_copy.index)
    
    # Compute OFI for each level
    for m in range(levels):
        bid_px_col = f'bid_px_{m:02d}'
        bid_sz_col = f'bid_sz_{m:02d}'
        ask_px_col = f'ask_px_{m:02d}'
        ask_sz_col = f'ask_sz_{m:02d}'
        
        # Skip if columns don't exist
        if not all(col in df_copy.columns for col in [bid_px_col, bid_sz_col, ask_px_col, ask_sz_col]):
            continue
        
        # Calculate order flow for bid at level m
        df_copy[f'bid_of_{m}'] = np.where(
            df_copy[bid_px_col] > df_copy[bid_px_col].shift(1),
            df_copy[bid_sz_col],
            np.where(
                df_copy[bid_px_col] == df_copy[bid_px_col].shift(1),
                df_copy[bid_sz_col] - df_copy[bid_sz_col].shift(1),
                -df_copy[bid_sz_col]
            )
        )
        
        # Calculate order flow for ask at level m
        df_copy[f'ask_of_{m}'] = np.where(
            df_copy[ask_px_col] < df_copy[ask_px_col].shift(1),
            df_copy[ask_sz_col],
            np.where(
                df_copy[ask_px_col] == df_copy[ask_px_col].shift(1),
                df_copy[ask_sz_col] - df_copy[ask_sz_col].shift(1),
                -df_copy[ask_sz_col]
            )
        )
        
        # Compute level m OFI
        df_copy[f'level_{m}_ofi'] = df_copy[f'bid_of_{m}'].fillna(0) - df_copy[f'ask_of_{m}'].fillna(0)
        multi_level_of[f'level_{m}_ofi'] = df_copy[f'level_{m}_ofi']
    
    # Compute average depth Q for normalization
    depth_columns = []
    for m in range(levels):
        bid_sz_col = f'bid_sz_{m:02d}'
        ask_sz_col = f'ask_sz_{m:02d}'
        if bid_sz_col in df_copy.columns and ask_sz_col in df_copy.columns:
            depth_columns.append(bid_sz_col)
            depth_columns.append(ask_sz_col)
    
    df_copy['Q'] = df_copy[depth_columns].mean(axis=1)
    
    # Resample
    Q_resampled = df_copy['Q'].resample(freq).mean()
    
    level_ofi_resampled = {}
    for col in multi_level_of.columns:
        level_ofi_resampled[col] = multi_level_of[col].resample(freq).sum()
    
    multi_level_of_resampled = pd.DataFrame(level_ofi_resampled)
    
    # Normalize by average depth
    multi_level_ofi = multi_level_of_resampled.sum(axis=1) / Q_resampled
    
    return multi_level_ofi, multi_level_of_resampled

def compute_integrated_ofi(multi_level_of):
    """Compute Integrated OFI using PCA"""
    multi_level_of_filled = multi_level_of.fillna(0)
    
    # Apply PCA to extract first principal component weights
    pca = PCA(n_components=1)
    pca.fit(multi_level_of_filled)
    
    # Extract weights and normalize by L1 norm
    weights = pca.components_[0]
    normalized_weights = weights / np.sum(np.abs(weights))
    
    # Apply weights to compute integrated OFI
    integrated_ofi = multi_level_of_filled.dot(normalized_weights)
    
    return integrated_ofi

def compute_cross_asset_ofi(df, freq='1min'):
    """Compute Cross-Asset OFI using LASSO regression"""
    # Group data by symbol
    symbols = df['symbol'].unique()
    
    if len(symbols) < 2:
        return pd.Series(np.nan, index=df.resample(freq).last().index)
    
    # Compute best-level OFI for each symbol
    ofi_by_symbol = {}
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol].copy()
        ofi_by_symbol[symbol] = compute_best_level_ofi(symbol_df, freq)
    
    # Create OFI matrix and fill NaN values
    ofi_matrix = pd.DataFrame(ofi_by_symbol).fillna(0)
    
    # Compute cross-asset OFI for each symbol using LASSO
    cross_ofi = {}
    for target_symbol in symbols:
        y = ofi_matrix[target_symbol]
        X = ofi_matrix.drop(columns=[target_symbol])
        
        # Fit LASSO model to estimate cross-impact coefficients
        model = Lasso(alpha=0.01)
        model.fit(X, y)
        
        # Predict cross-asset OFI
        cross_ofi[target_symbol] = model.predict(X)
    
    # Combine results (using mean across symbols)
    cross_ofi_df = pd.DataFrame(cross_ofi, index=ofi_matrix.index)
    return cross_ofi_df.mean(axis=1)

if __name__ == "__main__":
    # Compute and save OFI features
    file_path = 'first_25000_rows.csv'
    output_file = 'ofi_features.csv'
    ofi_features = compute_ofi_features(file_path, output_file)
    
    # Display results
    print("\nOFI Features (first 5 rows):")
    print(ofi_features.head())


