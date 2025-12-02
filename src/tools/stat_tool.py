import pandas as pd
import numpy as np
import os
from langchain_core.tools import tool

# --- Configuration ---
STATS_FILE = os.path.join("data", "structured_stats.csv")
# ---

def load_and_preprocess_stats():
    """
    Loads the player stats, performs data cleanup (forces numeric types), 
    and applies Min-Max normalization.
    """
    
    if not os.path.exists(STATS_FILE):
        print(f"Error: Stats file not found at {STATS_FILE}")
        return None, None

    df = pd.read_csv(STATS_FILE)
    
    # Use Player Name for re-alignment after cleanup
    player_names = df['Player Name']

    # Identify and drop non-numerical columns for the calculation DataFrame
    stats_df = df.drop(columns=['Player Name', 'Position', 'Team'], errors='ignore').copy()
    
    # --- CRITICAL FIX: Force all columns to be numeric (float) ---
    numerical_cols = stats_df.columns.tolist()
    for col in numerical_cols:
        # 'coerce' will turn any non-numeric entry (like a hyphen or empty string) into NaN
        stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce') 
        
    # Re-align and Drop rows with NaN values (removes players with corrupted stats)
    temp_df = pd.concat([player_names, stats_df], axis=1).dropna()
    
    # Extract the cleaned components
    cleaned_player_names = temp_df['Player Name']
    cleaned_stats_df = temp_df.drop(columns=['Player Name'])
    
    # Re-verify numerical columns after cleanup
    numerical_cols = cleaned_stats_df.select_dtypes(include=[np.number]).columns.tolist()

    # Perform Min-Max Scaling (Normalization) on cleaned data
    min_vals = cleaned_stats_df[numerical_cols].min()
    max_vals = cleaned_stats_df[numerical_cols].max()
    
    range_vals = max_vals - min_vals
    # Avoid division by zero
    range_vals[range_vals == 0] = 1 
    
    normalized_data = (cleaned_stats_df[numerical_cols] - min_vals) / range_vals
    
    # Recombine player names with normalized stats (resetting index for alignment)
    normalized_df = pd.concat([cleaned_player_names.reset_index(drop=True), normalized_data.reset_index(drop=True)], axis=1)
    
    # Return normalized data for calculation and original data for error checking
    return normalized_df, df 


def _get_statistical_match_logic(target_player_name: str, n: int = 3) -> list[dict]:
    """
    Core function to find the 'n' most statistically similar players 
    based on Euclidean distance across normalized metrics.
    """
    normalized_df, raw_df = load_and_preprocess_stats()
    
    if normalized_df is None:
        return [{"error": "Data loading failed or returned no valid players."}]
        
    # Find the target player's normalized row
    target_row = normalized_df[normalized_df['Player Name'].str.lower() == target_player_name.lower()]

    if target_row.empty:
         return [{"error": f"Target player '{target_player_name}' not found in the cleaned dataset. Check spelling."}]

    target_stats = target_row.iloc[0].drop('Player Name').values
    # EXPLICITLY ENSURE target_stats IS A 1D ARRAY
    target_stats = np.array(target_stats, dtype=np.float64).reshape(-1)
    
    # Exclude the target player from the comparison set
    comparison_df = normalized_df[normalized_df['Player Name'].str.lower() != target_player_name.lower()].copy()
    
    # Calculate Euclidean distance for all other players
    comparison_stats = comparison_df.iloc[:, 1:].values
    # EXPLICITLY ENSURE comparison_stats IS A 2D ARRAY (N rows, M columns)
    comparison_stats = np.array(comparison_stats, dtype=np.float64) 

    # Calculate the squared sum of differences (Squared Euclidean Distance)
    # This line is correct, but needs the input arrays to be guaranteed correct:
    squared_diff_sum = np.sum((comparison_stats - target_stats) ** 2, axis=1)
    
    # Final distance calculation: np.sqrt must receive an array
    distances = np.sqrt(squared_diff_sum)
    
    comparison_df['distance'] = distances
    top_matches = comparison_df.sort_values(by='distance').head(n)
    
    # Calculate similarity score (1.0 = max similarity)
    num_features = len(target_stats)
    max_possible_distance = np.sqrt(num_features) 
    
    top_matches['similarity_score'] = 1.0 - (top_matches['distance'] / max_possible_distance)
    
    results = top_matches[['Player Name', 'similarity_score']].to_dict('records')
    
    final_results = [
        {"player": row['Player Name'], "similarity_score": round(row['similarity_score'], 4)}
        for row in results
    ]
    
    return final_results

get_statistical_match_tool = tool(_get_statistical_match_logic)


if __name__ == '__main__':
    # Test cases now call the core logic function directly, NOT the tool object.
    print(f"--- Testing {STATS_FILE} ---")
    
    test_player = "Bukayo Saka"
    matches = _get_statistical_match_logic(test_player, n=3)
    print(f"\nStatistical Matches for {test_player}:")
    for match in matches:
        print(f"- {match['player']} (Score: {match['similarity_score']})")

    test_player_2 = "Declan Rice"
    matches_2 = _get_statistical_match_logic(test_player_2, n=3)
    print(f"\nStatistical Matches for {test_player_2}:")
    for match in matches_2:
        print(f"- {match['player']} (Score: {match['similarity_score']})")

    test_player_3 = "NonExistent Player"
    matches_3 = _get_statistical_match_logic(test_player_3, n=3)
    print(f"\nStatistical Matches for {test_player_3}:")
    print(matches_3)