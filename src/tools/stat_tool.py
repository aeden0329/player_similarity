import pandas as pd
import numpy as np
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# --- Configuration ---
# Path to the CSV file containing the structured player statistics
STATS_FILE_PATH = "data/structured_stats.csv"

# Columns to use for the statistical similarity calculation (must match your CSV headers exactly)
STAT_COLUMNS = [
    'Goals', 
    'Assists', 
    'xG', 
    'xA', 
    'Passes per 90', 
    'Tackles per 90', 
    'Successful Dribbles'
]
# The column containing the unique player name/identifier
PLAYER_NAME_COLUMN = 'Player Name'

# --- 1. Data Loading and Preprocessing ---

def _load_and_preprocess_data(stats_file_path: str):
    """Loads, cleans, and normalizes the statistical data."""
    print(f"Loading data from: {stats_file_path}")
    try:
        # Load the raw data
        df = pd.read_csv(stats_file_path)
        
        # Select the statistical columns
        df_stats = df[STAT_COLUMNS]
        
        # Drop rows with NaN values in the key statistical columns
        df_cleaned = df.dropna(subset=STAT_COLUMNS).copy()
        
        # Use only the statistical columns for calculation
        df_stats_cleaned = df_cleaned[STAT_COLUMNS]

        # Normalize the data using Min-Max Scaling
        min_vals = df_stats_cleaned.min()
        max_vals = df_stats_cleaned.max()
        
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1 
        
        df_normalized = (df_stats_cleaned - min_vals) / range_vals

        # Re-attach the player name column and set it as the index
        df_normalized[PLAYER_NAME_COLUMN] = df_cleaned[PLAYER_NAME_COLUMN]
        df_normalized.set_index(PLAYER_NAME_COLUMN, inplace=True)
        
        return df_normalized

    except FileNotFoundError:
        print(f"Error: Statistical file not found at {stats_file_path}")
        return None
    except KeyError as e:
        print(f"Error: Missing column(s) in CSV. Check if STAT_COLUMNS match your file headers. Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data preprocessing: {e}")
        return None

# Load the data once when the tool script is imported
NORMALIZED_STATS_DF = _load_and_preprocess_data(STATS_FILE_PATH)

# --- 2. Tool Logic (Euclidean Distance Calculation) ---

def _calculate_euclidean_distance(target_player_stats: np.ndarray, all_player_stats: pd.DataFrame) -> List[Dict[str, Any]]:
    """Calculates the Euclidean distance between the target player and all other players."""
    
    # Calculate squared difference for all stats
    squared_diff = (all_player_stats.values - target_player_stats) ** 2
    
    # Sum the squared differences across all statistical columns (axis=1)
    sum_squared_diff = squared_diff.sum(axis=1)
    
    # Take the square root to get the Euclidean distance
    distances = np.sqrt(sum_squared_diff)
    
    # Convert distances to similarity scores: lower distance = higher similarity
    similarity_scores = 1 / (1 + distances)
    
    # Create a DataFrame for results
    results = pd.DataFrame({
        'player': all_player_stats.index,
        'similarity_score': similarity_scores
    })
    
    results = results.sort_values(by='similarity_score', ascending=False)
    
    return results.to_dict('records')


def _get_statistical_match_logic(player_name: str, num_matches: int = 3) -> str:
    """Core function to find the N most statistically similar players."""
    if NORMALIZED_STATS_DF is None:
        return '{"error": "Statistical data could not be loaded or processed. Check console for details."}'

    try:
        player_name_lower = player_name.lower()
        
        lower_index = NORMALIZED_STATS_DF.index.str.lower()
        target_loc = lower_index.get_loc(player_name_lower)
        
        target_player = NORMALIZED_STATS_DF.index[target_loc]
        target_stats = NORMALIZED_STATS_DF.loc[target_player].values
        
        comparison_df = NORMALIZED_STATS_DF.drop(target_player)
        
        all_matches = _calculate_euclidean_distance(target_stats, comparison_df)
        top_matches = all_matches[:num_matches]
        
        return str(top_matches) 

    except KeyError:
        return f'{{"error": "Player \'{player_name}\' not found in the statistical dataset. Please check the spelling."}}'
    except Exception as e:
        return f'{{"error": "An unexpected error occurred during similarity calculation: {e}"}}'


# --- 3. LangChain Tool Definition ---

class StatisticalMatchInput(BaseModel):
    """Input for the statistical matching tool."""
    player_name: str = Field(description="The full name of the player to find statistical matches for.")
    num_matches: int = Field(description="The number of top statistically similar players to return. Default is 3.", default=3)


@tool(args_schema=StatisticalMatchInput)
def get_statistical_match_tool(player_name: str, num_matches: int = 3) -> str:
    """
    Finds the N most statistically similar professional football players 
    based on normalized performance metrics using Euclidean distance.
    The tool returns a list of dictionaries as a string, e.g., '[{...}, {...}]'.
    """
    return _get_statistical_match_logic(player_name, num_matches)