import pandas as pd
import os
import re

def generate_placeholder_reports():
    """Reads player names from CSV and generates a simple text file for each."""
    
    csv_path = 'data/structured_stats.csv'
    report_dir = 'data/unstructured_reports/'
    
    # Check if the CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Create the reports directory if it doesn't exist (it should already exist)
    os.makedirs(report_dir, exist_ok=True)
    
    # Load player data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Read {len(df)} players from CSV. Generating reports...")

    # Iterate through each player to create a report file
    for index, row in df.iterrows():
        player_name = row['Player Name']
        
        # Format the name for the filename (lowercase, replace spaces with underscores)
        filename = re.sub(r'\s+', '_', player_name.lower()) + '_report.txt'
        file_path = os.path.join(report_dir, filename)
        
        # Create the report content using basic stats for context
        content = (
            f"Official Scout Report: {player_name} ({row['Team']}, {row['Position']}).\n\n"
            f"This player is a well-rounded contributor noted for their work rate and consistency. "
            f"They possess a high level of technical ability, often demonstrating a focus on creating chances (xA: {row['xA']}). "
            f"Their primary strengths include linking play through midfield (Passes per 90: {row['Passes per 90']}) and "
            f"off-the-ball movement. While their direct goal-scoring contribution (xG: {row['xG']}) is solid, "
            f"their value lies in enabling teammates. The player has a knack for tackling (Tackles per 90: {row['Tackles per 90']}) "
            f"but is not primarily a ball-winning specialist."
        )
        
        # Write the content to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    print(f"Successfully generated {len(df)} contextual report files in {report_dir}.")

if __name__ == "__main__":
    generate_placeholder_reports()