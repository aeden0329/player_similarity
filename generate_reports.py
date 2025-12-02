import os
import random

# --- Configuration ---
REPORTS_DIR = "data/unstructured_reports"
NUM_REPORTS = 100
# ---

# List of 100 well-known or top-tier Premier League players (2024-2025 season included)
PLAYER_NAMES = [
    "Erling Haaland", "Mohamed Salah", "Kevin De Bruyne", "Bukayo Saka", 
    "Harry Kane", "Son Heung-min", "Virgil van Dijk", "Rodri", 
    "Phil Foden", "Declan Rice", "Bruno Fernandes", "Kylian Mbappe", # Using a highly-coveted player for variety
    "Ruben Dias", "Ollie Watkins", "Cole Palmer", "Eberechi Eze",
    "Alexander Isak", "Kai Havertz", "Anthony Gordon", "Dominik Szoboszlai",
    "Garnacho", "Julian Alvarez", "Bernardo Silva", "John Stones",
    "Gabriel Martinelli", "William Saliba", "Alisson Becker", "Ederson",
    "Andre Onana", "Vicario", "Sven Botman", "Josko Gvardiol",
    "Kieran Trippier", "Reece James", "Trent Alexander-Arnold", "Andy Robertson",
    "Luke Shaw", "Pau Torres", "Lucas Paqueta", "Douglas Luiz",
    "Pascal Gross", "Moises Caicedo", "Enzo Fernandez", "Alexis Mac Allister",
    "Martin Odegaard", "Bruno Guimaraes", "Emiliano Martinez", "Jarrod Bowen",
    "James Maddison", "Harvey Barnes", "Callum Wilson", "Darwin Nunez",
    "Cody Gakpo", "Diogo Jota", "Marcus Rashford", "Alejandro Garnacho",
    "Rasmus Hojlund", "Christopher Nkunku", "Ivan Toney", "Brennan Johnson",
    "Pedro Neto", "Matheus Cunha", "Yoane Wissa", "Bryan Mbeumo",
    "Leon Bailey", "Moussa Diaby", "Manuel Akanji", "Ezri Konsa",
    "Lewis Dunk", "Murillo", "Max Kilman", "Joao Gomes",
    "Pape Matar Sarr", "Yves Bissouma", "Tyler Adams", "Lewis Cook",
    "Vitaliy Mykolenko", "Destiny Udogie", "Tino Livramento", "Levi Colwill",
    "Marc Cucurella", "Malo Gusto", "Wataru Endo", "Romeo Lavia",
    "Curtis Jones", "Jacob Ramsey", "Michael Olise", "Simon Adingra",
    "Evan Ferguson", "Ben Doak", "Rico Lewis", "Kobbie Mainoo",
    "Carney Chukwuemeka", "Facundo Pellistri", "Gustavo Hamer", "Sandro Tonali",
    "Youri Tielemans", "Tariq Lamptey", "Sam Johnstone", "Nick Pope",
    "Bernd Leno", "Jose Sa", "Arijanet Muric", "Mark Flekken", 
    "Neto", "Alphonse Areola", "Matty Cash", "Ben White"
]

# Ensure we have exactly 100 players, padding or truncating if necessary
if len(PLAYER_NAMES) < NUM_REPORTS:
    PLAYER_NAMES.extend([f"Player_{i}" for i in range(len(PLAYER_NAMES), NUM_REPORTS)])
elif len(PLAYER_NAMES) > NUM_REPORTS:
    PLAYER_NAMES = PLAYER_NAMES[:NUM_REPORTS]


def generate_report_text(player_name):
    """Generates a structured, but 'unstructured' text report."""
    position_keywords = {
        "Forward": ["goalscoring instinct", "clinical finish", "runs behind", "xG overperformance", "shot accuracy", "first touch"],
        "Midfielder": ["engine room", "passing range", "progressive carries", "defensive contribution", "vision", "control the tempo"],
        "Defender": ["aerial dominance", "clearances", "tackling precision", "reading the game", "build-up play", "leadership at the back"],
        "Goalkeeper": ["shot-stopping ability", "commanding presence", "distribution skills", "saves percentage", "one-on-one"],
    }
    
    positions = list(position_keywords.keys())
    # Assign a position to the player
    position = random.choice(positions)
    
    # Generate some key statistics (mock data)
    goals = random.randint(0, 30)
    assists = random.randint(0, 20)
    tackles_p90 = round(random.uniform(0.5, 5.0), 1)
    
    # Select random descriptive phrases based on position
    descriptors = random.sample(position_keywords.get(position, []), k=3)
    
    # Construct the report
    report = f"""
    Player Name: {player_name}
    Position: {position}
    
    --- Scouting Report: 2024/2025 Season ---
    
    {player_name} has had a decisive season, displaying excellent {descriptors[0]} and a consistent {descriptors[1]}. 
    His work rate in the {position_keywords['Midfielder'][0]} is admirable, contributing defensively with an average of {tackles_p90} tackles per 90 minutes. 
    
    His offensive output is solid, tallying {goals} goals and {assists} assists this campaign. The analysis shows a high level of {descriptors[2]}, suggesting he has more to offer next season. 
    A notable highlight was his match against Man City where his {position_keywords['Defender'][1]} proved crucial.
    The primary area for improvement is his consistency against low-block defenses.

    Overall Assessment: A high-impact {position} with a strong blend of technical skill and physical presence.
    """
    return report.strip()


def create_reports():
    """Creates the data directory and writes the reports."""
    # 1. Create the directory if it doesn't exist
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        print(f"Created directory: {REPORTS_DIR}")

    print(f"Generating {NUM_REPORTS} player reports...")
    
    # 2. Generate and save reports
    for i, player in enumerate(PLAYER_NAMES):
        # Create a clean file name
        file_name = f"{player.lower().replace(' ', '_')}_report.txt"
        file_path = os.path.join(REPORTS_DIR, file_name)
        
        report_content = generate_report_text(player)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

    print(f"🎉 Successfully created {len(os.listdir(REPORTS_DIR))} reports in {REPORTS_DIR}.")

if __name__ == "__main__":
    create_reports()