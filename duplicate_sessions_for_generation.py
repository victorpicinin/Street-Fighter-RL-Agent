"""
Duplicate Training Sessions for Generation Training

This script:
1. Adds a 'generation_version' column to training_sessions table
2. For each existing session, creates 2 duplicate rows with new session_ids
3. Sets generation_version to "Gen 1" for all three rows (original + 2 duplicates)
4. Sets executed=False for the new duplicate rows
"""

import sqlite3
import uuid
from pathlib import Path
from training_db import TrainingDBSession
import json


def add_generation_version_column(db_path: str = "training_results.db"):
    """Add generation_version column to training_sessions table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("ALTER TABLE training_sessions ADD COLUMN generation_version TEXT DEFAULT NULL")
        conn.commit()
        print("✓ Added 'generation_version' column to training_sessions table")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("✓ Column 'generation_version' already exists")
        else:
            raise
    finally:
        conn.close()


def get_all_sessions(db_path: str = "training_results.db"):
    """Get all sessions from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all column names
    cursor.execute("PRAGMA table_info(training_sessions)")
    columns_info = cursor.fetchall()
    all_columns = [col[1] for col in columns_info]
    
    # Exclude generation_version from selection (we'll handle it separately)
    select_columns = [col for col in all_columns if col != 'generation_version']
    select_columns_str = ', '.join(select_columns)
    
    query = f"SELECT {select_columns_str} FROM training_sessions ORDER BY created_at"
    cursor.execute(query)
    rows = cursor.fetchall()
    
    # Build list of dictionaries
    sessions = []
    for row in rows:
        session_dict = {}
        for col, value in zip(select_columns, row):
            session_dict[col] = value
        sessions.append(session_dict)
    
    conn.close()
    return sessions, all_columns


def duplicate_sessions_for_generation(db_path: str = "training_results.db", generation: str = "Gen 1"):
    """
    Duplicate each existing session 2 times with new session_ids.
    All three sessions (original + 2 duplicates) will have the same generation_version.
    
    Args:
        db_path: Path to database file
        generation: Generation version string (default: "Gen 1")
    """
    # Add column if it doesn't exist
    add_generation_version_column(db_path)
    
    # Get all sessions
    sessions, all_columns = get_all_sessions(db_path)
    
    if not sessions:
        print("No sessions found in database. Nothing to duplicate.")
        return
    
    print(f"Found {len(sessions)} session(s) to duplicate")
    print(f"Will create 2 duplicates for each, resulting in {len(sessions) * 3} total sessions")
    print(f"All will be marked as generation_version = '{generation}'")
    print()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")
    
    duplicates_created = 0
    
    for idx, session in enumerate(sessions, 1):
        original_session_id = session['session_id']
        print(f"Processing session {idx}/{len(sessions)}: {original_session_id[:8]}...")
        
        # Update original session with generation_version
        try:
            cursor.execute(
                "UPDATE training_sessions SET generation_version = ? WHERE session_id = ?",
                (generation, original_session_id)
            )
            print(f"  ✓ Updated original session with generation_version='{generation}'")
        except Exception as e:
            print(f"  ⚠ Warning: Could not update original session: {e}")
        
        # Create 2 duplicates
        for dup_num in range(1, 3):
            # Generate new session_id
            new_session_id = str(uuid.uuid4())
            
            # Create new session_path based on new session_id
            original_path = session.get('session_path', f'session_{original_session_id[:8]}')
            new_session_path = Path(f'session_{new_session_id[:8]}')
            
            # Prepare columns and values for INSERT
            # We need to exclude session_id and session_path from the copy (we'll use new ones)
            # Also exclude created_at (will use CURRENT_TIMESTAMP) and updated_at
            exclude_cols = {'session_id', 'session_path', 'created_at', 'updated_at', 'generation_version'}
            
            insert_columns = []
            insert_values = []
            placeholders = []
            
            # Add new session_id and session_path first
            insert_columns.append('session_id')
            insert_values.append(new_session_id)
            placeholders.append('?')
            
            insert_columns.append('session_path')
            insert_values.append(str(new_session_path))
            placeholders.append('?')
            
            # Copy all other columns from original session
            for col in all_columns:
                if col not in exclude_cols:
                    insert_columns.append(col)
                    value = session.get(col)
                    # Handle None values
                    if value is None:
                        insert_values.append(None)
                    else:
                        insert_values.append(value)
                    placeholders.append('?')
            
            # Add generation_version
            insert_columns.append('generation_version')
            insert_values.append(generation)
            placeholders.append('?')
            
            # Set executed=False for duplicates
            if 'executed' in insert_columns:
                executed_idx = insert_columns.index('executed')
                insert_values[executed_idx] = 0
            else:
                insert_columns.append('executed')
                insert_values.append(0)
                placeholders.append('?')
            
            # Build and execute INSERT query
            columns_str = ', '.join(insert_columns)
            placeholders_str = ', '.join(placeholders)
            
            query = f"""
                INSERT INTO training_sessions ({columns_str})
                VALUES ({placeholders_str})
            """
            
            try:
                cursor.execute(query, insert_values)
                duplicates_created += 1
                print(f"  ✓ Created duplicate {dup_num}: {new_session_id[:8]}")
            except Exception as e:
                print(f"  ✗ Error creating duplicate {dup_num}: {e}")
                conn.rollback()
                continue
    
    conn.commit()
    conn.close()
    
    print()
    print("=" * 70)
    print(f"✓ Completed! Created {duplicates_created} duplicate sessions")
    print(f"✓ All sessions now have generation_version = '{generation}'")
    print(f"✓ Original sessions: {len(sessions)}")
    print(f"✓ Total sessions after duplication: {len(sessions) + duplicates_created}")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "training_results.db"
    generation = sys.argv[2] if len(sys.argv) > 2 else "Gen 1"
    
    print("=" * 70)
    print("Duplicate Training Sessions for Generation Training")
    print("=" * 70)
    print(f"Database: {db_path}")
    print(f"Generation: {generation}")
    print()
    
    duplicate_sessions_for_generation(db_path, generation)

