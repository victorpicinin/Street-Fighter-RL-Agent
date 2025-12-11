"""
Streamlit Training Dashboard
Visualizes training results from the SQLite database with interactive filtering and comprehensive charts.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import training_db

# Page configuration
st.set_page_config(
    page_title="Street Fighter Training Dashboard",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Action name mapping (from StreetFighter_Env2026.py)
ACTION_NAMES = {
    0: "UP_ARROW",
    1: "DOWN_ARROW",
    2: "LEFT_ARROW",
    3: "RIGHT_ARROW",
    4: "PUNCH",
    5: "KICK",
    6: "HURRICANE_KICK",
    7: "HADOUKEN"
}

# Database path
DB_PATH = "training_results.db"


@st.cache_data(ttl=1)  # Cache for 1 second to allow manual refresh
def load_fight_results(session_ids: List[str], iteration_range: Tuple[int, int], db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Load fight results from database filtered by session IDs and iteration range.
    
    Args:
        session_ids: List of session IDs to filter by
        iteration_range: Tuple of (min_iter, max_iter)
        db_path: Path to database file
        
    Returns:
        DataFrame with fight results
    """
    if not session_ids or not Path(db_path).exists():
        return pd.DataFrame()
    
    try:
        with training_db.TrainingDBSession(db_path) as db:
            conn = db.conn
            placeholders = ','.join(['?'] * len(session_ids))
            min_iter, max_iter = iteration_range
            
            query = f"""
                SELECT * FROM fight_results
                WHERE session_id IN ({placeholders})
                AND iteration >= ?
                AND iteration <= ?
                ORDER BY iteration, fight_num
            """
            
            params = list(session_ids) + [min_iter, max_iter]
            df = pd.read_sql_query(query, conn, params=params)
            return df
    except Exception as e:
        st.error(f"Error loading fight results: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)  # Cache for 60 seconds (sessions don't change often)
def get_available_sessions(db_path: str = DB_PATH) -> Tuple[List[str], Dict[str, str]]:
    """
    Get list of all available session IDs from training_sessions table with aliases.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Tuple of (list of session IDs ordered by created_at, dict mapping session_id to alias)
    """
    if not Path(db_path).exists():
        return [], {}
    
    try:
        with training_db.TrainingDBSession(db_path) as db:
            conn = db.conn
            query = "SELECT session_id, created_at FROM training_sessions ORDER BY created_at ASC"
            df = pd.read_sql_query(query, conn)
            
            session_ids = df['session_id'].tolist()
            # Create aliases: Session 1, Session 2, etc. based on creation order
            alias_map = {session_id: f"Session {idx + 1}" for idx, session_id in enumerate(session_ids)}
            
            return session_ids, alias_map
    except Exception as e:
        st.error(f"Error loading sessions: {e}")
        return [], {}


def get_session_alias(session_id: str, alias_map: Dict[str, str]) -> str:
    """
    Get alias for a session ID, or return shortened ID if not found.
    
    Args:
        session_id: Session ID
        alias_map: Dictionary mapping session_id to alias
        
    Returns:
        Alias or shortened session ID
    """
    return alias_map.get(session_id, session_id[:8])


def get_iteration_range(session_ids: List[str], db_path: str = DB_PATH) -> Tuple[int, int]:
    """
    Get min and max iteration values for selected sessions.
    
    Args:
        session_ids: List of session IDs
        db_path: Path to database file
        
    Returns:
        Tuple of (min_iter, max_iter)
    """
    if not session_ids or not Path(db_path).exists():
        return (0, 100)
    
    try:
        with training_db.TrainingDBSession(db_path) as db:
            conn = db.conn
            placeholders = ','.join(['?'] * len(session_ids))
            query = f"""
                SELECT MIN(iteration) as min_iter, MAX(iteration) as max_iter
                FROM fight_results
                WHERE session_id IN ({placeholders})
            """
            df = pd.read_sql_query(query, conn, params=list(session_ids))
            if df.empty or df['min_iter'].iloc[0] is None:
                return (0, 100)
            return (int(df['min_iter'].iloc[0]), int(df['max_iter'].iloc[0]))
    except Exception as e:
        st.error(f"Error getting iteration range: {e}")
        return (0, 100)


def calculate_action_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average damage per action.
    
    Args:
        df: DataFrame with fight results
        
    Returns:
        DataFrame with action names and average damage
    """
    if df.empty:
        return pd.DataFrame()
    
    action_data = []
    for action_num in range(8):
        action_col = f'action_{action_num}'
        damage_col = f'damage_{action_num}'
        
        if action_col in df.columns and damage_col in df.columns:
            # Filter rows where action was used (action count > 0)
            used_mask = df[action_col] > 0
            if used_mask.any():
                # Calculate average damage per action usage
                total_damage = df.loc[used_mask, damage_col].sum()
                total_actions = df.loc[used_mask, action_col].sum()
                
                if total_actions > 0:
                    avg_damage = total_damage / total_actions
                    action_data.append({
                        'action_num': action_num,
                        'action_name': ACTION_NAMES[action_num],
                        'avg_damage': avg_damage,
                        'total_uses': int(total_actions),
                        'total_damage': total_damage
                    })
    
    if action_data:
        return pd.DataFrame(action_data).sort_values('avg_damage', ascending=False)
    return pd.DataFrame()


def _create_donut_chart(df: pd.DataFrame, session_id: str, session_aliases: Dict[str, str], 
                         title: str, is_overall: bool = True):
    """
    Helper function to create a donut chart for win/loss distribution.
    
    Args:
        df: DataFrame with fight results
        session_id: Session ID
        session_aliases: Dictionary mapping session IDs to aliases
        title: Chart title
        is_overall: If True, use all data. If False, use last 5 iterations only.
    """
    session_alias = get_session_alias(session_id, session_aliases)
    st.markdown(f"<center><b>{session_alias}</b></center>", unsafe_allow_html=True)
    
    if not is_overall:
        # Get last 5 iterations
        max_iteration = df['iteration'].max()
        df = df[df['iteration'] > (max_iteration - 5)]
        
        if df.empty:
            st.info("No data for last 5 iterations")
            return
    
    wins = len(df[df['result'] == 1])
    losses = len(df[df['result'] == 0])
    incomplete = len(df[df['result'] == 2])
    
    pie_data = {
        'Result': [],
        'Count': [],
    }
    
    if wins > 0:
        pie_data['Result'].append('Wins')
        pie_data['Count'].append(wins)
    
    if losses > 0:
        pie_data['Result'].append('Losses')
        pie_data['Count'].append(losses)
    
    if incomplete > 0:
        pie_data['Result'].append('Incomplete')
        pie_data['Count'].append(incomplete)
    
    if pie_data['Count']:
        pie_df = pd.DataFrame(pie_data)
        fig_pie = px.pie(
            pie_df,
            values='Count',
            names='Result',
            title=title,
            color='Result',
            color_discrete_map={'Wins': 'green', 'Losses': 'red', 'Incomplete': 'gray'},
            hole=0.4  # Creates donut chart
        )
        # Increase height for better visibility
        fig_pie.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_pie, width='stretch')
    else:
        st.info("No data available")


def main():
    st.title("🥊 Street Fighter Training Dashboard")
    st.markdown("---")
    
    # Sidebar - Filters
    with st.sidebar:
        st.header("Filters")
        
        # Check if database exists
        if not Path(DB_PATH).exists():
            st.error(f"Database not found: {DB_PATH}")
            st.stop()
        
        # Session filter
        available_sessions, session_aliases = get_available_sessions()
        if not available_sessions:
            st.warning("No training sessions found in database.")
            st.stop()
        
        # Initialize session state for selected sessions
        if 'selected_sessions' not in st.session_state:
            # Default to first session selected
            st.session_state.selected_sessions = [available_sessions[0]] if available_sessions else []
        
        st.markdown("**Select Sessions**")
        st.markdown("---")
        
        # Create buttons for each session
        selected_sessions = []
        for session_id in available_sessions:
            session_alias = get_session_alias(session_id, session_aliases)
            is_selected = session_id in st.session_state.selected_sessions
            
            # Create button with appropriate type
            button_type = "primary" if is_selected else "secondary"
            button_label = f"{session_alias} ({session_id[:8]})"
            
            if st.button(
                button_label,
                key=f"session_btn_{session_id}",
                type=button_type,
                width='stretch'
            ):
                # Toggle selection on click
                if is_selected:
                    # Remove from selection
                    st.session_state.selected_sessions = [
                        sid for sid in st.session_state.selected_sessions if sid != session_id
                    ]
                else:
                    # Add to selection
                    st.session_state.selected_sessions.append(session_id)
                st.rerun()
            
            # Track selected sessions
            if session_id in st.session_state.selected_sessions:
                selected_sessions.append(session_id)
        
        # Get iteration range for selected sessions
        if selected_sessions:
            min_iter, max_iter = get_iteration_range(selected_sessions)
            
            # Iteration range slider
            iteration_range = st.slider(
                "Iteration Range",
                min_value=min_iter,
                max_value=max_iter,
                value=(min_iter, max_iter),
                help="Select range of iterations to analyze"
            )
        else:
            iteration_range = (0, 100)
            st.info("Please select at least one session")
        
        st.markdown("---")
        
        # Manual refresh button
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    if not selected_sessions:
        st.info("👈 Please select at least one session from the sidebar to view data.")
        return
    
    df = load_fight_results(selected_sessions, iteration_range)
    
    if df.empty:
        st.warning("No data found for the selected filters. Try adjusting your session or iteration range.")
        return
    
    # Get session aliases for display
    _, session_aliases = get_available_sessions()
    
    # Check if multiple sessions selected for comparison
    num_sessions = len(selected_sessions)
    
    if num_sessions == 1:
        # Single session view - original layout
        session_alias = get_session_alias(selected_sessions[0], session_aliases)
        display_single_session(selected_sessions[0], df, session_alias)
    else:
        # Multiple sessions - side by side comparison
        display_multiple_sessions(selected_sessions, iteration_range, session_aliases)


def display_single_session(session_id: str, df: pd.DataFrame, session_alias: str):
    """Display dashboard for a single session."""
    st.header(f"📊 {session_alias}")
    
    # Section 1: Summary Statistics (KPIs)
    st.subheader("Summary Statistics")
    
    total_fights = len(df)
    wins = len(df[df['result'] == 1])
    losses = len(df[df['result'] == 0])
    win_rate = (wins / total_fights * 100) if total_fights > 0 else 0
    
    avg_fight_time = df['fight_time'].mean()
    avg_reward = df['total_reward'].mean()
    
    # Calculate total damage dealt (sum of all damage_N columns)
    damage_columns = [f'damage_{i}' for i in range(8)]
    total_damage_dealt = df[damage_columns].sum(axis=1).sum()
    avg_damage_dealt = df[damage_columns].sum(axis=1).mean()
    
    avg_damage_taken = df['damage_taken'].mean()
    total_stun_time = df['stun_time'].sum()
    
    # Create summary statistics table
    stats_data = {
        'Category': [
            'Total Fights',
            'Win Rate',
            'Avg Fight Time',
            'Avg Reward',
            'Avg Damage Dealt',
            'Avg Damage Taken',
            'Total Stun Time'
        ],
        'Value': [
            f"{total_fights:,}",
            f"{win_rate:.1f}%",
            f"{avg_fight_time:.1f}s",
            f"{avg_reward:.1f}",
            f"{avg_damage_dealt:.1f}",
            f"{avg_damage_taken:.1f}",
            f"{total_stun_time:,}"
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    st.table(stats_df)
    
    # Add win/loss pie charts (overall and last 5 iterations)
    st.subheader("🏆 Win/Loss Distribution")
    col_pie_overall, col_pie_recent = st.columns(2)
    
    with col_pie_overall:
        st.markdown("**Overall**")
        wins = len(df[df['result'] == 1])
        losses = len(df[df['result'] == 0])
        incomplete = len(df[df['result'] == 2])
        
        pie_data = {
            'Result': [],
            'Count': [],
        }
        
        if wins > 0:
            pie_data['Result'].append('Wins')
            pie_data['Count'].append(wins)
        
        if losses > 0:
            pie_data['Result'].append('Losses')
            pie_data['Count'].append(losses)
        
        if incomplete > 0:
            pie_data['Result'].append('Incomplete')
            pie_data['Count'].append(incomplete)
        
        if pie_data['Count']:
            pie_df = pd.DataFrame(pie_data)
            fig_pie = px.pie(
                pie_df,
                values='Count',
                names='Result',
                title="Overall Win/Loss",
                color='Result',
                color_discrete_map={'Wins': 'green', 'Losses': 'red', 'Incomplete': 'gray'},
                hole=0.4  # Creates donut chart
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, width='stretch')
    
    with col_pie_recent:
        st.markdown("**Last 5 Iterations**")
        # Get last 5 iterations
        max_iteration = df['iteration'].max()
        last_5_iters_df = df[df['iteration'] > (max_iteration - 5)]
        
        if not last_5_iters_df.empty:
            wins_recent = len(last_5_iters_df[last_5_iters_df['result'] == 1])
            losses_recent = len(last_5_iters_df[last_5_iters_df['result'] == 0])
            incomplete_recent = len(last_5_iters_df[last_5_iters_df['result'] == 2])
            
            pie_data_recent = {
                'Result': [],
                'Count': [],
            }
            
            if wins_recent > 0:
                pie_data_recent['Result'].append('Wins')
                pie_data_recent['Count'].append(wins_recent)
            
            if losses_recent > 0:
                pie_data_recent['Result'].append('Losses')
                pie_data_recent['Count'].append(losses_recent)
            
            if incomplete_recent > 0:
                pie_data_recent['Result'].append('Incomplete')
                pie_data_recent['Count'].append(incomplete_recent)
            
            if pie_data_recent['Count']:
                pie_df_recent = pd.DataFrame(pie_data_recent)
                fig_pie_recent = px.pie(
                    pie_df_recent,
                    values='Count',
                    names='Result',
                    title="Last 5 Iterations Win/Loss",
                    color='Result',
                    color_discrete_map={'Wins': 'green', 'Losses': 'red', 'Incomplete': 'gray'},
                    hole=0.4  # Creates donut chart
                )
                fig_pie_recent.update_layout(height=400)
                st.plotly_chart(fig_pie_recent, width='stretch')
        else:
            st.info("No data for last 5 iterations")
    
    st.markdown("---")
    
    # Aggregate data by iteration
    action_cols = [f'action_{i}' for i in range(8)]
    damage_cols = [f'damage_{i}' for i in range(8)]
    
    iteration_stats = df.groupby('iteration').agg({
        'total_reward': 'mean',
        'fight_time': 'mean',
        'damage_taken': 'mean',
        'stun_time': 'sum',
        **{col: 'sum' for col in action_cols},
        **{col: 'sum' for col in damage_cols}
    }).reset_index()
    
    # Calculate total damage dealt per iteration
    iteration_stats['damage_dealt'] = iteration_stats[damage_cols].sum(axis=1)
    
    # Generate all charts for single session
    generate_all_charts(iteration_stats, df)


def generate_all_charts(iteration_stats: pd.DataFrame, df: pd.DataFrame):
    """Generate all charts for a session."""
    # Section 2: Action Count Stacked Bar Chart
    st.header("🎮 Action Usage Over Iterations")
    
    view_mode = st.radio(
        "View Mode",
        ["Counts", "Percentages"],
        index=1,  # Default to "Percentages"
        horizontal=True,
        help="Switch between absolute counts and percentage proportions",
        key=f"view_mode_{hash(str(df['session_id'].iloc[0]) if 'session_id' in df.columns else 'default')}"
    )
    
    # Prepare data for stacked bar chart
    action_data_for_chart = []
    for iter_val in iteration_stats['iteration']:
        row = iteration_stats[iteration_stats['iteration'] == iter_val].iloc[0]
        total_actions = sum(row[f'action_{i}'] for i in range(8))
        
        if total_actions > 0:
            for action_num in range(8):
                action_count = row[f'action_{action_num}']
                if view_mode == "Percentages":
                    value = (action_count / total_actions * 100) if total_actions > 0 else 0
                else:
                    value = action_count
                
                action_data_for_chart.append({
                    'iteration': iter_val,
                    'action': ACTION_NAMES[action_num],
                    'value': value
                })
    
    if action_data_for_chart:
        action_df = pd.DataFrame(action_data_for_chart)
        
        fig = px.bar(
            action_df,
            x='iteration',
            y='value',
            color='action',
            title=f"Action Usage by Iteration ({view_mode})",
            labels={'value': f'Action Count ({view_mode})', 'iteration': 'Iteration'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        # 20% bigger vertically: 400 * 1.2 = 480
        fig.update_layout(
            barmode='stack',
            height=480,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.5,  # Move down like 4 paragraphs
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No action data available for the selected range.")
    
    st.markdown("---")
    
    # Section 3: Damage Analysis Bar Chart
    st.header("⚔️ Damage Analysis")
    
    fig_damage = go.Figure()
    
    # Add bar charts
    fig_damage.add_trace(go.Bar(
        x=iteration_stats['iteration'],
        y=iteration_stats['damage_dealt'],
        name='Damage Dealt',
        marker_color='green'
    ))
    
    fig_damage.add_trace(go.Bar(
        x=iteration_stats['iteration'],
        y=iteration_stats['damage_taken'],
        name='Damage Taken',
        marker_color='red'
    ))
    
    # Add average lines
    avg_damage_dealt = iteration_stats['damage_dealt'].mean()
    avg_damage_taken = iteration_stats['damage_taken'].mean()
    
    fig_damage.add_trace(go.Scatter(
        x=iteration_stats['iteration'],
        y=[avg_damage_dealt] * len(iteration_stats),
        mode='lines',
        name=f'Avg Dealt ({avg_damage_dealt:.1f})',
        line=dict(color='red', width=2)
    ))
    
    fig_damage.add_trace(go.Scatter(
        x=iteration_stats['iteration'],
        y=[avg_damage_taken] * len(iteration_stats),
        mode='lines',
        name=f'Avg Taken ({avg_damage_taken:.1f})',
        line=dict(color='red', width=2)
    ))
    
    # Add trend lines
    if len(iteration_stats) > 1:
        # Trend line for damage dealt
        z_dealt = np.polyfit(iteration_stats['iteration'], iteration_stats['damage_dealt'], 1)
        p_dealt = np.poly1d(z_dealt)
        fig_damage.add_trace(go.Scatter(
            x=iteration_stats['iteration'],
            y=p_dealt(iteration_stats['iteration']),
            mode='lines',
            name='Trend Dealt',
            line=dict(color='#90EE90', width=2)  # Light pastel green
        ))
        
        # Trend line for damage taken
        z_taken = np.polyfit(iteration_stats['iteration'], iteration_stats['damage_taken'], 1)
        p_taken = np.poly1d(z_taken)
        fig_damage.add_trace(go.Scatter(
            x=iteration_stats['iteration'],
            y=p_taken(iteration_stats['iteration']),
            mode='lines',
            name='Trend Taken',
            line=dict(color='#FFB6C1', width=2)  # Light pastel red
        ))
    
    fig_damage.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Damage",
        height=400,
        barmode='group',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.42,
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig_damage, width='stretch')
    
    st.markdown("---")
    
    # Section 4 & 5 & 6: Multiple line plots
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("⏱️ Average Fight Time")
        fig_time = px.bar(
            iteration_stats,
            x='iteration',
            y='fight_time',
            labels={'fight_time': 'Fight Time (seconds)', 'iteration': 'Iteration'}
        )
        avg_fight_time = iteration_stats['fight_time'].mean()
        fig_time.add_trace(go.Scatter(
            x=iteration_stats['iteration'],
            y=[avg_fight_time] * len(iteration_stats),
            mode='lines',
            name=f'Average ({avg_fight_time:.1f}s)',
            line=dict(color='red', width=2)
        ))
        # Add trend line
        if len(iteration_stats) > 1:
            z = np.polyfit(iteration_stats['iteration'], iteration_stats['fight_time'], 1)
            p = np.poly1d(z)
            fig_time.add_trace(go.Scatter(
                x=iteration_stats['iteration'],
                y=p(iteration_stats['iteration']),
                mode='lines',
                name='Trend',
                line=dict(color='green', width=2)
            ))
        fig_time.update_layout(
            height=350,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.42,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_time, width='stretch')
        
        st.subheader("🎯 Average Reward")
        fig_reward = px.bar(
            iteration_stats,
            x='iteration',
            y='total_reward',
            labels={'total_reward': 'Reward', 'iteration': 'Iteration'},
            color_discrete_sequence=['#D3D3D3']
        )
        avg_reward_val = iteration_stats['total_reward'].mean()
        fig_reward.add_trace(go.Scatter(
            x=iteration_stats['iteration'],
            y=[avg_reward_val] * len(iteration_stats),
            mode='lines',
            name=f'Average ({avg_reward_val:.1f})',
            line=dict(color='red', width=2)
        ))
        # Add trend line
        if len(iteration_stats) > 1:
            z = np.polyfit(iteration_stats['iteration'], iteration_stats['total_reward'], 1)
            p = np.poly1d(z)
            fig_reward.add_trace(go.Scatter(
                x=iteration_stats['iteration'],
                y=p(iteration_stats['iteration']),
                mode='lines',
                name='Trend',
                line=dict(color='green', width=2)
            ))
        fig_reward.update_layout(
            height=350,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.42,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_reward, width='stretch')
    
    with col_right:
        st.subheader("😵 Stun Time")
        fig_stun = px.bar(
            iteration_stats,
            x='iteration',
            y='stun_time',
            labels={'stun_time': 'Stun Time (ticks)', 'iteration': 'Iteration'},
            color_discrete_sequence=['#FFE5B4']
        )
        avg_stun_time = iteration_stats['stun_time'].mean()
        fig_stun.add_trace(go.Scatter(
            x=iteration_stats['iteration'],
            y=[avg_stun_time] * len(iteration_stats),
            mode='lines',
            name=f'Average ({avg_stun_time:.1f})',
            line=dict(color='red', width=2)
        ))
        # Add trend line
        if len(iteration_stats) > 1:
            z = np.polyfit(iteration_stats['iteration'], iteration_stats['stun_time'], 1)
            p = np.poly1d(z)
            fig_stun.add_trace(go.Scatter(
                x=iteration_stats['iteration'],
                y=p(iteration_stats['iteration']),
                mode='lines',
                name='Trend',
                line=dict(color='green', width=2)
            ))
        fig_stun.update_layout(
            height=350,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.42,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_stun, width='stretch')
        
        # Win/Loss ratio chart
        win_loss_by_iter = df.groupby('iteration')['result'].agg(['count', lambda x: (x == 1).sum()]).reset_index()
        win_loss_by_iter.columns = ['iteration', 'total_fights', 'wins']
        win_loss_by_iter['win_rate'] = (win_loss_by_iter['wins'] / win_loss_by_iter['total_fights'] * 100)
        
        st.subheader("📈 Win Rate")
        fig_winrate = px.bar(
            win_loss_by_iter,
            x='iteration',
            y='win_rate',
            labels={'win_rate': 'Win Rate (%)', 'iteration': 'Iteration'},
            color_discrete_sequence=['#90EE90']
        )
        avg_win_rate = win_loss_by_iter['win_rate'].mean()
        fig_winrate.add_trace(go.Scatter(
            x=win_loss_by_iter['iteration'],
            y=[avg_win_rate] * len(win_loss_by_iter),
            mode='lines',
            name=f'Average ({avg_win_rate:.1f}%)',
            line=dict(color='red', width=2)
        ))
        # Add trend line
        if len(win_loss_by_iter) > 1:
            z = np.polyfit(win_loss_by_iter['iteration'], win_loss_by_iter['win_rate'], 1)
            p = np.poly1d(z)
            fig_winrate.add_trace(go.Scatter(
                x=win_loss_by_iter['iteration'],
                y=p(win_loss_by_iter['iteration']),
                mode='lines',
                name='Trend',
                line=dict(color='green', width=2)
            ))
        # Allow negative values on y-axis
        fig_winrate.update_layout(
            height=350,
            yaxis=dict(range=None),  # Allow automatic range including negative values
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.42,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_winrate, width='stretch')
    
    st.markdown("---")
    
    # Section 7: Average Damage per Action Bar Chart
    st.header("💥 Average Damage per Action")
    
    action_avg_df = calculate_action_averages(df)
    
    if not action_avg_df.empty:
        fig_action_damage = px.bar(
            action_avg_df,
            x='action_name',
            y='avg_damage',
            labels={'avg_damage': 'Average Damage per Use', 'action_name': 'Action'},
            color='avg_damage',
            color_continuous_scale='YlOrRd',
            text='avg_damage'
        )
        fig_action_damage.update_traces(
            texttemplate='%{text:.1f}',
            textposition='outside',
            textfont=dict(color='white', size=12)
        )
        # 30% bigger vertically: 400 * 1.3 = 520
        fig_action_damage.update_layout(
            height=520,
            xaxis_tickangle=-45,
            coloraxis_colorbar=dict(
                orientation="h",
                yanchor="top",
                y=-0.5,  # Move down like 4 paragraphs
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_action_damage, width='stretch')
        
        # Show detailed stats
        with st.expander("📋 Detailed Action Statistics"):
            st.dataframe(
                action_avg_df[['action_name', 'avg_damage', 'total_uses', 'total_damage']].style.format({
                    'avg_damage': '{:.2f}',
                    'total_uses': '{:,.0f}',
                    'total_damage': '{:.2f}'
                }),
                width='stretch'
            )
    else:
        st.info("No action damage data available for the selected range.")


def display_multiple_sessions(session_ids: List[str], iteration_range: Tuple[int, int], session_aliases: Dict[str, str]):
    """Display side-by-side comparison for multiple sessions."""
    st.header("🔀 Multi-Session Comparison")
    
    # Create comparison title with aliases
    alias_list = [get_session_alias(sid, session_aliases) for sid in session_ids]
    st.markdown(f"Comparing: {', '.join(alias_list)}")
    st.markdown("---")
    
    # Load data for each session
    session_data = {}
    for session_id in session_ids:
        session_df = load_fight_results([session_id], iteration_range)
        if not session_df.empty:
            session_data[session_id] = session_df
    
    if not session_data:
        st.warning("No data available for the selected sessions.")
        return
    
    # Display summary stats for all sessions as tables
    st.subheader("📊 Summary Statistics Comparison")
    
    # Prepare data for comparison table
    comparison_data = {'Category': [
        'Total Fights',
        'Win Rate',
        'Avg Fight Time',
        'Avg Reward',
        'Avg Damage Dealt',
        'Avg Damage Taken'
    ]}
    
    for session_id, df in session_data.items():
        session_alias = get_session_alias(session_id, session_aliases)
        total_fights = len(df)
        wins = len(df[df['result'] == 1])
        win_rate = (wins / total_fights * 100) if total_fights > 0 else 0
        avg_fight_time = df['fight_time'].mean()
        avg_reward = df['total_reward'].mean()
        damage_columns = [f'damage_{i}' for i in range(8)]
        avg_damage_dealt = df[damage_columns].sum(axis=1).mean()
        avg_damage_taken = df['damage_taken'].mean()
        
        comparison_data[session_alias] = [
            f"{total_fights:,}",
            f"{win_rate:.1f}%",
            f"{avg_fight_time:.1f}s",
            f"{avg_reward:.1f}",
            f"{avg_damage_dealt:.1f}",
            f"{avg_damage_taken:.1f}"
        ]
    
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)
    
    st.markdown("---")
    
    # Add win/loss pie charts for each session (overall and last 5 iterations)
    st.subheader("🏆 Win/Loss Distribution")
    
    # Limit columns per row to prevent charts from being too small
    max_cols_per_row = 3
    num_sessions = len(session_data)
    
    # Overall win/loss - display in rows if more than max_cols_per_row
    st.markdown("**Overall**")
    
    if num_sessions <= max_cols_per_row:
        # Single row
        pie_cols_overall = st.columns(num_sessions)
        for idx, (session_id, df) in enumerate(session_data.items()):
            with pie_cols_overall[idx]:
                _create_donut_chart(df, session_id, session_aliases, "Overall Win/Loss", is_overall=True)
    else:
        # Multiple rows
        session_items_list = list(session_data.items())
        for row_start in range(0, num_sessions, max_cols_per_row):
            row_end = min(row_start + max_cols_per_row, num_sessions)
            pie_cols_row = st.columns(max_cols_per_row)
            for col_idx in range(max_cols_per_row):
                if row_start + col_idx < num_sessions:
                    session_id, df = session_items_list[row_start + col_idx]
                    with pie_cols_row[col_idx]:
                        _create_donut_chart(df, session_id, session_aliases, "Overall Win/Loss", is_overall=True)
    
    st.markdown("---")
    
    # Last 5 iterations win/loss
    st.markdown("**Last 5 Iterations**")
    
    if num_sessions <= max_cols_per_row:
        # Single row
        pie_cols_recent = st.columns(num_sessions)
        for idx, (session_id, df) in enumerate(session_data.items()):
            with pie_cols_recent[idx]:
                _create_donut_chart(df, session_id, session_aliases, "Last 5 Iterations", is_overall=False)
    else:
        # Multiple rows
        session_items_list = list(session_data.items())
        for row_start in range(0, num_sessions, max_cols_per_row):
            row_end = min(row_start + max_cols_per_row, num_sessions)
            pie_cols_row = st.columns(max_cols_per_row)
            for col_idx in range(max_cols_per_row):
                if row_start + col_idx < num_sessions:
                    session_id, df = session_items_list[row_start + col_idx]
                    with pie_cols_row[col_idx]:
                        _create_donut_chart(df, session_id, session_aliases, "Last 5 Iterations", is_overall=False)
    
    st.markdown("---")
    
    # Display charts for each session side by side
    chart_types = [
        ('damage', '⚔️ Damage Analysis Comparison'),
        ('reward', '🎯 Average Reward Comparison'),
        ('fight_time', '⏱️ Average Fight Time Comparison'),
        ('win_rate', '📈 Win Rate Comparison'),
        ('stun_time', '😵 Stun Time Comparison'),
        ('action_usage', '🎮 Action Usage Comparison'),
        ('action_damage', '💥 Average Damage per Action Comparison')
    ]
    
    for chart_type, chart_title in chart_types:
        st.subheader(chart_title)
        chart_cols = st.columns(len(session_data))
        
        if chart_type == 'damage':
            for idx, (session_id, df) in enumerate(session_data.items()):
                with chart_cols[idx]:
                    session_alias = get_session_alias(session_id, session_aliases)
                    st.markdown(f"<center><b>{session_alias}</b></center>", unsafe_allow_html=True)
                    action_cols = [f'action_{i}' for i in range(8)]
                    damage_cols = [f'damage_{i}' for i in range(8)]
                    iter_stats = df.groupby('iteration').agg({
                        'damage_taken': 'mean',
                        **{col: 'sum' for col in damage_cols}
                    }).reset_index()
                    iter_stats['damage_dealt'] = iter_stats[damage_cols].sum(axis=1)
                    
                    # Calculate percentage change from previous point
                    iter_stats['pct_change_dealt'] = iter_stats['damage_dealt'].pct_change() * 100
                    iter_stats['pct_change_taken'] = iter_stats['damage_taken'].pct_change() * 100
                    
                    fig = go.Figure()
                    # Add dealt trace (without text, we'll use annotations for colored labels)
                    fig.add_trace(go.Scatter(
                        x=iter_stats['iteration'],
                        y=iter_stats['damage_dealt'],
                        mode='lines+markers',
                        name='Dealt',
                        line=dict(color='green', width=2)
                    ))
                    # Add annotations for dealt with conditional colors (only if >= 10% or <= -10%)
                    for idx, (iter_val, damage_val, pct) in enumerate(zip(
                        iter_stats['iteration'], 
                        iter_stats['damage_dealt'], 
                        iter_stats['pct_change_dealt']
                    )):
                        if not pd.isna(pct) and abs(pct) >= 10:
                            color = 'green' if pct >= 0 else 'red'
                            fig.add_annotation(
                                x=iter_val,
                                y=damage_val,
                                text=f'{pct:+.1f}%',
                                showarrow=False,
                                font=dict(size=10, color=color),
                                yshift=15
                            )
                    
                    # Add taken trace (without text, we'll use annotations for colored labels)
                    fig.add_trace(go.Scatter(
                        x=iter_stats['iteration'],
                        y=iter_stats['damage_taken'],
                        mode='lines+markers',
                        name='Taken',
                        line=dict(color='red', width=2)
                    ))
                    # Add annotations for taken with conditional colors (only if >= 10% or <= -10%)
                    for idx, (iter_val, damage_val, pct) in enumerate(zip(
                        iter_stats['iteration'], 
                        iter_stats['damage_taken'], 
                        iter_stats['pct_change_taken']
                    )):
                        if not pd.isna(pct) and abs(pct) >= 10:
                            color = 'green' if pct >= 0 else 'red'
                            fig.add_annotation(
                                x=iter_val,
                                y=damage_val,
                                text=f'{pct:+.1f}%',
                                showarrow=False,
                                font=dict(size=10, color=color),
                                yshift=-15
                            )
                    # 30% bigger vertically: 300 * 1.3 = 390
                    fig.update_layout(
                        height=390,
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.42,
                            xanchor="center",
                            x=0.5
                        )
                    )
                    st.plotly_chart(fig, width='stretch')
        
        elif chart_type == 'reward':
            for idx, (session_id, df) in enumerate(session_data.items()):
                with chart_cols[idx]:
                    session_alias = get_session_alias(session_id, session_aliases)
                    st.markdown(f"<center><b>{session_alias}</b></center>", unsafe_allow_html=True)
                    iter_stats = df.groupby('iteration')['total_reward'].mean().reset_index()
                    fig = px.bar(iter_stats, x='iteration', y='total_reward', color_discrete_sequence=['#D3D3D3'])
                    avg_reward = iter_stats['total_reward'].mean()
                    fig.add_trace(go.Scatter(
                        x=iter_stats['iteration'],
                        y=[avg_reward] * len(iter_stats),
                        mode='lines',
                        name=f'Avg ({avg_reward:.1f})',
                        line=dict(color='red', width=2)
                    ))
                    # Add trend line
                    if len(iter_stats) > 1:
                        z = np.polyfit(iter_stats['iteration'], iter_stats['total_reward'], 1)
                        p = np.poly1d(z)
                        fig.add_trace(go.Scatter(
                            x=iter_stats['iteration'],
                            y=p(iter_stats['iteration']),
                            mode='lines',
                            name='Trend',
                            line=dict(color='green', width=2)
                        ))
                    fig.update_layout(
                        height=300,
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.42,
                            xanchor="center",
                            x=0.5
                        )
                    )
                    st.plotly_chart(fig, width='stretch')
        
        elif chart_type == 'fight_time':
            for idx, (session_id, df) in enumerate(session_data.items()):
                with chart_cols[idx]:
                    session_alias = get_session_alias(session_id, session_aliases)
                    st.markdown(f"<center><b>{session_alias}</b></center>", unsafe_allow_html=True)
                    iter_stats = df.groupby('iteration')['fight_time'].mean().reset_index()
                    fig = px.bar(iter_stats, x='iteration', y='fight_time')
                    avg_fight_time = iter_stats['fight_time'].mean()
                    fig.add_trace(go.Scatter(
                        x=iter_stats['iteration'],
                        y=[avg_fight_time] * len(iter_stats),
                        mode='lines',
                        name=f'Avg ({avg_fight_time:.1f}s)',
                        line=dict(color='red', width=2)
                    ))
                    # Add trend line
                    if len(iter_stats) > 1:
                        z = np.polyfit(iter_stats['iteration'], iter_stats['fight_time'], 1)
                        p = np.poly1d(z)
                        fig.add_trace(go.Scatter(
                            x=iter_stats['iteration'],
                            y=p(iter_stats['iteration']),
                            mode='lines',
                            name='Trend',
                            line=dict(color='green', width=2)
                        ))
                    fig.update_layout(
                        height=300,
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.42,
                            xanchor="center",
                            x=0.5
                        )
                    )
                    st.plotly_chart(fig, width='stretch')
        
        elif chart_type == 'win_rate':
            for idx, (session_id, df) in enumerate(session_data.items()):
                with chart_cols[idx]:
                    session_alias = get_session_alias(session_id, session_aliases)
                    st.markdown(f"<center><b>{session_alias}</b></center>", unsafe_allow_html=True)
                    win_loss = df.groupby('iteration')['result'].agg(['count', lambda x: (x == 1).sum()]).reset_index()
                    win_loss.columns = ['iteration', 'total_fights', 'wins']
                    win_loss['win_rate'] = (win_loss['wins'] / win_loss['total_fights'] * 100) if win_loss['total_fights'].sum() > 0 else 0
                    fig = px.bar(win_loss, x='iteration', y='win_rate', color_discrete_sequence=['#90EE90'])
                    avg_win_rate = win_loss['win_rate'].mean()
                    fig.add_trace(go.Scatter(
                        x=win_loss['iteration'],
                        y=[avg_win_rate] * len(win_loss),
                        mode='lines',
                        name=f'Avg ({avg_win_rate:.1f}%)',
                        line=dict(color='red', width=2)
                    ))
                    # Add trend line
                    if len(win_loss) > 1:
                        z = np.polyfit(win_loss['iteration'], win_loss['win_rate'], 1)
                        p = np.poly1d(z)
                        fig.add_trace(go.Scatter(
                            x=win_loss['iteration'],
                            y=p(win_loss['iteration']),
                            mode='lines',
                            name='Trend',
                            line=dict(color='green', width=2)
                        ))
                    # Allow negative values on y-axis
                    fig.update_layout(
                        height=300,
                        yaxis=dict(range=None),  # Allow automatic range including negative values
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.42,
                            xanchor="center",
                            x=0.5
                        )
                    )
                    st.plotly_chart(fig, width='stretch')
        
        elif chart_type == 'stun_time':
            for idx, (session_id, df) in enumerate(session_data.items()):
                with chart_cols[idx]:
                    session_alias = get_session_alias(session_id, session_aliases)
                    st.markdown(f"<center><b>{session_alias}</b></center>", unsafe_allow_html=True)
                    iter_stats = df.groupby('iteration')['stun_time'].sum().reset_index()
                    fig = px.bar(iter_stats, x='iteration', y='stun_time', color_discrete_sequence=['#FFE5B4'])
                    avg_stun_time = iter_stats['stun_time'].mean()
                    fig.add_trace(go.Scatter(
                        x=iter_stats['iteration'],
                        y=[avg_stun_time] * len(iter_stats),
                        mode='lines',
                        name=f'Avg ({avg_stun_time:.1f})',
                        line=dict(color='red', width=2)
                    ))
                    # Add trend line
                    if len(iter_stats) > 1:
                        z = np.polyfit(iter_stats['iteration'], iter_stats['stun_time'], 1)
                        p = np.poly1d(z)
                        fig.add_trace(go.Scatter(
                            x=iter_stats['iteration'],
                            y=p(iter_stats['iteration']),
                            mode='lines',
                            name='Trend',
                            line=dict(color='green', width=2)
                        ))
                    fig.update_layout(
                        height=300,
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.42,
                            xanchor="center",
                            x=0.5
                        )
                    )
                    st.plotly_chart(fig, width='stretch')
        
        elif chart_type == 'action_usage':
            view_mode_global = st.radio(
                "View Mode (applies to all sessions)",
                ["Counts", "Percentages"],
                index=1,  # Default to "Percentages"
                horizontal=True,
                key=f"global_view_mode_{chart_type}"
            )
            for idx, (session_id, df) in enumerate(session_data.items()):
                with chart_cols[idx]:
                    session_alias = get_session_alias(session_id, session_aliases)
                    st.markdown(f"<center><b>{session_alias}</b></center>", unsafe_allow_html=True)
                    action_cols = [f'action_{i}' for i in range(8)]
                    iter_stats = df.groupby('iteration').agg({col: 'sum' for col in action_cols}).reset_index()
                    
                    action_data = []
                    for _, row in iter_stats.iterrows():
                        total = sum(row[f'action_{i}'] for i in range(8))
                        if total > 0:
                            for action_num in range(8):
                                count = row[f'action_{action_num}']
                                value = (count / total * 100) if view_mode_global == "Percentages" else count
                                action_data.append({
                                    'iteration': row['iteration'],
                                    'action': ACTION_NAMES[action_num],
                                    'value': value
                                })
                    
                    if action_data:
                        action_df = pd.DataFrame(action_data)
                        fig = px.bar(
                            action_df,
                            x='iteration',
                            y='value',
                            color='action',
                            barmode='stack',
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        # 20% bigger vertically: 300 * 1.2 = 360
                        fig.update_layout(
                            height=360,
                            legend=dict(
                                orientation="h",
                                yanchor="top",
                                y=-0.6,  # Move down like 4 paragraphs
                                xanchor="center",
                                x=0.5
                            )
                        )
                        st.plotly_chart(fig, width='stretch')
        
        elif chart_type == 'action_damage':
            for idx, (session_id, df) in enumerate(session_data.items()):
                with chart_cols[idx]:
                    session_alias = get_session_alias(session_id, session_aliases)
                    st.markdown(f"<center><b>{session_alias}</b></center>", unsafe_allow_html=True)
                    action_avg = calculate_action_averages(df)
                    if not action_avg.empty:
                        fig = px.bar(
                            action_avg,
                            x='action_name',
                            y='avg_damage',
                            color='avg_damage',
                            color_continuous_scale='YlOrRd',
                            text='avg_damage'
                        )
                        fig.update_traces(
                            texttemplate='%{text:.1f}',
                            textposition='outside',
                            textfont=dict(color='white', size=10)
                        )
                        # 30% bigger vertically: 300 * 1.3 = 390
                        fig.update_layout(
                            height=390,
                            xaxis_tickangle=-45,
                            coloraxis_colorbar=dict(
                                orientation="h",
                                yanchor="top",
                                y=-0.6,  # Move down like 4 paragraphs
                                xanchor="center",
                                x=0.5
                            )
                        )
                        st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")


if __name__ == "__main__":
    main()

