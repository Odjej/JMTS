"""
Generate a presentation-ready visualization of CarAgent behavior and architecture.

Usage:
    python tools/visualize_car_agent.py --out=results/car_agent_behavior.png

Creates a comprehensive diagram showing:
  - Agent state variables
  - Dynamic models (GFM, IDM, lane change)
  - Route planning (A*)
  - Interaction flows
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

# Set global font to sans-serif
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# Shared font sizes
TITLE_SIZE = 24
BOX_TITLE_SIZE = 20
BODY_SIZE = 16
FOOTER_SIZE = 12


def create_agent_behavior_figure(out_path):
    """Create 6 separate detailed visualizations of CarAgent behavior."""
    out_dir = os.path.dirname(out_path) if os.path.dirname(out_path) else '.'
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(out_path))[0]
    
    # ===== 1. Agent Architecture Overview =====
    fig1, ax1 = plt.subplots(figsize=(14, 10), facecolor='white')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.text(5, 9.8, 'CarAgent Architecture & Dynamics', fontsize=TITLE_SIZE, fontweight='bold', 
            ha='center', va='top', family='sans-serif')
    
    # Main agent box
    agent_box = FancyBboxPatch((1.5, 6.8), 2.2, 2.2, boxstyle="round,pad=0.1", 
                               edgecolor='#2E86AB', facecolor='#E8F4F8', linewidth=3)
    ax1.add_patch(agent_box)
    ax1.text(2.6, 8.5, 'CarAgent', ha='center', va='center', fontsize=BOX_TITLE_SIZE, fontweight='bold', family='sans-serif')
    ax1.text(2.6, 8.0, 'ID, position', ha='center', va='center', fontsize=BODY_SIZE, family='sans-serif')
    ax1.text(2.6, 7.6, 'velocity, route', ha='center', va='center', fontsize=BODY_SIZE, family='sans-serif')
    ax1.text(2.6, 7.1, 'lane, replans', ha='center', va='center', fontsize=BODY_SIZE, family='sans-serif')
    
    # Input: Environment & Network
    env_box = FancyBboxPatch((0.1, 4.5), 1.9, 1.4, boxstyle="round,pad=0.08", 
                            edgecolor='#A23B72', facecolor='#F5E6F0', linewidth=2)
    ax1.add_patch(env_box)
    ax1.text(1.05, 5.6, 'Environment', ha='center', va='center', fontsize=BOX_TITLE_SIZE, fontweight='bold', family='sans-serif')
    ax1.text(1.05, 5.1, 'Constructions', ha='center', va='center', fontsize=BODY_SIZE, family='sans-serif')
    
    net_box = FancyBboxPatch((2.2, 4.5), 1.9, 1.4, boxstyle="round,pad=0.08", 
                            edgecolor='#F18F01', facecolor='#FFE8CC', linewidth=2)
    ax1.add_patch(net_box)
    ax1.text(3.15, 5.6, 'Network', ha='center', va='center', fontsize=BOX_TITLE_SIZE, fontweight='bold', family='sans-serif')
    ax1.text(3.15, 5.1, 'Edges, nodes', ha='center', va='center', fontsize=BODY_SIZE, family='sans-serif')
    
    # Processes
    route_box = FancyBboxPatch((5.2, 7.6), 2.2, 1.4, boxstyle="round,pad=0.1", 
                              edgecolor='#06A77D', facecolor='#D4F1E4', linewidth=3)
    ax1.add_patch(route_box)
    ax1.text(6.3, 8.6, 'Route Planning', ha='center', va='center', fontsize=BOX_TITLE_SIZE, fontweight='bold', family='sans-serif')
    ax1.text(6.3, 8.1, 'A* + env costs', ha='center', va='center', fontsize=BODY_SIZE, family='sans-serif')
    
    accel_box = FancyBboxPatch((5.2, 5.9), 2.2, 1.4, boxstyle="round,pad=0.1", 
                              edgecolor='#D62828', facecolor='#FFE5E5', linewidth=3)
    ax1.add_patch(accel_box)
    ax1.text(6.3, 6.9, 'Acceleration', ha='center', va='center', fontsize=BOX_TITLE_SIZE, fontweight='bold', family='sans-serif')
    ax1.text(6.3, 6.4, 'GFM + IDM', ha='center', va='center', fontsize=BODY_SIZE, family='sans-serif')
    
    lane_box = FancyBboxPatch((5.2, 4.2), 2.2, 1.4, boxstyle="round,pad=0.1", 
                             edgecolor='#F77F00', facecolor='#FFE8CC', linewidth=3)
    ax1.add_patch(lane_box)
    ax1.text(6.3, 5.2, 'Lane Changes', ha='center', va='center', fontsize=BOX_TITLE_SIZE, fontweight='bold', family='sans-serif')
    ax1.text(6.3, 4.7, 'Overtaking', ha='center', va='center', fontsize=BODY_SIZE, family='sans-serif')
    
    # Arrows from inputs to agent
    arrow1 = FancyArrowPatch((1.05, 5.9), (2.0, 6.8), arrowstyle='->', 
                            mutation_scale=20, linewidth=2, color='#A23B72', alpha=0.8)
    ax1.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((3.15, 5.9), (3.2, 6.8), arrowstyle='->', 
                            mutation_scale=20, linewidth=2, color='#F18F01', alpha=0.8)
    ax1.add_patch(arrow2)
    
    # Arrows from agent to processes
    arrow3 = FancyArrowPatch((3.7, 8.3), (5.2, 8.3), arrowstyle='->', 
                            mutation_scale=20, linewidth=2, color='#06A77D', alpha=0.8)
    ax1.add_patch(arrow3)
    
    arrow4 = FancyArrowPatch((3.7, 7.8), (5.2, 6.6), arrowstyle='->', 
                            mutation_scale=20, linewidth=2, color='#D62828', alpha=0.8)
    ax1.add_patch(arrow4)
    
    arrow5 = FancyArrowPatch((3.7, 7.3), (5.2, 4.9), arrowstyle='->', 
                            mutation_scale=20, linewidth=2, color='#F77F00', alpha=0.8)
    ax1.add_patch(arrow5)
    
    # Outputs
    output_box = FancyBboxPatch((7.8, 6.5), 2.0, 2.5, boxstyle="round,pad=0.1", 
                               edgecolor='#55A630', facecolor='#E8F5E9', linewidth=3)
    ax1.add_patch(output_box)
    ax1.text(8.8, 8.7, 'Output Metrics', ha='center', va='center', fontsize=BOX_TITLE_SIZE, fontweight='bold', family='sans-serif')
    ax1.text(7.9, 8.2, '• Travel time', ha='left', va='center', fontsize=BODY_SIZE, family='sans-serif')
    ax1.text(7.9, 7.85, '• Distance', ha='left', va='center', fontsize=BODY_SIZE, family='sans-serif')
    ax1.text(7.9, 7.5, '• Avg speed', ha='left', va='center', fontsize=BODY_SIZE, family='sans-serif')
    ax1.text(7.9, 7.15, '• # Replans', ha='left', va='center', fontsize=BODY_SIZE, family='sans-serif')
    ax1.text(7.9, 6.8, '• Detour ratio', ha='left', va='center', fontsize=BODY_SIZE, family='sans-serif')
    
    # Arrow from processes to output
    arrow_out = FancyArrowPatch((7.4, 7.3), (7.8, 7.3), arrowstyle='->', 
                               mutation_scale=20, linewidth=2, color='#55A630', alpha=0.8)
    ax1.add_patch(arrow_out)
    
    fig1.text(0.5, 0.02, 'Generalized Force Model (GFM) + Intelligent Driver Model (IDM) + A* route planning',
            ha='center', fontsize=FOOTER_SIZE, style='italic', color='gray', family='sans-serif')
    
    # Save figure 1
    fig1_path = os.path.join(out_dir, f'{base_name}_01_architecture.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {fig1_path}')
    plt.close(fig1)
    
    # ===== 2. Dynamics equations =====
    fig2, ax2 = plt.subplots(figsize=(10, 8), facecolor='white')
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.text(5, 9.6, 'Core Equations', fontsize=TITLE_SIZE, fontweight='bold', ha='center', va='top', family='sans-serif')
    
    eq_text = """Relaxation (GFM):
a_tot = a_des + a_brake + ξ

a_des = (v₀ - v) / τ

a_brake = -a_max × [(gap_des - gap_act) / gap_des]

Pedestrian Avoidance:
If d ≤ 2m: a = -5.0 m/s²
If d ≤ 8m: linear fade

Stochastic Noise:
ξ ~ N(0, 0.05) [m/s²]"""
    
    ax2.text(5, 8.2, eq_text, fontsize=BODY_SIZE, family='monospace', ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.6, edgecolor='black', linewidth=2))
    
    fig2_path = os.path.join(out_dir, f'{base_name}_02_equations.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {fig2_path}')
    plt.close(fig2)
    
    # ===== 3. Per-Step Behavior =====
    fig3, ax3 = plt.subplots(figsize=(10, 8), facecolor='white')
    ax3.axis('off')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.text(5, 9.6, 'Per-Step Behavior', fontsize=TITLE_SIZE, fontweight='bold', ha='center', va='top', family='sans-serif')
    
    step_text = """1. Evaluate lane change feasibility
2. Compute desired acceleration
3. Find leader vehicle on lane
4. Find pedestrians in 30m radius
5. Compute braking distance
6. Add stochastic noise
7. Update velocity & position
8. Move along current route
9. Detect arrival at destination"""
    
    ax3.text(5, 8.2, step_text, fontsize=BODY_SIZE, family='monospace', ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.6, edgecolor='black', linewidth=2))
    
    fig3_path = os.path.join(out_dir, f'{base_name}_03_behavior.png')
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {fig3_path}')
    plt.close(fig3)
    
    # ===== 4. Route Planning =====
    fig4, ax4 = plt.subplots(figsize=(10, 8), facecolor='white')
    ax4.axis('off')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.text(5, 9.6, 'Route Planning (A*)', fontsize=TITLE_SIZE, fontweight='bold', ha='center', va='top', family='sans-serif')
    
    planning_text = """Start Node:
  Nearest node to current position

End Node:
  Nearest node to destination

Heuristic:
  Euclidean distance

Weight Function:
  Edge travel time (s)
  • Base: length / max_speed
  • Includes construction impact
  • Dynamic cost adaptation

Replanning:
  Triggered on blocked/slow edge detection"""
    
    ax4.text(5, 8.2, planning_text, fontsize=BODY_SIZE, family='monospace', ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFFACD', alpha=0.6, edgecolor='black', linewidth=2))
    
    fig4_path = os.path.join(out_dir, f'{base_name}_04_planning.png')
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {fig4_path}')
    plt.close(fig4)
    
    # ===== 5. Agent Parameters =====
    fig5, ax5 = plt.subplots(figsize=(10, 8), facecolor='white')
    ax5.axis('off')
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    ax5.text(5, 9.6, 'Agent Parameters', fontsize=TITLE_SIZE, fontweight='bold', ha='center', va='top', family='sans-serif')
    
    params_text = """Max Velocity:
  v₀ = 13.89 m/s

Initial Velocity:
  v_init = 8.33 m/s

Acceleration Limits:
  a_max = 2.0 m/s²
  b (decel) = 3.0 m/s²

Temporal Scale:
  τ = 1.0 s (relaxation time)
  dt = 0.1 s (simulation step)

Safety & Comfort:
  L = 4.5 m (vehicle length)
  T = 1.0 s (time headway)
  s₀ = 2.0 m (minimum gap)"""
    
    ax5.text(5, 8.2, params_text, fontsize=BODY_SIZE, family='monospace', ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE4E1', alpha=0.6, edgecolor='black', linewidth=2))
    
    fig5_path = os.path.join(out_dir, f'{base_name}_05_parameters.png')
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {fig5_path}')
    plt.close(fig5)
    
    # ===== 6. Environment Interactions =====
    fig6, ax6 = plt.subplots(figsize=(16, 10), facecolor='white')
    ax6.set_xlim(0, 15)
    ax6.set_ylim(0, 3.5)
    ax6.axis('off')
    ax6.text(7.5, 3.25, 'Environment Interactions', fontsize=TITLE_SIZE, fontweight='bold', ha='center', va='top', family='sans-serif')
    
    # Construction impact
    const_box = FancyBboxPatch((0.2, 0.8), 2.8, 2.0, boxstyle="round,pad=0.1", 
                              edgecolor='#DC143C', facecolor='#FFE4E1', linewidth=2.5)
    ax6.add_patch(const_box)
    ax6.text(1.6, 2.5, 'Constructions', ha='center', fontsize=BOX_TITLE_SIZE, fontweight='bold', family='sans-serif')
    ax6.text(1.6, 2.1, '• Increase cost', ha='center', fontsize=BODY_SIZE, family='sans-serif')
    ax6.text(1.6, 1.75, '• Trigger replans', ha='center', fontsize=BODY_SIZE, family='sans-serif')
    ax6.text(1.6, 1.4, '• Slow: 70%', ha='center', fontsize=BODY_SIZE, family='sans-serif')
    ax6.text(1.6, 1.05, '• Lane: 20%', ha='center', fontsize=BODY_SIZE, family='sans-serif')
    
    # Leader vehicle
    leader_box = FancyBboxPatch((3.2, 0.8), 2.8, 2.0, boxstyle="round,pad=0.1", 
                               edgecolor='#FF8C00', facecolor='#FFE8CC', linewidth=2.5)
    ax6.add_patch(leader_box)
    ax6.text(4.6, 2.5, 'Leader Vehicle', ha='center', fontsize=BOX_TITLE_SIZE, fontweight='bold', family='sans-serif')
    ax6.text(4.6, 2.1, '• Distance track', ha='center', fontsize=BODY_SIZE, family='sans-serif')
    ax6.text(4.6, 1.75, '• IDM braking', ha='center', fontsize=BODY_SIZE, family='sans-serif')
    ax6.text(4.6, 1.4, '• Safe gap', ha='center', fontsize=BODY_SIZE, family='sans-serif')
    ax6.text(4.6, 1.05, '• Lane evaluation', ha='center', fontsize=BODY_SIZE, family='sans-serif')
    
    # Pedestrians
    ped_box = FancyBboxPatch((6.2, 0.8), 2.8, 2.0, boxstyle="round,pad=0.1", 
                            edgecolor='#228B22', facecolor='#E8F5E9', linewidth=2.5)
    ax6.add_patch(ped_box)
    ax6.text(7.6, 2.5, 'Pedestrians', ha='center', fontsize=BOX_TITLE_SIZE, fontweight='bold', family='sans-serif')
    ax6.text(7.6, 2.1, '• 30m lookahead', ha='center', fontsize=BODY_SIZE, family='sans-serif')
    ax6.text(7.6, 1.75, '• Distance brake', ha='center', fontsize=BODY_SIZE, family='sans-serif')
    ax6.text(7.6, 1.4, '• Yield logic', ha='center', fontsize=BODY_SIZE, family='sans-serif')
    ax6.text(7.6, 1.05, '• Block lane change', ha='center', fontsize=BODY_SIZE, family='sans-serif')
    
    # Outputs
    output_detailed = FancyBboxPatch((9.2, 0.8), 5.5, 2.0, boxstyle="round,pad=0.1", 
                                    edgecolor='#4169E1', facecolor='#F0F8FF', linewidth=2.5)
    ax6.add_patch(output_detailed)
    ax6.text(11.95, 2.5, 'Observable Metrics', ha='center', fontsize=BOX_TITLE_SIZE, fontweight='bold', family='sans-serif')
    ax6.text(9.4, 2.1, '• Travel time (s)', ha='left', fontsize=BODY_SIZE, family='sans-serif')
    ax6.text(9.4, 1.75, '• Distance (m)', ha='left', fontsize=BODY_SIZE, family='sans-serif')
    ax6.text(9.4, 1.4, '• Avg speed (m/s)', ha='left', fontsize=BODY_SIZE, family='sans-serif')
    ax6.text(9.4, 1.05, '• Replans & detours', ha='left', fontsize=BODY_SIZE, family='sans-serif')
    ax6.text(12.0, 2.1, '• Arrival rate', ha='left', fontsize=BODY_SIZE, family='sans-serif')
    ax6.text(12.0, 1.75, '• Reroute %', ha='left', fontsize=BODY_SIZE, family='sans-serif')
    ax6.text(12.0, 1.4, '• Congestion', ha='left', fontsize=BODY_SIZE, family='sans-serif')
    ax6.text(12.0, 1.05, '• Construction impact', ha='left', fontsize=BODY_SIZE, family='sans-serif')
    
    fig6_path = os.path.join(out_dir, f'{base_name}_06_interactions.png')
    plt.savefig(fig6_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {fig6_path}')
    plt.close(fig6)
    
    print(f'\nGenerated 6 presentation figures in {out_dir}:')
    print(f'  01_architecture.png - System architecture & component flow')
    print(f'  02_equations.png - Core mathematical models')
    print(f'  03_behavior.png - Per-timestep decision process')
    print(f'  04_planning.png - Route planning & A* algorithm')
    print(f'  05_parameters.png - Agent configuration')
    print(f'  06_interactions.png - Environment interactions & metrics')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize CarAgent behavior for presentations.')
    parser.add_argument('--out', default='results/car_agent_behavior.png',
                       help='Output PNG file path')
    args = parser.parse_args()
    
    create_agent_behavior_figure(args.out)


if __name__ == '__main__':
    main()
    main()
