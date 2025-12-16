#!/usr/bin/env python3
"""
Bird's-Eye View (BEV) Ray Visualization for LOS Audit

Generates publication-standard BEV diagrams showing:
1. Per-pedestrian ray diagrams (one plot per pedestrian)
2. Combined scene overview with all rays
3. Critical occlusion zones highlighted
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Polygon
from pathlib import Path

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'clear': '#2ecc71',
    'occluded': '#e74c3c',
    'partial': '#f39c12',
    'pedestrian': '#3498db',
    'person': '#3498db',
    'car': '#e74c3c',
    'truck': '#27ae60',
    'cycle': '#9b59b6',
    'static_bg': '#bdc3c7',
}

VEHICLE_DIMS = {
    'car': (4.5, 1.8),
    'truck': (7.0, 2.5),
    'cycle': (1.8, 0.6),
    'person': (0.5, 0.5),
}


def load_report(path: str = 'los_audit_report.json') -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def draw_vehicle(ax, centroid, heading, veh_type, veh_id, bbox_size=None, alpha=0.8):
    """Draw a vehicle as a rectangle with heading indicator."""
    pos = np.array(centroid[:2])
    
    if bbox_size and len(bbox_size) >= 2 and bbox_size[0] > 0.5:
        length, width = bbox_size[0], bbox_size[1]
    else:
        length, width = VEHICLE_DIMS.get(veh_type, (4.0, 1.8))
    
    corners = np.array([
        [-length/2, -width/2],
        [length/2, -width/2],
        [length/2, width/2],
        [-length/2, width/2],
    ])
    
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    rotated = corners @ rotation.T + pos
    
    color = COLORS.get(veh_type, COLORS['car'])
    vehicle = Polygon(rotated, closed=True, facecolor=color, 
                     edgecolor='black', linewidth=1.5, alpha=alpha)
    ax.add_patch(vehicle)
    
    front = pos + np.array([np.cos(heading), np.sin(heading)]) * length/2 * 0.6
    ax.annotate('', xy=front, xytext=pos,
               arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    label = veh_id.replace('_', ' ')
    ax.text(pos[0], pos[1] - width/2 - 0.8, label, 
           ha='center', va='top', fontsize=8, fontweight='bold')


def draw_pedestrian(ax, centroid, ped_id, is_source=True):
    """Draw a pedestrian as a circle at their position."""
    pos = np.array(centroid[:2])
    radius = 0.6 if is_source else 0.4
    color = COLORS['pedestrian']
    
    circle = Circle(pos, radius, facecolor=color, edgecolor='black', 
                   linewidth=2, zorder=10)
    ax.add_patch(circle)
    
    if is_source:
        eye = Circle(pos, 0.2, facecolor='white', edgecolor='black', 
                    linewidth=1, zorder=11)
        ax.add_patch(eye)
    
    label = ped_id.replace('_', ' ')
    ax.text(pos[0], pos[1] - radius - 0.4, label, 
           ha='center', va='top', fontsize=9, fontweight='bold')


def draw_ray(ax, start, end, visibility_score, alpha=0.6):
    """Draw a ray from pedestrian to vehicle with color based on visibility."""
    start = np.array(start[:2])
    end = np.array(end[:2])
    
    if visibility_score >= 0.6:
        color = COLORS['clear']
    elif visibility_score >= 0.4:
        color = COLORS['partial']
    else:
        color = COLORS['occluded']
    
    ax.plot([start[0], end[0]], [start[1], end[1]], 
           color=color, linestyle='-', linewidth=2.5, 
           alpha=alpha, zorder=5)
    
    mid = (start + end) / 2
    ax.text(mid[0], mid[1], f'{visibility_score:.0%}', 
           ha='center', va='center', fontsize=8, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='gray'))


def plot_single_pedestrian_bev(report: dict, ped_id: str, output_dir: Path):
    """Create BEV diagram for a single pedestrian showing all rays to vehicles."""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    obj_pos = report.get('object_positions', {})
    pedestrians = obj_pos.get('pedestrians', {})
    vehicles = obj_pos.get('vehicles', {})
    
    if ped_id not in pedestrians:
        print(f"  Warning: {ped_id} not found in object_positions")
        plt.close()
        return
    
    ped_results = [r for r in report['all_results'] if r['pedestrian'] == ped_id]
    
    if not ped_results:
        plt.close()
        return
    
    ped_data = pedestrians[ped_id]
    ped_pos = np.array(ped_data['centroid'][:2])
    
    for r in ped_results:
        ped_pos_ray = np.array(r['pedestrian_pos'][:2])
        veh_pos_ray = np.array(r['vehicle_pos'][:2])
        draw_ray(ax, ped_pos_ray, veh_pos_ray, r['visibility_score'])
    
    for veh_id, veh_data in vehicles.items():
        vis_result = next((r for r in ped_results if r['vehicle'] == veh_id), None)
        alpha = 0.9 if vis_result and vis_result['is_visible'] else 0.6
        draw_vehicle(ax, veh_data['centroid'], veh_data['heading'], 
                    veh_data['class'], veh_id, 
                    bbox_size=veh_data.get('bbox_size'), alpha=alpha)
    
    draw_pedestrian(ax, ped_data['centroid'], ped_id, is_source=True)
    
    for other_ped, other_data in pedestrians.items():
        if other_ped != ped_id:
            draw_pedestrian(ax, other_data['centroid'], other_ped, is_source=False)
    
    all_x = [ped_pos[0]] + [v['centroid'][0] for v in vehicles.values()]
    all_y = [ped_pos[1]] + [v['centroid'][1] for v in vehicles.values()]
    
    margin = 5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'Bird\'s-Eye View: Line-of-Sight from {ped_id.replace("_", " ").title()}\n(Positions from PLY Scene)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    legend_elements = [
        mpatches.Patch(color=COLORS['clear'], label='Clear LOS (≥60%)'),
        mpatches.Patch(color=COLORS['partial'], label='Partial (40-60%)'),
        mpatches.Patch(color=COLORS['occluded'], label='Occluded (<40%)'),
        mpatches.Patch(color=COLORS['pedestrian'], label='Pedestrian'),
        mpatches.Patch(color=COLORS['car'], label='Car'),
        mpatches.Patch(color=COLORS['truck'], label='Truck'),
        mpatches.Patch(color=COLORS['cycle'], label='Cycle'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    visible_count = sum(1 for r in ped_results if r['is_visible'])
    total_count = len(ped_results)
    avg_score = np.mean([r['visibility_score'] for r in ped_results])
    
    stats_text = f"Visibility Summary for {ped_id}:\n"
    stats_text += f"  Vehicles visible: {visible_count}/{total_count}\n"
    stats_text += f"  Average score: {avg_score:.1%}\n"
    stats_text += f"  Position: ({ped_pos[0]:.1f}, {ped_pos[1]:.1f}) m"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_dir / f'bev_{ped_id}.png')
    plt.savefig(output_dir / f'bev_{ped_id}.pdf')
    plt.close()


def plot_combined_bev(report: dict, output_dir: Path):
    """Create combined BEV diagram showing all pedestrians and visibility rays."""
    fig, ax = plt.subplots(figsize=(16, 14))
    
    obj_pos = report.get('object_positions', {})
    pedestrians = obj_pos.get('pedestrians', {})
    vehicles = obj_pos.get('vehicles', {})
    
    if not pedestrians or not vehicles:
        print("  Error: No object_positions in report. Re-run los_audit.py")
        plt.close()
        return
    
    for r in report['all_results']:
        ped_pos = np.array(r['pedestrian_pos'][:2])
        veh_pos = np.array(r['vehicle_pos'][:2])
        
        if r['visibility_score'] >= 0.6:
            color = COLORS['clear']
        elif r['visibility_score'] >= 0.4:
            color = COLORS['partial']
        else:
            color = COLORS['occluded']
        
        ax.plot([ped_pos[0], veh_pos[0]], [ped_pos[1], veh_pos[1]], 
               color=color, linewidth=2, alpha=0.5, zorder=1)
    
    for veh_id, veh_data in vehicles.items():
        draw_vehicle(ax, veh_data['centroid'], veh_data['heading'], 
                    veh_data['class'], veh_id, 
                    bbox_size=veh_data.get('bbox_size'), alpha=0.85)
    
    for ped_id, ped_data in pedestrians.items():
        draw_pedestrian(ax, ped_data['centroid'], ped_id, is_source=True)
    
    all_x = [p['centroid'][0] for p in pedestrians.values()]
    all_x += [v['centroid'][0] for v in vehicles.values()]
    all_y = [p['centroid'][1] for p in pedestrians.values()]
    all_y += [v['centroid'][1] for v in vehicles.values()]
    
    if all_x and all_y:
        margin = 8
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Bird\'s-Eye View: Complete LOS Audit\nActual Scene Positions from PLY', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    legend_elements = [
        mpatches.Patch(color=COLORS['clear'], label='Clear (≥60%)'),
        mpatches.Patch(color=COLORS['partial'], label='Partial (40-60%)'),
        mpatches.Patch(color=COLORS['occluded'], label='Occluded (<40%)'),
        mpatches.Patch(color=COLORS['pedestrian'], label='Pedestrian'),
        mpatches.Patch(color=COLORS['car'], label='Car'),
        mpatches.Patch(color=COLORS['truck'], label='Truck'),
        mpatches.Patch(color=COLORS['cycle'], label='Cycle'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    summary = report['summary']
    stats = f"LOS Audit Summary\n"
    stats += f"─" * 22 + "\n"
    stats += f"Total pairs: {summary['total_pairs']}\n"
    stats += f"Visible: {summary['visible_pairs']} ({summary['visible_pairs']/summary['total_pairs']*100:.0f}%)\n"
    stats += f"Occluded: {summary['occluded_pairs']} ({summary['occluded_pairs']/summary['total_pairs']*100:.0f}%)\n"
    stats += f"Avg visibility: {summary['avg_visibility_score']:.1%}\n"
    stats += f"─" * 22 + "\n"
    stats += f"Scale factor: {summary['scale_factor']:.3f}"
    
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))
    
    print("\n  Object positions from PLY:")
    for ped_id, ped_data in pedestrians.items():
        pos = ped_data['centroid']
        print(f"    {ped_id}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    for veh_id, veh_data in vehicles.items():
        pos = veh_data['centroid']
        print(f"    {veh_id}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bev_combined.png')
    plt.savefig(output_dir / 'bev_combined.pdf')
    plt.close()


def plot_safety_zones(report: dict, output_dir: Path):
    """Create a top-down view highlighting occlusion danger zones."""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    obj_pos = report.get('object_positions', {})
    pedestrians = obj_pos.get('pedestrians', {})
    vehicles = obj_pos.get('vehicles', {})
    
    critical = [r for r in report['all_results'] 
               if r['distance'] < 15.0 and r['visibility_score'] < 0.4]
    
    for r in critical:
        ped_pos = np.array(r['pedestrian_pos'][:2])
        veh_pos = np.array(r['vehicle_pos'][:2])
        mid = (ped_pos + veh_pos) / 2
        
        danger = Circle(mid, 3, facecolor='red', alpha=0.15, 
                       edgecolor='red', linewidth=2, linestyle='--', zorder=0)
        ax.add_patch(danger)
    
    for r in report['all_results']:
        ped_pos = np.array(r['pedestrian_pos'][:2])
        veh_pos = np.array(r['vehicle_pos'][:2])
        
        is_critical = r['distance'] < 15.0 and r['visibility_score'] < 0.4
        
        if is_critical:
            ax.plot([ped_pos[0], veh_pos[0]], [ped_pos[1], veh_pos[1]], 
                   color='red', linewidth=3, alpha=0.8, zorder=2)
        else:
            color = COLORS['clear'] if r['is_visible'] else COLORS['occluded']
            ax.plot([ped_pos[0], veh_pos[0]], [ped_pos[1], veh_pos[1]], 
                   color=color, linewidth=1.5, alpha=0.4, zorder=1)
    
    for veh_id, veh_data in vehicles.items():
        draw_vehicle(ax, veh_data['centroid'], veh_data['heading'], 
                    veh_data['class'], veh_id, 
                    bbox_size=veh_data.get('bbox_size'), alpha=0.85)
    
    for ped_id, ped_data in pedestrians.items():
        draw_pedestrian(ax, ped_data['centroid'], ped_id, is_source=True)
    
    all_x = [p['centroid'][0] for p in pedestrians.values()]
    all_x += [v['centroid'][0] for v in vehicles.values()]
    all_y = [p['centroid'][1] for p in pedestrians.values()]
    all_y += [v['centroid'][1] for v in vehicles.values()]
    
    if all_x and all_y:
        margin = 8
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Safety Analysis: Critical Occlusion Zones\n(Distance < 15m, Visibility < 40%)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    legend_elements = [
        mpatches.Patch(color='red', alpha=0.3, label='Critical Zone'),
        mpatches.Patch(color=COLORS['clear'], label='Clear LOS'),
        mpatches.Patch(color=COLORS['occluded'], label='Occluded'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    if critical:
        text = "⚠️ CRITICAL OCCLUSIONS:\n"
        for c in critical[:5]:
            text += f"  • {c['pedestrian']} → {c['vehicle']}\n"
            text += f"    {c['visibility_score']:.0%} vis, {c['distance']:.1f}m\n"
    else:
        text = "✓ No critical occlusions detected"
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bev_safety_zones.png')
    plt.savefig(output_dir / 'bev_safety_zones.pdf')
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate BEV Ray Visualizations")
    parser.add_argument('--report', type=str, default='los_audit_report.json',
                       help='Path to LOS audit report JSON file')
    args = parser.parse_args()
    
    print("=" * 60)
    print("BIRD'S-EYE VIEW (BEV) RAY VISUALIZATION")
    print(f"Using positions from {args.report}")
    print("=" * 60)
    
    report_path = Path(args.report)
    if not report_path.exists():
        print(f"ERROR: {report_path} not found")
        return
    
    report = load_report(str(report_path))
    print(f"\n✓ Loaded report: {report['summary']['total_pairs']} pairs")
    
    if 'object_positions' not in report:
        print("\n⚠️  object_positions not found in report!")
        print("   Please re-run los_audit.py to generate updated report with positions.")
        print("   Command: python3 los_audit.py --scene scene_with_objects.ply")
        return
    
    obj_pos = report['object_positions']
    print(f"✓ Found {len(obj_pos['pedestrians'])} pedestrians, {len(obj_pos['vehicles'])} vehicles")
    
    output_dir = Path('los_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating BEV visualizations...")
    
    plot_combined_bev(report, output_dir)
    print("  ✓ bev_combined.png/pdf - All rays overview")
    
    for ped_id in obj_pos['pedestrians'].keys():
        plot_single_pedestrian_bev(report, ped_id, output_dir)
        print(f"  ✓ bev_{ped_id}.png/pdf - Individual view")
    
    plot_safety_zones(report, output_dir)
    print("  ✓ bev_safety_zones.png/pdf - Critical zone analysis")
    
    print(f"\n✓ All BEV figures saved to: {output_dir}/")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
