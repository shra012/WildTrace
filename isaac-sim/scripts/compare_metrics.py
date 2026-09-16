"""Compare baseline vs improved trajectory tracking metrics."""
import json
import sys
from pathlib import Path

def load_metrics(path):
    """Load metrics JSON file."""
    p = Path(path)
    if not p.is_file():
        print(f"❌ File not found: {path}")
        return None
    with open(p) as f:
        return json.load(f)

def format_change(before, after, lower_is_better=True):
    """Format percentage change with color coding."""
    if before == 0:
        return "N/A"
    pct_change = ((after - before) / before) * 100
    if lower_is_better:
        if pct_change < -5:
            return f"✅ {pct_change:+.1f}%"
        elif pct_change > 5:
            return f"⚠️ {pct_change:+.1f}%"
        else:
            return f"➡️ {pct_change:+.1f}%"
    else:
        if pct_change > 5:
            return f"✅ {pct_change:+.1f}%"
        elif pct_change < -5:
            return f"⚠️ {pct_change:+.1f}%"
        else:
            return f"➡️ {pct_change:+.1f}%"

def main():
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"
    
    # Default file paths
    baseline_path = outputs_dir / "run_metrics_baseline.json"
    current_path = outputs_dir / "run_metrics.json"
    
    # Allow command-line override
    if len(sys.argv) >= 2:
        baseline_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        current_path = Path(sys.argv[2])
    
    print("\n" + "="*70)
    print("  xArm 7 Trajectory Tracking - Metrics Comparison")
    print("="*70 + "\n")
    
    print(f"📊 Baseline: {baseline_path.name}")
    print(f"📊 Current:  {current_path.name}\n")
    
    baseline = load_metrics(baseline_path)
    current = load_metrics(current_path)
    
    if baseline is None or current is None:
        print("\n⚠️  Could not load metrics files.")
        print("\nTo create baseline:")
        print("  1. Run: python scripts\\run_two_stroke_drawing.py")
        print("  2. Copy: cp outputs/run_metrics.json outputs/run_metrics_baseline.json")
        print("  3. Make parameter changes")
        print("  4. Run: python scripts\\run_two_stroke_drawing.py")
        print("  5. Compare: python scripts\\compare_metrics.py\n")
        return 1
    
    # Status check
    print("Status:")
    print(f"  Baseline: {baseline['status']}")
    print(f"  Current:  {current['status']}")
    if baseline['status'] != 'finished' or current['status'] != 'finished':
        print("\n⚠️  One or both runs did not complete successfully.\n")
        return 1
    print()
    
    # Position tracking
    print("Position Tracking Error (pen-down only):")
    print("-" * 70)
    
    b_pos = baseline['pen_down_tracking_error']
    c_pos = current['pen_down_tracking_error']
    
    print(f"  RMSE:")
    print(f"    Baseline: {b_pos['rmse_m']*1000:.3f} mm")
    print(f"    Current:  {c_pos['rmse_m']*1000:.3f} mm  {format_change(b_pos['rmse_m'], c_pos['rmse_m'])}")
    print()
    
    print(f"  Max Error:")
    print(f"    Baseline: {b_pos['max_error_m']*1000:.3f} mm")
    print(f"    Current:  {c_pos['max_error_m']*1000:.3f} mm  {format_change(b_pos['max_error_m'], c_pos['max_error_m'])}")
    print()
    
    print(f"  Mean Error:")
    print(f"    Baseline: {b_pos['mean_error_m']*1000:.3f} mm")
    print(f"    Current:  {c_pos['mean_error_m']*1000:.3f} mm  {format_change(b_pos['mean_error_m'], c_pos['mean_error_m'])}")
    print()
    
    print(f"  P95 Error:")
    print(f"    Baseline: {b_pos['p95_error_m']*1000:.3f} mm")
    print(f"    Current:  {c_pos['p95_error_m']*1000:.3f} mm  {format_change(b_pos['p95_error_m'], c_pos['p95_error_m'])}")
    print()
    
    # Per-stroke breakdown
    print("\nPer-Stroke Position RMSE:")
    print("-" * 70)
    
    for stroke_id in sorted(baseline.get('per_stroke_tracking', {}).keys()):
        b_stroke = baseline['per_stroke_tracking'][stroke_id]
        c_stroke = current['per_stroke_tracking'][stroke_id]
        print(f"  Stroke {stroke_id}:")
        print(f"    Baseline: {b_stroke['RMSE_position_m']*1000:.3f} mm")
        print(f"    Current:  {c_stroke['RMSE_position_m']*1000:.3f} mm  {format_change(b_stroke['RMSE_position_m'], c_stroke['RMSE_position_m'])}")
    print()
    
    # Heading errors (informational - not a control objective)
    if 'heading_error' in baseline and 'heading_error' in current:
        print("\nHeading Error (informational - pen orientation is fixed):")
        print("-" * 70)
        
        b_head = baseline['heading_error']
        c_head = current['heading_error']
        
        print(f"  RMSE:")
        print(f"    Baseline: {b_head['rmse_deg']:.1f}°")
        print(f"    Current:  {c_head['rmse_deg']:.1f}°  {format_change(b_head['rmse_deg'], c_head['rmse_deg'])}")
        print()
        
        print(f"  P95:")
        print(f"    Baseline: {b_head['p95_abs_error_deg']:.1f}°")
        print(f"    Current:  {c_head['p95_abs_error_deg']:.1f}°  {format_change(b_head['p95_abs_error_deg'], c_head['p95_abs_error_deg'])}")
        print()
    
    # Execution time
    print("\nExecution Time:")
    print("-" * 70)
    b_time = baseline['simulation_duration_s']
    c_time = current['simulation_duration_s']
    print(f"  Baseline: {b_time:.1f} s")
    print(f"  Current:  {c_time:.1f} s  {format_change(b_time, c_time, lower_is_better=True)}")
    print(f"  Steps: {baseline['simulation_steps']} → {current['simulation_steps']}")
    print()
    
    # System stability
    print("\nSystem Stability:")
    print("-" * 70)
    
    b_clamp = baseline.get('clamped_command_fraction', 0.0)
    c_clamp = current.get('clamped_command_fraction', 0.0)
    print(f"  Joint slew clamping:")
    print(f"    Baseline: {b_clamp*100:.2f}%")
    print(f"    Current:  {c_clamp*100:.2f}%")
    if c_clamp > 0.05:
        print("    ⚠️  Clamping > 5% - consider relaxing joint velocity limits")
    else:
        print("    ✅ Acceptable (<5%)")
    print()
    
    # New velocity gate metrics
    if 'velocity_settled_fraction' in current:
        print("\nVelocity Gate (new feature):")
        print("-" * 70)
        v_settled = current['velocity_settled_fraction']
        print(f"  Velocity settled fraction: {v_settled*100:.1f}%")
        if v_settled < 0.90:
            print("    ⚠️  <90% settled - consider increasing max_tip_speed_mm_s")
        else:
            print("    ✅ Good settling behavior")
        
        if 'tip_speed_mm_s_stats' in current:
            stats = current['tip_speed_mm_s_stats']
            print(f"  Tip speed (mm/s):")
            print(f"    Mean:   {stats['mean']:.2f}")
            print(f"    Median: {stats['median']:.2f}")
            print(f"    P95:    {stats['p95']:.2f}")
            print(f"    Max:    {stats['max']:.2f}")
            if 'max_tip_speed_mm_s' in current['tolerances'] and current['tolerances']['max_tip_speed_mm_s'] is not None:
                gate = current['tolerances']['max_tip_speed_mm_s']
                print(f"    Gate:   {gate:.2f} mm/s")
        print()
    
    # Summary
    print("\n" + "="*70)
    print("  Summary")
    print("="*70 + "\n")
    
    rmse_improvement = (1 - c_pos['rmse_m'] / b_pos['rmse_m']) * 100
    max_improvement = (1 - c_pos['max_error_m'] / b_pos['max_error_m']) * 100
    time_cost = (c_time / b_time - 1) * 100
    
    if rmse_improvement > 5:
        print(f"✅ Position RMSE improved by {rmse_improvement:.1f}%")
    elif rmse_improvement < -5:
        print(f"⚠️  Position RMSE degraded by {-rmse_improvement:.1f}%")
    else:
        print(f"➡️  Position RMSE changed by {rmse_improvement:+.1f}% (minimal change)")
    
    if max_improvement > 5:
        print(f"✅ Max error improved by {max_improvement:.1f}%")
    elif max_improvement < -5:
        print(f"⚠️  Max error degraded by {-max_improvement:.1f}%")
    else:
        print(f"➡️  Max error changed by {max_improvement:+.1f}% (minimal change)")
    
    if time_cost > 20:
        print(f"⚠️  Execution time increased by {time_cost:.1f}% (consider relaxing tolerances)")
    elif time_cost > 0:
        print(f"➡️  Execution time increased by {time_cost:.1f}% (acceptable for accuracy gain)")
    else:
        print(f"✅ Execution time improved by {-time_cost:.1f}%")
    
    print()
    
    # Overall assessment
    if rmse_improvement > 10 and max_improvement > 10:
        print("🎉 Excellent improvement! Parameters are working well.")
    elif rmse_improvement > 5 and time_cost < 30:
        print("👍 Good improvement with reasonable time cost.")
    elif rmse_improvement > 0:
        print("📊 Minor improvement. Consider further tuning if needed.")
    else:
        print("⚠️  No improvement or degradation. Review parameter changes.")
    
    print("\n" + "="*70 + "\n")
    
    # Next steps
    print("Next Steps:")
    if rmse_improvement < 10:
        print("  - Review tuning guide: outputs/TUNING_QUICK_REFERENCE.md")
        print("  - Check diagnostics: outputs/tracking_diagnostics.png")
        if c_clamp > 0.05:
            print("  - High clamping detected - check max_joint_delta_rad")
        if 'velocity_settled_fraction' in current and current['velocity_settled_fraction'] < 0.90:
            print("  - Low velocity settling - check max_tip_speed_mm_s")
    else:
        print("  - Document your final parameter configuration")
        print("  - Consider testing on different drawings to validate")
    
    print()
    return 0

if __name__ == "__main__":
    sys.exit(main())
