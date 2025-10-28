#!/usr/bin/env python3
"""
GPU Profile Visualizer for NVIDIA Nsight Systems output
Parses nsys stats output and creates performance visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import re
import sys
import os

def parse_nsys_stats(filename):
    """Parse nsys stats text file"""
    data = []
    
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found")
        return []
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        print(f"Parsing nsys stats output...")
        
        # Parse CUDA GPU Kernel Summary section - ONLY KERNELS
        kernel_match = re.search(r'\*\* CUDA GPU Kernel Summary.*?\n(.*?)(?=\n\n|\*\*|Processing)', content, re.DOTALL)
        if kernel_match:
            print("Found CUDA GPU Kernel Summary")
            section = kernel_match.group(1)
            lines = section.split('\n')
            
            for line in lines:
                # Skip headers and separator lines
                if 'Time (%)' in line or '--------' in line or not line.strip():
                    continue
                
                # Parse data lines - format: Time(%)  TotalTime(ns)  Instances  Avg  Med  Min  Max  StdDev  Name
                parts = line.split()
                if len(parts) >= 9:
                    try:
                        time_percent = float(parts[0])
                        total_time_ns = int(parts[1].replace(',', ''))
                        instances = int(parts[2].replace(',', ''))
                        # Name is everything from index 8 onwards
                        name = ' '.join(parts[8:])
                        # Extract just function name (before parenthesis)
                        if '(' in name:
                            name = name.split('(')[0]
                        
                        # ONLY ADD KERNELS (skip CUDA API functions)
                        if not name.startswith('cuda'):
                            data.append({
                                'name': name,
                                'time_ms': total_time_ns / 1e6,  # Convert ns to ms
                                'time_percent': time_percent,
                                'calls': instances
                            })
                            print(f"  ✓ {name}: {time_percent}% ({total_time_ns/1e6:.2f}ms) {instances} calls")
                    except Exception as e:
                        continue
                        
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    print(f"\n✓ Total GPU operations parsed: {len(data)}")
    return data

def create_gpu_call_graph(data, output_file='gpu_call_graph.png'):
    """Create a call graph visualization"""
    
    if not data:
        print("No data to visualize")
        return
    
    # Sort by time
    data.sort(key=lambda x: x['time_ms'], reverse=True)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Color scheme
    colors = {
        'kernel': '#FF6B6B',
        'memcpy': '#4ECDC4',
        'api': '#45B7D1',
        'other': '#96CEB4'
    }
    
    y_pos = 0
    node_height = 0.8
    node_spacing = 1.2
    
    # Show ALL kernels found (no limit)
    num_items = len(data)
    
    for i, func in enumerate(data[:num_items]):
        # Determine color
        name_lower = func['name'].lower()
        if 'kernel' in name_lower or 'convolve' in name_lower or 'compute' in name_lower:
            color = colors['kernel']
        elif 'memcpy' in name_lower or 'memory' in name_lower:
            color = colors['memcpy']
        elif 'cuda' in name_lower:
            color = colors['api']
        else:
            color = colors['other']
        
        # Create rectangle - scale by percentage
        max_percent = data[0]['time_percent'] if data[0]['time_percent'] > 0 else 100
        width = (func['time_percent'] / max_percent) * 8 if func['time_percent'] > 0 else 2
        width = max(width, 1.5)  # Minimum width for visibility
        
        rect = FancyBboxPatch(
            (0, y_pos), width, node_height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add text
        label = f"{func['name'][:30]}\n{func['time_percent']:.1f}% ({func['time_ms']:.2f}ms)\n{func['calls']} calls"
        ax.text(width/2, y_pos + node_height/2, label, 
                ha='center', va='center', fontsize=8, weight='bold', wrap=True)
        
        y_pos += node_spacing
    
    # Customize
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, y_pos)
    ax.set_title('GPU Performance Profile - KLT Feature Tracking', fontsize=16, weight='bold')
    ax.set_xlabel('Relative Execution Time', fontsize=12)
    ax.set_ylabel('GPU Functions', fontsize=12)
    
    # Legend
    legend_elements = [
        patches.Patch(color=colors['kernel'], label='GPU Kernels'),
        patches.Patch(color=colors['memcpy'], label='Memory Operations'),
        patches.Patch(color=colors['api'], label='CUDA API'),
        patches.Patch(color=colors['other'], label='Other Operations')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ GPU call graph saved as {output_file}")
    
    try:
        plt.show()
    except:
        print("  (Unable to display plot - saved to file only)")

def create_summary_stats(data):
    """Create summary statistics"""
    if not data:
        return
    
    total_time = sum(d['time_ms'] for d in data)
    kernel_time = sum(d['time_ms'] for d in data 
                     if 'kernel' in d['name'].lower() or 'convolve' in d['name'].lower() or 'compute' in d['name'].lower())
    memcpy_time = sum(d['time_ms'] for d in data if 'memcpy' in d['name'].lower() or 'memory' in d['name'].lower())
    
    print("\n" + "="*60)
    print("GPU PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Total GPU Time:         {total_time:.2f} ms")
    
    if total_time > 0:
        print(f"Kernel Time:            {kernel_time:.2f} ms ({kernel_time/total_time*100:.1f}%)")
        print(f"Memory Transfer Time:   {memcpy_time:.2f} ms ({memcpy_time/total_time*100:.1f}%)")
        print(f"Compute Efficiency:     {kernel_time/total_time*100:.1f}%")
    
    print("\n" + "="*60)
    print("ALL GPU KERNEL FUNCTIONS")
    print("="*60)
    
    sorted_data = sorted(data, key=lambda x: x['time_ms'], reverse=True)
    for i, func in enumerate(sorted_data):
        print(f"{i+1:2d}. {func['name'][:40]:<40} {func['time_ms']:>8.2f}ms ({func['time_percent']:>5.1f}%) {func['calls']:>6d} calls")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Check command line arguments
    filename = 'klt_gpu_stats.txt'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    print(f"Parsing GPU profile from: {filename}")
    print("="*60)
    
    data = parse_nsys_stats(filename)
    
    if data:
        create_summary_stats(data)
        create_gpu_call_graph(data)
    else:
        print("\n❌ No GPU profiling data found in file")
        print("\nExpected file format: nsys stats output")
        print("\nTo generate the profile on GPU server:")
        print("  1. nsys profile -o klt_gpu_profile --stats=true ./example3_gpu")
        print("  2. nsys stats klt_gpu_profile.nsys-rep > klt_gpu_stats.txt")
        print("  3. scp 23I-0743@172.17.170.89:~/klt/klt_gpu_stats.txt .")
        print("  4. python3 gpu_profile_visualizer.py")


