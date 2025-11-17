#!/usr/bin/env python3
"""
GPU Profile Visualizer for NVIDIA Nsight Systems output
Parses nsys stats output (including OS Runtime, CUDA API, GPU Kernels, MemOps)
and creates performance visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import re
import sys
import os

def parse_section(content, section_title):
    """Generic parser for a section in nsys stats output"""
    match = re.search(rf'\*\* {section_title}.*?\n(.*?)(?=\n\n|\*\*|Processing)', content, re.DOTALL)
    data = []
    if not match:
        return data
    
    section = match.group(1)
    lines = section.split('\n')

    for line in lines:
        if 'Time (%)' in line or '--------' in line or not line.strip():
            continue

        parts = line.split()
        if len(parts) >= 9:
            try:
                time_percent = float(parts[0])
                total_time_ns = int(parts[1].replace(',', ''))
                instances = int(parts[2].replace(',', ''))
                name = ' '.join(parts[8:])
                if '(' in name:
                    name = name.split('(')[0]

                data.append({
                    'name': name,
                    'time_ms': total_time_ns / 1e6,
                    'time_percent': time_percent,
                    'calls': instances,
                    'section': section_title
                })
            except Exception:
                continue
    return data

def parse_nsys_stats(filename):
    """Parse multiple sections of nsys stats text file"""
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found")
        return []
    
    with open(filename, 'r') as f:
        content = f.read()

    print(f"Parsing nsys stats output...\n")

    sections = [
        "OS Runtime Summary",
        "CUDA API Summary",
        "CUDA GPU Kernel Summary",
        "CUDA GPU MemOps Summary"
    ]
    all_data = []
    for s in sections:
        parsed = parse_section(content, s)
        if parsed:
            print(f"✓ Found {s} ({len(parsed)} entries)")
            all_data.extend(parsed)
    print(f"\n✓ Total operations parsed: {len(all_data)}")
    return all_data

def create_gpu_call_graph(data, output_file='gpu_call_graph.png'):
    """Create a call graph visualization"""
    if not data:
        print("No data to visualize")
        return
    
    # Separate GPU kernel data
    kernel_data = [d for d in data if d['section'] == 'CUDA GPU Kernel Summary']
    other_data = [d for d in data if d['section'] != 'CUDA GPU Kernel Summary']
    
    # Sort by execution time
    kernel_data.sort(key=lambda x: x['time_ms'], reverse=True)
    other_data.sort(key=lambda x: x['time_ms'], reverse=True)
    
    # Keep all kernels + top 5 others
    data = kernel_data + other_data[:10]
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    colors = {
        'CUDA GPU Kernel Summary': '#FF6B6B',
        'CUDA GPU MemOps Summary': '#4ECDC4',
        'CUDA API Summary': '#45B7D1',
        'OS Runtime Summary': '#96CEB4',
        'other': '#C7CEEA'
    }
    
    y_pos = 0
    node_height = 0.8
    node_spacing = 1.2
    
    for func in data:
        color = colors.get(func['section'], colors['other'])
        max_percent = data[0]['time_percent'] if data[0]['time_percent'] > 0 else 100
        width = (func['time_percent'] / max_percent) * 8
        width = max(width, 1.5)
        
        rect = FancyBboxPatch(
            (0, y_pos), width, node_height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        label = f"{func['name'][:30]}\n{func['time_percent']:.1f}% ({func['time_ms']:.2f}ms)\n{func['calls']} calls"
        ax.text(width/2, y_pos + node_height/2, label, 
                ha='center', va='center', fontsize=9, weight='bold', wrap=True)
        
        y_pos += node_spacing

    ax.set_xlim(0, 10)
    ax.set_ylim(-1, y_pos)
    ax.set_title(' Graph', fontsize=16, weight='bold')
    ax.set_xlabel('Relative Execution Time', fontsize=12)
    ax.set_ylabel('Functions', fontsize=12)
    
    legend_elements = [
        patches.Patch(color=colors['CUDA GPU Kernel Summary'], label='GPU Kernels'),
        patches.Patch(color=colors['CUDA GPU MemOps Summary'], label='Memory Operations'),
        patches.Patch(color=colors['CUDA API Summary'], label='CUDA API'),
        patches.Patch(color=colors['OS Runtime Summary'], label='OS Runtime')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graph saved as {output_file}")
    
    try:
        plt.show()
    except:
        print("  (Unable to display plot - saved to file only)")

def create_summary_stats(data):
    """Print performance summary"""
    total_time = sum(d['time_ms'] for d in data)
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Total Time: {total_time:.2f} ms\n")
    
    sections = sorted(set(d['section'] for d in data))
    for s in sections:
        section_time = sum(d['time_ms'] for d in data if d['section'] == s)
        percent = (section_time / total_time) * 100 if total_time else 0
        print(f"{s:<35} {section_time:>10.2f} ms ({percent:>5.1f}%)")
    print("="*60)

if __name__ == "__main__":
    filename = 'gpu_profile.txt'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    print(f"Parsing GPU profile from: {filename}")
    print("="*60)
    
    data = parse_nsys_stats(filename)
    if data:
        create_summary_stats(data)
        create_gpu_call_graph(data)
    else:
        print("❌ No profiling data found.")
