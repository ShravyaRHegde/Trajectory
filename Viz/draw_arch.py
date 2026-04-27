import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import io

# Force UTF-8 encoding for Windows terminals
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def draw_kasmu_v4_architecture():
    """
    Generates a high-quality schematic diagram of the KASMU v4 architecture
    using Matplotlib (No Graphviz dependency required).
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor('#0d0d0f')
    fig.patch.set_facecolor('#0d0d0f')
    
    # 1. Helper for rounded boxes
    def draw_box(x, y, w, h, label, color, text_color='white'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                      linewidth=2, edgecolor=color, facecolor='#1c1c1f', zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, color=text_color, ha='center', va='center', 
                fontsize=10, fontweight='bold', zorder=4)

    # 2. Draw Components
    # --- Input Stage ---
    draw_box(1, 6, 2, 1, "STATE HISTORY\n(20, 5)", '#38bdf8')
    
    # --- Encoder Stage ---
    draw_box(4, 6, 2, 1, "LSTM ENCODER\n(2 Layers)", '#38bdf8')
    ax.annotate('', xy=(4, 6.5), xytext=(3, 6.5), arrowprops=dict(arrowstyle='->', color='#38bdf8', lw=2))

    # --- Context Stage ---
    draw_box(4, 3, 2, 1, "LANE CONTEXT\n(1, 1)", '#10b981')
    draw_box(7, 3, 2, 1, "FiLM HEAD\n(Gamma, Beta)", '#10b981')
    ax.annotate('', xy=(7, 3.5), xytext=(6, 3.5), arrowprops=dict(arrowstyle='->', color='#10b981', lw=2))

    # --- Combining Block (FiLM Modulation) ---
    draw_box(7, 6, 2, 1, "LATENT MIXER\n(FiLM Modulation)", '#d946ef')
    # From Encoder to Mixer
    ax.annotate('', xy=(7, 6.5), xytext=(6, 6.5), arrowprops=dict(arrowstyle='->', color='#d946ef', lw=2))
    # From Context to Mixer
    ax.annotate('', xy=(8, 6), xytext=(8, 4), arrowprops=dict(arrowstyle='->', color='#d946ef', lw=2))

    # --- Decoder Stage ---
    draw_box(10, 6, 2, 1, "GRU DECODER\n(Recursive Cell)", '#d946ef')
    ax.annotate('', xy=(10, 6.5), xytext=(9, 6.5), arrowprops=dict(arrowstyle='->', color='#d946ef', lw=2))

    # --- Output Head ---
    draw_box(13, 6, 2, 1, "JERK HEADS\n(Multi-Head)", '#f59e0b')
    ax.annotate('', xy=(13, 6.5), xytext=(12, 6.5), arrowprops=dict(arrowstyle='->', color='#f59e0b', lw=2))

    # --- Integration Chain ---
    draw_box(13, 3, 2, 1, "PHYSICS ENGINE\n(Integration ΔJ)", '#f59e0b')
    ax.annotate('', xy=(14, 4), xytext=(14, 6), arrowprops=dict(arrowstyle='->', color='#f59e0b', lw=2))
    
    # --- Final Output ---
    draw_box(16, 3, 2.5, 1, "PREDICTED TRAJECTORY\n(30, 2, 3)", '#d946ef', text_color='#d946ef')
    ax.annotate('', xy=(16, 3.5), xytext=(15, 3.5), arrowprops=dict(arrowstyle='->', color='#d946ef', lw=2))

    # Feedback Loop logic
    ax.annotate('', xy=(11, 7), xytext=(14, 7), arrowprops=dict(arrowstyle='->', color='#d946ef', lw=1, connectionstyle="arc3,rad=.5"))
    ax.text(12.5, 7.8, "Recursive Feed (3.0s)", color='#d946ef', alpha=0.6, fontsize=8, ha='center')

    # Formatting
    ax.set_xlim(0, 20)
    ax.set_ylim(1, 9)
    plt.axis('off')
    plt.title("KASMU v4 : Kinematic-Aware Predictive Architecture", color='white', pad=20, fontsize=16, fontweight='bold')
    
    # Save
    output_path = "KASMU_Architecture_v4_Schematic.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0d0d0f')
    print(f"Architecture schematic saved successfully: {os.path.abspath(output_path)}")
    plt.close()

if __name__ == "__main__":
    draw_kasmu_v4_architecture()
