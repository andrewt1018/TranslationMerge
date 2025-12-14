# bleu_charts.py

import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# MarianMT
models_marian = ["Base", "Expert", "Naive Avg", "WUDI"]
bleu_marian_zh = [34.46, 39.37, 24.03, 10.01]
bleu_marian_ja = [13.75, 19.91, 6.69, 3.12]

# Qwen 2.5-0.5B
models_qwen = ["Base", "Expert", "Naive Avg", "WUDI"]
bleu_qwen_zh = [12.90, 30.99, 14.05, 23.73]
bleu_qwen_ja = [4.13, 17.23, 3.91, 3.94]

def plot_bleu_grouped(models, bleu_zh, bleu_ja, title, out_path):
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 4.8))
    plt.bar(x - width/2, bleu_zh, width, label="En→Zh BLEU")
    plt.bar(x + width/2, bleu_ja, width, label="En→Ja BLEU")
    plt.xticks(x, models)
    plt.ylabel("BLEU")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_retention(models, expert_zh, expert_ja, merged_zh, merged_ja, title, out_path):
    """
    Retention = merged BLEU / expert BLEU.
    We plot retention only for methods that produce a merged model
    (Naive Avg, WUDI). Base/Expert are shown as reference lines.
    """
    methods = ["Naive Avg", "WUDI"]

    idx_avg = models.index("Naive Avg")
    idx_wudi = models.index("WUDI")

    retention_zh = [
        merged_zh[idx_avg] / expert_zh,
        merged_zh[idx_wudi] / expert_zh,
    ]
    retention_ja = [
        merged_ja[idx_avg] / expert_ja,
        merged_ja[idx_wudi] / expert_ja,
    ]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(7.2, 4.8))
    plt.bar(x - width/2, retention_zh, width, label="En→Zh retention")
    plt.bar(x + width/2, retention_ja, width, label="En→Ja retention")
    plt.xticks(x, methods)
    plt.ylabel("Retention (Merged / Expert)")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # BLEU bar charts
    plot_bleu_grouped(
        models_marian,
        bleu_marian_zh,
        bleu_marian_ja,
        "BLEU Comparison — MarianMT (Encoder–Decoder)",
        os.path.join(OUT_DIR, "bleu_marian.png"),
    )

    plot_bleu_grouped(
        models_qwen,
        bleu_qwen_zh,
        bleu_qwen_ja,
        "BLEU Comparison — Qwen 2.5-0.5B (Decoder-Only)",
        os.path.join(OUT_DIR, "bleu_qwen.png"),
    )

    # Optional: retention plots (merged vs expert)
    plot_retention(
        models_marian,
        expert_zh=39.37,
        expert_ja=19.91,
        merged_zh=bleu_marian_zh,
        merged_ja=bleu_marian_ja,
        title="Retention — MarianMT (Merged / Expert BLEU)",
        out_path=os.path.join(OUT_DIR, "retention_marian.png"),
    )

    plot_retention(
        models_qwen,
        expert_zh=30.99,
        expert_ja=17.23,
        merged_zh=bleu_qwen_zh,
        merged_ja=bleu_qwen_ja,
        title="Retention — Qwen 2.5-0.5B (Merged / Expert BLEU)",
        out_path=os.path.join(OUT_DIR, "retention_qwen.png"),
    )
