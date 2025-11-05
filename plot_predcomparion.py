#!/usr/bin/env python
# coding: utf-8

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
if 'Arial' in _available_fonts:
    mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
})

values = [1.2753, 1.9812, 1.8880]           
labels = ['w/o', '5 genes', '10 genes']   
ylabel = 'Mean Squared Error'                        

fig, ax = plt.subplots(figsize=(6, 4))
x = range(len(values))
bars = ax.bar(x, values, width=0.4)

ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel(ylabel)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2.0, val,
            f'{val:.2f}', ha='center', va='bottom')

ymax = max(values) if values else 1.0
ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)

plt.tight_layout()

plt.savefig('pred_for_sfig7.png', dpi=300)
plt.show()
