#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 20:06:09 2026

@author: keli
"""


import pandas as pd
import numpy as np

# ===================== 1. load data =====================

df_known = pd.read_csv(r'/Users/keli/Workstation/München/canon-shelf-main/lists/merged_all_books.csv', sep='\t')
rank_cols = [col for col in df_known.columns if col not in ['Title', 'Work Wikidata ID', 'Publication year (EDTF)']]
rank_year_map = {col: int(col.split("_")[0]) for col in rank_cols}
sorted_ranks = sorted(rank_cols, key=lambda x: rank_year_map[x])

prior_strength = 1

# ===================== 2. canon score calculation =====================
all_results = []

for i in range(1, len(sorted_ranks) + 1):
    current_ranks = sorted_ranks[:i]
    stage_name = "+".join(current_ranks)
    
    # years of lists, which are used for calculation
    current_rank_years = np.array([rank_year_map[col] for col in current_ranks])
    
    # only if a book's publication year <= years of the lists, the book can be considered for calculation
    valid_mask = df_known['Publication year (EDTF)'].values.reshape(-1, 1) <= current_rank_years
    
    temp = df_known[['Title', 'Publication year (EDTF)']].copy()
    temp["could_be_listed_count"] = valid_mask.sum(axis=1)
    temp["in_list_count"] = (df_known[current_ranks].values * valid_mask).sum(axis=1)
    temp["stage"] = stage_name
    
    # Filter out books that are not eligible for consideration in a / several lists
    temp = temp[temp["could_be_listed_count"] > 0].copy()
    
    # calculate global prior
    global_avg_rate = temp["in_list_count"].sum() / temp["could_be_listed_count"].sum()
    alpha = prior_strength * global_avg_rate
    beta = prior_strength * (1 - global_avg_rate)
    
    # calculate canon_score
    temp["canon_score"] = (alpha + temp["in_list_count"]) / (alpha + beta + temp["could_be_listed_count"])
    
    all_results.append(temp)

# ===================== 3. output results  =====================
full_result = pd.concat(all_results, ignore_index=True)
score_pivot = full_result.pivot(index="Title", columns="stage", values="canon_score")
score_pivot = score_pivot.sort_values(by="+".join(sorted_ranks), ascending=False)


score_pivot.to_csv(r'/Users/keli/Workstation/München/canon-shelf-main/canon_scores.csv', sep='\t', index=True)



# ===================== 4. Visualization  =====================

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'/Users/keli/Workstation/München/canon-shelf-main/canon_score_examples.csv', sep='\t')

book_col = "Title"

df_long = df.melt(
    id_vars=book_col,
    var_name="time(aggregated)",
    value_name="canonicity score"
)


# 提取阶段顺序（保持原CSV中的列顺序）
stage_labels = df.columns.drop(book_col).tolist()

sns.set_style("whitegrid")
g = sns.FacetGrid(
    df_long,
    col=book_col,
    col_wrap=3,
    height=3,
    sharex=True,
    sharey=True
)

# 2. 用 sns.lineplot 替代 plt.plot，对分类x轴兼容性更好
g.map_dataframe(
    sns.lineplot,
    x="time(aggregated)",
    y="canonicity score",
    marker="o",
    linewidth=2
)

# 3. 核心修复：遍历每个子图，强制设置刻度和标签
for ax in g.axes.flat:
    ax.set_xticks(range(len(stage_labels)))       # 固定刻度位置
    ax.set_xticklabels(stage_labels, rotation=90, ha="right", fontsize=8)  # 手动写入标签

# 4. 设置轴标题和边距
g.set_axis_labels("Cumulative Stages", "Bayesian Score")
g.set_titles(col_template="{col_name}")

# 调大底部和子图纵向间距，确保长标签不被截断
g.fig.subplots_adjust(bottom=0.28)#, hspace=0.6)

plt.show()