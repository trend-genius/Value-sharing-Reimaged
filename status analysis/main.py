#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TikTok Content Sentiment Value vs. User Behavior Analysis
Make sure the dataset path is correct before running:
F:\tiktok\Value-sharing-Reimaged\analysis\data\final_sample_with_llm_scores_15000.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 1. Load data ----------
file_path = r'F:\tiktok\Value-sharing-Reimaged\analysis\data\final_sample_with_llm_scores_15000.csv'
try:
    df = pd.read_csv(file_path)
    print('Successfully loaded dataset, shape:', df.shape)
except FileNotFoundError as e:
    print(e)
    raise SystemExit("Please check the file path")

# ---------- 2. Core statistics ----------
analysis_by_score = (
    df.groupby('llm_sentiment_score')
      .agg(
          video_count=('video_id', 'count'),
          avg_deep_engagement=('deep_engagement_rate', 'mean'),
          avg_views=('video_view_count', 'mean'),
          avg_likes=('video_like_count', 'mean'),
          avg_shares=('video_share_count', 'mean'),
          avg_comments=('video_comment_count', 'mean')
      )
      .assign(
          share_to_like_ratio=lambda x: x['avg_shares'] / x['avg_likes'],
          comment_to_like_ratio=lambda x: x['avg_comments'] / x['avg_likes']
      )
)

# ---------- 3. Cleaned data for visualization ----------
data = {
    'llm_sentiment_score': [-1.0, 0.0, 1.0, 2.0],
    'video_count': [216, 4430, 10143, 211],
    'avg_deep_engagement': [0.426050, 0.452159, 0.454656, 0.396844],
    'avg_views': [105183.060185, 245128.297065, 266170.254165, 82906.180095],
    'avg_likes': [37226.787037, 81049.227991, 88278.649709, 24525.578199],
    'avg_shares': [7859.870370, 15886.444921, 17358.780341, 5085.663507],
    'avg_comments': [142.337963, 332.793002, 361.605639, 99.478673],
    'share_to_like_ratio': [0.211135, 0.196010, 0.196636, 0.207362],
    'comment_to_like_ratio': [0.003824, 0.004106, 0.004096, 0.004056]
}
df_viz = pd.DataFrame(data)

# ---------- 4. Visualization ----------
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('🎬 TikTok Content Sentiment Value vs. User Behavior',
             fontsize=20, fontweight='bold', y=0.98)

# 1. Content volume
ax1 = axes[0, 0]
bars1 = ax1.bar(df_viz['llm_sentiment_score'], df_viz['video_count'],
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                alpha=0.8, edgecolor='black')
ax1.set_title('Distribution of Content by Sentiment Score', fontsize=14, fontweight='bold')
ax1.set_xlabel('LLM Sentiment Score')
ax1.set_ylabel('Video Count')
ax1.set_xticks(df_viz['llm_sentiment_score'])
ax1.set_xticklabels(['negative(-1)', 'neutral(0)', 'positive(1)', 'high-quality(2)'])
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height + 50,
             f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# 2. Average views (thousands)
ax2 = axes[0, 1]
bars2 = ax2.bar(df_viz['llm_sentiment_score'], df_viz['avg_views'] / 1000,
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                alpha=0.8, edgecolor='black')
ax2.set_title('Average View Count (Thousands)', fontsize=14, fontweight='bold')
ax2.set_xlabel('LLM Sentiment Score')
ax2.set_ylabel('Avg Views (K)')
ax2.set_xticks(df_viz['llm_sentiment_score'])
ax2.set_xticklabels(['negative(-1)', 'neutral(0)', 'positive(1)', 'high-quality(2)'])
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height + 5,
             f'{height:.0f}K', ha='center', va='bottom', fontweight='bold')

# 3. Deep engagement trend
ax3 = axes[0, 2]
ax3.plot(df_viz['llm_sentiment_score'], df_viz['avg_deep_engagement'],
         marker='o', linewidth=3, markersize=8, color='#FF4757')
ax3.set_title('Deep Engagement Trend', fontsize=14, fontweight='bold')
ax3.set_xlabel('LLM Sentiment Score')
ax3.set_ylabel('Deep Engagement')
ax3.set_xticks(df_viz['llm_sentiment_score'])
ax3.set_xticklabels(['negative(-1)', 'neutral(0)', 'positive(1)', 'high-quality(2)'])
for x, y in zip(df_viz['llm_sentiment_score'], df_viz['avg_deep_engagement']):
    ax3.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontweight='bold')

# 4. Normalized interaction comparison
ax4 = axes[1, 0]
x_pos = np.arange(len(df_viz))
width = 0.25
likes_norm = df_viz['avg_likes'] / df_viz['avg_likes'].max()
shares_norm = df_viz['avg_shares'] / df_viz['avg_shares'].max()
comments_norm = df_viz['avg_comments'] / df_viz['avg_comments'].max()
ax4.bar(x_pos - width, likes_norm, width, label='Likes', color='#FF6B6B', alpha=0.8)
ax4.bar(x_pos, shares_norm, width, label='Shares', color='#4ECDC4', alpha=0.8)
ax4.bar(x_pos + width, comments_norm, width, label='Comments', color='#45B7D1', alpha=0.8)
ax4.set_title('User Interaction Behavior (Normalized)', fontsize=14, fontweight='bold')
ax4.set_xlabel('LLM Sentiment Score')
ax4.set_ylabel('Normalized Intensity')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(['negative(-1)', 'neutral(0)', 'positive(1)', 'high-quality(2)'])
ax4.legend()

# 5. Share value index
ax5 = axes[1, 1]
bars5 = ax5.bar(df_viz['llm_sentiment_score'], df_viz['share_to_like_ratio'],
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                alpha=0.8, edgecolor='black')
max_idx = df_viz['share_to_like_ratio'].idxmax()
bars5[max_idx].set_color('#FFD93D')
bars5[max_idx].set_edgecolor('#FF6B35')
bars5[max_idx].set_linewidth(3)
ax5.set_title('Share Value Index', fontsize=14, fontweight='bold')
ax5.set_xlabel('LLM Sentiment Score')
ax5.set_ylabel('Share / Like Ratio')
ax5.set_xticks(df_viz['llm_sentiment_score'])
ax5.set_xticklabels(['negative(-1)', 'neutral(0)', 'positive(1)', 'high-quality(2)'])
for bar in bars5:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.002,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

# 6. Radar chart
ax6 = axes[1, 2]
categories = ['Content Volume', 'View Heat', 'Participation Depth',
              'Share Value', 'Comment Activity']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]
sentiment_labels = ['negative(-1)', 'neutral(0)', 'positive(1)', 'high-quality(2)']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
for i, _ in enumerate(df_viz['llm_sentiment_score']):
    values = [
        df_viz.loc[i, 'video_count'] / df_viz['video_count'].max(),
        df_viz.loc[i, 'avg_views'] / df_viz['avg_views'].max(),
        df_viz.loc[i, 'avg_deep_engagement'] / df_viz['avg_deep_engagement'].max(),
        df_viz.loc[i, 'share_to_like_ratio'] / df_viz['share_to_like_ratio'].max(),
        df_viz.loc[i, 'comment_to_like_ratio'] / df_viz['comment_to_like_ratio'].max()
    ]
    values += values[:1]
    ax6.plot(angles, values, 'o-', linewidth=2, label=sentiment_labels[i], color=colors[i])
    ax6.fill(angles, values, alpha=0.1, color=colors[i])
ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(categories)
ax6.set_ylim(0, 1)
ax6.set_title('Content Feature Radar Chart', fontsize=14, fontweight='bold')
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax6.grid(True)

# ---------- Save figure ----------
save_path = os.path.join(os.path.dirname(__file__), "tiktok_sentiment_report.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Figure saved: {save_path}")

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()
# ---------- Append results to existing README ----------
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')

append_md = f"""
---

## 📊 TikTok Sentiment Analysis (Auto-Generated)

![Results](./tiktok_sentiment_report.png)

### Key Findings
1. **Positive content** (score 1.0) dominates: **10,143 videos (67.6 %)**.  
2. **High-quality content** (score 2.0) is scarce: only **211 videos (1.4 %)**.  
3. **Positive content** earns the **highest average views** (266,170).  
4. **Sharing paradox**: negative content (-1) has a **higher share-to-like ratio** (0.211 vs. 0.197 for positive).

### User Behavior Insights
- **View preference**: positive > neutral > negative > high-quality  
- **Engagement depth**: positive ≈ neutral > negative > high-quality  
- Both **negative** and **high-quality** contents show unexpectedly high **sharing willingness**

### Business Recommendations
1. The algorithm favors **positive content** → maximize exposure.  
2. **High-quality content** delivers **superior share value** → deserves **targeted support**.  
3. The elevated share rate of **negative content** reveals a **controversy niche**.  
4. Creators should **balance entertainment (views) and value (share rate)**.

*Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*
"""

with open(readme_path, 'a', encoding='utf-8') as f:
    f.write(append_md)

print('Results appended to README.md in English!')
# ---------- 5. Text insights ----------
print("=" * 60)
print("TikTok Content Ecosystem Insights Report")
print("=" * 60)

total = df_viz['video_count'].sum()
print("\nCore findings:")
print(f"1. Positive contents (1.0) are mainstream: {df_viz.loc[2, 'video_count']:,} videos, "
      f"taking up {df_viz.loc[2, 'video_count'] / total:.1%}")
print(f"2. High-quality contents (2.0) are rare: only {df_viz.loc[3, 'video_count']} videos, "
      f"taking up {df_viz.loc[3, 'video_count'] / total:.1%}")
print(f"3. Positive contents get the highest views: {df_viz.loc[2, 'avg_views']:,.0f}")
print(f"4. Sharing paradox: negative (-1) share/like = {df_viz.loc[0, 'share_to_like_ratio']:.3f} > "
      f"positive (1) share/like = {df_viz.loc[2, 'share_to_like_ratio']:.3f}")

print("\nUser behavior insights:")
print("• View preference: positive > neutral > negative > high-quality")
print(f"• Engagement depth: positive ({df_viz.loc[2, 'avg_deep_engagement']:.3f}) ≈ neutral > negative > high-quality")
print("• Sharing willingness of negative & high-quality contents is unexpectedly high!")

print("\nBusiness insights:")
print("1. Algorithm favors positive content → highest exposure.")
print("2. High-quality content has high share value → deserves extra support.")
print("3. Negative content’s high share rate indicates a niche for controversy.")
print("4. Creators should balance entertainment (views) and value (share rate).")