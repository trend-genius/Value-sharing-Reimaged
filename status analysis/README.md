README.md
=============

A concise, end-to-end analysis of **15 000 TikTok videos** grouped by LLM-generated sentiment scores (`-1`, `0`, `1`, `2`).  
The script loads data, aggregates key metrics, plots six charts, and prints actionable business insights—**all in one click**.

---

📁 Project Layout
-----------------
```
status analysis/
├── tiktok_sentiment_analysis.py   # main script
├── README.md                      
└── data/
    └── final_sample_with_llm_scores_15000.csv
```

🚀 Quick Start
--------------
1. **Place your dataset**  
   Ensure `final_sample_with_llm_scores_15000.csv` is inside the `data/` folder (or edit `file_path` in the script).

2. **Run the analysis**  
   ```bash
   python tiktok_sentiment_analysis.py
   ```

3. **Results**  
   - **Console**: key findings & business insights.  
   - **Figure**: `tiktok_sentiment_report.png` saved in the same directory (300 dpi).

---

📊 What You Get
---------------
| Plot | Description |
|------|-------------|
| **Content Volume** | Number of videos per sentiment class. |
| **Average Views** | Mean view counts (in thousands). |
| **Deep Engagement Trend** | Line chart of average deep-engagement rate. |
| **Interaction Comparison** | Normalized likes, shares, comments side-by-side. |
| **Share Value Index** | Share-to-like ratio highlighting the “sharing paradox”. |
| **Radar Chart** | Five-dimension feature profile for each class. |

---

🔍 Key Insights Preview
-----------------------
- **Positive (1.0)** content dominates: ~67 % of all videos and highest average views.  
- **High-quality (2.0)** content is scarce (~1 %) but drives the **highest share value**.  
- **Negative (-1)** content shows a **higher share/like ratio** despite lower views—evidence of controversy appeal.  
- Platform algorithm clearly favors positive sentiment; creators should balance **reach vs. shareability**.

---

🛠️ Customization Tips
---------------------
- **Change colors**: edit the `color=` lists in each plotting block.  
- **Add new metrics**: extend the `.agg()` dictionary in section 2.  
- **Save individual subplots**: uncomment the loop provided in the script.

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

---

## 📄 License

m13052784890@163.com

Feel free to open issues or PRs!
