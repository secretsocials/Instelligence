
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instagram Analytics Pipeline
--------------------------------
Run this script regularly with a new Excel export to generate:
- Post-level & profile-level engagement metrics
- Caption clustering and engagement by theme
- Hashtag performance (incl. vs no-hashtags by post type)
- Posting time analysis + heatmap
- 90-day posting calendar (Excel, CSV, ICS)

Usage:
  python ig_analytics_pipeline.py --input /path/to/instagram_data.xlsx --start-date 2025-10-01 --output ./output --tz Europe/London

Notes:
- Charts use matplotlib only (no seaborn).
- Requires: pandas, numpy, scikit-learn, matplotlib, pytz
"""

import argparse
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pytz


def extract_number(text):
    """Extract integer from strings like '117.000follower' -> 117000; '1,930' -> 1930"""
    if pd.isna(text):
        return None
    if isinstance(text, (int, float)):
        try:
            return int(text)
        except Exception:
            return None
    s = str(text)
    nums = re.findall(r"[0-9\.,]+", s)
    if not nums:
        return None
    n = nums[0].replace(".", "").replace(",", "")
    try:
        return int(n)
    except Exception:
        try:
            return int(float(n))
        except Exception:
            return None


def load_all_posts(excel_path):
    """Parse the multi-sheet Excel export into a unified posts DataFrame and followers per profile."""
    xls = pd.ExcelFile(excel_path)
    all_posts = []
    followers_map = {}

    for sheet in xls.sheet_names:
        df = xls.parse(sheet)

        # Followers (second column of the 'Followers' row)
        try:
            followers_val = df[df.iloc[:, 0] == "Followers"].iloc[0, 1]
            followers_map[sheet] = extract_number(followers_val)
        except Exception:
            followers_map[sheet] = None

        # Find the "Post #" header row to locate post table
        try:
            start_idx_list = df.index[df.iloc[:, 0] == "Post #"].tolist()
            if not start_idx_list:
                continue
            start_idx = start_idx_list[0]
            posts_df = df.iloc[start_idx + 1 :].copy()
            posts_df.columns = [
                "Post #",
                "Caption",
                "Likes",
                "Comments",
                "Upload Date",
                "Upload Time",
                "Post Type",
                "Media Link",
                "Hashtags",
            ]

            # Clean types
            posts_df["Likes"] = posts_df["Likes"].apply(extract_number)
            posts_df["Comments"] = pd.to_numeric(posts_df["Comments"], errors="coerce")
            posts_df["Upload Date"] = pd.to_datetime(posts_df["Upload Date"], errors="coerce")
            posts_df["Upload Time"] = pd.to_datetime(posts_df["Upload Time"], errors="coerce").dt.time
            posts_df["Creator"] = sheet
            all_posts.append(posts_df)
        except Exception:
            continue

    if not all_posts:
        raise ValueError("No post tables found. Ensure each sheet has a 'Post #' section.")

    all_posts_df = pd.concat(all_posts, ignore_index=True)
    all_posts_df["Followers"] = all_posts_df["Creator"].map(followers_map)

    # Engagement Rate per post
    all_posts_df["Engagement Rate (%)"] = (
        (all_posts_df["Likes"].fillna(0) + all_posts_df["Comments"].fillna(0)) / all_posts_df["Followers"]
    ) * 100

    return all_posts_df, followers_map


def caption_clustering(all_posts_df, n_clusters=4, random_state=42):
    captions = all_posts_df["Caption"].dropna().astype(str).tolist()
    if len(captions) < n_clusters:
        n_clusters = max(1, min(2, len(captions)))  # fallback

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(captions)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X)

    # Map clusters back
    all_posts_df["Caption Cluster"] = np.nan
    all_posts_df.loc[all_posts_df["Caption"].notna(), "Caption Cluster"] = clusters

    # Top terms per cluster
    terms = vectorizer.get_feature_names_out()
    cluster_keywords = {}
    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        top_idx = center.argsort()[-12:][::-1]
        cluster_keywords[i] = [terms[j] for j in top_idx]

    # Engagement by cluster
    engagement_by_cluster = (
        all_posts_df.groupby("Caption Cluster")["Engagement Rate (%)"].mean().sort_values(ascending=False)
    )

    return all_posts_df, cluster_keywords, engagement_by_cluster


def hashtag_analysis(all_posts_df):
    # Flag presence
    all_posts_df["Has Hashtags"] = all_posts_df["Hashtags"].notna() & (all_posts_df["Hashtags"] != "No Hashtags")

    # Expand hashtags
    records = []
    hashtag_posts = all_posts_df[all_posts_df["Has Hashtags"]]
    for _, row in hashtag_posts.iterrows():
        tags = [t.strip().lower() for t in str(row["Hashtags"]).split(",") if t.strip()]
        for tag in tags:
            records.append(
                {
                    "Creator": row["Creator"],
                    "Hashtag": tag,
                    "Likes": row["Likes"],
                    "Comments": row["Comments"],
                    "Engagement Rate (%)": row["Engagement Rate (%)"],
                    "Post Type": row["Post Type"],
                }
            )
    hashtags_df = pd.DataFrame(records)

    # Summary tables
    hashtag_counts = hashtags_df["Hashtag"].value_counts()
    valid_tags = hashtag_counts[hashtag_counts >= 2].index  # filter out singletons to reduce outliers
    top_hashtags = (
        hashtags_df[hashtags_df["Hashtag"].isin(valid_tags)]
        .groupby("Hashtag")["Engagement Rate (%)"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    hashtag_vs_type = (
        all_posts_df.groupby(["Post Type", "Has Hashtags"])["Engagement Rate (%)"].mean().reset_index()
    )

    top_hashtag_posts = (
        hashtag_posts.sort_values("Engagement Rate (%)", ascending=False)
        .head(20)[["Creator", "Caption", "Hashtags", "Likes", "Comments", "Engagement Rate (%)", "Post Type"]]
    )

    return top_hashtags, hashtag_vs_type, top_hashtag_posts


def posting_time_analysis(all_posts_df, tz_name="Europe/London", out_dir="."):
    # Hour extraction
    all_posts_df["Upload Hour"] = pd.to_datetime(all_posts_df["Upload Time"], format="%H:%M:%S", errors="coerce").dt.hour

    hourly_engagement = (
        all_posts_df.groupby("Upload Hour")["Engagement Rate (%)"].mean().dropna().sort_index()
    ).reset_index()

    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    heat = (
        all_posts_df.groupby([all_posts_df["Upload Date"].dt.day_name(), "Upload Hour"])["Engagement Rate (%)"]
        .mean()
        .unstack()
        .reindex(dow_order)
    )

    # Plot heatmap with matplotlib only
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(heat.values, aspect="auto")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    ax.set_title("Engagement Rate Heatmap (Day × Hour)")
    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(list(heat.columns))
    ax.set_yticks(range(heat.shape[0]))
    ax.set_yticklabels(list(heat.index))
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Avg Engagement Rate (%)")
    plt.tight_layout()
    heatmap_path = os.path.join(out_dir, "engagement_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close(fig)

    return hourly_engagement, heat, heatmap_path


def profile_engagement(all_posts_df):
    # Average per post per profile
    avg_post = (
        all_posts_df.groupby("Creator")["Engagement Rate (%)"].mean().sort_values(ascending=False).reset_index()
    )
    # Total across all posts per profile
    def total_er(group):
        followers = group["Followers"].iloc[0]
        return ((group["Likes"].sum() + group["Comments"].sum()) / followers) * 100
    total_profile = (
        all_posts_df.groupby("Creator").apply(total_er).sort_values(ascending=False).reset_index(name="Total Engagement Across Posts (%)")
    )
    # Post counts
    post_counts = all_posts_df.groupby("Creator")["Post #"].count().reset_index(name="Posts")

    summary = avg_post.merge(total_profile, on="Creator").merge(post_counts, on="Creator")
    summary = summary.rename(columns={"Engagement Rate (%)": "Avg Engagement per Post (%)"})
    return summary


def generate_calendar(out_dir, start_date="2025-10-01", tz_name="Europe/London", days=90):
    tz = pytz.timezone(tz_name)
    start_dt = tz.localize(datetime.strptime(start_date, "%Y-%m-%d"))
    end_dt = start_dt + timedelta(days=days-1)

    # Weekly schedule: (weekday, hour, minute, slot, post_type, theme, use_hashtags, hashtag_set)
    weekly_slots = [
        (1, 7, 0,  "Morning Prime", "Video", "Release/Mix Teaser", True,  ["#newmusic", "#techhouse", "#housemusic"]),
        (2, 17, 0, "Evening Prime", "Video", "Event Recap / Gig Footage", False, []),
        (4, 17, 0, "Evening Prime", "Image/Carousel", "Lifestyle / Tour Story", False, []),
        (6, 9, 0,  "Morning Prime", "Video", "Collab / Highlight Reel", True,  ["#deeptech", "#minimaltech", "#techhouse"]),
    ]

    # Build schedule
    dates = []
    cur = start_dt
    while cur.date() <= end_dt.date():
        dates.append(cur)
        cur += timedelta(days=1)

    rows = []
    for dt in dates:
        for wd, hour, minute, slot, post_type, theme, use_hashtags, tagset in weekly_slots:
            if dt.weekday() == wd:
                post_dt = tz.localize(datetime(dt.year, dt.month, dt.day, hour, minute))
                rows.append({
                    "Date": post_dt.strftime("%Y-%m-%d"),
                    "Day": post_dt.strftime("%A"),
                    "Local Time": post_dt.strftime("%H:%M"),
                    "Slot": slot,
                    "Post Type": post_type,
                    "Content Theme": theme,
                    "Use Hashtags?": "Yes" if use_hashtags else "No",
                    "Suggested Hashtags": " ".join(tagset),
                    "Caption Style": "Hype & concise" if "Event" in theme or post_type=="Video" else "Storytelling & reflective",
                    "Notes": "Tag collaborators" if "Collab" in theme else ("Add CTA: save/share" if "Release" in theme else ""),
                })

    schedule_df = pd.DataFrame(rows)
    if schedule_df.empty:
        return None, None, None

    # Save files
    excel_path = os.path.join(out_dir, f"schedule_{start_date}_{days}d.xlsx")
    csv_path = os.path.join(out_dir, f"schedule_{start_date}_{days}d.csv")
    ics_path = os.path.join(out_dir, f"schedule_{start_date}_{days}d.ics")
    schedule_df.to_excel(excel_path, index=False)
    schedule_df.to_csv(csv_path, index=False)

    # ICS
    def make_ics(df, filename):
        def escape(text):
            return text.replace(",", "\\,").replace(";", "\\;")
        lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//AI//IG Strategy//EN"]
        for _, r in df.iterrows():
            dt_local = tz.localize(datetime.strptime(f"{r['Date']} {r['Local Time']}", "%Y-%m-%d %H:%M"))
            dt_utc = dt_local.astimezone(pytz.utc)
            dtend_utc = (dt_local + timedelta(hours=1)).astimezone(pytz.utc)
            dtstart_str = dt_utc.strftime("%Y%m%dT%H%M%SZ")
            dtend_str = dtend_utc.strftime("%Y%m%dT%H%M%SZ")
            uid = f"{dtstart_str}-{r['Post Type'].replace('/', '')}@ig-strategy"
            title = f"{r['Post Type']} – {r['Content Theme']}"
            desc = f"Slot: {r['Slot']}\\nUse Hashtags? {r['Use Hashtags?']}\\nSuggested: {r['Suggested Hashtags']}\\nCaption: {r['Caption Style']}\\nNotes: {r['Notes']}"
            lines += [
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTART:{dtstart_str}",
                f"DTEND:{dtend_str}",
                f"SUMMARY:{escape(title)}",
                f"DESCRIPTION:{escape(desc)}",
                "END:VEVENT"
            ]
        lines.append("END:VCALENDAR")
        with open(filename, "w") as f:
            f.write("\n".join(lines))

    make_ics(schedule_df, ics_path)

    return excel_path, csv_path, ics_path


def main():
    parser = argparse.ArgumentParser(description="Run Instagram analytics pipeline.")
    parser.add_argument("--input", required=True, help="Path to Excel file (multi-sheet Instagram export).")
    parser.add_argument("--output", default="./output", help="Output directory.")
    parser.add_argument("--start-date", default=None, help="Start date for 90-day calendar (YYYY-MM-DD). Optional.")
    parser.add_argument("--tz", default="Europe/London", help="Timezone for calendar (IANA tz database name).")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load & clean
    print("[1/6] Loading data...")
    posts_df, followers_map = load_all_posts(args.input)

    # Save raw unified posts
    posts_path = os.path.join(args.output, "all_posts_unified.csv")
    posts_df.to_csv(posts_path, index=False)

    # Profile engagement
    print("[2/6] Computing engagement metrics...")
    profile_summary = profile_engagement(posts_df)
    profile_summary_path = os.path.join(args.output, "engagement_by_profile.csv")
    profile_summary.to_csv(profile_summary_path, index=False)

    # Caption clustering
    print("[3/6] Clustering captions & scoring themes...")
    posts_df, cluster_keywords, engagement_by_cluster = caption_clustering(posts_df.copy())
    cluster_kw_path = os.path.join(args.output, "caption_cluster_keywords.csv")
    pd.DataFrame(
        [(cid, ", ".join(words)) for cid, words in cluster_keywords.items()],
        columns=["Cluster", "Top Keywords"]
    ).to_csv(cluster_kw_path, index=False)
    engagement_by_cluster_path = os.path.join(args.output, "engagement_by_caption_cluster.csv")
    engagement_by_cluster.reset_index().to_csv(engagement_by_cluster_path, index=False)

    # Hashtag analysis
    print("[4/6] Analyzing hashtags...")
    top_hashtags, hashtag_vs_type, top_hashtag_posts = hashtag_analysis(posts_df.copy())
    top_hashtags_path = os.path.join(args.output, "top_hashtags.csv")
    top_hashtags.to_csv(top_hashtags_path, index=False)
    hashtag_vs_type_path = os.path.join(args.output, "hashtag_vs_post_type.csv")
    hashtag_vs_type.to_csv(hashtag_vs_type_path, index=False)
    top_hashtag_posts_path = os.path.join(args.output, "top_hashtag_posts.csv")
    top_hashtag_posts.to_csv(top_hashtag_posts_path, index=False)

    # Posting time analysis
    print("[5/6] Computing posting time analytics...")
    hourly_engagement, heat_df, heatmap_path = posting_time_analysis(posts_df.copy(), args.tz, args.output)
    hourly_engagement_path = os.path.join(args.output, "hourly_engagement.csv")
    hourly_engagement.to_csv(hourly_engagement_path, index=False)
    heat_df_path = os.path.join(args.output, "engagement_heatmap_data.csv")
    heat_df.to_csv(heat_df_path)

    # Calendar (optional)
    cal_paths = (None, None, None)
    if args.start_date:
        print("[6/6] Generating 90-day calendar...")
        cal_paths = generate_calendar(args.output, args.start_date, args.tz, days=90)

    # Write a simple README
    readme = f"""Instagram Analytics Pipeline — Outputs

Input file: {args.input}

Generated files in: {os.path.abspath(args.output)}

1) Engagement
- all_posts_unified.csv
- engagement_by_profile.csv

2) Caption Themes
- caption_cluster_keywords.csv
- engagement_by_caption_cluster.csv

3) Hashtags
- top_hashtags.csv
- hashtag_vs_post_type.csv
- top_hashtag_posts.csv

4) Posting Time
- hourly_engagement.csv
- engagement_heatmap_data.csv
- engagement_heatmap.png

5) Calendar (if --start-date was provided)
- schedule_{{start-date}}_90d.xlsx / .csv / .ics

How to run:
python ig_analytics_pipeline.py --input /path/to/instagram_data.xlsx --start-date 2025-10-01 --output ./output --tz Europe/London
"""
    with open(os.path.join(args.output, "README.txt"), "w") as f:
        f.write(readme)

    print("Done.")
    print(f"Outputs written to: {os.path.abspath(args.output)}")
    if cal_paths[0]:
        print("Calendar files:", cal_paths)


if __name__ == "__main__":
    main()
