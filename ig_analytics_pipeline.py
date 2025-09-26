#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, re
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pytz

def extract_number(text) -> Optional[int]:
    if pd.isna(text): return None
    if isinstance(text, (int, float)):
        try: return int(text)
        except Exception: return None
    s = str(text)
    nums = re.findall(r"[0-9\.,]+", s)
    if not nums: return None
    n = nums[0].replace(".", "").replace(",", "")
    try: return int(n)
    except Exception:
        try: return int(float(n))
        except Exception: return None

def load_all_posts(excel_path: str) -> Tuple[pd.DataFrame, Dict[str, Optional[int]]]:
    xls = pd.ExcelFile(excel_path)  # requires openpyxl
    all_posts: List[pd.DataFrame] = []
    followers_map: Dict[str, Optional[int]] = {}
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        # Followers
        try:
            followers_val = df[df.iloc[:, 0] == "Followers"].iloc[0, 1]
            followers_map[sheet] = extract_number(followers_val)
        except Exception:
            followers_map[sheet] = None
        # Posts table
        try:
            start_idx_list = df.index[df.iloc[:, 0] == "Post #"].tolist()
            if not start_idx_list: continue
            start_idx = start_idx_list[0]
            posts_df = df.iloc[start_idx+1:].copy()
            posts_df.columns = ["Post #","Caption","Likes","Comments","Upload Date","Upload Time","Post Type","Media Link","Hashtags"]
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
    followers_nonzero = all_posts_df["Followers"].replace(0, np.nan)
    all_posts_df["Engagement Rate (%)"] = ((all_posts_df["Likes"].fillna(0)+all_posts_df["Comments"].fillna(0))/followers_nonzero)*100
    return all_posts_df, followers_map

def caption_clustering(all_posts_df: pd.DataFrame, n_clusters:int=4, random_state:int=42):
    import numpy as np
    captions_series = all_posts_df["Caption"].dropna().astype(str)
    captions = captions_series.tolist()
    if len(captions)==0:
        all_posts_df["Caption Cluster"] = np.nan
        return all_posts_df, {}, pd.Series(dtype=float)
    if len(captions) < n_clusters:
        n_clusters = max(1, min(2, len(captions)))
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(captions)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X)
    all_posts_df = all_posts_df.copy()
    all_posts_df["Caption Cluster"] = np.nan
    all_posts_df.loc[captions_series.index, "Caption Cluster"] = clusters
    terms = vectorizer.get_feature_names_out()
    cluster_keywords = {}
    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        top_idx = center.argsort()[-12:][::-1]
        cluster_keywords[i] = [terms[j] for j in top_idx]
    engagement_by_cluster = all_posts_df.groupby("Caption Cluster")["Engagement Rate (%)"].mean().sort_values(ascending=False)
    return all_posts_df, cluster_keywords, engagement_by_cluster

def hashtag_analysis(all_posts_df: pd.DataFrame):
    df = all_posts_df.copy()
    df["Has Hashtags"] = df["Hashtags"].notna() & (df["Hashtags"]!="No Hashtags")
    records = []
    hashtag_posts = df[df["Has Hashtags"]]
    for _, row in hashtag_posts.iterrows():
        tags = [t.strip().lower() for t in str(row["Hashtags"]).split(",") if t.strip()]
        for tag in tags:
            records.append({"Creator":row["Creator"],"Hashtag":tag,"Likes":row["Likes"],"Comments":row["Comments"],"Engagement Rate (%)":row["Engagement Rate (%)"],"Post Type":row["Post Type"]})
    hashtags_df = pd.DataFrame(records)
    if not hashtags_df.empty:
        counts = hashtags_df["Hashtag"].value_counts()
        valid = counts[counts>=2].index
        top_hashtags = hashtags_df[hashtags_df["Hashtag"].isin(valid)].groupby("Hashtag")["Engagement Rate (%)"].mean().sort_values(ascending=False).reset_index()
    else:
        top_hashtags = pd.DataFrame(columns=["Hashtag","Engagement Rate (%)"])
    hashtag_vs_type = df.groupby(["Post Type","Has Hashtags"])["Engagement Rate (%)"].mean().reset_index()
    top_hashtag_posts = hashtag_posts.sort_values("Engagement Rate (%)", ascending=False).head(20)[["Creator","Caption","Hashtags","Likes","Comments","Engagement Rate (%)","Post Type"]]
    return top_hashtags, hashtag_vs_type, top_hashtag_posts

def posting_time_analysis(all_posts_df: pd.DataFrame, tz_name:str="Europe/London", out_dir:str="."):
    df = all_posts_df.copy()
    df["Upload Hour"] = pd.to_datetime(df["Upload Time"], format="%H:%M:%S", errors="coerce").dt.hour
    hourly_engagement = df.groupby("Upload Hour")["Engagement Rate (%)"].mean().dropna().sort_index().reset_index()
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    heat = df.groupby([df["Upload Date"].dt.day_name(),"Upload Hour"])["Engagement Rate (%)"].mean().unstack().reindex(dow_order)
    fig, ax = plt.subplots(figsize=(12,5))
    im = ax.imshow(heat.values, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Day of Week"); ax.set_title("Engagement Rate Heatmap (Day × Hour)")
    ax.set_xticks(range(heat.shape[1])); ax.set_xticklabels(list(heat.columns))
    ax.set_yticks(range(heat.shape[0])); ax.set_yticklabels(list(heat.index))
    cbar = plt.colorbar(im, ax=ax); cbar.set_label("Avg Engagement Rate (%)")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    heatmap_path = os.path.join(out_dir, "engagement_heatmap.png")
    plt.savefig(heatmap_path, dpi=150); plt.close(fig)
    return hourly_engagement, heat, heatmap_path

def profile_engagement(all_posts_df: pd.DataFrame) -> pd.DataFrame:
    df = all_posts_df.copy()
    avg_post = df.groupby("Creator")["Engagement Rate (%)"].mean().sort_values(ascending=False).reset_index()
    def total_er(group: pd.DataFrame) -> float:
        followers = group["Followers"].iloc[0]
        if not followers or followers==0 or pd.isna(followers): return float("nan")
        return ((group["Likes"].sum()+group["Comments"].sum())/followers)*100
    total_profile = df.groupby("Creator").apply(total_er).sort_values(ascending=False).reset_index(name="Total Engagement Across Posts (%)")
    post_counts = df.groupby("Creator")["Post #"].count().reset_index(name="Posts")
    summary = avg_post.merge(total_profile, on="Creator").merge(post_counts, on="Creator")
    summary = summary.rename(columns={"Engagement Rate (%)":"Avg Engagement per Post (%)"})
    return summary

def generate_calendar(out_dir:str, start_date:str="2025-10-01", tz_name:str="Europe/London", days:int=90):
    tz = pytz.timezone(tz_name)
    start_dt = tz.localize(datetime.strptime(start_date, "%Y-%m-%d"))
    end_dt = start_dt + timedelta(days=days-1)
    weekly_slots = [
        (1,7,0,"Morning Prime","Video","Release/Mix Teaser",True,["#newmusic","#techhouse","#housemusic"]),
        (2,17,0,"Evening Prime","Video","Event Recap / Gig Footage",False,[]),
        (4,17,0,"Evening Prime","Image/Carousel","Lifestyle / Tour Story",False,[]),
        (6,9,0,"Morning Prime","Video","Collab / Highlight Reel",True,["#deeptech","#minimaltech","#techhouse"]),
    ]
    dates=[]; cur=start_dt
    while cur.date()<=end_dt.date():
        dates.append(cur); cur += timedelta(days=1)
    rows=[]
    for dt in dates:
        for wd,hour,minute,slot,post_type,theme,use_hashtags,tagset in weekly_slots:
            if dt.weekday()==wd:
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
    if schedule_df.empty: return None, None, None
    os.makedirs(out_dir, exist_ok=True)
    excel_path = os.path.join(out_dir, f"schedule_{start_date}_{days}d.xlsx")
    csv_path = os.path.join(out_dir, f"schedule_{start_date}_{days}d.csv")
    ics_path = os.path.join(out_dir, f"schedule_{start_date}_{days}d.ics")
    schedule_df.to_excel(excel_path, index=False)
    schedule_df.to_csv(csv_path, index=False)
    def make_ics(df: pd.DataFrame, filename: str):
        def escape(t:str)->str: return t.replace(",","\\,").replace(";","\\;")
        lines=["BEGIN:VCALENDAR","VERSION:2.0","PRODID:-//AI//IG Strategy//EN"]
        for _,r in df.iterrows():
            dt_local = tz.localize(datetime.strptime(f"{r['Date']} {r['Local Time']}", "%Y-%m-%d %H:%M"))
            dt_utc = dt_local.astimezone(pytz.utc)
            dtend_utc = (dt_local + timedelta(hours=1)).astimezone(pytz.utc)
            dtstart_str = dt_utc.strftime("%Y%m%dT%H%M%SZ")
            dtend_str = dtend_utc.strftime("%Y%m%dT%H%M%SZ")
            uid = f"{dtstart_str}-{r['Post Type'].replace('/', '')}@ig-strategy"
            title = f"{r['Post Type']} – {r['Content Theme']}"
            desc = f"Slot: {r['Slot']}\\nUse Hashtags? {r['Use Hashtags?']}\\nSuggested: {r['Suggested Hashtags']}\\nCaption: {r['Caption Style']}\\nNotes: {r['Notes']}"
            lines += ["BEGIN:VEVENT",f"UID:{uid}",f"DTSTART:{dtstart_str}",f"DTEND:{dtend_str}",f"SUMMARY:{escape(title)}",f"DESCRIPTION:{escape(desc)}","END:VEVENT"]
        lines.append("END:VCALENDAR")
        with open(filename,"w") as f: f.write("\n".join(lines))
    make_ics(schedule_df, ics_path)
    return excel_path, csv_path, ics_path

def main():
    parser = argparse.ArgumentParser(description="Run Instagram analytics pipeline.")
    parser.add_argument("--input", required=True); parser.add_argument("--output", default="./output")
    parser.add_argument("--start-date", default=None); parser.add_argument("--tz", default="Europe/London")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    posts_df,_ = load_all_posts(args.input)
    posts_df.to_csv(os.path.join(args.output,"all_posts_unified.csv"), index=False)
    summary = profile_engagement(posts_df)
    summary.to_csv(os.path.join(args.output,"engagement_by_profile.csv"), index=False)
    posts_df, cluster_keywords, engagement_by_cluster = caption_clustering(posts_df.copy())
    pd.DataFrame([(cid, ", ".join(words)) for cid, words in cluster_keywords.items()], columns=["Cluster","Top Keywords"]).to_csv(os.path.join(args.output,"caption_cluster_keywords.csv"), index=False)
    engagement_by_cluster.reset_index().to_csv(os.path.join(args.output,"engagement_by_caption_cluster.csv"), index=False)
    top_hashtags, hashtag_vs_type, top_hashtag_posts = hashtag_analysis(posts_df.copy())
    top_hashtags.to_csv(os.path.join(args.output,"top_hashtags.csv"), index=False)
    hashtag_vs_type.to_csv(os.path.join(args.output,"hashtag_vs_post_type.csv"), index=False)
    top_hashtag_posts.to_csv(os.path.join(args.output,"top_hashtag_posts.csv"), index=False)
    hourly_engagement, heat_df, heatmap_path = posting_time_analysis(posts_df.copy(), args.tz, args.output)
    hourly_engagement.to_csv(os.path.join(args.output,"hourly_engagement.csv"), index=False)
    heat_df.to_csv(os.path.join(args.output,"engagement_heatmap_data.csv"))
    if args.start_date: generate_calendar(args.output, args.start_date, args.tz, days=90)
    print("Done. Outputs in:", os.path.abspath(args.output))

if __name__=="__main__":
    main()
