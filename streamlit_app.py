
import os
import io
import zipfile
from datetime import datetime
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import functions from the pipeline
import sys
sys.path.append("/mnt/data")
from ig_analytics_pipeline import (
    load_all_posts,
    profile_engagement,
    caption_clustering,
    hashtag_analysis,
    posting_time_analysis,
    generate_calendar,
)

st.set_page_config(page_title="Instagram Analytics Dashboard", layout="wide")

st.title("📊 Instagram Analytics Dashboard")

with st.sidebar:
    st.header("Upload & Settings")
    uploaded = st.file_uploader("Upload Excel export (.xlsx)", type=["xlsx"])
    tz = st.selectbox("Timezone", ["Europe/London", "UTC", "America/New_York", "Europe/Berlin", "Asia/Dubai"], index=0)
    start_date = st.date_input("Start date for 90-day calendar (optional)")
    gen_calendar = st.checkbox("Generate 90-day calendar", value=True)
    run_btn = st.button("Run Analysis")

if run_btn and uploaded is not None:
    with st.spinner("Processing data..."):
        # Save uploaded file to a temp path
        input_path = "/mnt/data/_uploaded.xlsx"
        with open(input_path, "wb") as f:
            f.write(uploaded.read())

        # Load and compute core datasets
        posts_df, followers_map = load_all_posts(input_path)

        # Profile engagement summary
        prof_summary = profile_engagement(posts_df)

        # Caption clustering
        posts_df_clustered, cluster_keywords, engagement_by_cluster = caption_clustering(posts_df.copy())

        # Hashtag analysis
        top_hashtags, hashtag_vs_type, top_hashtag_posts = hashtag_analysis(posts_df_clustered.copy())

        # Posting time analysis
        hourly_engagement, heat_df, heatmap_path = posting_time_analysis(posts_df_clustered.copy(), tz_name=tz, out_dir="/mnt/data")

        # Calendar (optional)
        cal_files = (None, None, None)
        if gen_calendar:
            # Allow empty start date -> skip generation
            try:
                start_str = start_date.strftime("%Y-%m-%d")
                cal_files = generate_calendar("/mnt/data", start_date=start_str, tz_name=tz, days=90)
            except Exception:
                pass

    st.success("Analysis complete!")

    # Tabs for results
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Caption Themes", "Hashtags", "Timing", "Calendar"]
    )

    with tab1:
        st.subheader("Profile Engagement Summary")
        st.dataframe(prof_summary)

        st.download_button(
            "Download profile engagement CSV",
            data=prof_summary.to_csv(index=False).encode("utf-8"),
            file_name="engagement_by_profile.csv",
            mime="text/csv",
        )

        st.subheader("All Posts (cleaned)")
        st.dataframe(posts_df_clustered.head(200))

        st.download_button(
            "Download all posts CSV",
            data=posts_df_clustered.to_csv(index=False).encode("utf-8"),
            file_name="all_posts_unified.csv",
            mime="text/csv",
        )

    with tab2:
        st.subheader("Caption Cluster Keywords")
        ck_df = pd.DataFrame([(cid, ", ".join(words)) for cid, words in cluster_keywords.items()], columns=["Cluster", "Top Keywords"])
        st.dataframe(ck_df)

        st.subheader("Engagement by Caption Theme")
        st.dataframe(engagement_by_cluster.reset_index())

    with tab3:
        st.subheader("Top Hashtags by Engagement Rate")
        st.dataframe(top_hashtags.head(50))

        st.subheader("Hashtags vs No Hashtags (by Post Type)")
        st.dataframe(hashtag_vs_type)

        st.subheader("Top Hashtag Posts")
        st.dataframe(top_hashtag_posts)

    with tab4:
        st.subheader("Hourly Engagement (Avg ER %)")
        st.dataframe(hourly_engagement)

        st.subheader("Engagement Heatmap (Day × Hour)")
        if os.path.exists(heatmap_path):
            st.image(heatmap_path, use_column_width=True)

        st.subheader("Heatmap Data")
        st.dataframe(heat_df)

    with tab5:
        if cal_files[0]:
            excel_path, csv_path, ics_path = cal_files
            st.success("Calendar generated.")
            with open(excel_path, "rb") as f:
                st.download_button("Download calendar (Excel)", f, file_name=os.path.basename(excel_path))
            with open(csv_path, "rb") as f:
                st.download_button("Download calendar (CSV)", f, file_name=os.path.basename(csv_path))
            with open(ics_path, "rb") as f:
                st.download_button("Download calendar (.ics)", f, file_name=os.path.basename(ics_path))
        else:
            st.info("Enable 'Generate 90-day calendar' and choose a start date to build a schedule.")
else:
    st.info("Upload an Excel export and click 'Run Analysis' to begin.")
