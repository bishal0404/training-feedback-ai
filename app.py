import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from analyzer_engine import classify_columns, compute_stats, group_into_sections
# Added get_sentiment to the import
from nlp_processor import summarize_feedback, get_sentiment 

st.set_page_config(page_title="Training Feedback Intelligence", layout="wide")

st.title("📊 Training Feedback Intelligence")
st.markdown("Upload your training feedback Excel file for instant AI-driven insights.")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
        
    rating_cols, text_cols, ignored_cols = classify_columns(df)
    stats = compute_stats(df, rating_cols)
    valid_rating_qs = list(stats.keys())
    sections = group_into_sections(valid_rating_qs)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview", "📊 Question Analysis", "🧩 Section Insights", "📝 Text Feedback", "📄 Raw Data"
    ])

    with tab1:
        st.subheader("Analysis Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Responses", len(df))
        col2.metric("Rating Questions Analyzed", len(valid_rating_qs))
        
        if stats:
            avg_score = np.mean([s['mean'] for s in stats.values()])
            col3.metric("Global Average Score", f"{avg_score:.2f} / 5")

    with tab2:
        st.subheader("Question-Wise Detailed Analysis")
        if stats:
            # 1. Create a consolidated Summary Table for all questions
            summary_data = []
            for q_name, q_data in stats.items():
                row = {
                    "Question": q_name,
                    "Mean": q_data['mean'],
                    "Std Dev": q_data['std'],
                }
                # Add individual counts (1-5) to the row
                row.update({f"{i}s": q_data['counts'][i] for i in range(1, 6)})
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            
            st.write("#### Consolidated Summary Table")
            st.table(summary_df.set_index("Question"))

            st.divider()

            # 2. Create a single Bar Chart for all questions
            st.write("#### Mean Scores by Question")
            
            fig = px.bar(
                summary_df, 
                x='Question', 
                y='Mean',
                text='Mean',  # Displays the mean value above/on the bar
                color='Mean',
                color_continuous_scale='RdYlGn',
                range_color=[1, 5],
                labels={'Mean': 'Average Rating', 'Question': 'Feedback Question'},
                height=600
            )

            # Improve label visibility
            fig.update_traces(textposition='outside', textfont_size=12)
            fig.update_layout(
                yaxis_range=[0, 5.5],
                xaxis_tickangle=-45, # Slant text if questions are long
                margin=dict(b=150)    # Extra bottom margin for long question text
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("No rating data available.")

    with tab3:
        st.subheader("Section-Wise Performance")
        sec_rows = []
        
        with st.expander("🔍 See Question Grouping Logic", expanded=False):
            for sec, qs in sections.items():
                st.markdown(f"**{sec}**")
                for q in qs:
                    st.write(f"  • {q}")

        for sec, qs in sections.items():
            valid_qs = [q for q in qs if q in stats]
            if valid_qs:
                all_raw_data = []
                total_counts = {1:0, 2:0, 3:0, 4:0, 5:0}
                for q in valid_qs:
                    raw_values = pd.to_numeric(df[q], errors='coerce').dropna()
                    raw_values = raw_values[(raw_values >= 1) & (raw_values <= 5)]
                    all_raw_data.extend(raw_values.tolist())
                    for val in range(1, 6):
                        total_counts[val] += stats[q]['counts'][val]
                
                if all_raw_data:
                    sec_rows.append({
                        "Section": sec, "1s": total_counts[1], "2s": total_counts[2], 
                        "3s": total_counts[3], "4s": total_counts[4], "5s": total_counts[5],
                        "Mean": round(float(np.mean(all_raw_data)), 2),
                        "Std Dev": round(float(np.std(all_raw_data)), 2)
                    })

        if sec_rows:
            res_df = pd.DataFrame(sec_rows)
            st.write("#### Section Summary Table")
            st.table(res_df.set_index("Section"))
            
            # --- FIX: Narrowing the Section Chart ---
            st.write("#### Mean Rating by Section")
            col_l, col_m, col_r = st.columns([1, 2, 1]) # 2 is the relative width of the center
            with col_m:
                fig_sec = px.bar(
                    res_df, x='Section', y='Mean', color='Mean', 
                    color_continuous_scale='RdYlGn', range_color=[1, 5], 
                    text='Mean'
                )
                fig_sec.update_traces(textposition='outside')
                fig_sec.update_layout(yaxis_range=[0, 5.5], height=450)
                st.plotly_chart(fig_sec, use_container_width=True, key="section_bar_chart")

    with tab4:
        st.subheader("Textual Feedback Insights")
        if text_cols:
            for i, col in enumerate(text_cols):
                with st.expander(f"Analysis for: {col}", expanded=(i==0)):
                    # CLEANUP: Drop NA before processing to avoid errors
                    clean_text = df[col].dropna().astype(str)
                    if not clean_text.empty:
                        sentiments = clean_text.apply(get_sentiment)
                        sent_counts = sentiments.value_counts()
                        
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            fig_pie = px.pie(
                                values=sent_counts.values, 
                                names=sent_counts.index, 
                                color=sent_counts.index,
                                # Custom colors for sentiment
                                color_discrete_map={'Positive':'#2ca02c', 'Neutral':'#ff7f0e', 'Negative':'#d62728'},
                                title="Sentiment Distribution"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{i}")
                            
                        with c2:
                            st.write("**Key Suggestions / Feedback:**")
                            summaries = summarize_feedback(clean_text)
                            for s in summaries:
                                st.markdown(f"- {s}")
                    else:
                        st.write("No text data in this column.")
        else:
            st.info("No text-based feedback columns detected.")

    with tab5:
        st.subheader("Raw Data")
        st.dataframe(df)

else:
    st.info("Please upload an Excel file to begin.")