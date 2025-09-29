import streamlit as st
import pandas as pd
import os
from io import StringIO
from agents.cleaner import CleanerAgent
from agents.analyst import AnalystAgent
from agents.visualizer import VisualizerAgent
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AnalysisState(TypedDict):
    raw_data: pd.DataFrame
    cleaned_data: pd.DataFrame
    cleaning_metadata: dict
    validation_report: dict
    statistics: dict
    insights: list
    outliers: dict
    visualizations: dict
    report_path: str
    messages: Annotated[list, operator.add]


def load_data_node(state: AnalysisState) -> AnalysisState:
    state["messages"].append("✓ Data loaded successfully")
    return state


def cleaning_node(state: AnalysisState) -> AnalysisState:
    cleaner = CleanerAgent()
    cleaning_result = cleaner.clean_data(state["raw_data"])
    state["cleaned_data"] = cleaning_result["cleaned_data"]
    state["cleaning_metadata"] = cleaning_result["metadata"]
    state["validation_report"] = cleaner.validate_data(state["cleaned_data"])
    state["messages"].append(
        f"✓ Data cleaned: {state['cleaning_metadata']['rows_removed']} rows removed"
    )
    return state


def analysis_node(state: AnalysisState) -> AnalysisState:
    analyst = AnalystAgent()
    state["statistics"] = analyst.compute_statistics(state["cleaned_data"])
    state["insights"] = analyst.generate_insights(state["cleaned_data"])
    state["outliers"] = analyst.detect_outliers(state["cleaned_data"])
    state["messages"].append(
        f"✓ Analysis complete: {len(state['insights'])} insights generated"
    )
    return state


def visualization_node(state: AnalysisState) -> AnalysisState:
    visualizer = VisualizerAgent()
    dist_plots = visualizer.create_distribution_plots(state["cleaned_data"])
    heatmap = visualizer.create_correlation_heatmap(state["cleaned_data"])
    state["visualizations"] = {
        "distributions": dist_plots,
        "correlation_heatmap": heatmap
    }
    state["report_path"] = visualizer.create_summary_report(
        state["statistics"], state["insights"]
    )
    state["messages"].append(
        f"✓ Visualizations created: {len(dist_plots)} plots + heatmap"
    )
    return state


def create_workflow() -> StateGraph:
    workflow = StateGraph(AnalysisState)
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("clean_data", cleaning_node)
    workflow.add_node("analyze_data", analysis_node)
    workflow.add_node("visualize_data", visualization_node)
    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "clean_data")
    workflow.add_edge("clean_data", "analyze_data")
    workflow.add_edge("analyze_data", "visualize_data")
    workflow.add_edge("visualize_data", END)
    return workflow.compile()


st.set_page_config(
    page_title="Multi-Agent Data Analysis",
    page_icon="",
    layout="wide"
)

st.title(" Multi-Agent Data Analysis System")
st.markdown("**LangGraph | 3 Specialized Agents**")

with st.sidebar:
    st.header(" Agents")
    st.markdown("""
    -  CleanerAgent: Data validation & cleaning
    -  AnalystAgent: Statistical analysis
    -  VisualizerAgent: Chart generation
    """)
    
    st.header(" Configuration")
    st.info("Upload your CSV file to begin analysis")

tab1, tab2, tab3, tab4 = st.tabs([" Upload Data", " Analysis", " Visualizations", " Report"])

with tab1:
    st.header("Upload Your Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with at least one numeric column"
        )
        
        use_sample = st.checkbox("Use sample data (E-commerce dataset)", value=False)
    
    with col2:
        st.info("""
        **Requirements:**
        - CSV format
        - Headers in first row
        - At least 1 numeric column
        """)
    
    if uploaded_file or use_sample:
        try:
            if use_sample:
                df = pd.read_csv("sample_data.csv")
                st.success(" Sample data loaded!")
            else:
                df = pd.read_csv(uploaded_file)
                st.success(" File uploaded successfully!")
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10), width='stretch')
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", len(df))
            col2.metric("Columns", len(df.columns))
            col3.metric("Numeric Columns", len(df.select_dtypes(include=['number']).columns))
            
            st.session_state['df'] = df
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

with tab2:
    st.header("Analysis Results")
    
    if 'df' in st.session_state:
        if st.button("🚀 Run Analysis Pipeline", type="primary"):
            with st.spinner("Running multi-agent analysis..."):
                initial_state = {
                    "raw_data": st.session_state['df'],
                    "cleaned_data": None,
                    "cleaning_metadata": {},
                    "validation_report": {},
                    "statistics": {},
                    "insights": [],
                    "outliers": {},
                    "visualizations": {},
                    "report_path": "",
                    "messages": []
                }
                
                app = create_workflow()
                final_state = app.invoke(initial_state)
                
                st.session_state['results'] = final_state
            
            st.success(" Analysis complete!")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            
            st.subheader(" Pipeline Execution Log")
            for msg in results["messages"]:
                st.info(msg)
            
            st.subheader(" Key Insights")
            for i, insight in enumerate(results["insights"], 1):
                st.markdown(f"**{i}.** {insight}")
            
            st.subheader(" Data Quality Report")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Cleaning Summary**")
                meta = results["cleaning_metadata"]
                st.metric("Original Rows", meta["original_rows"])
                st.metric("Cleaned Rows", meta["cleaned_rows"])
                st.metric("Rows Removed", meta["rows_removed"])
            
            with col2:
                st.markdown("**Statistics Summary**")
                if results["statistics"].get("numeric_columns_stats"):
                    stats_df = pd.DataFrame(results["statistics"]["numeric_columns_stats"]).T
                    st.dataframe(stats_df, width='stretch')
            
            if results["outliers"]:
                st.subheader(" Outlier Detection")
                outlier_data = []
                for col, info in results["outliers"].items():
                    outlier_data.append({
                        "Column": col,
                        "Outliers": info["count"],
                        "Percentage": f"{info['percentage']:.2f}%"
                    })
                st.dataframe(pd.DataFrame(outlier_data), width='stretch')
    else:
        st.info(" Please upload data in the 'Upload Data' tab first")

with tab3:
    st.header("Data Visualizations")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        if results["visualizations"].get("correlation_heatmap"):
            st.subheader(" Correlation Heatmap")
            st.image(results["visualizations"]["correlation_heatmap"], width='stretch')
        
        if results["visualizations"].get("distributions"):
            st.subheader(" Distribution Plots")
            dist_plots = results["visualizations"]["distributions"]
            
            cols = st.columns(2)
            for idx, (col_name, filepath) in enumerate(dist_plots.items()):
                with cols[idx % 2]:
                    st.markdown(f"**{col_name}**")
                    st.image(filepath, width='stretch')
    else:
        st.info(" Run analysis first to see visualizations")

with tab4:
    st.header("Analysis Report")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        if results.get("report_path") and os.path.exists(results["report_path"]):
            with open(results["report_path"], 'r') as f:
                report_content = f.read()
            
            st.text_area("Full Report", report_content, height=400)
            
            st.download_button(
                label=" Download Report",
                data=report_content,
                file_name="analysis_report.txt",
                mime="text/plain"
            )
            
            if results.get("cleaned_data") is not None:
                csv = results["cleaned_data"].to_csv(index=False)
                st.download_button(
                    label=" Download Cleaned Data",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
    else:
        st.info(" Run analysis first to generate report")

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p> Streamlit & LangGraph | Multi-Agent System</p>
</div>
""", unsafe_allow_html=True)