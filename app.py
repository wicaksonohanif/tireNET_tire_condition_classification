# Import libraries
import streamlit as st
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# Import utils
from utils.model import load_model, load_config, load_test_results, get_model_info
from utils.image_processor import preprocess_image, predict, get_class_name

# Page configuration
st.set_page_config(
    page_title="TireNET | Tire Condition Monitoring System",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
    <style>
    .main {
        padding-top: 0px;
        max-width: 100%;
    }
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .result-container {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 20px 0;
    }
    .result-container.defective {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def initialize_resources():
    """Initialize model dan configuration"""
    device = get_device()
    
    # Get paths
    project_root = Path(__file__).parent
    model_path = project_root / "models" / "best_model.pth"
    config_path = project_root / "config" / "train_config.json"
    results_path = project_root / "results" / "test_results.json"
    
    # Check paths exist
    missing_files = []
    if not model_path.exists():
        missing_files.append(f"Model: {model_path}")
    if not config_path.exists():
        missing_files.append(f"Config: {config_path}")
    if not results_path.exists():
        missing_files.append(f"Results: {results_path}")
    
    if missing_files:
        error_msg = "Missing required files:\n" + "\n".join(missing_files)
        st.error(error_msg)
        return None, None, None, None, device
    
    # Load resources
    try:
        model = load_model(str(model_path), device=device)
        config = load_config(str(config_path))
        test_results = load_test_results(str(results_path))
        model_info = get_model_info(config, test_results)
        return model, config, test_results, model_info, device
    except Exception as e:
        st.error(f"❌ Error loading resources: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
        return None, None, None, None, device

# Load resources
model, config, test_results, model_info, device = initialize_resources()

# Sidebar Configuration
with st.sidebar:
    
    if model_info:
        # Model Selection (dropdown)
        st.markdown("### Model Selection")
        selected_model = st.selectbox(
            "Select Model",
            ["MobileNetV2 + CBAM"],
            label_visibility="collapsed"
        )
        
        # Model Information Box
        st.markdown("### Model Details")
        with st.container(border=False):
            st.markdown("""
            <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 18px; margin-right: 10px;">🧪</span>
                    <span><strong>Model:</strong> MobileNetV2 + CBAM</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 18px; margin-right: 10px;">🖼️</span>
                    <span><strong>Input Size:</strong> 224×224 pixels</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 18px; margin-right: 10px;">📂</span>
                    <span><strong>Classes:</strong> 2 Tire Types</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        
        # Tire Type Information
        st.markdown("### Tire Type Information")
        
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 15px;">
            <div style="display: flex; align-items: flex-start;">
                <span style="color: #4CAF50; font-size: 24px; margin-right: 15px;">●</span>
                <div>
                    <h3 style="margin: 0 0 10px 0; color: #333;">Good</h3>
                    <p style="margin: 0; color: #666; font-size: 14px;">Tire in optimal condition with adequate tread depth. The tire surface is even with no cracks, deep cuts, or uneven wear. The tire is safe to use and has excellent grip performance.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: flex-start;">
                <span style="color: #f44336; font-size: 24px; margin-right: 15px;">●</span>
                <div>
                    <h3 style="margin: 0 0 10px 0; color: #333;">Defective</h3>
                    <p style="margin: 0; color: #666; font-size: 14px;">Tire shows significant signs of wear including reduced tread depth, cracks, deep cuts, or uneven wear patterns. The tire may not be safe to use and requires immediate replacement to ensure driving safety.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        # System Information
        st.markdown("### System Info")
        st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 18px; margin-right: 10px;">🖥️</span>
                <span><strong>Device:</strong> {str(device).upper()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Main Content Area
st.write("")
st.write("")

# Banner
banner_path = Path(__file__).parent / "assets" / "banner.png"
if banner_path.exists():
    st.image(str(banner_path))
else:
    # Create placeholder banner if not exists
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 40px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0; font-size: 3em;">🚘 Tire Wear Classification</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 1.2em;">
            AI-Powered Tire Condition Monitoring System
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

if model is None:
    st.error("❌ Failed to load model. Please check the model file path and configuration.")
else:
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown("### 📂 Upload Image")
        uploaded_file = st.file_uploader(
            "Upload tire image for classification",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, width=300, caption="Uploaded Tire Image")
    
    with col2:
        st.markdown("### 🔍 Classification Result")
        
        if uploaded_file is not None:
            try:
                image_tensor, processed_image = preprocess_image(
                    image, 
                    image_size=config['image_size'],
                    device=device
                )
                
                predictions, predicted_class, confidence_score = predict(
                    model,
                    image_tensor,
                    device=device
                )
                
                class_names = {0: "Good", 1: "Defective"}
                predicted_class_name = get_class_name(predicted_class, class_names)
                
                result_color = "green" if predicted_class == 0 else "red"
                result_icon = "✅" if predicted_class == 0 else "⚠️"
                
                st.markdown(f"""
                <div class="result-container {'defective' if predicted_class == 1 else ''}">
                    <h2 style="margin: 0; color: {result_color};">
                        {result_icon} {predicted_class_name}
                    </h2>
                    <h1 style="margin: 10px 0; color: {result_color};">
                        {confidence_score:.2%}
                    </h1>
                    <p style="margin: 0; color: #666;">Confidence Score</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.predictions = predictions
                st.session_state.predicted_class = predicted_class
                st.session_state.confidence_score = confidence_score
                st.session_state.class_names = class_names
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
        else:
            st.markdown("""
            <div style="border: 3px dashed #1d67a1; padding: 15px; border-radius: 10px; background-color: #e5e5e5; color: #7f7f7f; text-align: center; font-size: 16px;">
                👆 Upload an image to see classification results
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Confidence Visualization Section
    if uploaded_file is not None and hasattr(st.session_state, 'predictions'):
        
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            st.markdown("### 🕡 Confidence Gauge")
            
            # Create gauge chart (speedometer style)
            predictions = st.session_state.predictions
            predicted_class = st.session_state.predicted_class
            confidence_score = st.session_state.confidence_score
            class_names = st.session_state.class_names
            
            predicted_class_name = class_names[predicted_class]
            
            fig_gauge = go.Figure(data=[go.Indicator(
                mode="gauge+number+delta",
                value=confidence_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': predicted_class_name, 'font': {'size': 20}},
                delta={'reference': 97.12, 'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#23a71f" if predicted_class == 0 else "#ff3c42"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 25], 'color': "#ffcdd2"},
                        {'range': [25, 50], 'color': "#fff9c4"},
                        {'range': [50, 75], 'color': "#c8e6c9"},
                        {'range': [75, 100], 'color': "#a5d6a7"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            )])
            
            fig_gauge.update_layout(
                font={'size': 15},
                height=320,
                margin=dict(l=20, r=20, t=60, b=40),
                autosize=True
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Class Confidence Comparison")
            
            # Create bar chart
            class_names = st.session_state.class_names
            predictions = st.session_state.predictions
            
            class_labels = list(class_names.values())
            confidence_values = [pred * 100 for pred in predictions]
            colors = ['#23a71f', '#ff3c42']
            
            fig_bar = go.Figure(data=[
                go.Bar(
                    y=class_labels,
                    x=confidence_values,
                    orientation='h',
                    marker=dict(
                        color=colors,
                        cornerradius="50%"
                    ),
                    text=[f'{val:.2f}%' for val in confidence_values],
                    textposition='outside',
                    hovertemplate='%{y}: %{x:.2f}%<extra></extra>'
                )
            ])
            
            fig_bar.update_layout(
                xaxis_title="Confidence (%)",
                yaxis_title="Class",
                height=400,
                margin=dict(l=20, r=20, t=60, b=20),
                xaxis=dict(range=[0, 100]),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                autosize=True
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
    
    else:
        st.info("ℹ️ Visualizations will appear here after uploading an image.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p><strong>TireNET V1.1</strong> | Tire Condition Monitoring System</p>
    <p>Made by <a href="https://github.com/wicaksonohanif" target="_blank">@wicaksonohanif</a></p>
</div>
""", unsafe_allow_html=True)
