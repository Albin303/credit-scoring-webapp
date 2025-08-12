import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import traceback

# Configure page
st.set_page_config(
    page_title="AI Credit Score Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6 0%, #e8ecf0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .error-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 1px solid #f5c6cb;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .debug-info {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def check_model_files():
    """Check if all required model files exist"""
    model_files = {
        'models/credit_score_mlp_model.h5': 'MLP Neural Network Model',
        'models/quantile_transformer.pkl': 'Quantile Transformer',
        'models/poly_features.pkl': 'Polynomial Features',
        'models/final_scaler.pkl': 'Final Scaler',
        'models/top_feature_indices.pkl': 'Top Feature Indices',
        'models/feature_names.pkl': 'Feature Names'
    }
    
    missing_files = []
    existing_files = []
    
    for file_path, description in model_files.items():
        if os.path.exists(file_path):
            existing_files.append((file_path, description))
        else:
            missing_files.append((file_path, description))
    
    return existing_files, missing_files

@st.cache_resource
def load_models():
    """Load all trained models and transformers with enhanced error handling"""
    
    # Check file existence first
    existing_files, missing_files = check_model_files()
    
    if missing_files:
        st.sidebar.error(f"❌ Missing {len(missing_files)} model files")
        for file_path, description in missing_files:
            st.sidebar.write(f"• {description}")
        return None
    
    try:
        with st.spinner("🤖 Loading AI models..."):
            # Load model and recompile
            model = load_model('models/credit_score_mlp_model.h5')
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Load transformers
            quantile_transformer = joblib.load('models/quantile_transformer.pkl')
            poly_features = joblib.load('models/poly_features.pkl')
            final_scaler = joblib.load('models/final_scaler.pkl')
            top_feature_indices = joblib.load('models/top_feature_indices.pkl')
            feature_names = joblib.load('models/feature_names.pkl')
            
            # Verify loaded components
            st.sidebar.success("✅ All models loaded successfully!")
            
            # Test model with dummy data
            dummy_data = np.zeros((1, 100))
            test_prediction = model.predict(dummy_data, verbose=0)
            
            return {
                'model': model,
                'quantile_transformer': quantile_transformer,
                'poly_features': poly_features,
                'final_scaler': final_scaler,
                'top_feature_indices': top_feature_indices,
                'feature_names': feature_names
            }
            
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {str(e)}")
        with st.sidebar.expander("🔍 Error Details"):
            st.code(traceback.format_exc())
        return None

def predict_credit_score(customer_data, models, debug_mode=False):
    """Enhanced prediction function with debugging and fallback logic"""
    
    debug_info = []
    
    try:
        debug_info.append("🚀 Starting prediction process...")
        
        # Input validation
        income = max(0, customer_data.get('INCOME', 0))
        debt = max(0, customer_data.get('DEBT', 0))
        savings = max(0, customer_data.get('SAVINGS', 0))
        
        debug_info.append(f"📊 Input validation: Income=₹{income:,}, Debt=₹{debt:,}, Savings=₹{savings:,}")
        
        # Convert customer data to DataFrame
        df_input = pd.DataFrame([customer_data])
        debug_info.append(f"✅ DataFrame created: {df_input.shape}")
        
        # Ensure all required features are present
        original_features = len(df_input.columns)
        for feature in models['feature_names']:
            if feature not in df_input.columns:
                df_input[feature] = 0
        
        debug_info.append(f"✅ Features expanded: {original_features} → {len(df_input.columns)}")
        
        # Reorder columns to match training data
        df_input = df_input[models['feature_names']]
        debug_info.append(f"✅ Features aligned: {df_input.shape}")
        
        # Apply preprocessing pipeline
        debug_info.append("🔧 Applying preprocessing pipeline...")
        
        X_quantile = models['quantile_transformer'].transform(df_input.values)
        debug_info.append(f"   • Quantile transform: {X_quantile.shape}")
        
        X_top = df_input.values[:, models['top_feature_indices']]
        debug_info.append(f"   • Top features selected: {X_top.shape}")
        
        X_poly = models['poly_features'].transform(X_top)
        debug_info.append(f"   • Polynomial features: {X_poly.shape}")
        
        X_enhanced = np.concatenate([X_quantile, X_poly], axis=1)
        debug_info.append(f"   • Combined features: {X_enhanced.shape}")
        
        X_final = models['final_scaler'].transform(X_enhanced)
        debug_info.append(f"   • Final scaling: {X_final.shape}")
        
        # Get prediction from MLP model
        debug_info.append("🧠 Making AI prediction...")
        raw_prediction = models['model'].predict(X_final, verbose=0)[0][0]
        debug_info.append(f"✅ Raw AI prediction: {raw_prediction:.2f}")
        
        # Apply business logic adjustments
        debug_info.append("💼 Applying business logic...")
        
        net_worth = savings - debt
        debt_coverage = (savings / debt) if debt > 0 else float('inf')
        
        adjusted_score = raw_prediction
        adjustments = []
        
        # Debt coverage adjustment
        if debt_coverage >= 1.0 and net_worth >= 0:
            boost = min(100, 50 + (net_worth / income * 20)) if income > 0 else 50
            adjusted_score += boost
            adjustments.append(f"Debt fully covered by savings: +{boost:.0f}")
            debug_info.append(f"   • Debt coverage bonus: +{boost:.0f}")
        
        # High net worth adjustment
        if income > 0:
            net_worth_ratio = net_worth / income
            if net_worth_ratio > 3:
                boost = min(75, net_worth_ratio * 15)
                adjusted_score += boost
                adjustments.append(f"Excellent net worth: +{boost:.0f}")
                debug_info.append(f"   • Net worth bonus: +{boost:.0f}")
        
        final_score = min(850, max(300, adjusted_score))
        debug_info.append(f"🎯 Final score: {final_score:.0f}")
        
        # Determine approval status
        if final_score >= 740:
            approval = "Excellent"
            apr = "3-7%"
            color = "green"
            risk_level = "Very Low"
        elif final_score >= 670:
            approval = "High"
            apr = "6-12%"
            color = "lightgreen"
            risk_level = "Low"
        elif final_score >= 580:
            approval = "Medium"
            apr = "10-18%"
            color = "orange"
            risk_level = "Medium"
        else:
            approval = "Low"
            apr = "15-25%"
            color = "red"
            risk_level = "High"
        
        debug_info.append(f"✅ Prediction completed successfully!")
        
        result = {
            'raw_score': round(raw_prediction, 0),
            'final_score': round(final_score, 0),
            'adjustments': adjustments,
            'approval': approval,
            'apr': apr,
            'color': color,
            'risk_level': risk_level,
            'net_worth': net_worth,
            'debt_coverage': debt_coverage,
            'debug_info': debug_info if debug_mode else [],
            'success': True
        }
        
        return result
        
    except Exception as e:
        error_msg = f"❌ Prediction failed: {str(e)}"
        debug_info.append(error_msg)
        debug_info.append(f"Error type: {type(e).__name__}")
        
        # Fallback calculation
        debug_info.append("🔄 Using fallback calculation...")
        
        debt_ratio = (debt / income * 100) if income > 0 else 999
        savings_ratio = (savings / income * 100) if income > 0 else 0
        
        # Simple rule-based scoring
        if income >= 60000 and debt_ratio < 40 and savings_ratio > 10:
            fallback_score = 680 + min(70, int(savings_ratio/2))
        elif income >= 30000 and debt_ratio < 60:
            fallback_score = 600 + min(50, int(savings_ratio/3))
        else:
            fallback_score = 520 + min(30, int(savings_ratio/5))
        
        fallback_score = min(850, max(300, fallback_score))
        
        # Determine approval for fallback
        if fallback_score >= 670:
            approval = "High"
            apr = "6-12%"
            color = "lightgreen"
        elif fallback_score >= 580:
            approval = "Medium"
            apr = "10-18%"
            color = "orange"
        else:
            approval = "Low"
            apr = "15-25%"
            color = "red"
        
        return {
            'raw_score': fallback_score,
            'final_score': fallback_score,
            'adjustments': ["Fallback calculation used due to model error"],
            'approval': approval,
            'apr': apr,
            'color': color,
            'risk_level': "Unknown",
            'net_worth': savings - debt,
            'debt_coverage': (savings / debt) if debt > 0 else float('inf'),
            'debug_info': debug_info if debug_mode else [],
            'success': False,
            'error': str(e)
        }

def create_enhanced_gauge(score, color):
    """Create an enhanced gauge chart for credit score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Credit Score", 'font': {'size': 24, 'color': '#1f77b4'}},
        delta = {'reference': 650, 'font': {'size': 16}},
        gauge = {
            'axis': {
                'range': [None, 850],
                'tickwidth': 1,
                'tickcolor': "darkblue",
                'tickfont': {'size': 12}
            },
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [300, 580], 'color': "#ffcccc", 'name': 'Poor'},
                {'range': [580, 670], 'color': "#fff3cd", 'name': 'Fair'},
                {'range': [670, 740], 'color': "#d4edda", 'name': 'Good'},
                {'range': [740, 850], 'color': "#c3e6cb", 'name': 'Excellent'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 700
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    
    return fig

def display_financial_insights(debt, income, savings, net_worth):
    """Display financial health insights"""
    
    st.markdown("### 📈 Financial Health Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        debt_ratio = (debt / income * 100) if income > 0 else 0
        savings_ratio = (savings / income * 100) if income > 0 else 0
        
        st.markdown(f"**💳 Debt-to-Income Ratio:** {debt_ratio:.1f}%")
        if debt_ratio < 20:
            st.success("✅ Excellent debt management")
        elif debt_ratio < 40:
            st.warning("⚠️ Manageable debt level")
        else:
            st.error("🔴 High debt burden")
        
        st.markdown(f"**💰 Savings-to-Income Ratio:** {savings_ratio:.1f}%")
        if savings_ratio > 30:
            st.success("✅ Excellent savings rate")
        elif savings_ratio > 15:
            st.info("💙 Good savings discipline")
        else:
            st.warning("⚠️ Consider increasing savings")
    
    with col2:
        st.markdown(f"**🏦 Net Worth:** ₹{net_worth:,.0f}")
        if net_worth > income * 2:
            st.success("✅ Strong financial position")
        elif net_worth > 0:
            st.info("💙 Positive net worth")
        else:
            st.error("🔴 Negative net worth - focus on debt reduction")
        
        emergency_months = (savings / (income / 12)) if income > 0 else 0
        st.markdown(f"**🚨 Emergency Fund:** {emergency_months:.1f} months")
        if emergency_months >= 6:
            st.success("✅ Excellent emergency preparedness")
        elif emergency_months >= 3:
            st.info("💙 Adequate emergency fund")
        else:
            st.warning("⚠️ Build emergency savings")

def main():
    """Enhanced main Streamlit application"""
    
    # Header with animation
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 class="main-header">🏦 AI Credit Score Predictor</h1>
        <p class="sub-header">Powered by Advanced Machine Learning Neural Network</p>
        <p style="color: #888; font-size: 0.9rem;">Professional credit assessment in seconds</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model status check
    with st.container():
        existing_files, missing_files = check_model_files()
        
        if missing_files:
            st.error("🚨 **Model files are missing!** The AI prediction system cannot function without these files.")
            
            with st.expander("📋 Required Files Status"):
                col1, col2 = st.columns(2)
                
                with col1:
                    if existing_files:
                        st.success("✅ **Available Files:**")
                        for _, description in existing_files:
                            st.write(f"• {description}")
                
                with col2:
                    st.error("❌ **Missing Files:**")
                    for file_path, description in missing_files:
                        st.write(f"• {description}")
                        st.code(file_path, language=None)
            
            st.info("💡 **Note:** Without model files, the system will use a basic rule-based calculation as fallback.")
    
    # Load models
    models = load_models()
    
    # Create main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📝 Enter Your Financial Information")
        
        # Input form with better styling
        with st.form("financial_form"):
            income = st.number_input(
                "💰 Annual Income (₹)",
                min_value=0,
                max_value=10000000,
                value=50000,
                step=5000,
                help="Your total annual income from all sources"
            )
            
            debt = st.number_input(
                "💳 Total Debt (₹)",
                min_value=0,
                max_value=5000000,
                value=15000,
                step=1000,
                help="Total outstanding debt (loans, credit cards, etc.)"
            )
            
            savings = st.number_input(
                "🏦 Total Savings (₹)",
                min_value=0,
                max_value=10000000,
                value=25000,
                step=2500,
                help="Total savings and liquid investments"
            )
            
            # Advanced options
            st.markdown("---")
            col_a, col_b = st.columns(2)
            
            with col_a:
                show_details = st.checkbox("🔍 Show Technical Details", value=False)
            
            with col_b:
                debug_mode = st.checkbox("🐛 Debug Mode", value=False)
            
            # Submit button
            submitted = st.form_submit_button("🔮 Predict Credit Score", type="primary")
    
    with col2:
        # Results area
        if submitted or st.session_state.get('show_results', False):
            st.session_state.show_results = True
            
            # Input validation
            if income <= 0:
                st.error("⚠️ Please enter a valid annual income greater than 0")
                st.stop()
            
            # Create customer data
            customer_data = {
                'INCOME': income,
                'DEBT': debt,
                'SAVINGS': savings
            }
            
            # Make prediction
            with st.spinner("🤖 AI is analyzing your financial profile..."):
                result = predict_credit_score(customer_data, models, debug_mode)
            
            # Display debug info if requested
            if debug_mode and result.get('debug_info'):
                with st.expander("🐛 Debug Information"):
                    for info in result['debug_info']:
                        st.markdown(f'<div class="debug-info">{info}</div>', unsafe_allow_html=True)
            
            # Main results display
            st.markdown("## 📊 Credit Score Analysis Results")
            
            # Show warning if fallback was used
            if not result.get('success', True):
                st.warning("⚠️ **Fallback Mode:** AI model unavailable, using rule-based calculation")
            
            # Main metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                delta_value = result['final_score'] - result['raw_score']
                st.metric(
                    label="🎯 Credit Score",
                    value=f"{result['final_score']:.0f}",
                    delta=f"{delta_value:.0f}" if abs(delta_value) > 0 else None,
                    help="Your predicted credit score (300-850 range)"
                )
            
            with col2:
                st.metric(
                    label="📈 Approval Chance",
                    value=result['approval'],
                    help="Likelihood of loan approval"
                )
            
            with col3:
                st.metric(
                    label="💸 Expected APR",
                    value=result['apr'],
                    help="Estimated annual percentage rate"
                )
            
            # Gauge chart and financial insights
            col1, col2 = st.columns([3, 2])
            
            with col1:
                fig = create_enhanced_gauge(result['final_score'], result['color'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                display_financial_insights(debt, income, savings, result['net_worth'])
            
            # Credit explanation
            st.markdown("### 🎯 Credit Assessment")
            
            if result['final_score'] >= 740:
                st.markdown('<div class="success-box">🌟 <strong>Excellent Credit!</strong><br>You qualify for premium financial products with the best terms available. Banks will compete for your business.</div>', unsafe_allow_html=True)
            elif result['final_score'] >= 670:
                st.markdown('<div class="success-box">✅ <strong>Good Credit!</strong><br>You have access to most financial products with competitive rates and favorable terms.</div>', unsafe_allow_html=True)
            elif result['final_score'] >= 580:
                st.markdown('<div class="warning-box">🟡 <strong>Fair Credit</strong><br>Loans are possible but may come with higher interest rates. Focus on improving your financial profile.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-box">🔴 <strong>Poor Credit</strong><br>Limited financial options available. Priority should be debt reduction and income improvement.</div>', unsafe_allow_html=True)
            
            # Show AI adjustments
            if result.get('adjustments') and len(result['adjustments']) > 0:
                st.markdown("### 🎁 AI Adjustments Applied")
                for adjustment in result['adjustments']:
                    if "Fallback" not in adjustment:
                        st.success(f"• {adjustment}")
                    else:
                        st.info(f"• {adjustment}")
            
            # Technical details
            if show_details:
                st.markdown("### 🤖 Technical Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🧠 Model Architecture:**")
                    st.write("• Multi-Layer Perceptron (MLP) Neural Network")
                    st.write("• 5 layers with BatchNormalization")
                    st.write("• Dropout regularization")
                    st.write("• Huber loss function")
                    st.write("• R² Score: 0.5609 (56% explained variance)")
                    st.write("• MAE: 31.93 points")
                
                with col2:
                    st.markdown("**⚙️ Processing Pipeline:**")
                    st.write("• Input features: 55 financial variables")
                    st.write("• Quantile transformation for normalization")
                    st.write("• Polynomial feature engineering")
                    st.write("• Standard scaling")
                    st.write("• Enhanced features: 100 total")
                    st.write("• Business logic adjustments")
                
                st.markdown(f"**📊 Prediction Breakdown:**")
                st.write(f"• Raw AI Score: {result['raw_score']:.0f}")
                st.write(f"• Final Score: {result['final_score']:.0f}")
                st.write(f"• Risk Level: {result.get('risk_level', 'Unknown')}")
                st.write(f"• Net Worth: ₹{result['net_worth']:,.0f}")
                st.write(f"• Debt Coverage: {result['debt_coverage']:.2f}x" if result['debt_coverage'] != float('inf') else "• Debt Coverage: ∞ (no debt)")
        
        else:
            # Welcome message when no prediction has been made
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f0f2f6 0%, #e8ecf0 100%); border-radius: 15px; margin: 2rem 0;">
                <h3>👈 Enter your financial information to get started</h3>
                <p>Our advanced AI will analyze your profile and provide:</p>
                <ul style="text-align: left; display: inline-block;">
                    <li>🎯 Accurate credit score prediction</li>
                    <li>📈 Loan approval probability</li>
                    <li>💰 Expected interest rates</li>
                    <li>💡 Financial health insights</li>
                    <li>🎁 Personalized recommendations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Model Performance**")
        st.write("• 56% Explained Variance")
        st.write("• 31.93 Points MAE")
        st.write("• 1,000 Customer Training Data")
    
    with col2:
        st.markdown("**⚡ Features**")
        st.write("• Real-time AI Predictions")
        st.write("• Advanced Feature Engineering")
        st.write("• Business Logic Integration")
    
    with col3:
        st.markdown("**🔒 Disclaimer**")
        st.write("Educational purposes only")
        st.write("Actual scores may vary")
        st.write("Consult financial advisors")
    
    # About section
    with st.expander("ℹ️ About This AI Credit Scoring System"):
        st.markdown("""
        ### 🎯 **Project Overview**
        This is a comprehensive AI-powered credit scoring system that demonstrates advanced machine learning engineering combined with practical business applications.
        
        ### 🧠 **Technology Stack**
        - **Machine Learning:** TensorFlow/Keras MLP Neural Network
        - **Feature Engineering:** QuantileTransformer + PolynomialFeatures (100 enhanced features)
        - **Performance Metrics:** R² = 0.5609 (56% explained variance), MAE = 31.93 points
        - **Business Logic:** Realistic financial adjustments for edge cases
        - **Frontend:** Streamlit with advanced visualizations
        
        ### 📚 **Model Training Details**
        - **Dataset:** 1,000 customer profiles with 55 financial features
        - **Architecture:** Enhanced MLP with BatchNormalization, Dropout regularization
        - **Training:** 200+ epochs with early stopping and learning rate scheduling
        - **Validation:** Proper train/test split with robust evaluation methodology
        
        ### 🎨 **Key Features**
        - Real-time credit score predictions
        - Interactive financial health analysis
        - Advanced data visualizations
        - Fallback mechanisms for robustness
        - Debug mode for transparency
        - Professional UI/UX design
        
        ### 👨‍💼 **Business Applications**
        - Loan origination automation
        - Risk assessment and management
        - Customer segmentation
        - Financial product recommendations
        
        **Created by:** [Your Name] | **Year:** 2024 | **Portfolio Project**
        """)

if __name__ == "__main__":
    main()
