import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle

st.set_page_config(page_title="Bank Risk Controller System", layout="wide", page_icon="🏦")

df = pd.read_csv("model_data.csv")

@st.cache_resource
def load_model():
    with open("rf_model.pkl", "rb") as f:
        return pickle.load(f)
model = load_model()

@st.cache_resource
def load_label_encoders():
    encoders = {}
    columns = [
        "NAME_CLIENT_TYPE", "FLAG_OWN_REALTY", "OCCUPATION_TYPE",
        "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS"
    ]
    files = [
        "NAME_CLIENT_TYPE_enc.pkl", "FLAG_OWN_REALTY_enc.pkl", "OCCUPATION_TYPE_enc.pkl",
        "NAME_INCOME_TYPE_enc.pkl", "NAME_FAMILY_STATUS_enc.pkl"
    ]

    for col, file in zip(columns, files):
        with open(file, 'rb') as f:
            encoders[col] = pickle.load(f)

    return encoders
encoders = load_label_encoders()

education_mapping = {
    'Lower secondary': 0,
    'Secondary / secondary special': 1,
    'Incomplete higher': 2,
    'Higher education': 3,
    'Academic degree': 4
    }


if 'active_section' not in st.session_state:
    st.session_state.active_section = "Home"  # Default to "Home"


with st.sidebar:
    img="https://static.vecteezy.com/system/resources/thumbnails/005/051/138/small_2x/applying-for-loan-at-the-bank-illustration-concept-flat-illustration-isolated-on-white-background-vector.jpg"
    st.image(img,width=250)
    st.header("🧭Navigation")
    if st.button("🏠Home"):
        st.session_state.active_section = "Home"
    if st.button("📊Model Preformance and Data"):
        st.session_state.active_section = "Model Preformance and Data"
    if st.button("📈EDA Visual"):
        st.session_state.active_section = "EDA Visual"
    if st.button("🧮Prediction"):
        st.session_state.active_section = "Prediction"
    if st.button("🎬Movie Recommendation"):
        st.session_state.active_section = "Movie Recommendation"


if st.session_state.active_section == "Home":
    st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://www.icicidirect.com/images/loan7-202306271644570347759.jpg' 
             style='width: 100%; height: 250px; object-fit: cover; border-radius: 10px;' />
        <p style='color: grey;'></p>
    </div>
    """,
    unsafe_allow_html=True)

    st.title("🏦 Bank :green[Loan Prediction] and :red[Risk Analysis]")
    st.markdown("""
        Welcome to the **Bank Loan Prediction and Risk Analysis** application! This interactive platform is designed to provide insights into loan approval predictions and risk analysis. It also includes a bonus feature: a **Movie Recommendation System**.
    """)

    st.markdown("---")

    st.header("📊 Project Overview")
    st.markdown("""
        This project aims to assist financial institutions in making informed decisions about loan approvals by analyzing customer data. The application leverages **machine learning models**  to predict loan defaults and explore loan risks.
    """)

    st.markdown("---")

    st.header("🔍 Key Features")
    st.markdown("""
    1. **Exploratory Data Analysis (EDA)**:
       - Visualize trends and patterns in customer data.
       - Analyze the distribution of key features like income, age, and employment years.

    2. **Loan Prediction**:
       - Predict the likelihood of loan approval using advanced machine learning models.
       - Models include Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost, and CatBoost.

    3. **Risk Analysis**:
       - Understand the factors contributing to loan defaults.
       - Explore feature importance and correlations.

    4. **Movie Recommendation System**:
       - Get personalized movie recommendations based on your preferences.
       - Powered by **TF-IDF Vectorization** and **Cosine Similarity**.
    """)

    st.markdown("---")

    st.header("🚀 How to Use")
    st.markdown("""
    1. Navigate through the sidebar to explore different sections:
       - **EDA visuals**: View interactive visualizations of the dataset.
       - **Model Performance and Data**: Compare the accuracy and metrics of various models.
       - **Prediction**: Input customer details to predict loan approval.
       - **Movie Recommendations**: Discover movies tailored to your taste.

    2. Use the **interactive widgets** to customize your inputs and explore the results.
    """)

    st.markdown("---")

    st.header("🛠️ Technologies Used")
    st.markdown("""
    - **Python**: Core programming language.
    - **Pandas, NumPy**: Data manipulation and analysis.
    - **Seaborn, Matplotlib, Plotly**: Data visualization tools.
    - **Scikit-learn, XGBoost, CatBoost**: Machine learning libraries.
    - **Streamlit**: Framework for building the interactive app. 
    """)

    st.markdown("---")

    st.success("Thank you for using the **Bank Loan Prediction and Risk Analysis** app! 🎉")

if st.session_state.active_section == "Model Preformance and Data":
    st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://media.geeksforgeeks.org/wp-content/uploads/20230808130011/Machine-Learning-Algorithms1-(1).webp' 
             style='width: 100%; height: 250px; object-fit: cover; border-radius: 10px;' />
        <p style='color: grey;'></p>
    </div>
    """,
    unsafe_allow_html=True)
    st.header("All :blue[Model Preformance] Scores and sample of the :orange[Dataset]")
    st.subheader("🤖Model Performance Metrics")
    model_metrics = pd.DataFrame({
    'Model': ['Logistic Regression','Decision Tree', 'Random Forest', 'XGBoost'],
    'Accuracy': [56.86, 93.33, 96.78, 75.93],
    'Precision': [56.69, 92.58, 96.80, 76.23],
    'Recall': [58.53, 94.22, 96.77, 75.44],
    'F1-Score': [57.59, 93.39, 96.78, 75.83],
    'ROC-AUC Score': [59.12, 93.33, 99.61, 76.64]
    },index=[1,2,3,4])
    st.dataframe(model_metrics)
    st.write(":red[Random Forest] is the **best model** with :green[96% above] scores in all metrics.")
    st.write("")

    st.header("📂 About the Dataset")
    st.markdown("""
    The dataset includes customer information such as:
    - **Income**, **Age**, **Employment Years**, **Loan Amount**, and more.
    - Target variable: **Loan Default (1: Default, 0: No Default)**.
    """)

    st.subheader("Here is the :violet[dataset used] for loan prediction:")
    st.dataframe(df.head(20))

if st.session_state.active_section == "EDA Visual":
    st.title("Exploratory Data Analysis (EDA) :blue[Visualizations]")
    
    # --- Title and Subheader ---
    st.markdown("<h1 style='color:#1F618D;'>📊 Data Visualization</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#117864;'>Explore trends, patterns, and insights from the data</h4>", unsafe_allow_html=True)

    st.markdown("---")

    # --- Visualizations ---

    # Target Variable Distribution (Pie Chart)
    st.markdown("<h5 style='color:#884EA0;'>🔹 Target Variable Distribution</h5>", unsafe_allow_html=True)
    target_counts = df['TARGET'].value_counts()
    fig = px.pie(
        names=["Non-Defaulter", "Defaulter"],
        values=target_counts.values,
        title="Distribution of Target Variable",
        color_discrete_sequence=['skyblue', 'salmon']
    )
    st.plotly_chart(fig, use_container_width=True)

    # Age Distribution (Histogram and Pie Chart)
    st.markdown("<h5 style='color:#2471A3;'>🔹 Age Distribution</h5>", unsafe_allow_html=True)
    fig_hist = px.histogram(
        df,
        x='AGE',
        color='TARGET',
        barmode='stack',
        color_discrete_map={0: 'skyblue', 1: 'salmon'},
        title="Age Distribution by Target",
        labels={'AGE': 'Age', 'count': 'Frequency'}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Pie Chart for Age Categories
    st.markdown("<h5 style='color:#1F618D;'>🔹 Age Categories Distribution</h5>", unsafe_allow_html=True) 
    age_category_counts = df['AGE_CATEGORY'].value_counts()
    fig_pie = px.pie(
        names=age_category_counts.index,
        values=age_category_counts.values,
        title="Age Category Distribution (Pie Chart)",
        color_discrete_sequence=['skyblue', 'salmon', 'lightgreen']
    )
    st.plotly_chart(fig_pie, use_container_width=True)


    # AMT_INCOME_TOTAL Distribution with KDE
    st.markdown("<h5 style='color:#117A65;'>🔹 Total Income Distribution</h5>", unsafe_allow_html=True)  
    fig = px.histogram(df, x='AMT_INCOME_TOTAL', nbins=50, marginal='box', histnorm='density')
    fig.update_traces(marker_color='blue', opacity=0.7)
    fig.add_scatter(x=df['AMT_INCOME_TOTAL'], 
                    y=df['AMT_INCOME_TOTAL'].value_counts(normalize=True).sort_index(),
                    mode='lines', name='KDE', line=dict(color='red'))
    fig.update_layout(title='AMT_INCOME_TOTAL Distribution with KDE',
                      xaxis_title='AMT_INCOME_TOTAL',
                      yaxis_title='Density',
                      showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # Bar plot for Top 10 Occupation Types
    st.markdown("<h5 style='color:#AF601A;'>🔹 Top 10 Occupation Types</h5>", unsafe_allow_html=True) 
    occupation_counts = df['OCCUPATION_TYPE'].value_counts().reset_index()
    occupation_counts.columns = ['OCCUPATION_TYPE', 'COUNT']
    fig = px.bar(occupation_counts, y='OCCUPATION_TYPE', x='COUNT', color="COUNT", 
                 title='Occupation Type Counts', color_continuous_scale='PiYG')
    st.plotly_chart(fig, use_container_width=True)

    # Line plot for Income Type Counts
    st.markdown("<h5 style='color:#884EA0;'>🔹 Income Type Counts</h5>", unsafe_allow_html=True) 
    income_counts = df['NAME_INCOME_TYPE'].value_counts().reset_index()
    income_counts.columns = ['NAME_INCOME_TYPE', 'COUNT']
    fig = px.line(income_counts, x='NAME_INCOME_TYPE', y='COUNT', title='Income Type Counts')
    st.plotly_chart(fig, use_container_width=True)

    # Pie chart for Family Status Distribution
    st.markdown("<h5 style='color:#BA4A00;'>🔹 Family Status Distribution</h5>", unsafe_allow_html=True) 
    family_counts = df['NAME_FAMILY_STATUS'].value_counts().reset_index()
    family_counts.columns = ['NAME_FAMILY_STATUS', 'COUNT']
    fig = px.pie(family_counts, names='NAME_FAMILY_STATUS', values='COUNT', 
                 title='Family Status Distribution')
    st.plotly_chart(fig, use_container_width=True)

    # Bar plot for Education Type Counts
    st.markdown("<h5 style='color:#2471A3;'>🔹 Education counts</h5>", unsafe_allow_html=True)  
    education_counts = df['NAME_EDUCATION_TYPE'].value_counts().reset_index()
    education_counts.columns = ['NAME_EDUCATION_TYPE', 'COUNT']
    fig = px.bar(education_counts, x='NAME_EDUCATION_TYPE', y='COUNT', color='COUNT',
                 color_continuous_scale='Viridis', title='Education Type Counts')
    fig.update_layout(legend_title_text='Education Type')
    st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap
    st.markdown("<h5 style='color:#A93226;'>🔹 Correlation Heatmap</h5>", unsafe_allow_html=True) 
    dff = df[['TARGET', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'EXT_SOURCE_2',"EXT_SOURCE_3",'AGE', 'EMPLOYED_YEARS']]
    corr = dff.corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu",
                    title="Correlation Matrix Heatmap")
    st.plotly_chart(fig, use_container_width=True)

    # Hist plot for OCCUPATION_TYPE by TARGET
    st.markdown("<h5 style='color:#196F3D;'>🔹 Occupation with Target</h5>", unsafe_allow_html=True) 
    fig = px.histogram(df, x='OCCUPATION_TYPE', color='TARGET', barmode='group')
    fig.update_layout(title='Countplot of TARGET by OCCUPATION_TYPE', 
                      xaxis_title='OCCUPATION_TYPE', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance Scores (Bar Plot)
    st.markdown("<h5 style='color:#196F3D;'>🔹 Feature Importance Scores</h5>", unsafe_allow_html=True)
    importances = model.feature_importances_
    feature_names = model.feature_names_in_
    importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig = px.bar(importance_df.head(10),
             x='Importance', y='Feature',
             orientation='h',
             title='Top 10 Feature Importances',
             color='Importance',
             color_continuous_scale='Greens')

    fig.update_layout(yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig, use_container_width=True)

    
if st.session_state.active_section == "Prediction":
    st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://lausconsult.com/wp-content/uploads/How-To-Secure-That-Bank-Loan-For-Your-Business-810x423.jpg' 
             style='width: 75%; height: 300px; object-fit: cover; border-radius: 10px;' />
        <p style='color: grey;'></p>
    </div>
    """,
    unsafe_allow_html=True)
    st.title("💳Bank Loan Prediction")

    # Collect user input for prediction
    st.subheader("Enter Applicant Information")
    col1, col2 = st.columns([6, 6])

    with col1:
        AMT_ANNUITY = st.number_input("Annuity Amount", min_value=3114.0, max_value=61281.0,value=10000.0, step=500.0)
        AMT_INCOME_TOTAL = st.number_input("Total Income", min_value=25650.0,max_value=337500.0,value=100000.0, step=10000.0)
        EXT_SOURCE_2 = st.slider("External Source 2", min_value=0.00, max_value=0.85, step=0.01)
        EXT_SOURCE_3 = st.slider("External Source 3",min_value=0.04, max_value=0.89, step=0.01)
        AGE = st.slider("Age",min_value=21, max_value=69, step=1)
        EMPLOYED_YEARS = st.slider("Years Employed",min_value=0, max_value=35, step=1)

    with col2:
        OCCUPATION_TYPE = st.selectbox("Occupation Type", ['Laborers', 'Core staff', 'Drivers', 'Sales staff',
                                                           'Cleaning staff', 'Not mentioned', 'Private service staff',
                                                            'Managers', 'Medicine staff', 'Security staff', 'Cooking staff',
                                                            'High skill tech staff', 'Waiters/barmen staff',
                                                            'Low-skill Laborers', 'Realty agents', 'Accountants',
                                                            'Secretaries', 'HR staff', 'IT staff'])
        NAME_CLIENT_TYPE = st.selectbox("Client Type", ['New', 'Repeater', 'Refreshed', 'XNA'])
        FLAG_OWN_REALTY = st.selectbox("Own Realty", ["Y","N"])
        NAME_INCOME_TYPE = st.selectbox("Income Type", ['Working', 'State servant', 'Commercial associate', 'Student','Pensioner', 'Maternity leave'])
        NAME_EDUCATION_TYPE = st.selectbox("Education Type", ['Secondary / secondary special', 'Higher education','Incomplete higher', 'Lower secondary', 'Academic degree'])
        NAME_FAMILY_STATUS = st.selectbox("Family Status", ['Single / not married', 'Married', 'Civil marriage', 'Widow','Separated'])

    data = {
        "AMT_ANNUITY": AMT_ANNUITY,
        "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
        "EXT_SOURCE_2": EXT_SOURCE_2,
        "EXT_SOURCE_3": EXT_SOURCE_3,
        "NAME_CLIENT_TYPE": NAME_CLIENT_TYPE,
        "FLAG_OWN_REALTY": FLAG_OWN_REALTY,
        'OCCUPATION_TYPE': OCCUPATION_TYPE,
        "NAME_INCOME_TYPE": NAME_INCOME_TYPE,
        "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE,
        "NAME_FAMILY_STATUS": NAME_FAMILY_STATUS,
        "AGE": AGE,
        "EMPLOYED_YEARS": EMPLOYED_YEARS,
        }
    input_df = pd.DataFrame(data, index=[0])
    st.subheader("Form Details")
    st.dataframe(input_df)


    input_df["NAME_EDUCATION_TYPE"] = input_df["NAME_EDUCATION_TYPE"].map(education_mapping)

    for col in encoders:
        input_df[col] = encoders[col].transform(input_df[[col]])
    
    if st.checkbox("Show Processed Input Data"):
        st.subheader("Processed Input Data")
        st.dataframe(input_df)

    if st.button("Predict"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        st.write("### Prediction")
        if prediction[0] == 1:
            st.error("🚫 Prediction: Defaulter — The user is **likely to default** on the loan.")
        else:
            st.success("✅ Prediction: Non-Defaulter — The user is **likely to repay** the loan.")

        st.info(f"🧠 Model Confidence (Default Probability): **{prediction_proba[0][1]:.2%}**")


with open("similarity.pkl", "rb") as f:
    similarity = pickle.load(f)

with open("movie_df.pkl", "rb") as f:
    movie_df = pickle.load(f)

if st.session_state.active_section == "Movie Recommendation":
    st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://platform.vox.com/wp-content/uploads/sites/2/chorus/uploads/chorus_asset/file/9871033/Movies_end_of_year_2017.jpg?quality=90&strip=all&crop=1.1195652173913,0,97.760869565217,100' 
             style='width: 100%; height: 300px; object-fit: cover; border-radius: 10px;' />
        <p style='color: grey;'></p>
    </div>
    """,
    unsafe_allow_html=True)

    st.title("🎬 **Movie Recommendation System**")
    st.subheader("💡 _Find :red[great picks]_ based on your favorite :blue[movies].")

    selected_movie=st.selectbox("select movie", movie_df["title"].values, key="selected_movie")
    num_recs=st.selectbox("select no.of.recommendation", [5,10,15,20,25], key="number_of_recommendations")

    def recommend(movie_title,num_recommendations):
        if movie_title not in movie_df['title'].values:
            st.warning(f"Movie '{movie_title}' not found in the database.")
            return None, None, []

        index = movie_df[movie_df['title'] == movie_title].index[0]
        movie_genre = movie_df.iloc[index]['genres']
        distances = list(enumerate(similarity[index]))
        sorted_movies = sorted(distances, reverse=True, key=lambda x: x[1])

        recommended_titles = []
        for i in sorted_movies[1:num_recommendations+1]:
            recommended_titles.append(movie_df.iloc[i[0]].title)

        return movie_title, movie_genre, recommended_titles
    
    if st.button("Show Recommend"):
                    movie_title, movie_genre, recommended_titles = recommend(selected_movie, num_recs)

                    if recommended_titles:
                        st.markdown(f"## 🎥 You selected: :violet[{movie_title}]")
                        st.markdown(f"### 🎭 Genre: :blue[{movie_genre}]")
                        st.markdown("---")
                        st.markdown(f"## 🌟 Recommended Movies for You:")
                        
                        count = 1
                        for rec_title in recommended_titles:
                            rec_movie = movie_df[movie_df['title'] == rec_title].iloc[0]
                            rec_genre = rec_movie['genres']
                            rec_overview = rec_movie['overview']
                            st.markdown(f"#### 🎬 Title {count}: :orange[{rec_title}]")
                            st.markdown(f"#### 🎭 Genre: :blue[{rec_genre}]")
                            st.markdown(f"#### 🎭 overview: :green[{rec_overview}]")
                            st.write("----")
                            count +=1
