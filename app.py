# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# import joblib

# # PAGE CONFIG
# st.set_page_config(page_title="📊 Student Performance Dashboard", layout="wide")

# # LOAD DATA
# @st.cache_data
# def load_data():
#     df = pd.read_csv('data/stud.csv')
#     df['total score'] = df['math_score'] + df['reading_score'] + df['writing_score']
#     df['average'] = df['total score'] / 3
#     return df

# df = load_data()

# # THEME AND PAGE STATE
# if 'theme' not in st.session_state:
#     st.session_state.theme = 'light'
# if 'page' not in st.session_state:
#     st.session_state.page = 'Dashboard'

# # CSS STYLING
# st.markdown("""
#     <style>
#     .block-container {
#         padding: 0 !important;
#         margin: 0 !important;
#         max-width: 100% !important;
#     }
#     header, footer, .stDeployButton {
#         display: none !important;
#     }
#     .top-navbar {
#         background-color: #0c4a6e;
#         color: white;
#       margin-top:0;
#         display: flex;
#         justify-content: space-between;
#         align-items: center;
#         font-family: 'Segoe UI', sans-serif;
#     }
#     .top-navbar h1 {
#         margin: 0;
#         font-size: 1.5rem;
#     }

#   .sidebar button {
#     /* REMOVE or comment this */
#     width: 100%;
#     background-color: transparent;
#     text-align: center;
#     font-size: 1rem;
#     color: #0c4a6e;
#     cursor: pointer;
#     font-weight: 600;
#     transition: all 0.3s ease;
# }
# .sidebar button:hover {
#     background-color: #e2e8f0;
#     transform: translateX(5px);
# }
# .sidebar button.active {
#     background-color: #0c4a6e;
#     color: white;
# }

#     /* Remove default top padding/margin caused by Streamlit */
#     section.main > div {
#         padding-top: 0rem !important;
#         margin-top: 0rem !important;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # TOP NAVBAR
# st.markdown(f"""
# <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 1rem;">
#     <h1>📊 Student Performance Dashboard</h1>
# </div>

# """, unsafe_allow_html=True)

# # LAYOUT: Sidebar + Content
# layout_col1, layout_col2 = st.columns([1, 6], gap="small")

# # Sidebar
# with layout_col1:
#     st.markdown("""
#         <div class="sidebar">
      
#     """, unsafe_allow_html=True)

#     dash_class = 'active' if st.session_state.page == 'Dashboard' else ''
#     train_class = 'active' if st.session_state.page == 'Model Training' else ''
#     pred_class = 'active' if st.session_state.page == 'Prediction' else ''

#     if st.button("📈 Dashboard", key="dashboard"):
#         st.session_state.page = 'Dashboard'
#     if st.button("🛠️ Model Training", key="model_training"):
#         st.session_state.page = 'Model Training'
#     if st.button("🔮 Prediction", key="prediction"):
#         st.session_state.page = 'Prediction'

#     st.markdown("</div>", unsafe_allow_html=True)

# # Content
# with layout_col2:
#     # DASHBOARD
#     if st.session_state.page == "Dashboard":
#         st.header("📈 Dashboard Overview")

#         total_students = df.shape[0]
#         avg_score = df['average'].mean()
#         min_score = df['average'].min()
#         max_score = df['average'].max()

#         kpi1, kpi2, kpi3, kpi4 = st.columns(4)
#         kpi1.metric("Total Students", total_students)
#         kpi2.metric("Average Score", f"{avg_score:.2f}")
#         kpi3.metric("Min Score", f"{min_score:.2f}")
#         kpi4.metric("Max Score", f"{max_score:.2f}")

#         st.markdown("---")

#         # Top row: 3 smaller graphs
#         col_a, col_b, col_c = st.columns(3, gap="small")
#         with col_a:
#             st.subheader("Avg. Score by Parent Ed.")
#             edu_avg = df.groupby('parental_level_of_education')['average'].mean().reset_index()
#             edu_avg = edu_avg.sort_values(by='average', ascending=True)
#             fig_bar = px.bar(
#                 edu_avg,
#                 x='average', y='parental_level_of_education',
#                 orientation='h', color='average',
#                 color_continuous_scale='Viridis',
#                 labels={'average': 'Average Score', 'parental_level_of_education': 'Parental Education'},
#                 height=300
#             )
#             st.plotly_chart(fig_bar, use_container_width=True)

#         with col_b:
#             st.subheader("Score by Gender")
#             fig_box = px.box(
#                 df, x='gender', y='average', color='gender',
#                 points="all",
#                 labels={'gender': 'Gender', 'average': 'Average Score'},
#                 height=300
#             )
#             st.plotly_chart(fig_box, use_container_width=True)

#         with col_c:
#             st.subheader("Performance Gauge")
#             fig_gauge = go.Figure(go.Indicator(
#                 mode="gauge+number",
#                 value=avg_score,
#                 domain={'x': [0, 1], 'y': [0, 1]},
#                 title={'text': "Average Score"},
#                 gauge={'axis': {'range': [0, 100]},
#                        'bar': {'color': "blue"},
#                        'steps': [
#                            {'range': [0, 60], 'color': 'red'},
#                            {'range': [60, 80], 'color': 'yellow'},
#                            {'range': [80, 100], 'color': 'green'}
#                        ]}
#             ))
#             fig_gauge.update_layout(height=300)
#             st.plotly_chart(fig_gauge, use_container_width=True)

#         st.markdown("---")

#         # Bottom row: 2 smaller graphs
#         col_d, col_e = st.columns(2, gap="small")
#         with col_d:
#             st.subheader("📌 Correlation Heatmap")
#             fig_corr, ax = plt.subplots(figsize=(10, 4))
#             sns.heatmap(
#                 df[['math_score', 'reading_score', 'writing_score', 'average']].corr(),
#                 annot=True, cmap='coolwarm', linewidths=0.5
#             )
#             st.pyplot(fig_corr)

#         with col_e:
#             st.subheader("Reading vs Writing")
#             fig_scatter = px.scatter(
#                 df, x='reading_score', y='writing_score', color='gender',
#                 size='average', hover_data=['parental_level_of_education'],
#                 labels={'reading_score': 'Reading Score', 'writing_score': 'Writing Score'},
#                 height=300
#             )
#             st.plotly_chart(fig_scatter, use_container_width=True)

#     # MODEL TRAINING
#     elif st.session_state.page == "Model Training":
#         st.header("🛠️ Model Training Module")
#         X = df.drop(columns=['math_score', 'total score', 'average'])
#         y = df['math_score']
#         X_encoded = pd.get_dummies(X)
#         X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

#         if st.button("🚀 Train Model"):
#             model = RandomForestRegressor(n_estimators=100, random_state=42)
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             score = r2_score(y_test, y_pred) * 100
#             joblib.dump(model, 'model.joblib')
#             st.success("✅ Model trained and saved successfully!")
#             st.metric("R² Score on Test Set", f"{score:.2f}%")

#             fig4, ax = plt.subplots(figsize=(6, 4))
#             sns.scatterplot(x=y_test, y=y_pred, color='purple')
#             plt.xlabel("Actual Exam Scores")
#             plt.ylabel("Predicted Exam Scores")
#             plt.title("Actual vs. Predicted Exam Scores")
#             st.pyplot(fig4)

#     # PREDICTION
#     elif st.session_state.page == "Prediction":
#         st.header("🔮 Predict Student Performance")
#         try:
#             model = joblib.load('model.joblib')
#         except:
#             st.error("⚠️ Model not trained yet. Please train it in the Model Training section.")
#             st.stop()

#         st.write("### Input Student Data:")
#         col1, col2 = st.columns(2)
#         with col1:
#             gender = st.selectbox("Gender", ["female", "male"])
#             race = st.selectbox("Race/Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
#             lunch = st.selectbox("Lunch", ['standard', 'free/reduced'])
#             test_prep = st.selectbox("Test Preparation Course", ['none', 'completed'])
#         with col2:
#             parent_edu = st.selectbox("Parental Education", [
#                 "some high school", "high school", "some college",
#                 "associate's degree", "bachelor's degree", "master's degree"
#             ])
#             reading_score = st.slider("Reading Score", 0, 100, 50)
#             writing_score = st.slider("Writing Score", 0, 100, 50)

#         input_df = pd.DataFrame({
#             'gender': [gender],
#             'race/ethnicity': [race],
#             'parental_level_of_education': [parent_edu],
#             'lunch': [lunch],
#             'test_preparation_course': [test_prep],
#             'reading_score': [reading_score],
#             'writing_score': [writing_score]
#         })

#         input_encoded = pd.get_dummies(input_df)
#         missing_cols = set(model.feature_names_in_) - set(input_encoded.columns)
#         for col in missing_cols:
#             input_encoded[col] = 0
#         input_encoded = input_encoded[model.feature_names_in_]

#         if st.button("🔍 Predict Exam Score"):
#             prediction = model.predict(input_encoded)[0]
#             st.metric("Predicted Exam Score", f"{prediction:.2f}")
#             fig5 = px.bar(
#                 x=['Predicted Exam Score', 'Reading Score', 'Writing Score'],
#                 y=[prediction, reading_score, writing_score],
#                 color=['Predicted Exam Score', 'Reading Score', 'Writing Score'],
#                 labels={'x': 'Scores', 'y': 'Values'},
#                 title="📊 Predicted vs. Input Scores"
#             )
#       
#       st.plotly_chart(fig5, use_container_width=True)
# Paste this in your .py file and run using: streamlit run your_file.py



# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# import joblib

# # PAGE CONFIG
# st.set_page_config(
#     page_title=" Student Performance Dashboard",
#     page_icon="📊",  # or use a path to a PNG file like "favicon.png"
#     layout="wide"
# )

# # LOAD DATA
# @st.cache_data
# def load_data():
#     df = pd.read_csv('data/stud.csv')
#     df['total score'] = df['math_score'] + df['reading_score'] + df['writing_score']
#     df['average'] = df['total score'] / 3
#     return df

# df = load_data()

# # READ PAGE FROM QUERY PARAM
# query_page = st.query_params.get("page")
# if query_page:
#     st.session_state.page = query_page.replace("+", " ")
# if 'page' not in st.session_state:
#     st.session_state.page = "Dashboard"

# st.markdown("""
# <style>
# /* Make body and main container full width */
# html, body {
#     margin: 0;
#     padding: 0;
#     width: 100%;
#     background-color: #f3f4f6 !important;
#     overflow-x: hidden;
# }

# /* Streamlit main container full width */
# .block-container {
#     padding: 1rem 2rem !important;
#     margin: 0 !important;
#     width: 100% !important;
#     max-width: 100% !important;
#     background-color: #ffffff  !important;
# }

# /* Remove header/footer decorations */
# div[data-testid="stDecoration"],
# header[data-testid="stHeader"] {
#     display: none !important;
# }

# /* Add top spacing for navbar */
# section.main {
#     padding-top: 120px !important;
#     padding-left: 1rem !important;
#     padding-right: 1rem !important;
#     width: 100%;
# }

# /* Navbar styling */
# .top-navbar {
#     position: fixed;
#     top: 0;
#     left: 0;
#     width: 100vw;
#     z-index: 999;
#     background-color: #0c4a6e;
#     color: white;
#     padding: 1.5rem 2.5rem;
#     display: flex;
#     align-items: center;
#     justify-content: space-between;
#     font-family: 'Segoe UI', sans-serif;
#     box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
# }

# .top-navbar .title {
#     font-size: 1.5rem;
#     font-weight: bold;
#     color: white;
#     display: flex;
#     align-items: center;
#     gap: 0.5rem;
# }

# .top-navbar .title-icon {
#     font-size: 1.75rem;
# }

# .top-navbar a {
#     text-decoration: none;
#     font-weight: 500;
#     font-size: 1rem;
#     color: white;
#     background-color: transparent;
#     padding: 0.5rem 1rem;
#     border-radius: 6px;
#     transition: background 0.3s, color 0.3s;
#     margin-left: 1rem;
# }
# .top-navbar a:hover {
#     background-color: #1e40af;
#     color: #e0f2fe;
# }
# .top-navbar a.active {
#     background-color: white;
#     color: #0c4a6e;
# }

# /* Headings and spacing */
# section.main h1, section.main h2, section.main h3, section.main h4 {
#     margin-bottom: 0.4rem !important;
# }
# section.main .element-container {
#     margin-bottom: 1rem !important;
# }

# .dashboard-subtitle {
#     font-size: 1rem;
#     color: #555;
#     margin-top: -0.4rem;
#     margin-bottom: 1.5rem;
#     text-align: left;
#     padding-left: 0.25rem;
# }
# </style>
# """, unsafe_allow_html=True)


# # NAVBAR
# st.markdown(f"""
# <div class="top-navbar">
#     <div class="title"><span class="title-icon">📊</span> Student Academic Performance</div>
#     <div>
#         <a href="?page=Dashboard" class="{'active' if st.session_state.page == 'Dashboard' else ''}">Dashboard</a>
#         <a href="?page=Model+Training" class="{'active' if st.session_state.page == 'Model Training' else ''}">Model Training</a>
#         <a href="?page=Prediction" class="{'active' if st.session_state.page == 'Prediction' else ''}">Prediction</a>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# # PAGE CONTENT
# if st.session_state.page == "Dashboard":
#     st.header("")
#     st.header("📈 Dashboard Overview")
#     st.markdown("""<link rel="shortcut icon" href="data:image/x-icon;base64,=" />""", unsafe_allow_html=True)

#     total_students = df.shape[0]
#     avg_score = df['average'].mean()
#     min_score = df['average'].min()
#     max_score = df['average'].max()

#     kpi1, kpi2, kpi3, kpi4 = st.columns(4)
#     kpi1.metric("Total Students", total_students)
#     kpi2.metric("Average Score", f"{avg_score:.2f}")
#     kpi3.metric("Min Score", f"{min_score:.2f}")
#     kpi4.metric("Max Score", f"{max_score:.2f}")
#     st.markdown("---")

#     col_a, col_b, col_c = st.columns(3)
#     with col_a:
#         st.subheader("Avg. Score by Parent Ed.")
#         edu_avg = df.groupby('parental_level_of_education')['average'].mean().reset_index()
#         fig_bar = px.bar(edu_avg, x='average', y='parental_level_of_education', orientation='h', color='average', color_continuous_scale='Viridis', height=300)
#         st.plotly_chart(fig_bar, use_container_width=True)

#     with col_b:
#         st.subheader("Score by Gender")
#         fig_box = px.box(df, x='gender', y='average', color='gender', points="all", height=300)
#         st.plotly_chart(fig_box, use_container_width=True)

#     with col_c:
#         st.subheader("Performance Gauge")
#         fig_gauge = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=avg_score,
#             domain={'x': [0, 1], 'y': [0, 1]},
#             title={'text': "Average Score"},
#             gauge={
#                 'axis': {'range': [0, 100]},
#                 'bar': {'color': "blue"},
#                 'steps': [
#                     {'range': [0, 60], 'color': 'red'},
#                     {'range': [60, 80], 'color': 'yellow'},
#                     {'range': [80, 100], 'color': 'green'}
#                 ]
#             }
#         ))
#         st.plotly_chart(fig_gauge, use_container_width=True)

#     st.markdown("---")
#     col_d, col_e = st.columns(2)
#     with col_d:
#         st.subheader("📌 Correlation Heatmap")
#         fig_corr, ax = plt.subplots(figsize=(10, 4))
#         sns.heatmap(df[['math_score', 'reading_score', 'writing_score', 'average']].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
#         st.pyplot(fig_corr)

#     with col_e:
#         st.subheader("Reading vs Writing")
#         fig_scatter = px.scatter(df, x='reading_score', y='writing_score', color='gender', size='average', hover_data=['parental_level_of_education'], height=300)
#         st.plotly_chart(fig_scatter, use_container_width=True)

# elif st.session_state.page == "Model Training":
#     st.header("")
#     st.header("🛠️ Model Training")
#     X = df.drop(columns=['math_score', 'total score', 'average'])
#     y = df['math_score']
#     X_encoded = pd.get_dummies(X)
#     X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

#     if st.button("🚀 Train Model"):
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         score = r2_score(y_test, y_pred) * 100
#         joblib.dump(model, 'model.joblib')
#         st.success("✅ Model trained and saved successfully!")
#         st.metric("R² Score on Test Set", f"{score:.2f}%")

#         fig4, ax = plt.subplots(figsize=(6, 4))
#         sns.scatterplot(x=y_test, y=y_pred, color='purple')
#         plt.xlabel("Actual Exam Scores")
#         plt.ylabel("Predicted Exam Scores")
#         plt.title("Actual vs. Predicted Exam Scores")
#         st.pyplot(fig4)

# elif st.session_state.page == "Prediction":
#     st.header("")
#     st.header("🔮 Prediction")
#     try:
#         model = joblib.load('model.joblib')
#     except:
#         st.error("⚠️ Model not trained yet. Please train it in the Model Training section.")
#         st.stop()

#     st.write("### Input Student Data:")
#     col1, col2 = st.columns(2)
#     with col1:
#         gender = st.selectbox("Gender", ["female", "male"])
#         race = st.selectbox("Race/Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
#         lunch = st.selectbox("Lunch", ['standard', 'free/reduced'])
#         test_prep = st.selectbox("Test Preparation Course", ['none', 'completed'])
#     with col2:
#         parent_edu = st.selectbox("Parental Education", [
#             "some high school", "high school", "some college",
#             "associate's degree", "bachelor's degree", "master's degree"
#         ])
#         reading_score = st.slider("Reading Score", 0, 100, 50)
#         writing_score = st.slider("Writing Score", 0, 100, 50)

#     input_df = pd.DataFrame({
#         'gender': [gender],
#         'race/ethnicity': [race],
#         'parental_level_of_education': [parent_edu],
#         'lunch': [lunch],
#         'test_preparation_course': [test_prep],
#         'reading_score': [reading_score],
#         'writing_score': [writing_score]
#     })

#     input_encoded = pd.get_dummies(input_df)
#     missing_cols = set(model.feature_names_in_) - set(input_encoded.columns)
#     for col in missing_cols:
#         input_encoded[col] = 0
#     input_encoded = input_encoded[model.feature_names_in_]

#     if st.button("🔍 Predict Exam Score"):
#         prediction = model.predict(input_encoded)[0]
#         st.metric("Predicted Exam Score", f"{prediction:.2f}")
#         fig5 = px.bar(
#             x=['Predicted Exam Score', 'Reading Score', 'Writing Score'],
#             y=[prediction, reading_score, writing_score],
#             color=['Predicted Exam Score', 'Reading Score', 'Writing Score'],
#             labels={'x': 'Scores', 'y': 'Values'},
#             title="📊 Predicted vs. Input Scores"
#         )
#         st.plotly_chart(fig5, use_container_width=True) 





# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# import joblib

# # PAGE CONFIG
# st.set_page_config(
#     page_title=" Student Performance Dashboard",
#     page_icon="📊",
#     layout="wide"
# )

# # LOAD DATA
# @st.cache_data
# def load_data():
#     df = pd.read_csv('data/stud.csv')
#     df['total score'] = df['math_score'] + df['reading_score'] + df['writing_score']
#     df['average'] = df['total score'] / 3
#     return df

# df = load_data()

# # READ PAGE FROM QUERY PARAM
# query_page = st.query_params.get("page")
# if query_page:
#     st.session_state.page = query_page.replace("+", " ")
# if 'page' not in st.session_state:
#     st.session_state.page = "Dashboard"

# # STYLING
# st.markdown("""<style>html, body {margin: 0; padding: 0; width: 100%; background-color: #f3f4f6 !important;}
# .block-container {padding: 1rem 2rem !important; width: 100% !important; max-width: 100% !important; background-color: #ffffff  !important;}
# div[data-testid="stDecoration"], header[data-testid="stHeader"] {display: none !important;}
# section.main {padding-top: 120px !important; padding-left: 1rem !important; padding-right: 1rem !important; width: 100%;}
# .top-navbar {position: fixed; top: 0; left: 0; width: 100vw; z-index: 999; background-color: #0c4a6e; color: white; padding: 1.5rem 2.5rem; display: flex; align-items: center; justify-content: space-between; font-family: 'Segoe UI', sans-serif; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);}
# .top-navbar .title {font-size: 1.5rem; font-weight: bold; color: white; display: flex; align-items: center; gap: 0.5rem;}
# .top-navbar .title-icon {font-size: 1.75rem;}
# .top-navbar a {text-decoration: none; font-weight: 500; font-size: 1rem; color: white; padding: 0.5rem 1rem; border-radius: 6px; transition: background 0.3s, color 0.3s; margin-left: 1rem;}
# .top-navbar a:hover {background-color: #1e40af; color: #e0f2fe;}
# .top-navbar a.active {background-color: white; color: #0c4a6e;}
# section.main h1, section.main h2, section.main h3, section.main h4 {margin-bottom: 0.4rem !important;}
# section.main .element-container {margin-bottom: 1rem !important;}
# .dashboard-subtitle {font-size: 1rem; color: #555; margin-top: -0.4rem; margin-bottom: 1.5rem; text-align: left; padding-left: 0.25rem;}
# </style>""", unsafe_allow_html=True)

# # NAVBAR
# st.markdown(f"""
# <div class="top-navbar">
#     <div class="title"><span class="title-icon">📊</span> Student Academic Performance</div>
#     <div>
#         <a href="?page=Dashboard" class="{'active' if st.session_state.page == 'Dashboard' else ''}">Dashboard</a>
#         <a href="?page=Model+Training" class="{'active' if st.session_state.page == 'Model Training' else ''}">Model Training</a>
#         <a href="?page=Prediction" class="{'active' if st.session_state.page == 'Prediction' else ''}">Prediction</a>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# # PAGE CONTENT
# if st.session_state.page == "Dashboard":
#     st.header("")
#     st.header("📈 Dashboard Overview")
#     total_students = df.shape[0]
#     avg_score = df['average'].mean()
#     min_score = df['average'].min()
#     max_score = df['average'].max()

#     kpi1, kpi2, kpi3, kpi4 = st.columns(4)
#     kpi1.metric("Total Students", total_students)
#     kpi2.metric("Average Score", f"{avg_score:.2f}")
#     kpi3.metric("Min Score", f"{min_score:.2f}")
#     kpi4.metric("Max Score", f"{max_score:.2f}")
#     st.markdown("---")

#     col_a, col_b, col_c = st.columns(3)
#     with col_a:
#         st.subheader("Avg. Score by Parent Ed.")
#         edu_avg = df.groupby('parental_level_of_education')['average'].mean().reset_index()
#         fig_bar = px.bar(edu_avg, x='average', y='parental_level_of_education', orientation='h', color='average', color_continuous_scale='Viridis', height=300)
#         st.plotly_chart(fig_bar, use_container_width=True)

#     with col_b:
#         st.subheader("Score by Gender")
#         fig_box = px.box(df, x='gender', y='average', color='gender', points="all", height=300)
#         st.plotly_chart(fig_box, use_container_width=True)

#     with col_c:
#         st.subheader("Performance Gauge")
#         fig_gauge = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=avg_score,
#             domain={'x': [0, 1], 'y': [0, 1]},
#             title={'text': "Average Score"},
#             gauge={
#                 'axis': {'range': [0, 100]},
#                 'bar': {'color': "blue"},
#                 'steps': [
#                     {'range': [0, 60], 'color': 'red'},
#                     {'range': [60, 80], 'color': 'yellow'},
#                     {'range': [80, 100], 'color': 'green'}
#                 ]
#             }
#         ))
#         st.plotly_chart(fig_gauge, use_container_width=True)

#     st.markdown("---")
#     col_d, col_e = st.columns(2)
#     with col_d:
#         st.subheader("📌 Correlation Heatmap")
#         fig_corr, ax = plt.subplots(figsize=(10, 4))
#         sns.heatmap(df[['math_score', 'reading_score', 'writing_score', 'average']].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
#         st.pyplot(fig_corr)

#     with col_e:
#         st.subheader("Reading vs Writing")
#         fig_scatter = px.scatter(df, x='reading_score', y='writing_score', color='gender', size='average', hover_data=['parental_level_of_education'], height=300)
#         st.plotly_chart(fig_scatter, use_container_width=True)

#     st.markdown("---")
#     col_x, col_y, col_z = st.columns(3)
#     with col_x:
#         st.subheader("Gender Distribution")
#         gender_counts = df['gender'].value_counts().reset_index()
#         gender_counts.columns = ['gender', 'count']
#         fig_pie = px.pie(gender_counts, names='gender', values='count', title='', height=300)
#         st.plotly_chart(fig_pie, use_container_width=True)

#     with col_y:
#         st.subheader("Total Score Distribution")
#         fig_hist = px.histogram(df, x='total score', nbins=20, title='', height=300)
#         st.plotly_chart(fig_hist, use_container_width=True)

#     with col_z:
#         st.subheader("Avg. Score by Lunch")
#         fig_lunch = px.box(df, x='lunch', y='average', color='lunch', height=300)
#         st.plotly_chart(fig_lunch, use_container_width=True)

# elif st.session_state.page == "Model Training":
#     st.header("")
#     st.header("🛠️ Model Training")
#     X = df.drop(columns=['math_score', 'total score', 'average'])
#     y = df['math_score']
#     X_encoded = pd.get_dummies(X)
#     X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

#     if st.button("🚀 Train Model"):
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         score = r2_score(y_test, y_pred) * 100
#         joblib.dump(model, 'model.joblib')
#         st.success("✅ Model trained and saved successfully!")
#         st.metric("R² Score on Test Set", f"{score:.2f}%")

#         fig4, ax = plt.subplots(figsize=(6, 4))
#         sns.scatterplot(x=y_test, y=y_pred, color='purple')
#         plt.xlabel("Actual Exam Scores")
#         plt.ylabel("Predicted Exam Scores")
#         plt.title("Actual vs. Predicted Exam Scores")
#         st.pyplot(fig4)

# elif st.session_state.page == "Prediction":
#     st.header("")
#     st.header("🔮 Prediction")
#     try:
#         model = joblib.load('model.joblib')
#     except:
#         st.error("⚠️ Model not trained yet. Please train it in the Model Training section.")
#         st.stop()

#     st.write("### Input Student Data:")
#     col1, col2 = st.columns(2)
#     with col1:
#         gender = st.selectbox("Gender", ["female", "male"])
#         race = st.selectbox("Race/Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
#         lunch = st.selectbox("Lunch", ['standard', 'free/reduced'])
#         test_prep = st.selectbox("Test Preparation Course", ['none', 'completed'])
#     with col2:
#         parent_edu = st.selectbox("Parental Education", [
#             "some high school", "high school", "some college",
#             "associate's degree", "bachelor's degree", "master's degree"
#         ])
#         reading_score = st.slider("Reading Score", 0, 100, 50)
#         writing_score = st.slider("Writing Score", 0, 100, 50)

#     input_df = pd.DataFrame({
#         'gender': [gender],
#         'race/ethnicity': [race],
#         'parental_level_of_education': [parent_edu],
#         'lunch': [lunch],
#         'test_preparation_course': [test_prep],
#         'reading_score': [reading_score],
#         'writing_score': [writing_score]
#     })

#     input_encoded = pd.get_dummies(input_df)
#     missing_cols = set(model.feature_names_in_) - set(input_encoded.columns)
#     for col in missing_cols:
#         input_encoded[col] = 0
#     input_encoded = input_encoded[model.feature_names_in_]

#     if st.button("🔍 Predict Exam Score"):
#         prediction = model.predict(input_encoded)[0]
#         st.metric("Predicted Exam Score", f"{prediction:.2f}")
#         fig5 = px.bar(
#             x=['Predicted Exam Score', 'Reading Score', 'Writing Score'],
#             y=[prediction, reading_score, writing_score],
#             color=['Predicted Exam Score', 'Reading Score', 'Writing Score'],
#             labels={'x': 'Scores', 'y': 'Values'},
#             title="📊 Predicted vs. Input Scores"
#         )
#         st.plotly_chart(fig5, use_container_width=True)




import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# PAGE CONFIG
st.set_page_config(
    page_title=" Student Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv('data/stud.csv')
    df['total score'] = df['math_score'] + df['reading_score'] + df['writing_score']
    df['average'] = df['total score'] / 3
    return df

df = load_data()

# READ PAGE FROM QUERY PARAM
query_page = st.query_params.get("page")
if query_page:
    st.session_state.page = query_page.replace("+", " ")
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"

st.markdown("""<style>html, body {margin: 0; padding: 0; width: 100%; background-color: #f3f4f6 !important;}
.block-container {padding: 1rem 2rem !important; width: 100% !important; max-width: 100% !important; background-color: #ffffff  !important;}
div[data-testid="stDecoration"], header[data-testid="stHeader"] {display: none !important;}
section.main {padding-top: 120px !important; padding-left: 1rem !important; padding-right: 1rem !important; width: 100%;}
.top-navbar {position: fixed; top: 0; left: 0; width: 100vw; z-index: 999; background-color: #0c4a6e; color: white; padding: 1.5rem 2.5rem; display: flex; align-items: center; justify-content: space-between; font-family: 'Segoe UI', sans-serif; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);}
.top-navbar .title {font-size: 1.5rem; font-weight: bold; color: white; display: flex; align-items: center; gap: 0.5rem;}
.top-navbar .title-icon {font-size: 1.75rem;}
.top-navbar a {text-decoration: none; font-weight: 500; font-size: 1rem; color: white; padding: 0.5rem 1rem; border-radius: 6px; transition: background 0.3s, color 0.3s; margin-left: 1rem;}
.top-navbar a:hover {background-color: #1e40af; color: #e0f2fe;}
.top-navbar a.active {background-color: white; color: #0c4a6e;}
section.main h1, section.main h2, section.main h3, section.main h4 {margin-bottom: 0.4rem !important;}
section.main .element-container {margin-bottom: 1.2rem !important;}
.dashboard-subtitle {font-size: 1rem; color: #555; margin-top: -0.4rem; margin-bottom: 1.5rem; text-align: left; padding-left: 0.25rem;}
</style>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="top-navbar">
    <div class="title"><span class="title-icon">📊</span> Student Academic Performance</div>
    <div>
        <a href="?page=Dashboard" class="{'active' if st.session_state.page == 'Dashboard' else ''}">Dashboard</a>
        <a href="?page=Model+Training" class="{'active' if st.session_state.page == 'Model Training' else ''}">Model Training</a>
        <a href="?page=Prediction" class="{'active' if st.session_state.page == 'Prediction' else ''}">Prediction</a>
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.page == "Dashboard":
    st.header("")
    st.header("📈 Dashboard Overview")
    total_students = df.shape[0]
    avg_score = df['average'].mean()
    min_score = df['average'].min()
    max_score = df['average'].max()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Students", total_students)
    kpi2.metric("Average Score", f"{avg_score:.2f}")
    kpi3.metric("Min Score", f"{min_score:.2f}")
    kpi4.metric("Max Score", f"{max_score:.2f}")
    st.markdown("---")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.subheader("Avg. Score by Parent Ed.")
        edu_avg = df.groupby('parental_level_of_education')['average'].mean().reset_index()
        fig_bar = px.bar(edu_avg, x='average', y='parental_level_of_education', orientation='h', color='average', color_continuous_scale='Viridis', height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.subheader("Score by Gender")
        fig_box = px.box(df, x='gender', y='average', color='gender', points="all", height=300)
        st.plotly_chart(fig_box, use_container_width=True)

    with col_c:
        st.subheader("Performance Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Average Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "blue"},
                'steps': [
                    {'range': [0, 60], 'color': 'red'},
                    {'range': [60, 80], 'color': 'yellow'},
                    {'range': [80, 100], 'color': 'green'}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")
    col_d, col_e = st.columns(2)
    with col_d:
        st.subheader("📌 Correlation Heatmap")
        fig_corr, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(df[['math_score', 'reading_score', 'writing_score', 'average']].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        st.pyplot(fig_corr)

    with col_e:
        st.subheader("Reading vs Writing")
        fig_scatter = px.scatter(df, x='reading_score', y='writing_score', color='gender', size='average', hover_data=['parental_level_of_education'], height=300)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        st.subheader("🎨 Gender Distribution")
        gender_counts = df['gender'].value_counts().reset_index()
        gender_counts.columns = ['gender', 'count']
        fig_pie = px.pie(gender_counts, names='gender', values='count', color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_v2:



        st.subheader("🍱 Avg. by Lunch Type")
        fig_lunch = px.box(df, x='lunch', y='average', color='lunch', color_discrete_sequence=['#1f77b4', '#ff7f0e'])
        st.plotly_chart(fig_lunch, use_container_width=True)

    with col_v3:
        st.subheader("🎯 Total Score Histogram")
        fig_hist = px.histogram(df, x='total score', nbins=20, color_discrete_sequence=['#6a0dad'])
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    # img_col1, img_col2, img_col3 = st.columns(3)
    # with img_col1:
    #     st.image("https://via.placeholder.com/300x200.png?text=Student+1", caption="Student Achievement", use_column_width=True)
    # with img_col2:
    #     st.image("https://via.placeholder.com/300x200.png?text=Student+2", caption="Learning Activity", use_column_width=True)
    # with img_col3:
    #     st.image("https://via.placeholder.com/300x200.png?text=Student+3", caption="Group Work", use_column_width=True)

elif st.session_state.page == "Model Training":
    st.header("")
    st.header("🛠️ Model Training")
    X = df.drop(columns=['math_score', 'total score', 'average'])
    y = df['math_score']
    X_encoded = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    if st.button("🚀 Train Model"):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred) * 100
        joblib.dump(model, 'model.joblib')
        st.success("✅ Model trained and saved successfully!")
        st.metric("R² Score on Test Set", f"{score:.2f}%")

        fig4, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=y_test, y=y_pred, color='purple')
        plt.xlabel("Actual Exam Scores")
        plt.ylabel("Predicted Exam Scores")
        plt.title("Actual vs. Predicted Exam Scores")
        st.pyplot(fig4)

elif st.session_state.page == "Prediction":
    st.header("")
    st.header("🔮 Prediction")
    try:
        model = joblib.load('model.joblib')
    except:
        st.error("⚠️ Model not trained yet. Please train it in the Model Training section.")
        st.stop()

    st.write("### Input Student Data:")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["female", "male"])
        race = st.selectbox("Race/Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
        lunch = st.selectbox("Lunch", ['standard', 'free/reduced'])
        test_prep = st.selectbox("Test Preparation Course", ['none', 'completed'])
    with col2:
        parent_edu = st.selectbox("Parental Education", [
            "some high school", "high school", "some college",
            "associate's degree", "bachelor's degree", "master's degree"
        ])
        reading_score = st.slider("Reading Score", 0, 100, 50)
        writing_score = st.slider("Writing Score", 0, 100, 50)

    input_df = pd.DataFrame({
        'gender': [gender],
        'race/ethnicity': [race],
        'parental_level_of_education': [parent_edu],
        'lunch': [lunch],
        'test_preparation_course': [test_prep],
        'reading_score': [reading_score],
        'writing_score': [writing_score]
    })

    input_encoded = pd.get_dummies(input_df)
    missing_cols = set(model.feature_names_in_) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0
    input_encoded = input_encoded[model.feature_names_in_]

    if st.button("🔍 Predict Exam Score"):
        prediction = model.predict(input_encoded)[0]
        st.metric("Predicted Exam Score", f"{prediction:.2f}")
        fig5 = px.bar(
            x=['Predicted Exam Score', 'Reading Score', 'Writing Score'],
            y=[prediction, reading_score, writing_score],
            color=['Predicted Exam Score', 'Reading Score', 'Writing Score'],
            labels={'x': 'Scores', 'y': 'Values'},
            title="📊 Predicted vs. Input Scores"
        )
        st.plotly_chart(fig5, use_container_width=True)
