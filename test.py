import streamlit as st
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

customer_data = pd.read_csv("Mall_Customers.csv")

X = customer_data.loc[:,['Annual Income (k$)','Spending Score (1-100)']].values

st.set_page_config(page_title="Customer Segment Prediction App", page_icon=":guardsman:", layout="wide")

# Define a session state variable for login
class SessionState:
    def __init__(self, session):
        self._session = session
        self.logged_in = self._get_state('logged_in', False)

    def _get_state(self, key, default):
        return self._session.get(key, default)

    def _set_state(self, key, value):
        self._session[key] = value

    def clear(self):
        self._session.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

session_state = SessionState.get(get_report_ctx().session_id)

# Define a function for login
def login():
    password = st.text_input("Enter password", type="password")
    if st.button("Login") and _CodeHasher.verify_password(password, "password_hash"):
        session_state.logged_in = True

# Define a function for logout
def logout():
    session_state.logged_in = False

# Define the main app
def main():
    if not session_state.logged_in:
        login()
    else:
        st.title('Customer Segment Prediction')
        st.image("""https://cdn.qualitygurus.com/wp-content/uploads/2022/09/Customer-Segmentation.jpg?compress=true&quality=80&w=1366&dpr=1.0""")

        #display how the clusters are generated

        # Run K-means clustering
        kmeans = KMeans(n_clusters=5,random_state=42)
        labels = kmeans.fit_predict(X)

        # Create the scatter plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(X[:,0], X[:,1], c=labels, cmap='rainbow')
        centers = ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='black')
        ax.set_title('Distribution')
        ax.set_xlabel('Annual Income')
        ax.set_ylabel('Spending Score (1-100)')

        # Create the legend
        legend1 = ax.legend(*scatter.legend_elements(), title="Segments")
        legend2 = ax.legend([centers], ['Cluster centers'], loc="lower right")

        # Add the legends to the plot
        ax.add_artist(legend1)
        ax.add_artist(legend2)

        # Display the plot in Streamlit
        st.pyplot(fig)

        st.sidebar.title("Enter input values ")
        form = st.sidebar.form(key='my_form')
        age=form.number_input(label="Age")
        gender=form.radio("Gender",["Male","Female"])
        annual_income = form.number_input(label="annual_income")
        spending_score = form.number_input(label="spending_score(0-100)")

        Select=form.selectbox("Algorithm",["KNN"])
        submit_button = form.form_submit_button(label='Submit')
        if gender=="Male":
            gender=1
       
