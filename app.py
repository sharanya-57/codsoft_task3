import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# App title
st.markdown("<h1 style='text-align: center; color: purple;'>🌸 Iris Flower Prediction App</h1>", unsafe_allow_html=True)

st.write("### Enter flower measurements:")

# Layout (2 columns)
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8)
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.2)
    petal_length = st.slider("Petal Length", 1.0, 7.0, 4.8)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 1.5)

# Prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)

species = ["Setosa", "Versicolor", "Virginica"]

with col2:
    st.write("## Prediction:")
    st.success(species[prediction[0]])

    # Show image based on prediction
    if species[prediction[0]] == "Setosa":
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg")
    elif species[prediction[0]] == "Versicolor":
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg")
    else:
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg")

st.write("---")
st.caption("Made with ❤️ for CodSoft Internship")