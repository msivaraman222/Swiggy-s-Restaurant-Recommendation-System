import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    cleaned_data = pd.read_csv("cleaned_data.csv")
    encoded_data = pd.read_csv("encoded_data.csv")

    cleaned_data = cleaned_data.reset_index(drop=True)
    encoded_data = encoded_data.reset_index(drop=True)

    # Keep only numeric columns and handle missing values
    numeric_data = encoded_data.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy='mean')
    numeric_data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data),
                                        columns=numeric_data.columns)
    return cleaned_data, numeric_data_imputed

cleaned_data, numeric_data = load_data()
similarity_matrix = cosine_similarity(numeric_data)

# -------------------- RECOMMENDATION FUNCTION --------------------
def recommend_restaurants(restaurant_name, top_n=5):
    if restaurant_name not in cleaned_data["name"].values:
        st.error(f"Restaurant '{restaurant_name}' not found.")
        return pd.DataFrame()

    idx = cleaned_data[cleaned_data["name"] == restaurant_name].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    indices = [i[0] for i in scores]

    cols_to_show = ["name", "city", "cuisine", "rating", "cost", "link", "menu"]
    return cleaned_data.loc[indices, cols_to_show].reset_index(drop=True)

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="🍴 Restaurant Recommender", layout="wide")

st.title("🍽️ Restaurant Recommendation System")
st.write("Find similar restaurants based on cuisine, rating, and price — with links to their pages & menus!")

# ---- User Filters ----
col1, col2, col3, col4 = st.columns(4)
with col1:
    city_filter = st.selectbox("Select City", ["All"] + sorted(cleaned_data["city"].dropna().unique().tolist()))
with col2:
    cuisine_filter = st.selectbox("Select Cuisine", ["All"] + sorted(cleaned_data["cuisine"].dropna().unique().tolist()))
with col3:
    min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
with col4:
    max_cost = st.number_input("Maximum Price for Two", value=1000, step=100)

# ---- Apply Filters ----
filtered_data = cleaned_data.copy()
if city_filter != "All":
    filtered_data = filtered_data[filtered_data["city"] == city_filter]
if cuisine_filter != "All":
    filtered_data = filtered_data[filtered_data["cuisine"].str.contains(cuisine_filter, case=False, na=False)]
filtered_data = filtered_data[(filtered_data["rating"] >= min_rating) & (filtered_data["cost"] <= max_cost)]

st.write(f"### 🍛 Showing {len(filtered_data)} matching restaurants")

# ---- Restaurant Selection ----
restaurant_list = filtered_data["name"].unique().tolist()
if len(restaurant_list) > 0:
    restaurant_name = st.selectbox("Select a restaurant for recommendations:", restaurant_list)

    if st.button("🔍 Get Recommendations"):
        recommendations = recommend_restaurants(restaurant_name, top_n=5)
        if not recommendations.empty:
            st.subheader(f"🍴 Restaurants similar to **{restaurant_name}**:")

            # Display nicely formatted cards
            for _, row in recommendations.iterrows():
                with st.container():
                    st.markdown(f"### 🏠 {row['name']}")
                    st.markdown(f"**📍 City:** {row['city']}  |  **⭐ Rating:** {row['rating']}  |  **💰 Cost for two:** ₹{row['cost']}")
                    st.markdown(f"**🍽️ Cuisine:** {row['cuisine']}")
                    if pd.notna(row.get("link")):
                        st.markdown(f"[🔗 Visit Restaurant Page]({row['link']})", unsafe_allow_html=True)
                    if pd.notna(row.get("menu")):
                        st.markdown(f"[📜 View Menu]({row['menu']})", unsafe_allow_html=True)
                    st.divider()
        else:
            st.warning("No recommendations found.")
else:
    st.warning("No restaurants match your filters.")

# ---- Footer ----
st.markdown("---")
st.caption("Developed by Siva Raman | Restaurant Recommendation System using Cosine Similarity")
