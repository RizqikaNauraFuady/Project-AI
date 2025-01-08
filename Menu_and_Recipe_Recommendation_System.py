import streamlit as st
st.sidebar.title("Welcome🙌")
page = st.sidebar.radio("select the menu below", ["Tutorial", "Recommendation System"])

if page == "Tutorial":
    st.title("Tutorial: How to Use This Application🍴")
    st.markdown("""
        ### Welcome to the Menu and Recipe Recommendation System!
        This application helps you find recipes based on your nutritional needs.
        
        **Steps to use the application:**
        1. Go to the 'Recommendation System' page.
        2. Input your desired nutritional values such as calories, fat, etc.
        3. (Optional) Add any specific ingredients you want in your recipe.
        4. Click on "Find Recommendation" to get your results.
        5. Explore the recommended recipes, their nutritional details, and instructions.

        Enjoy using the application!
    """)

elif page == "Recommendation System":

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from numpy.linalg import norm
    from streamlit_echarts import st_echarts
    import re

    def manual_standardization(dataframe):
        nutrition_columns = dataframe.columns[6:15]
        data = dataframe[nutrition_columns].to_numpy()
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(data)
        return standardized_data, scaler

    def cosine_distance(a, b):
        return 1 - np.dot(a, b) / (norm(a) * norm(b))

    def manual_knn_predictor(input_data, prep_data, k=10):
        distances = [(cosine_distance(input_data, data_point), idx)
                    for idx, data_point in enumerate(prep_data)]
        distances.sort(key=lambda x: x[0])
        return [idx for _, idx in distances[:k]]

    def manual_extract_data(dataframe, ingredient_filter, max_nutritional_values):
        extracted_data = dataframe.copy()
        for column, maximum in zip(extracted_data.columns[6:15], max_nutritional_values):
            extracted_data = extracted_data[extracted_data[column] < maximum]
        if ingredient_filter:
            for ingredient in ingredient_filter:
                extracted_data = extracted_data[extracted_data['RecipeIngredientParts'].str.contains(ingredient, regex=False)]
        return extracted_data
    
    def apply_manual_pipeline(input_data, dataframe, k=5, max_nutritional_values=None, ingredient_filter=None):
        extracted_data = manual_extract_data(dataframe, ingredient_filter, max_nutritional_values)
        prep_data, scaler = manual_standardization(extracted_data)
        input_data_scaled = scaler.transform(input_data.reshape(1, -1))
        nearest_indices = manual_knn_predictor(input_data_scaled[0], prep_data, k)
        return extracted_data.iloc[nearest_indices]

    def recommend(dataframe, input_data, max_nutritional_values, ingredient_filter=None, k=5):
        return apply_manual_pipeline(input_data, dataframe, k, max_nutritional_values, ingredient_filter)

    st.title("Menu and Recipe Recommendation System Based on Nutritional Needs")
    st.markdown("""
        This system is designed to provide menu recommendations and food recipes that match the user's nutritional needs, 
        such as calories, fat, carbohydrates, protein, and others. The user can input the desired nutritional values, 
        and the system will select menus and recipes that meet these criteria.
    """)

    @st.cache_data
    def load_data():
        file_path = 'recipes_data.csv'
        data = pd.read_csv(file_path)
        return data

    data = load_data()
    columns = ['RecipeId', 'Name', 'CookTime', 'PrepTime', 'TotalTime',
            'RecipeIngredientParts', 'Calories', 'FatContent', 'SaturatedFatContent',
            'CholesterolContent', 'SodiumContent', 'CarbohydrateContent',
            'FiberContent', 'SugarContent', 'ProteinContent', 'RecipeInstructions', 'Images', 'RecipeIngredientQuantities']
    dataset = data[columns]

    max_list = [2000, 100, 13, 300, 2300, 325, 40, 40, 200]

    with st.form("recommendation_form"):
        st.subheader("Input Nutritional values:")
        calories = st.slider("Calories", 0, 2000, 500, step=1)
        fat = st.slider("Fat Content", 0, 100, 20, step=1)
        saturated_fat = st.slider("Saturated Fat Content", 0, 13, 5, step=1)
        cholesterol = st.slider("Cholesterol Content", 0, 300, 50, step=1)
        sodium = st.slider("Sodium Content", 0, 2300, 500, step=1)
        carbohydrate = st.slider("Carbohydrate Content", 0, 325, 100, step=1)
        fiber = st.slider("Fiber Content", 0, 50, 10, step=1)
        sugar = st.slider("Sugar Content", 0, 40, 5, step=1)
        protein = st.slider("Protein Content", 0, 40, 10, step=1)
        
        input_nutritional_values = np.array([calories, fat, saturated_fat, cholesterol, sodium, carbohydrate, fiber, sugar, protein])

        st.subheader("Recommendation options (OPTIONAL):")
        ingredient_filter = st.text_input("Specify ingredients to include in the recommendations separated by ';' :", placeholder="Egg;Milk;Butter")
        ingredients = ingredient_filter.split(";") if ingredient_filter else None

        k = st.slider("Number of recommendations", 5, 20, step=5)
        generated = st.form_submit_button("Find Recommendation")

    if generated:
        with st.spinner("Looking for suitable recipes ..."):
            recommendations = recommend(dataset, input_nutritional_values, max_list, ingredients, k)
            st.session_state.recommendations = recommendations

    if "recommendations" in st.session_state:
        recommendations = st.session_state.recommendations
        if not recommendations.empty:
            st.subheader("Recommended recipes:")
            for idx, row in recommendations.iterrows():
                with st.expander(row["Name"]):
                    if 'Images' in row and pd.notna(row['Images']):
                        image_urls = re.findall(r'\"(https?://[^\"]+)\"', row['Images'])
                        if image_urls:
                            st.image(image_urls[0], caption=row['Name'], use_container_width=True)

                    st.write(f"**Nutrition Values** (g):")
                    nutritional_values = row[['Calories', 'FatContent', 'SaturatedFatContent',
                            'CholesterolContent', 'SodiumContent', 'CarbohydrateContent',
                            'FiberContent', 'SugarContent', 'ProteinContent']]
                    nutritional_df = pd.DataFrame(nutritional_values).transpose()
                    st.table(nutritional_df)

                    st.write(f"**Ingredients:**")
                    ingredients_list = row['RecipeIngredientParts'].strip("c()").replace('"', '').split(", ")
                    for ingredient in ingredients_list:
                        st.write(f"- {ingredient}")

                    st.write(f"**Ingredient Quantities:**")
                    if pd.notna(row['RecipeIngredientQuantities']):
                        quantities_list = row['RecipeIngredientQuantities'].strip("c()").replace('"', '').split(", ")
                        for quantity in quantities_list:
                            st.write(f"- {quantity}")
                    else:
                        st.write("Not available")

                    st.write(f"**Recipe Instructions:**")
                    instructions_list = row['RecipeInstructions'].strip("c()").replace('"', '').split(", ")
                    for idx, instruction in enumerate(instructions_list, start=1):
                        st.write(f"{idx}. {instruction}")

                    st.write(f"**Cooking and Preparation Time:**")
                    st.write(f"- Cook Time: {row['CookTime']}")
                    st.write(f"- Preparation Time: {row['PrepTime']}")
                    st.write(f"- Total Time: {row['TotalTime']}")

            st.subheader("Overview")
            selected_recipe = st.selectbox("Select a recipe", recommendations["Name"])
            selected_data = recommendations[recommendations["Name"] == selected_recipe].iloc[0]
            options = {
        "title": {
            "text": "Nutrition Overview",
            "subtext": selected_recipe,
            "left": "center"
        },
        "tooltip": {
            "trigger": "item",
            "formatter": "{a} <br/>{b}: {c} ({d}%)"
        },
        "legend": {
            "orient": "vertical",
            "left": "left",
        },
        "series": [{
            "name": "Nutrition values",
            "type": "pie",
            "radius": "50%",
            "data": [
                {"value": selected_data[col], "name": col} for col in dataset.columns[6:15]
            ],
            "emphasis": {
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowOffsetX": 0,
                    "shadowColor": "rgba(0, 0, 0, 0.5)"
                }
            }
        }]
    }

            st_echarts(options, height="500px")
        else:
            st.error("Couldn't find any recipes with the specified ingredients")