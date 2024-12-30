import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from IPython.display import Image, display, HTML
import difflib
import kagglehub
import os

# Unduh dataset menggunakan kagglehub
path = kagglehub.dataset_download("anasfikrihanif/indonesian-food-and-drink-nutrition-dataset")
# Dapatkan daftar file CSV di folder dataset
file_names = [file for file in os.listdir(path) if file.endswith('.csv')]

# List untuk menyimpan DataFrame sementara
data_frames = []

# Membaca setiap file CSV dan menambahkannya ke dalam list
for file_name in file_names:
    file_path = os.path.join(path, file_name)
    df = pd.read_csv(file_path)
    data_frames.append(df)

# Menggabungkan semua DataFrame menjadi satu
combined_data = pd.concat(data_frames, ignore_index=True)

# Pilih kolom yang relevan
columns_to_keep = ['id', 'calories', 'proteins', 'fat', 'carbohydrate', 'name', 'image']
combined_data = combined_data[columns_to_keep]

# Isi nilai null
numerical_cols = ['calories', 'proteins', 'fat', 'carbohydrate']
combined_data[numerical_cols] = combined_data[numerical_cols].fillna(combined_data[numerical_cols].mean())
categorical_cols = ['name', 'image']
for col in categorical_cols:
    combined_data[col] = combined_data[col].fillna("Unknown")

# Gabungkan data dengan nama yang sama
combined_data = combined_data.groupby('name', as_index=False).agg({
    'calories': 'mean',
    'proteins': 'mean',
    'fat': 'mean',
    'carbohydrate': 'mean',
    'id': 'first',
    'image': 'first'
})

# Normalisasi kolom numerik
scaler = MinMaxScaler()
combined_data[numerical_cols] = scaler.fit_transform(combined_data[numerical_cols])

# Gunakan data gabungan untuk rekomendasi makanan
data = combined_data
nutritional_features = data[['calories', 'proteins', 'fat', 'carbohydrate']]
normalized_data = scaler.fit_transform(nutritional_features)

# Fungsi untuk memberikan saran makanan berdasarkan nama umum
def suggest_foods(food_name, data):
    """
    Memberikan saran makanan berdasarkan nama umum atau substring.
    Prioritaskan hasil dengan Similarity tertinggi jika ada duplikasi.
    """
    food_name_lower = food_name.lower()
    matches = data[data['name'].str.lower().str.contains(food_name_lower)]

    # Jika terdapat lebih dari satu hasil dengan nama yang sama, pilih berdasarkan Similarity tertinggi
    if len(matches) > 1:
        similarity_scores = cosine_similarity(
            normalized_data[matches.index],
            normalized_data[matches.index].mean(axis=0).reshape(1, -1)
        ).flatten()
        matches['Similarity'] = similarity_scores
        matches = matches.sort_values(by='Similarity', ascending=False)

    return matches

# Fungsi untuk menampilkan detail makanan dengan gambar
def display_food_details_with_image(food_details):
    """
    Menampilkan detail makanan beserta gambar.
    """
    st.write(food_details[['name', 'calories', 'proteins', 'fat', 'carbohydrate']])
    image_url = food_details['image'].values[0]
    if image_url != "Unknown":
        display(Image(url=image_url, width=300, height=300))
    else:
        st.write("Gambar tidak tersedia untuk makanan ini.")


# Fungsi rekomendasi makanan berdasarkan nutrisi
def recommend_food(user_food, top_n=3, preferences=None):
    if user_food not in data['name'].values:
        return "Makanan tidak ditemukan dalam database."

    user_index = data[data['name'] == user_food].index[0]
    similarity_scores = cosine_similarity([normalized_data[user_index]], normalized_data).flatten()
    data['Similarity'] = similarity_scores

    filtered_data = data.copy()
    if preferences:
        for key, value in preferences.items():
            if key in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[key] <= value]

    recommended = filtered_data.sort_values(by='Similarity', ascending=False).iloc[1:top_n+1]
    return recommended[['name', 'calories', 'fat', 'carbohydrate', 'Similarity', 'image']]


# Fungsi validasi input float
def get_valid_float(prompt):
    while True:
        user_input = st.text_input(prompt).replace(',', '.')
        try:
            return float(user_input)
        except ValueError:
            st.write(f"Input '{user_input}' tidak valid. Harap masukkan angka.")

# Workflow Hybrid untuk pilihan awal pengguna
def get_initial_food_choice(data):
    """
    Pilihan awal untuk rekomendasi makanan.
    """
    st.write("1. Tampilkan daftar makanan secara acak.")
    st.write("2. Masukkan nama makanan favorit Anda untuk rekomendasi.")
    st.write("3. Gunakan preferensi nutrisi untuk rekomendasi awal.")

    choice = st.text_input("Masukkan pilihan Anda (1/2/3): ").strip()

    if choice == '1':
        st.write("Daftar makanan secara acak:")
        random_foods = data.sample(10)
        for _, row in random_foods.iterrows():
            st.write(f"Nama: {row['name']}")
            if row['image'] != "Unknown":
                display(HTML(f'<img src="{row["image"]}" alt="{row["name"]}" style="width:300px;height:auto;">'))
            else:
                st.write("Gambar tidak tersedia.")

        while True:
            food_choice = st.text_input("Masukkan nama makanan favorit Anda dari daftar di atas: ").strip()
            if food_choice != "":
                suggestions = suggest_foods(food_choice, random_foods)
                if len(suggestions) > 0:
                    st.write("Detail makanan yang sesuai:")
                    for _, row in suggestions.iterrows():
                        st.write(f"Nama: {row['name']}, Kalori: {row['calories']}, Lemak: {row['fat']}, Karbohidrat: {row['carbohydrate']}")
                        st.image(row['image'], width=300)
                    return suggestions['name'].iloc[0]
                else:
                    st.write(f"Makanan '{food_choice}' tidak ditemukan dalam daftar acak. Coba lagi.")
            else:
                st.stop()

    elif choice == '2':
        while True:
            food_choice = st.text_input("Masukkan nama makanan favorit Anda: ").strip()
            if food_choice != "":
                suggestions = suggest_foods(food_choice, data)
                if len(suggestions) > 0:
                    st.write("Detail makanan yang sesuai:")
                    for _, row in suggestions.iterrows():
                        st.write(f"Nama: {row['name']}, Kalori: {row['calories']}, Lemak: {row['fat']}, Karbohidrat: {row['carbohydrate']}")
                        st.image(row['image'], width=300)
                    return suggestions['name'].iloc[0]
                else:
                    st.write(f"Makanan '{food_choice}' tidak ditemukan dalam database. Coba lagi.")

    elif choice == '3':
        preferences = {
            'calories': get_valid_float("Maksimum kalori: "),
            'fat': get_valid_float("Maksimum lemak (g): "),
            'carbohydrate': get_valid_float("Maksimum karbohidrat (g): ")
        }
        recommended_data = recommend_food(data.iloc[0]['name'], preferences=preferences)
        if isinstance(recommended_data, str):
            st.write(recommended_data)
            return None
        else:
            food_choice = recommended_data.iloc[0]['name']
            food_details = data[data['name'] == food_choice]
            st.write("Detail makanan yang dipilih berdasarkan preferensi Anda:")
            display_food_details_with_image(food_details)
            st.write(f"Rekomendasi awal berdasarkan preferensi: {food_choice}")
            return food_choice

# Main Program

user_food_choice = get_initial_food_choice(data)
if user_food_choice:
    recommended_foods = recommend_food(user_food_choice, top_n=3)
    show_details = st.text_input("Apakah Anda ingin melihat detail setiap rekomendasi? (ya/tidak): ").strip().lower()

    if show_details == 'ya':
        for _, row in recommended_foods.iterrows():
            st.write(f"Nama: {row['name']}, Kalori: {row['calories']}, Lemak: {row['fat']}, Karbohidrat: {row['carbohydrate']}, Similarity: {row['Similarity']}")
            display(Image(url=row['image'], width=300, height=300))
    else:
        st.write(recommended_foods[['name', 'Similarity', 'calories', 'fat', 'carbohydrate']].to_string(index=False))

