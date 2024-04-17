import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pickle
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
import os # Import SimpleImputer


# Menampilkan markdown hanya jika opsi yang dipilih bukanlah "Dashboard"
# Sidebar
st.sidebar.title('Halaman')
selected_option = st.sidebar.selectbox('Select an option:', ['Dashboard', 'Distribution', 'Comparison', 'Composition', 'Relationship', 'Clustering'])
# Load data
url = 'https://raw.githubusercontent.com/amaliakartikasari/Mini-Project-Data-Mining/main/Check%20Point%203/Data%20Cleaned%20Fix.csv'
df_file = pd.read_csv(url)

# Tampilkan konten berdasarkan opsi yang dipilih
if selected_option == 'Dashboard':
    # Misalnya, jika Anda memiliki gambar dalam variabel img
    img = open('pelamar.jpg', 'rb').read()
    st.image(img)

    st.markdown("""
    # Analisis Faktor-Faktor yang Mempengaruhi Ketertarikan Pelamar terhadap Citra Perusahaan di Indonesia pada 13 Desember 2021
    """)
    st.write(df_file)  # Menampilkan seluruh data pada halaman "Dashboard"
    # Menampilkan teks dengan rata kanan kiri menggunakan markdown dan HTML
    st.markdown(
        """
        <div style="text-align: justify">
        Tabel diatas menggambarkan data fiktif yang menunjukkan tingkat ketertarikan pelamar terhadap citra perusahaan di Indonesia, serta faktor-faktor yang mempengaruhi ketertarikan tersebut, seperti lingkungan kerja dan pengembangan karir. Data tersebut disajikan dalam bentuk tabel dengan nama perusahaan, skor ketertarikan, skor lingkungan kerja, dan skor pengembangan karir. Analisis lebih lanjut terhadap data tersebut dapat memberikan wawasan yang berguna dalam memahami faktor-faktor yang memengaruhi persepsi pelamar terhadap citra perusahaan di Indonesia.
        </div>
        """,
        unsafe_allow_html=True
    )

if selected_option == 'Distribution':
    st.markdown("<h1 style='text-align: center;'>DISTRIBUTION</h1>", unsafe_allow_html=True)

    selected = st.selectbox('Pilih Data:', ['adType', 'Salary Currency', 'Salary Period', 'Location Category'])
    #    Create a figure and axis object
    if selected == 'adType':
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the bar chart for adType distribution
        sns.countplot(data=df_file, x='adType', ax=ax)
        ax.set_title('Distribution of Ad Types')
        ax.set_xlabel('Ad Type')
        ax.set_ylabel('Count')

        # Show the plot using Streamlit
        st.pyplot(fig)
        # Menampilkan caption dengan spasi di antara angka menggunakan markdown dan HTML
        st.caption("""
        <div style="text-align: center; margin-top: 10px;">
        0 : Standard &nbsp;&nbsp;&nbsp; 1 : Standout &nbsp;&nbsp;&nbsp; 2 : auto_increment
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
        """
        <div style="text-align: justify">
        Pada gambar di atas dapat dilihat bahwa terdapat tiga jenis data yang berada pada Ad Type yaitu, 0 adalah pekerjaan standard yang mengacu pada tugas rutin yang terstruktur dengan produksi yang jelas. Dapat dilihat pada barplot diatas bahwa pekerjaan dengan tipe standard menjadi jenis pekerjaan tertinggi dengan range 14.000-16.000 lamaran pekerjaan. Pekerjaan tipe 1 adalah pekerjaan standout dimana pekerjaan ini menuntut karyawan untuk menunjukan kinerja luar biasa dengan tanggung jawab yang lebih besar dengan tantangan yang lebih kompleks. Pada barplot di atas menunjukan bahwa pekerjaan tipe ini cenderung banyak dengan range 10.000 sampai 12.000 lamaran pekerjaan. Sedangkan untuk pekerjaan dengan tipe auto_increment ini berjumlah lebih sedikit daripada yang lain yaitu dibawah 2.000 lamaran pekerjaan. Pekerjaan dengan tipe auto_increment ini dirancang untuk memberikan peningkatan yang bertahap dengan tanggung jawab yang kompleks dengan memberikan kesempatan bagi karyawan dalam berkembang dan meningkatkan keterampilan. hal itu yang mempengaruhi jumlah pekerjaan dengan jenis auto_increment ini lebih sedikit dibanding jenis yang lain.
        </div>
        """,
        unsafe_allow_html=True
    )

    if selected == 'Salary Currency':
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the bar chart for adType distribution
        sns.countplot(data=df_file, x='salarycurrency', ax=ax)
        ax.set_title('Distribution of salarycurrency')
        ax.set_xlabel('salarycurrency')
        ax.set_ylabel('Count')

        # Show the plot using Streamlit
        st.pyplot(fig)
        st.caption("""
        <div style="text-align: center; margin-top: 10px;">
        0 : IDR
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
        """
        <div style="text-align: justify">
        Pada barplot di atas dapat dilihat bahwa seluruh data yang ada merupakan tipe 0 yaitu Salary Currency berupa IDR
        </div>
        """,
        unsafe_allow_html=True
    )

    if selected == 'Salary Period':
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the bar chart for adType distribution
        sns.countplot(data=df_file, x='salaryPeriod', ax=ax)
        ax.set_title('Distribution of salaryPeriod')
        ax.set_xlabel('salaryPeriod')
        ax.set_ylabel('Count')

        # Show the plot using Streamlit
        st.pyplot(fig)
        st.caption("""
        <div style="text-align: center; margin-top: 10px;">
        0 : Monthly
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
        """
        <div style="text-align: justify">
        Pada barplot di atas dapat dilihat bahwa seluruh data yang ada merupakan tipe 0 yaitu Salary Period berupa Monthly dimana seluruh lamaran pekerjaan memiliki gaji yang diberikan dalam rentang waktu bulanan.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if selected == 'Location Category':
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the bar chart for adType distribution
        sns.countplot(data=df_file, x='LocationCategory', ax=ax)
        ax.set_title('Distribution of LocationCategory')
        ax.set_xlabel('LocationCategory')
        ax.set_ylabel('Count')

        # Show the plot using Streamlit
        st.pyplot(fig)
        st.caption("""
        <div style="text-align: center; margin-top: 10px;">
        0 : Jakarta &nbsp;&nbsp;&nbsp; 1 : Luar Jakarta 
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
        """
        <div style="text-align: justify">
        Pada gambar di atas dapat dilihat bahwa terdapat dua jenis Location Category yaitu 0 yang berada di Jakarta dan 1 yang berada di Luar Jakarta. Pada barplot di atas dapat dilihat bahwa perubahan signifikan untuk Location Category tidak terlalu signifikan dan hanya terdapat sedikit perbedaan untuk jumlah kedua data.
        </div>
        """,
        unsafe_allow_html=True
    )

if selected_option == 'Comparison':
    st.markdown("<h1 style='text-align: center;'>COMPARISON</h1>", unsafe_allow_html=True)

    # Group by LocationCategory and calculate the counts of each adType
    comparison_data = df_file.groupby('LocationCategory')['adType'].value_counts().unstack()

    # Plot the grouped bar chart
    fig, ax = plt.subplots()
    comparison_data.plot(kind='bar', ax=ax, stacked=False)
    ax.set_xlabel('Location Category')
    ax.set_ylabel('Count')
    ax.set_title('Comparison of Ad Types by Location Category')
    ax.legend(title='Ad Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)  # Menampilkan plot menggunakan Streamlit
    st.caption("""
    <div style="text-align: center; margin-top: 10px;">
    Location Category &nbsp;&nbsp;&nbsp;=&nbsp;&nbsp;&nbsp; 0 : Jakarta &nbsp;&nbsp;&nbsp; 1 : Luar Jakarta <br>
    Ad Type &nbsp;&nbsp;&nbsp;=&nbsp;&nbsp;&nbsp; 0 : Standard &nbsp;&nbsp;&nbsp; 1 : Standout &nbsp;&nbsp;&nbsp; 2 : auto_increment
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
    """
    <div style="text-align: justify">
    Pada perbandingan di atas merupakan perbandingan Location Category dan Ad Type. Dapat dilihat bahwa perbandingan antara Location Category 0 (Jakarta) dan 1 (Luar Jakarta) mengenai tipe pekerjaan standard terdapat perbedaan. Pada Location Category di Jakarta, jumlah lamaran pekerjaan bertipe standar sekitar 7.000 lamaran pekerjaan. sedangkan untuk lokasi di luar jakarta berada di angka lebih dari 8.000 lamaran pekerjaan. Untuk tipe pekerjaan standout, lokasi di jakarta lebih banyak dibanding pada luar jakarta. Di Jakarta terdapat 5.000 lamaran pekerjaan untuk tipe standout, sedangkan di luar jakarta berada di bawah 5.000 lamaran pekerjaan. Untuk tipe pekerjaan terakhir yaitu auto_increment kedua lokasi memiliki jumlah yang sedikit yaitu di bawah 1.000 lamaran pekerjaan, namun dapat dilihat bahwa untuk lokasi di Jakarta lamaran pekerjaan bertipe auto_increment lebih tinggi dibandingkan luar jakarta.
    </div>
    """,
    unsafe_allow_html=True
)

if selected_option == 'Composition':
    st.markdown("<h1 style='text-align: center;'>COMPOSITION</h1>", unsafe_allow_html=True)

    selected_comparison = st.selectbox('Pilih Data:', ['adType', 'Salary Currency', 'Salary Period', 'Location Category'])
    #    Create a figure and axis object
    if selected_comparison == 'adType':
        # Get the counts of each unique ad type
        ad_type_counts = df_file["adType"].value_counts()

        # Extract labels and values for the pie chart
        ad_types = list(ad_type_counts.index)
        ad_type_values = list(ad_type_counts.values)

        # Define a list of custom colors (replace with your preferred colors)
        colors = ["red", "silver", "grey"]  # Adjust the number of colors based on ad_types

        # Create the pie chart with custom colors
        fig, ax = plt.subplots()
        ax.pie(ad_type_values, labels=ad_types, autopct="%1.1f%%", colors=colors)
        ax.set_title("Distribution of Ad Types")

        # Show the plot using Streamlit
        st.pyplot(fig)
        st.caption("""
        <div style="text-align: center; margin-top: 10px;">
        0 : Standard &nbsp;&nbsp;&nbsp; 1 : Standout &nbsp;&nbsp;&nbsp; 2 : auto_increment
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
        """
        <div style="text-align: justify">
        Pada piechart di atas dapat dilihat bahwa untuk komposisi dari tabel Adtype didominasi untuk tipe Standard dengan presentase 55.8%, sedangkan presentase untuk pekerjaan dengan jenis standout sebanyak 38.4%, dan jenis pekerjaan auto_increment sebesar 5.7%.
        </div>
        """,
        unsafe_allow_html=True
    )


    if selected_comparison == 'Salary Currency':
        # Get the counts of each unique salary currency
        salary_currency_counts = df_file["salarycurrency"].value_counts()

        # Extract labels and values for the pie chart
        salary_currencies = list(salary_currency_counts.index)
        salary_currency_values = list(salary_currency_counts.values)

        # Define a list of custom colors (replace with your preferred colors)
        colors = ["purple", "green", "orange"]  # Adjust the number of colors based on salary_currencies

        # Create the pie chart with custom colors
        fig, ax = plt.subplots()
        ax.pie(salary_currency_values, labels=salary_currencies, autopct="%1.1f%%", colors=colors)
        ax.set_title("Distribution of Salary Currency")

        # Show the plot using Streamlit
        st.pyplot(fig)
        st.caption("""
        <div style="text-align: center; margin-top: 10px;">
        0 : IDR
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
        """
        <div style="text-align: justify">
        Pada Pie Chart di atas dapat dilihat bahwa keseluruhan data merupakan data yang sama yaitu berisi 0 yang berarti memiliki record data berupa IDR.
        """,
        unsafe_allow_html=True
    )


    if selected_comparison == 'Salary Period':
        # Get the counts of each unique salary currency
        salary_period_counts = df_file["salaryPeriod"].value_counts()

        # Extract labels and values for the pie chart
        salary_period = list(salary_period_counts.index)
        salary_period_values = list(salary_period_counts.values)

        # Define a list of custom colors (replace with your preferred colors)
        colors = ["pink", "green", "orange"]  # Adjust the number of colors based on salary_currencies

        # Create the pie chart with custom colors
        fig, ax = plt.subplots()
        ax.pie(salary_period_values, labels=salary_period, autopct="%1.1f%%", colors=colors)
        ax.set_title("Distribution of Salary Currency")

        # Show the plot using Streamlit
        st.pyplot(fig)
        st.caption("""
        <div style="text-align: center; margin-top: 10px;">
        0 : Monthly
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
        """
        <div style="text-align: justify">
        Pada Pie Chart di atas dapat dilihat bahwa keseluruhan data merupakan data yang sama yaitu berisi 0 yang berarti memiliki record data berupa Monthly.
        """,
        unsafe_allow_html=True
    )

    if selected_comparison == 'Location Category':
        # Get the counts of each unique salary currency
        location_counts = df_file["LocationCategory"].value_counts()

        # Extract labels and values for the pie chart
        locationcategory = list(location_counts.index)
        locationcategory_values = list(location_counts.values)

        # Define a list of custom colors (replace with your preferred colors)
        colors = ["pink", "green", "orange"]  # Adjust the number of colors based on salary_currencies

        # Create the pie chart with custom colors
        fig, ax = plt.subplots()
        ax.pie(locationcategory_values, labels=locationcategory, autopct="%1.1f%%", colors=colors)
        ax.set_title("Distribution of Salary Currency")

        # Show the plot using Streamlit
        st.pyplot(fig)
        st.caption("""
        <div style="text-align: center; margin-top: 10px;">
        0 : Jakarta &nbsp;&nbsp;&nbsp; 1 : Luar Jakarta 
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
        """
        <div style="text-align: justify">
        Pada piechart di atas dapat dilihat bahwa untuk komposisi dari tabel Location Category terbagi menjadi dua yaitu Jakarta dan Luar Jakarta. Dari Pie Chart di atas dapat dilihat bahwa presentase perbedaan antara lokasi jakarta dan luar jakarta memiliki selisih yang tipis yaitu 50.1% untuk pekerjaan yang berlokasi di Jakarta dan 49.9% untuk pekerjaan berlokasi di Luar Jakarta.
        </div>
        """,
        unsafe_allow_html=True
    )

if selected_option == 'Relationship':
    st.markdown("<h1 style='text-align: center;'>RELATIONSHIP</h1>", unsafe_allow_html=True)
    numeric_cols = df_file.select_dtypes(include=['int', 'float'])
    correlation_matrix = numeric_cols.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap of Column Correlations')
    plt.show()
    st.pyplot(plt)
    st.markdown(
        """
        <div style="text-align: justify">
        Nilai dalam matriks ini merepresentasikan koefisien korelasi antara variabel-variabel tersebut. Koefisien korelasi bernilai 1 menunjukkan korelasi positif sempurna, artinya kedua variabel berbanding lurus. Nilai -1 menunjukkan korelasi negatif sempurna, artinya kedua variabel berbanding terbalik. Nilai 0 menunjukkan tidak ada korelasi antar variabel.

        1. Jenis Iklan (adtype) dan Jenis Iklan (adtype): Korelasi antara Jenis Iklan dengan dirinya sendiri selalu bernilai 1, menunjukkan korelasi positif sempurna.
        2. Mata Uang Gaji (salary currency) dan Jenis Iklan (adtype): Korelasinya berwarna putih, artinya data untuk hubungan ini tidak tersedia atau tidak signifikan.
        3. Gaji Minimum (salarymin) dan Jenis Iklan (adtype): Korelasinya -0.01, menunjukkan korelasi negatif yang sangat lemah. Ini menunjukkan sedikit kecenderungan iklan dengan gaji lebih rendah terkait dengan jenis iklan tertentu.
        4. Jenis Iklan (adtype) dan Gaji Maksimum (salarymax): Korelasinya -0.00, mendekati 0, dan menunjukkan tidak ada korelasi yang signifikan.  Artinya, tidak ada hubungan jelas antara jenis iklan dan gaji maksimum yang ditawarkan.
        5. Periode Gaji (salary period) dan Jenis Iklan (adtype): Korelasinya berwarna putih, artinya data untuk hubungan ini tidak tersedia atau tidak signifikan.
        6. Kategori Lokasi (locationcategory) dan Jenis Iklan (adtype): Korelasinya -0.08, menunjukkan korelasi negatif yang sangat lemah. Ini menunjukkan sedikit kecenderungan jenis iklan tertentu terkait dengan lokasi dengan rata-rata gaji lebih rendah.

        Korelasi antar variabel lainnya:

        1. Gaji Minimum (salarymin) dan Gaji Minimum (salarymin): Korelasi selalu bernilai 1, menunjukkan korelasi positif sempurna (sama seperti Jenis Iklan).
        2. Gaji Minimum (salarymin) dan Gaji Maksimum (salarymax): Korelasinya 0.99, menunjukkan korelasi positif yang sangat kuat. Ini berarti ada hubungan yang erat antara gaji minimum dan gaji maksimum yang ditawarkan.
        3. Gaji Minimum (salarymin) dan Kategori Lokasi (locationcategory): Korelasinya 0.07, menunjukkan korelasi positif yang sangat lemah. Ini menunjukkan sedikit kecenderungan gaji minimum lebih tinggi terkait dengan lokasi tertentu.

        Korelasi yang tersisa memiliki interpretasi serupa (korelasi lemah positif atau negatif) dan menunjukkan hubungan yang tidak terlalu kuat antara variabel-variabel tersebut.
        </div>
        """,
        unsafe_allow_html=True
    )

if selected_option == 'Clustering':
    st.subheader('Clustering Analysis based on Selected Features')
    st.write("For clustering analysis, we'll focus on the selected features.")

    # Selecting features for clustering
    selected_features = ['adType', 'salaryMin', 'salaryMax', 'LocationCategory']
    clustering_data = df_file[selected_features]

    # Handle NaN values by replacing them with mean
    imputer = SimpleImputer(strategy='mean')
    clustering_data_imputed = pd.DataFrame(imputer.fit_transform(clustering_data), columns=clustering_data.columns)

    # Perform one-hot encoding for categorical variables
    clustering_data_encoded = pd.get_dummies(clustering_data_imputed)

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data_encoded)

    # Selecting number of clusters with slider
    num_clusters = st.slider("Select number of clusters (2-8):", min_value=2, max_value=8, value=4, step=1)


    # Load the pre-trained models
    with open('kmeans.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('hierarchical.pkl', 'rb') as f:
        hierarchical = pickle.load(f)

    # Selecting features for clustering
    selected_features = ['adType', 'salaryMin', 'salaryMax', 'LocationCategory']
    clustering_data = df_file[selected_features]

    # Handle NaN values by replacing them with mean
    imputer = SimpleImputer(strategy='mean')
    clustering_data_imputed = pd.DataFrame(imputer.fit_transform(clustering_data), columns=clustering_data.columns)

    # Perform one-hot encoding for categorical variables
    clustering_data_encoded = pd.get_dummies(clustering_data_imputed)

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data_encoded)

    # Fit KMeans model
    kmeans.fit(scaled_data)

    # Get cluster labels from KMeans model
    kmeans_cluster_labels = kmeans.predict(scaled_data)
    hierarchical_cluster_labels = hierarchical.fit_predict(scaled_data)

    # Visualizing the clusters
    plt.figure(figsize=(16, 6))

    # Plot KMeans clustering
    plt.subplot(1, 2, 1)
    plt.scatter(clustering_data_encoded['salaryMin'], clustering_data_encoded['salaryMax'], c=kmeans_cluster_labels, cmap='viridis', s=50)
    plt.title(f'KMeans Clustering (Number of Clusters: {kmeans.n_clusters})')
    plt.xlabel('salaryMin')
    plt.ylabel('salaryMax')
    plt.grid(True)

    # Plot Hierarchical clustering
    plt.subplot(1, 2, 2)
    plt.scatter(clustering_data_encoded['salaryMin'], clustering_data_encoded['salaryMax'], c=hierarchical_cluster_labels, cmap='viridis', s=50)
    plt.title(f'Hierarchical Clustering (Number of Clusters: {hierarchical.n_clusters})')
    plt.xlabel('salaryMin')
    plt.ylabel('salaryMax')
    plt.grid(True)

    st.pyplot(plt)

    # Interpretation of clusters
    st.write(f"*Number of Clusters (KMeans): {kmeans.n_clusters}*")
    st.write(f"*Number of Clusters (Hierarchical): {hierarchical.n_clusters}*")
    st.markdown(
        """
        <div style="text-align: justify">
        Diagram di atas menampilkan analisis clustering berdasarkan fitur-fitur yang telah dipilih sebelumnya. Pertama-tama, terdapat dua scatter plots yang menggambarkan hasil clustering dari dua metode, yaitu KMeans dan Hierarchical. Scatter plot pertama menampilkan hasil clustering menggunakan KMeans dengan warna titik-titik yang mewakili label kluster berdasarkan 'salaryMin' dan 'salaryMax'. Sementara scatter plot kedua menunjukkan hasil clustering menggunakan metode Hierarchical dengan konfigurasi yang serupa. informasi tentang jumlah kluster yang terbentuk dari kedua metode clustering juga disajikan dalam output. Hal ini memberikan gambaran tentang kompleksitas struktur data dan variasi kluster yang mungkin terjadi.
        </div>
        """,
        unsafe_allow_html=True
    )
