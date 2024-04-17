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
df= pd.read_csv(url)
df_file = df.head(2700)

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
Analisis faktor-faktor yang memengaruhi ketertarikan pelamar terhadap citra perusahaan di Indonesia pada 13 Desember 2021 memiliki dasar yang kuat dalam kebutuhan perusahaan untuk memahami dinamika pasar kerja. Di tengah persaingan yang semakin ketat, citra perusahaan menjadi kunci utama dalam menarik bakat-bakat terbaik. Faktor-faktor seperti jenis pekerjaan yang ditawarkan, besaran gaji, periode pembayaran gaji, dan lokasi pekerjaan di Jakarta atau di luar Jakarta menjadi pertimbangan penting bagi calon karyawan dalam menilai ketertarikan terhadap suatu perusahaan. Dengan memahami preferensi dan kecenderungan pelamar terkait faktor-faktor ini, perusahaan dapat mengoptimalkan strategi mereka untuk membangun citra yang menarik bagi calon karyawan, sehingga meningkatkan kemungkinan untuk menarik bakat-bakat yang berkualitas dan sesuai dengan kebutuhan perusahaan.
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
            <strong>Interpretasi:</strong><br>
            Data barplot menunjukkan bahwa tipe pekerjaan standar (data 0) memiliki lebih dari 1000 record data, tipe pekerjaan standout (data 1) hampir mencapai 1000 record data, dan tipe pekerjaan auto_increment (data 2) memiliki lebih dari 600 record data. Hal ini mengindikasikan bahwa tipe pekerjaan standar memiliki representasi yang paling besar dalam dataset, diikuti oleh tipe pekerjaan standout, dan tipe pekerjaan auto_increment.<br><br>

            <strong>Insight:</strong><br>
            - Tipe pekerjaan standar mungkin merupakan tipe pekerjaan yang paling umum atau banyak dijumpai dalam dataset, mengingat representasinya yang paling besar.
            - Meskipun tipe pekerjaan standout memiliki jumlah record data yang cukup besar, namun masih sedikit dibandingkan dengan tipe pekerjaan standar, menunjukkan bahwa tipe pekerjaan standout mungkin lebih spesifik atau jarang dijumpai.
            - Tipe pekerjaan auto_increment memiliki jumlah record data yang signifikan, namun jumlahnya masih lebih sedikit dibandingkan dengan tipe pekerjaan standar dan standout, menunjukkan bahwa tipe pekerjaan ini mungkin juga spesifik atau jarang dijumpai seperti tipe pekerjaan standout.<br><br>

            <strong>Action Insight:</strong><br>
            - Lakukan analisis lebih lanjut terhadap data-data yang terkait dengan tipe pekerjaan standar, standout, dan auto_increment, seperti analisis karakteristik pekerjaan dan distribusi data pekerjaan di dalam dataset.
            - Perhatikan apakah ada pola atau tren yang menarik dalam data pekerjaan, seperti perbedaan dalam rata-rata gaji atau tingkat kepuasan kerja antara tipe pekerjaan yang berbeda.
            - Jika dataset ini digunakan untuk analisis lebih lanjut atau pembuatan model prediktif, pastikan untuk memperhitungkan perbedaan dalam representasi tipe pekerjaan terhadap analisis yang akan dilakukan.
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
            <strong>Interpretasi:</strong><br>
            Jumlah record data yang lebih dari 2500 menunjukkan bahwa data gaji (salary) dengan mata uang IDR (Rupiah Indonesia) memiliki representasi yang signifikan dalam dataset. Hal ini mengindikasikan bahwa mayoritas data gaji yang terdapat dalam dataset menggunakan mata uang IDR.<br><br>

            <strong>Insight:</strong><br>
            Mata uang IDR mungkin merupakan mata uang yang paling umum digunakan dalam konteks dataset ini, terutama untuk menggambarkan data gaji. Diperlukan pemahaman lebih lanjut tentang distribusi dan karakteristik data gaji dengan mata uang IDR untuk mengidentifikasi tren atau pola yang mungkin terjadi dalam gaji-gaji ini.<br><br>

            <strong>Action Insight:</strong><br>
            - Lakukan analisis lebih lanjut terhadap data gaji dengan mata uang IDR, seperti analisis distribusi gaji, median, dan rata-rata gaji untuk memahami gambaran yang lebih lengkap.<br>
            - Periksa apakah ada kesenjangan atau perbedaan signifikan dalam gaji dengan mata uang IDR dibandingkan dengan mata uang lainnya, jika ada, perlu diperhatikan penyebabnya.<br>
            - Jika dataset ini digunakan untuk analisis yang lebih mendalam atau pembuatan model prediktif, pastikan untuk memperhitungkan konversi mata uang dengan tepat dan menyeluruh agar hasil analisis atau prediksi lebih akurat.<br>
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
            <strong>Interpretasi:</strong><br>
            Jumlah record data yang lebih dari 2500 menunjukkan bahwa data gaji (salary) dengan periode bulanan (monthly) memiliki representasi yang signifikan dalam dataset. Hal ini menunjukkan bahwa mayoritas data gaji dalam dataset diukur secara bulanan.<br><br>

            <strong>Insight:</strong><br>
            Data gaji yang diukur secara bulanan mungkin memiliki pola atau tren yang berbeda dengan data gaji yang diukur dengan periode lainnya, seperti tahunan atau harian. Diperlukan analisis lebih lanjut untuk memahami karakteristik dan distribusi data gaji bulanan ini.<br><br>

            <strong>Action Insight:</strong><br>
            - Lakukan analisis statistik deskriptif terhadap data gaji bulanan, seperti rata-rata, median, dan distribusi gaji bulanan untuk mendapatkan pemahaman yang lebih mendalam.<br>
            - Identifikasi apakah ada tren atau pola tertentu dalam data gaji bulanan, seperti kenaikan atau penurunan secara periodik, dan analisis faktor-faktor yang mungkin memengaruhi pola tersebut.<br>
            - Jika dataset ini digunakan untuk pembuatan model prediktif, pertimbangkan untuk memperhitungkan efek dari data gaji bulanan terhadap prediksi atau analisis yang akan dilakukan.<br>
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
            <strong>Interpretasi:</strong><br>
            Data barplot menunjukkan bahwa kategori lokasi Jakarta (location category Jakarta) memiliki lebih dari 1400 record data, sedangkan kategori lokasi luar Jakarta (location category luar Jakarta) memiliki lebih dari 1000 record data. Hal ini menunjukkan bahwa data yang berkaitan dengan lokasi di Jakarta memiliki representasi yang lebih banyak dalam dataset dibandingkan dengan lokasi di luar Jakarta.<br><br>

            <strong>Insight:</strong><br>
            - Kategori lokasi Jakarta mungkin memiliki keterkaitan yang lebih kuat atau lebih relevan dengan subjek atau variabel yang sedang diteliti dalam dataset ini, sehingga data yang terkait dengan lokasi Jakarta lebih banyak terwakili.<br>
            - Data lokasi Jakarta dan luar Jakarta dapat memberikan informasi yang berbeda dalam analisis atau pemodelan, tergantung pada tujuan analisis atau prediksi yang dilakukan.<br><br>

            <strong>Action Insight:</strong><br>
            - Lakukan analisis lebih lanjut terhadap data-data yang terkait dengan lokasi Jakarta dan luar Jakarta, seperti analisis perbandingan antara kedua lokasi tersebut terhadap variabel lain dalam dataset.<br>
            - Perhatikan apakah terdapat pola atau tren yang menarik atau informasi penting yang dapat diambil dari data lokasi Jakarta dan luar Jakarta, seperti perbedaan perilaku atau kecenderungan antara kedua lokasi tersebut.<br>
            - Jika dataset ini digunakan untuk tujuan prediktif, pastikan untuk memperhitungkan efek dari data lokasi Jakarta dan luar Jakarta terhadap model prediktif yang akan dibuat.<br>
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
        <strong>Interpretasi:</strong><br><br>
        Data barplot perbandingan menunjukkan bahwa:<br>
        - Lokasi kategori Jakarta memiliki hampir 600 pekerjaan dengan tipe data standar, di atas 500 pekerjaan dengan tipe pekerjaan standout, dan hampir 400 pekerjaan dengan tipe pekerjaan auto_increment.<br>
        - Lokasi kategori luar Jakarta memiliki di atas 500 pekerjaan dengan tipe data standar, hampir 500 pekerjaan dengan tipe pekerjaan standout, dan di atas 200 pekerjaan dengan tipe pekerjaan auto_increment.<br><br>

        <strong>Insight:</strong><br>
        - Jumlah pekerjaan berdasarkan lokasi kategori Jakarta dan luar Jakarta memberikan gambaran tentang distribusi pekerjaan di dalam dataset.
        - Lokasi kategori Jakarta memiliki jumlah pekerjaan yang signifikan dengan tipe data standar dan standout, menunjukkan fokus yang lebih besar pada jenis pekerjaan ini di Jakarta.
        - Lokasi kategori luar Jakarta juga memiliki jumlah pekerjaan yang signifikan dengan tipe data standar dan standout, namun lebih sedikit dibandingkan dengan Jakarta, menunjukkan perbedaan distribusi pekerjaan berdasarkan lokasi.<br><br>

        <strong>Action Insight:</strong><br>
        - Lakukan analisis lebih lanjut terhadap data-data pekerjaan berdasarkan lokasi kategori Jakarta dan luar Jakarta, seperti analisis karakteristik pekerjaan dan distribusi data pekerjaan di dalam dataset.
        - Identifikasi faktor-faktor yang memengaruhi jumlah pekerjaan pada masing-masing lokasi, seperti kebutuhan pasar atau kebijakan perusahaan terkait penempatan pekerjaan.
        - Jika dataset ini digunakan untuk analisis lebih lanjut atau pembuatan model prediktif, pastikan untuk memperhitungkan perbedaan dalam distribusi pekerjaan berdasarkan lokasi terhadap analisis yang akan dilakukan.
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
            <strong>Interpretasi:</strong><br>
            Data barplot komposisi menunjukkan bahwa:
            - Tipe pekerjaan standar (data 0) menyumbang sekitar 40.7% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan standout (data 1) menyumbang sekitar 36.6% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan auto_increment (data 2) menyumbang sekitar 22.6% dari keseluruhan pekerjaan dalam dataset.<br><br>

            <strong>Insight:</strong><br>
            - Komposisi tipe pekerjaan dalam dataset menunjukkan proporsi masing-masing tipe pekerjaan terhadap total pekerjaan.
            - Tipe pekerjaan standar dan standout merupakan tipe pekerjaan yang dominan dalam dataset, dengan masing-masing menyumbang lebih dari 30% dari total pekerjaan.
            - Tipe pekerjaan auto_increment, meskipun memiliki proporsi yang lebih kecil, tetap merupakan bagian yang signifikan dalam distribusi tipe pekerjaan.<br><br>

            <strong>Action Insight:</strong><br>
            - Lakukan analisis lebih lanjut terhadap karakteristik pekerjaan berdasarkan tipe, seperti analisis kompensasi, tingkat kepuasan kerja, atau prospek karir untuk setiap tipe pekerjaan.
            - Identifikasi apakah terdapat perbedaan signifikan dalam kinerja atau karakteristik pekerjaan antara tipe pekerjaan standar, standout, dan auto_increment.
            - Jika dataset ini digunakan untuk pengambilan keputusan atau pembuatan model prediktif, pastikan untuk mempertimbangkan proporsi tipe pekerjaan ini dalam analisis atau prediksi yang akan dilakukan.
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
            <strong>Action Insight:</strong><br>
            Karena proporsi data 0 (salary currency IDR) adalah 100%, tidak ada variasi atau komposisi lain dalam data tersebut. Namun, Anda dapat melakukan analisis lebih lanjut terhadap data gaji dalam mata uang IDR untuk mendapatkan wawasan yang lebih mendalam. Berikut beberapa saran tindakan yang dapat dilakukan:<br>
            - Lakukan analisis statistik deskriptif untuk data gaji dalam mata uang IDR, seperti rata-rata, median, dan distribusi gaji.
            - Identifikasi pola atau tren dalam gaji berdasarkan level pekerjaan, industri, atau lokasi.
            - Lakukan pembandingan antara gaji dalam mata uang IDR dengan mata uang lain jika data tersebut tersedia.
            - Jika tujuannya adalah untuk membuat model prediktif, pastikan untuk memperhitungkan faktor-faktor yang memengaruhi gaji dalam mata uang IDR, seperti pengalaman kerja, tingkat pendidikan, atau spesialisasi pekerjaan.
            </div>
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
            <strong>Action Insight:</strong><br>
            Karena proporsi data 0 (salary period Monthly) adalah 100%, tidak ada variasi atau komposisi lain dalam data tersebut. Namun, Anda dapat melakukan analisis lebih lanjut terhadap data gaji yang disajikan secara bulanan untuk mendapatkan wawasan yang lebih mendalam. Berikut beberapa saran tindakan yang dapat dilakukan:<br>
            - Lakukan analisis statistik deskriptif untuk data gaji yang disajikan secara bulanan, seperti rata-rata, median, dan distribusi gaji.
            - Identifikasi pola atau tren dalam gaji berdasarkan level pekerjaan, industri, atau lokasi.
            - Jika dataset mencakup informasi waktu, lakukan analisis tren seiring waktu untuk memahami perubahan gaji bulanan dari waktu ke waktu.
            - Jika tujuannya adalah untuk membuat model prediktif, pastikan untuk mempertimbangkan faktor-faktor yang memengaruhi gaji bulanan, seperti pengalaman kerja, tingkat pendidikan, atau lokasi kerja.
            </div>
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
            <strong>Interpretasi:</strong><br>
            Data barplot komposisi menunjukkan bahwa:<br>
            - Lokasi kategori Jakarta (data 0) menyumbang sekitar 56.3% dari total data dalam dataset.<br>
            - Lokasi kategori luar Jakarta (data 1) menyumbang sekitar 43.7% dari total data dalam dataset.<br><br>

            <strong>Insight:</strong><br>
            - Komposisi berdasarkan lokasi kategori Jakarta dan luar Jakarta memberikan gambaran proporsi masing-masing lokasi dalam dataset.
            - Jakarta merupakan lokasi yang dominan dalam dataset, menyumbang lebih dari setengah dari total data.
            - Lokasi luar Jakarta, meskipun memiliki proporsi yang lebih kecil, tetap merupakan bagian yang signifikan dalam distribusi lokasi.<br><br>

            <strong>Action Insight:</strong><br>
            - Lakukan analisis lebih lanjut terhadap data-data berdasarkan lokasi kategori Jakarta dan luar Jakarta, seperti analisis karakteristik responden atau variabel lainnya yang berkaitan dengan lokasi.
            - Identifikasi apakah terdapat perbedaan signifikan dalam pola atau tren data antara lokasi kategori Jakarta dan luar Jakarta.
            - Jika dataset ini digunakan untuk pengambilan keputusan atau pembuatan model prediktif, pastikan untuk mempertimbangkan proporsi lokasi ini dalam analisis atau prediksi yang akan dilakukan.
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
        4. Jenis Iklan (adtype) dan Gaji Maksimum (salarymax): Korelasinya -0.01, mendekati 0, dan menunjukkan tidak ada korelasi yang signifikan.  Artinya, tidak ada hubungan jelas antara jenis iklan dan gaji maksimum yang ditawarkan.
        5. Periode Gaji (salary period) dan Jenis Iklan (adtype): Korelasinya berwarna putih, artinya data untuk hubungan ini tidak tersedia atau tidak signifikan.
        6. Kategori Lokasi (locationcategory) dan Jenis Iklan (adtype): Korelasinya -0.07, menunjukkan korelasi negatif yang sangat lemah. Ini menunjukkan sedikit kecenderungan jenis iklan tertentu terkait dengan lokasi dengan rata-rata gaji lebih rendah.

        Korelasi antar variabel lainnya:

        1. Gaji Minimum (salarymin) dan Gaji Minimum (salarymin): Korelasi selalu bernilai 1, menunjukkan korelasi positif sempurna (sama seperti Jenis Iklan).
        2. Gaji Minimum (salarymin) dan Gaji Maksimum (salarymax): Korelasinya 0.96, menunjukkan korelasi positif yang sangat kuat. Ini berarti ada hubungan yang erat antara gaji minimum dan gaji maksimum yang ditawarkan.
        3. Gaji Minimum (salarymin) dan Kategori Lokasi (locationcategory): Korelasinya 0.06, menunjukkan korelasi positif yang sangat lemah. Ini menunjukkan sedikit kecenderungan gaji minimum lebih tinggi terkait dengan lokasi tertentu.

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
