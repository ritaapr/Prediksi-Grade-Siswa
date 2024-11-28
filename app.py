import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# Memuat model dan scaler
loaded_model = pickle.load(open('Student_Grades_Prediction.pkl', 'rb'))
load_scaler = pickle.load(open('Standard_Scalar.pkl', 'rb'))

# Memuat dataset
df = pd.read_csv('student-por.csv', sep=";")  # Sesuaikan nama file dataset Anda

# Fungsi untuk prediksi
def student_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(load_scaler.transform(input_data_reshaped))
    if prediction[0] <= 2:
        return 'Grade Rendah: ', prediction[0]
    elif prediction[0] <= 6:
        return 'Grade Cukup: ', prediction[0]
    else:
        return 'Grade Tinggi: ', prediction[0]

# Fungsi untuk halaman Deskripsi
def show_deskripsi():
    st.title("Deskripsi")
    st.write("""
    Selamat datang di aplikasi prediksi kinerja siswa berbasis web.

Aplikasi ini menggunakan teknologi Machine Learning untuk memberikan prediksi yang akurat terkait performa akademik siswa berdasarkan berbagai faktor yang mempengaruhi hasil belajar mereka. Dengan memasukkan data seperti nilai mata pelajaran sebelumnya, absensi, kegiatan ekstrakurikuler, dukungan les privat, dan waktu belajar mingguan, pengguna dapat dengan mudah memprediksi kemungkinan nilai akhir yang akan diperoleh siswa pada akhir semester.
    """)
    st.write("Dataset: https://archive.ics.uci.edu/dataset/320/student+performance")

# Fungsi untuk halaman Dataset
def show_dataset():
    st.title("Dataset")
    st.dataframe(df)
    st.write("Informasi dataset digunakan untuk melatih model prediksi nilai siswa.")
    st.markdown("""
    ( 1 ) **G1**:
       - Nilai ujian pertama siswa (skor antara 0-20).
       - Mencerminkan performa awal siswa dalam pelajaran.

    ( 2 ) **G2**:
       - Nilai ujian kedua siswa (skor antara 0-20).
       - Memberikan gambaran lebih lanjut tentang pemahaman siswa setelah beberapa waktu belajar.

    ( 3 ) **Absences**:
       - Jumlah hari absen siswa selama periode tertentu.
       - Kehadiran yang rendah dapat mempengaruhi pemahaman pelajaran.

    ( 4 ) **Activities**:
       - Partisipasi dalam kegiatan ekstrakurikuler (Yes/No).
       - Keterlibatan dalam kegiatan di luar sekolah dapat memberikan pengalaman tambahan yang berharga.

    ( 5 ) **Paid**:
       - Menunjukkan apakah siswa mengikuti les privat (Yes/No).
       - Les privat biasanya membantu siswa lebih memahami materi pelajaran.

    ( 6 ) **Failures**:
       - Jumlah kegagalan siswa pada ujian sebelumnya.
       - Kegagalan yang lebih sering mungkin menunjukkan kebutuhan untuk bantuan tambahan.

    ( 7 ) **StudyTime**:
       - Jumlah jam belajar siswa di luar sekolah setiap minggu (skor antara 1-4).
       - Waktu belajar yang lebih banyak biasanya berhubungan dengan performa akademis yang lebih baik.
    """)

# Fungsi untuk halaman Grafik
def show_grafik():
    st.title("Visualisasi")
    st.write("Visualisasi data untuk membantu memahami distribusi nilai dan karakteristik siswa.")
    
    # Loop untuk membuat distplot untuk setiap kolom numerik
    for col in df.select_dtypes(include=['number']).columns:
        st.subheader(f"Distribusi {col}")  # Subjudul untuk tiap grafik
        plt.figure(figsize=(7, 4))
        sns.distplot(df[col], kde=True, hist=True, bins=20)
        plt.title(f"Distribusi {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        
        # Tampilkan plot di Streamlit
        st.pyplot(plt)
        plt.clf()

# Fungsi untuk halaman Prediksi
def show_prediksi():
    st.title("Prediksi Nilai Siswa")
    st.write("Masukkan nilai untuk variabel berikut untuk memprediksi grade akhir:")
    
    G1 = st.number_input('Masukkan Nilai G1:', min_value=0, max_value=20, value=10)
    G2 = st.number_input('Masukkan Nilai G2:', min_value=0, max_value=20, value=10)
    Absences = st.number_input('Masukkan Jumlah Absensi:', min_value=0, value=5)
    Activities = st.selectbox('Ikut Kegiatan Ekstrakurikuler?', ['Yes', 'No'])
    Paid = st.selectbox('Ikut Les Privat?', ['Yes', 'No'])
    Failures = st.number_input('Jumlah Kegagalan Sebelumnya:', min_value=0, max_value=5, value=0)
    StudyTime = st.slider('Jam Belajar Mingguan:', min_value=1, max_value=4, value=2)
    
    # Konversi categorical input
    Activities = 1 if Activities == 'Yes' else 0
    Paid = 1 if Paid == 'Yes' else 0
    
    if st.button('Prediksi Grade'):
        result = student_prediction([G1, G2, Absences, Activities, Paid, Failures, StudyTime])
        st.success(result)

# Menu sidebar
# Menampilkan judul dan nama di sidebar
st.sidebar.title("Sistem Prediksi Grade Berdasarkan Nilai Siswa")
st.sidebar.write("by Rita Aprilia")

# Menu sidebar
add_selectbox = st.sidebar.selectbox(
    "Pilih Halaman",  # Teks label untuk dropdown
    ("Deskripsi", "Dataset", "Visualisasi Data", "Prediksi")  # Pilihan dropdown
)

# Menampilkan halaman berdasarkan pilihan
if add_selectbox == "Deskripsi":
    show_deskripsi()
elif add_selectbox == "Dataset":
    show_dataset()
elif add_selectbox == "Visualisasi Data":
    show_grafik()
elif add_selectbox == "Prediksi":
    show_prediksi()

