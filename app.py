import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import silhouette_score, accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import plotly.graph_objects as go
import io
import os
import seaborn as sns

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="DataScope | Veri Analiz Platformu",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/datascope',
        'Report a bug': 'https://github.com/yourusername/datascope/issues',
        'About': '''
        ### DataScope - Veri Analiz Platformu
        Verilerinizi kolayca analiz edin ve görselleştirin.
        
        Version: 1.0.0
        '''
    }
)

# URL'yi özelleştir
st.markdown("""
    <script>
        window.history.replaceState({}, '', '/datascope');
    </script>
""", unsafe_allow_html=True)

# CSS Stilleri
def load_css():
    if os.path.exists("static/style.css"):
        with open("static/style.css") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


load_css()

# Ana başlık ve açıklama
st.markdown('<h1>📊 DataScope</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2em; color: #e6e6e6;">CSV verinizi yükleyin ve kümeleri görselleştirin</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h2 style="color: #fff;">Veri Yükleme</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Veri Dosyası Seçin (CSV, XLSX)", type=["csv", "xlsx"])

    st.markdown('<h2 style="color: #fff;">Analiz Seçenekleri</h2>', unsafe_allow_html=True)
    analysis_type = st.radio(
        "Analiz Yöntemi",
        ["Kümeleme", "Sınıflandırma"]
    )

    if analysis_type == "Kümeleme":
        st.markdown("""
        <div class="cluster-tips">
            <h4>Kümeleme Analizi için Öneriler:</h4>
            <ol>
                <li>Küme sayısını 2 ile 9 arasında belirleyiniz.</li>
                <li>Veri setinizdeki doğal grupları analiz ediniz.</li>
                <li>Çok fazla küme seçimi, sonuçların yorumlanabilirliğini azaltabilir.</li>
                <li>Her kümede yeterli sayıda veri noktası bulunduğundan emin olunuz.</li>
            </ol>
            <h4 style="margin-top: 1em;">Veri Seti Gereksinimleri:</h4>
            <ol>
                <li>En az 2 sayısal sütun içermelidir.</li>
                <li>Eksik veriler temizlenmiş olmalıdır.</li>
                <li>Aykırı değerler analiz edilmelidir.</li>
                <li>Özellikler uygun şekilde ölçeklendirilmelidir.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        clustering_method = st.selectbox(
            "Kümeleme Algoritması",
            ["K-Means", "DBSCAN", "Hierarchical Clustering"]
        )
        
        if clustering_method == "K-Means":
            n_clusters = st.slider("Küme Sayısı", min_value=2, max_value=9, value=3, key="kmeans_n_clusters")
        elif clustering_method == "DBSCAN":
            eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="dbscan_eps")
            min_samples = st.slider("Minimum Örnek Sayısı", min_value=2, max_value=10, value=5, key="dbscan_min_samples")
        else:  # Hierarchical Clustering
            n_clusters = st.slider("Küme Sayısı", min_value=2, max_value=9, value=3, key="hierarchical_n_clusters")
            linkage = st.selectbox("Bağlantı Yöntemi", ["ward", "complete", "average", "single"], key="hierarchical_linkage")
    else:
        st.markdown("""
        <div class="cluster-tips">
            <h4>Sınıflandırma Analizi için Öneriler:</h4>
            <ol>
                <li>Veri setinizde 'Label' isimli bir sütun bulunmalıdır.</li>
                <li>Label sütunu sınıf etiketlerini içermelidir.</li>
                <li>Dengeli bir sınıf dağılımı olmalıdır.</li>
                <li>Test seti oranını veriye göre ayarlayınız.</li>
            </ol>
            <h4 style="margin-top: 1em;">Veri Seti Gereksinimleri:</h4>
            <ol>
                <li>En az 2 sayısal özellik içermelidir.</li>
                <li>Eksik veriler doldurulmalıdır.</li>
                <li>Kategorik değişkenler dönüştürülmelidir.</li>
                <li>Özellik ölçeklendirmesi yapılmalıdır.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        model_type = st.selectbox(
            "Model Seçimi",
            ["Decision Tree", "Random Forest", "SVM", "XGBoost", "LightGBM"],
            key="model_type_select"
        )
        
        test_size = st.slider("Test Veri Seti Oranı (%)", min_value=10, max_value=40, value=20, key="test_size_slider_global")
        
        # Hiperparametre optimizasyonu seçeneği
        optimize_hyperparams = st.checkbox("Hiperparametre Optimizasyonu Yap", value=False, key="optimize_hyperparams_checkbox")

    # Geliştirici Bilgileri
    st.markdown("---")
    st.markdown("""
    <div style='background: #2a2a4a; padding: 1.5em; border-radius: 10px; border: 1px solid #3a3a5a; margin-top: 2em;'>
        <h3 style='color: #9d84ff; font-size: 1.2em; margin-bottom: 1em;'>Geliştirici</h3>
        <p style='color: #e6e6e6; margin-bottom: 0.5em; display: flex; align-items: center; gap: 8px;'>
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#9d84ff" viewBox="0 0 24 24">
                <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
            </svg>
            <a href="https://www.linkedin.com/in/faik-döner" target="_blank" style="color: #e6e6e6; text-decoration: none; border-bottom: 1px dashed #9d84ff;">Faik Döner</a>
        </p>
        <p style='color: #9d84ff; font-size: 0.9em; margin-top: 1em;'>© 2025 DataScope. Tüm hakları saklıdır.</p>
    </div>
    """, unsafe_allow_html=True)

# Ana içerik
if uploaded_file is not None:
    try:
        # Veri setini yükle
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ Veri seti başarıyla yüklendi!")

        # Veri seti bilgilerini göster
        st.write("### 📊 Veri Seti Bilgileri")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Satır Sayısı", df.shape[0])
        with col2:
            st.metric("Sütun Sayısı", df.shape[1])
        with col3:
            st.metric("Sayısal Sütunlar", len(df.select_dtypes(include=[np.number]).columns))

        # Veri önizleme
        st.write("### 👀 Veri Seti Önizleme")
        st.dataframe(df.head())

        # Özellik seçimi
        st.write("### 🎯 Analiz için Özellik Seçimi")
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_columns) < 2:
            st.error("⚠️ Veri setinde en az 2 sayısal sütun bulunmalıdır!")
        else:
            selected_features = st.multiselect(
                "Analiz edilecek özellikleri seçin:",
                numeric_columns,
                default=numeric_columns[:2] if len(numeric_columns) >= 2 else numeric_columns
            )

            if len(selected_features) < 2:
                st.warning("⚠️ Lütfen en az 2 özellik seçin!")
            else:
                X = df[selected_features]

                if analysis_type == "Kümeleme":
                    # K-Means kümeleme
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    if clustering_method == "K-Means":
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        clusters = kmeans.fit_predict(X_scaled)
                    elif clustering_method == "DBSCAN":
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        clusters = dbscan.fit_predict(X_scaled)
                    else:  # Hierarchical Clustering
                        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                        clusters = agg_clustering.fit_predict(X_scaled)

                    # Sonuçları görselleştirme
                    st.write("### 📈 Kümeleme Sonuçları")

                    # 2D görselleştirme
                    if len(selected_features) == 2:
                        fig = px.scatter(
                            df,
                            x=selected_features[0],
                            y=selected_features[1],
                            color=clusters.astype(str),
                            title="Kümeleme Sonuçları",
                            color_discrete_sequence=["#7b61ff", "#9d84ff", "#b4a7ff", "#cbc3ff", "#e2deff"]
                        )
                        fig.update_layout(
                            plot_bgcolor="#2a2a4a",
                            paper_bgcolor="#2a2a4a",
                            font_color="#e6e6e6"
                        )
                        st.plotly_chart(fig)

                    # Silhouette skoru
                    silhouette_avg = silhouette_score(X_scaled, clusters)
                    st.metric("Silhouette Skoru", f"{silhouette_avg:.3f}")

                    # Küme merkezleri
                    st.write("### 📍 Küme Merkezleri")
                    if clustering_method == "K-Means":
                        centers = scaler.inverse_transform(kmeans.cluster_centers_)
                    elif clustering_method == "DBSCAN":
                        centers = scaler.inverse_transform(dbscan.components_)
                    else:  # Hierarchical Clustering
                        centers = scaler.inverse_transform(agg_clustering.children_)
                    centers_df = pd.DataFrame(centers, columns=selected_features)
                    centers_df.index = [f"Küme {i}" for i in range(n_clusters)]
                    st.dataframe(centers_df)

                else:  # Sınıflandırma
                    # Etiket sütunu seçimi
                    label_columns = df.columns.tolist()
                    selected_label = st.selectbox(
                        "Etiket Sütununu Seçin",
                        label_columns,
                        index=label_columns.index("Label") if "Label" in label_columns else 0
                    )
                    
                    if selected_label not in df.columns:
                        st.error(f"⚠️ Seçilen etiket sütunu ({selected_label}) veri setinde bulunamadı!")
                    else:
                        # Veri ön işleme seçenekleri
                        st.write("### 🔧 Veri Ön İşleme Seçenekleri")
                        preprocess_options = st.multiselect(
                            "Uygulanacak Ön İşlemler",
                            ["Eksik Verileri Doldur", "Aykırı Değerleri Temizle", "Özellik Ölçeklendirme"],
                            default=["Eksik Verileri Doldur", "Özellik Ölçeklendirme"],
                            key="preprocess_options_multiselect"
                        )

                        # Etiket sütununu işle
                        if df[selected_label].dtype in ['float64', 'int64']:
                            st.write("### 📊 Etiket Dönüşümü")
                            n_categories = st.slider("Kategori Sayısı", min_value=2, max_value=5, value=3, key="n_categories_slider_classification")
                            
                            # Kategori isimlerini al
                            category_names = []
                            for i in range(n_categories):
                                name = st.text_input(f"Kategori {i+1} İsmi", 
                                                   value=f"Kategori {i+1}" if i == 0 else f"Kategori {i+1}",
                                                   key=f"category_name_input_{i}")
                                category_names.append(name)
                            
                            if st.button("Etiketleri Dönüştür", key="convert_labels_button"):
                                # Sayısal değerleri kategorilere dönüştür
                                y = pd.qcut(df[selected_label], q=n_categories, labels=category_names)
                                st.success("✅ Etiketler başarıyla dönüştürüldü!")
                        else:
                            y = df[selected_label]

                        # Veri ön işleme
                        X_processed = X.copy()
                        
                        if "Eksik Verileri Doldur" in preprocess_options:
                            X_processed = X_processed.fillna(X_processed.mean())
                            
                        if "Aykırı Değerleri Temizle" in preprocess_options:
                            for col in X_processed.columns:
                                Q1 = X_processed[col].quantile(0.25)
                                Q3 = X_processed[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                X_processed[col] = X_processed[col].clip(lower_bound, upper_bound)
                            
                        if "Özellik Ölçeklendirme" in preprocess_options:
                            scaler = StandardScaler()
                            X_processed = pd.DataFrame(scaler.fit_transform(X_processed), columns=X_processed.columns)

                        # Veriyi eğitim ve test setlerine ayırma
                        test_size = st.slider("Test Veri Seti Oranı (%)", min_value=10, max_value=40, value=20, key="test_size_slider_classification")
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_processed, y, test_size=test_size/100, random_state=42, stratify=y
                        )

                        # Model seçimi ve eğitim
                        st.write("### 🤖 Model Seçimi ve Eğitim")
                        model_type = st.selectbox(
                            "Model Seçimi",
                            ["Decision Tree", "Random Forest", "SVM", "XGBoost", "LightGBM"],
                            key="model_type_select_classification"
                        )

                        # Model parametreleri
                        if model_type == "Decision Tree":
                            max_depth = st.slider("Maksimum Derinlik", min_value=1, max_value=20, value=5, key="dt_max_depth_classification")
                            min_samples_split = st.slider("Minimum Bölünme Örneği", min_value=2, max_value=20, value=2, key="dt_min_samples_classification")
                            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
                        elif model_type == "Random Forest":
                            n_estimators = st.slider("Ağaç Sayısı", min_value=10, max_value=200, value=100, key="rf_n_estimators_classification")
                            max_depth = st.slider("Maksimum Derinlik", min_value=1, max_value=20, value=5, key="rf_max_depth_classification")
                            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                        elif model_type == "SVM":
                            kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], key="svm_kernel_classification")
                            C = st.slider("C Değeri", min_value=0.1, max_value=10.0, value=1.0, key="svm_c_classification")
                            model = SVC(kernel=kernel, C=C, random_state=42)
                        elif model_type == "XGBoost":
                            n_estimators = st.slider("Ağaç Sayısı", min_value=10, max_value=200, value=100, key="xgb_n_estimators_classification")
                            learning_rate = st.slider("Öğrenme Oranı", min_value=0.01, max_value=0.3, value=0.1, key="xgb_learning_rate_classification")
                            model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                        else:  # LightGBM
                            n_estimators = st.slider("Ağaç Sayısı", min_value=10, max_value=200, value=100, key="lgb_n_estimators_classification")
                            learning_rate = st.slider("Öğrenme Oranı", min_value=0.01, max_value=0.3, value=0.1, key="lgb_learning_rate_classification")
                            model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                        # Model eğitimi
                        with st.spinner("Model eğitiliyor..."):
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

                        # Model performansı
                        st.write("### 📊 Model Performansı")
                        accuracy = accuracy_score(y_test, y_pred)
                        st.metric("Doğruluk (Accuracy)", f"{accuracy:.3f}")

                        # Detaylı sınıflandırma raporu
                        st.write("### 📑 Detaylı Sınıflandırma Raporu")
                        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df)

                        # Karışıklık matrisi
                        st.write("### 🔄 Karışıklık Matrisi")
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        plt.title("Karışıklık Matrisi")
                        ax.set_facecolor("#2a2a4a")
                        fig.patch.set_facecolor("#2a2a4a")
                        plt.xlabel("Tahmin Edilen Sınıf")
                        plt.ylabel("Gerçek Sınıf")
                        for text in ax.texts:
                            text.set_color("white")
                        ax.tick_params(colors='white')
                        plt.title("Karışıklık Matrisi", color='white')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        st.pyplot(fig)

                        # ROC eğrisi (eğer predict_proba varsa)
                        if y_pred_proba is not None:
                            st.write("### 📈 ROC Eğrisi")
                            from sklearn.preprocessing import label_binarize
                            from sklearn.metrics import roc_curve, auc
                            
                            # Çok sınıflı ROC için
                            classes = np.unique(y)
                            y_test_bin = label_binarize(y_test, classes=classes)
                            
                            fig = go.Figure()
                            for i in range(len(classes)):
                                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                                roc_auc = auc(fpr, tpr)
                                fig.add_trace(go.Scatter(
                                    x=fpr, y=tpr,
                                    name=f'ROC {classes[i]} (AUC = {roc_auc:.2f})',
                                    mode='lines'
                                ))
                            
                            fig.add_trace(go.Scatter(
                                x=[0, 1], y=[0, 1],
                                name='Random',
                                mode='lines',
                                line=dict(dash='dash')
                            ))
                            
                            fig.update_layout(
                                title='ROC Eğrisi',
                                xaxis_title='False Positive Rate',
                                yaxis_title='True Positive Rate',
                                plot_bgcolor="#2a2a4a",
                                paper_bgcolor="#2a2a4a",
                                font_color="#e6e6e6"
                            )
                            st.plotly_chart(fig)

                        # Özellik önem dereceleri
                        if hasattr(model, "feature_importances_"):
                            st.write("### 🎯 Özellik Önem Dereceleri")
                            feature_importance = pd.DataFrame({
                                'Özellik': selected_features,
                                'Önem': model.feature_importances_
                            })
                            feature_importance = feature_importance.sort_values('Önem', ascending=False)

                            fig = px.bar(
                                feature_importance,
                                x='Özellik',
                                y='Önem',
                                title='Özellik Önem Dereceleri'
                            )
                            fig.update_layout(
                                plot_bgcolor="#2a2a4a",
                                paper_bgcolor="#2a2a4a",
                                font_color="#e6e6e6"
                            )
                            st.plotly_chart(fig)

                        # Model kaydetme seçeneği
                        if st.button("Modeli Kaydet", key="save_model_button"):
                            import joblib
                            model_path = "model.joblib"
                            joblib.dump(model, model_path)
                            st.success(f"✅ Model başarıyla kaydedildi: {model_path}")
    except Exception as e:
        st.error(f"⚠️ İşlem sırasında bir hata oluştu: {e}")
else:
    st.info("📤 Lütfen bir veri seti yükleyiniz.")

    st.header("Nasıl Kullanılır?")
    st.markdown("""
    1. Sol menüden bir CSV veya Excel dosyası yükleyiniz.
    2. Analiz türünü seçiniz. (Kümeleme veya Sınıflandırma)  
    3. Analiz için özellikleri seçiniz.
    4. Sonuçları inceleyiniz ve değerlendiriniz.
    """)

    st.header("Örnek Veri Seti Oluşturun.")
    if st.button("Örnek Veri Seti Oluştur"):
        np.random.seed(42)
        n_samples = 200

        # Üç farklı grup için merkez noktalar
        centers = [
            [-3, -3],  # Grup 1
            [0, 0],    # Grup 2
            [3, 3]     # Grup 3
        ]

        # Veri noktaları ve etiketleri oluştur
        X = []
        y = []

        for i, center in enumerate(centers):
            cluster_samples = np.random.randn(n_samples // len(centers), len(center)) + center
            X.append(cluster_samples)
            y.extend([i] * (n_samples // len(centers)))

        X = np.vstack(X)

        # Veri setini DataFrame'e dönüştür
        df_example = pd.DataFrame(X, columns=["Özellik 1", "Özellik 2"])

        # Etiketleri anlamlı kategorik değerlere dönüştür
        label_mapping = {
            0: "A Grubu",
            1: "B Grubu",
            2: "C Grubu"
        }
        df_example["Label"] = [label_mapping[label] for label in y]

        # Veri setini göster
        st.write("### Örnek Veri Seti Önizleme")
        st.dataframe(df_example.head())

        # Veri setini indir
        csv = df_example.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Örnek Veri Setini İndir (CSV)",
            data=csv,
            file_name="ornek_veri_seti.csv",
            mime="text/csv"
        )

        # Veri dağılımını görselleştir
        st.write("### Veri Dağılımı")
        fig = px.scatter(
            df_example,
            x="Özellik 1",
            y="Özellik 2",
            color="Label",
            title="Örnek Veri Seti Dağılımı",
            color_discrete_sequence=["#7b61ff", "#9d84ff", "#b4a7ff"]
        )
        fig.update_layout(
            plot_bgcolor="#2a2a4a",
            paper_bgcolor="#2a2a4a",
            font_color="#e6e6e6"
        )
        st.plotly_chart(fig)
