# iris_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score, classification_report
from scipy.cluster import hierarchy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ------------- Настройка страницы -------------
st.set_page_config(
    page_title="Iris Flower Classifier Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили для matplotlib/ seaborn единообразные
sns.set_style("whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# ------------- Функции и загрузка данных -------------
@st.cache_data
def load_data():
    iris = load_iris()
    df_local = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_local['species'] = iris.target
    df_local['species_name'] = df_local['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df_local

@st.cache_data
def compute_basic_stats(df):
    numerical_cols = df.columns[:4]
    stats_df = pd.DataFrame({
        "Признак": numerical_cols,
        "Среднее": df[numerical_cols].mean().round(3).values,
        "Медиана": df[numerical_cols].median().round(3).values,
        "Ст. отклонение": df[numerical_cols].std().round(3).values
    })
    return stats_df

@st.cache_data
def train_logistic_model(X_train, y_train, max_iter=200):
    model = LogisticRegression(random_state=42, max_iter=max_iter, multi_class='auto')
    model.fit(X_train, y_train)
    return model

# Загружаем данные
df = load_data()

# ------------- Sidebar -------------
st.sidebar.title("🌸 Iris Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "📌 **Выберите раздел:**",
    ["📊 Визуализация данных", "🔍 Анализ данных", "🤖 Кластеризация", "🎯 Классификация", "📈 Метрики / Выводы"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Дашборд для анализа датасета Iris**  
Классический датасет: 150 наблюдений, 4 признака, 3 вида ирисов.
""")

# ------------- Объявление фильтра по видам (используется на многих страницах) -------------
with st.sidebar.expander("Фильтры данных"):
    species_filter = st.multiselect(
        "Фильтр по виду ириса:",
        options=df['species_name'].unique(),
        default=list(df['species_name'].unique())
    )

# Применяем фильтр на уровне всего приложения
df_filtered = df[df['species_name'].isin(species_filter)].reset_index(drop=True)

# ------------- Страница: Визуализация данных -------------
if page == "📊 Визуализация данных":
    st.title("📊 Визуализация исходных данных Iris")

    # Экспандер с описанием датасета
    with st.expander("ℹ Описание датасета Iris", expanded=True):
        st.markdown("""
        **Iris Dataset** — датасет содержит измерения длины/ширины чашелистика и лепестка (в см) для трёх видов ириса:
        *setosa*, *versicolor*, *virginica*.
        """)
        st.write("Количество записей (после фильтра):", df_filtered.shape[0])

    # Показать данные
    with st.expander("📋 Показать таблицу данных", expanded=False):
        st.dataframe(df_filtered, use_container_width=True, height=300)

    # KPI-метрики: простые карточки
    st.subheader("📈 Основные KPI")
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.metric("Всего записей", df_filtered.shape[0])
    with col_k2:
        st.metric("Признаков (числовых)", len(df_filtered.columns[:4]))
    with col_k3:
        st.metric("Видов (после фильтра)", df_filtered['species_name'].nunique())
    with col_k4:
        st.metric("Дубликатов", int(df_filtered.duplicated().sum()))

    # Дополнительные KPI: пропуски
    st.subheader("📌 Пропуски и качество данных")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        missing_by_col = df_filtered.isna().sum()
        st.dataframe(pd.DataFrame({'Колонка': missing_by_col.index, 'Пропуски': missing_by_col.values}),
                     use_container_width=True)
    with col_b:
        st.metric("Всего пропусков", int(df_filtered.isna().sum().sum()))
        st.metric("Колонок с пропусками", int((df_filtered.isna().sum() > 0).sum()))

    # KPI-таблица: средние, медианы, std
    st.subheader("📌 Базовые статистики (KPI)")
    stats_df = compute_basic_stats(df_filtered)
    st.dataframe(stats_df, use_container_width=True)

    # Кнопка скачать
    st.download_button(
        "⬇ Скачать данные (CSV)",
        df_filtered.to_csv(index=False).encode('utf-8'),
        file_name="iris_dataset_filtered.csv",
        mime="text/csv"
    )

    # Гистограмма для выбранного признака
    st.subheader("📊 Распределение признаков")
    feature = st.selectbox("Выберите признак:", df_filtered.columns[:4])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df_filtered[feature], bins=15, edgecolor='black', alpha=0.7)
    ax.set_title(f'Распределение {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Частота')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Pie chart (распределение по видам)
    st.subheader("🥧 Распределение по видам")
    species_counts = df_filtered['species_name'].value_counts()
    col1, col2 = st.columns([1, 1])
    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=90, explode=[0.03]*len(species_counts))
        ax1.set_title("Доля видов (после фильтра)")
        st.pyplot(fig1)
    with col2:
        st.dataframe(species_counts.rename("Количество").to_frame(), use_container_width=True)

    # Корреляционная матрица
    st.subheader("🔥 Корреляционная матрица")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_filtered.iloc[:, :4].corr(), annot=True, cmap='coolwarm', center=0, ax=ax2, fmt='.2f')
    ax2.set_title('Корреляция между признаками')
    st.pyplot(fig2)

    # Интерактивный Plotly scatter
    st.subheader("🧭 Интерактивный график (Plotly)")
    fig_px = px.scatter(
        df_filtered,
        x="petal length (cm)",
        y="petal width (cm)",
        color="species_name",
        size="sepal length (cm)",
        hover_data=df_filtered.columns,
        title="Интерактивный Scatter: Длина лепестка vs Ширина лепестка"
    )
    st.plotly_chart(fig_px, use_container_width=True)

# ------------- Страница: Анализ данных -------------
elif page == "🔍 Анализ данных":
    st.title("🔍 Глубокий анализ данных")

    st.subheader("📊 Pairplot (Seaborn)")
    pairplot_fig = sns.pairplot(df_filtered, hue='species_name', diag_kind='hist', palette='Set2')
    pairplot_fig.fig.suptitle('Pairplot всех признаков по видам', y=1.02)
    st.pyplot(pairplot_fig.fig)

    st.subheader("🎯 Scatter plot с выбором осей")
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Ось X:", df_filtered.columns[:4], index=0, key='x_scatter')
    with col2:
        y_axis = st.selectbox("Ось Y:", df_filtered.columns[:4], index=1, key='y_scatter')

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
    for species in df_filtered['species_name'].unique():
        subset = df_filtered[df_filtered['species_name'] == species]
        ax.scatter(subset[x_axis], subset[y_axis], label=species, alpha=0.7, s=60, color=palette.get(species, None))
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f'{x_axis} vs {y_axis}')
    ax.legend()
    st.pyplot(fig)

    st.subheader("📦 Boxplot по видам")
    box_feature = st.selectbox("Выберите признак для boxplot:", df_filtered.columns[:4], key='box_feature')
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df_filtered, x='species_name', y=box_feature, palette='pastel', ax=ax)
    ax.set_title(f'Распределение {box_feature} по видам')
    st.pyplot(fig)

# ------------- Страница: Кластеризация -------------
elif page == "🤖 Кластеризация":
    st.title("🤖 Кластеризация")

    X = df_filtered.iloc[:, :4].values

    st.subheader("1) Метод локтя для определения k")
    max_k = st.slider("Максимальное количество кластеров (для метода локтя):", 2, 10, 6)
    inertias = []
    for k in range(1, max_k + 1):
        kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_tmp.fit(X)
        inertias.append(kmeans_tmp.inertia_)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, max_k + 1), inertias, 'bo-', linewidth=2, markersize=6)
    ax.set_xlabel('k')
    ax.set_ylabel('Inertia')
    ax.set_title('Метод локтя')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.info("Оптимальное k обычно в точке перегиба (elbow) — часто 2 или 3 для Iris.")

    st.subheader("2) KMeans кластеризация и визуализация")
    selected_k = st.slider("Выберите k для KMeans:", 2, 6, 3)
    kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    df_filtered['cluster'] = cluster_labels

    col1, col2 = st.columns([1, 1])
    with col1:
        x_cluster = st.selectbox("Ось X для кластеров:", df_filtered.columns[:4], index=2, key='x_cluster')
    with col2:
        y_cluster = st.selectbox("Ось Y для кластеров:", df_filtered.columns[:4], index=3, key='y_cluster')

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(df_filtered[x_cluster], df_filtered[y_cluster], c=df_filtered['cluster'], cmap='viridis', alpha=0.8, s=70)
    centers = kmeans.cluster_centers_
    # Найдём индекс колонок для центров
    xi = list(df_filtered.columns[:4]).index(x_cluster)
    yi = list(df_filtered.columns[:4]).index(y_cluster)
    ax.scatter(centers[:, xi], centers[:, yi], c='red', s=200, marker='X', label='Центроиды')
    ax.set_xlabel(x_cluster)
    ax.set_ylabel(y_cluster)
    ax.set_title(f'KMeans (k={selected_k})')
    ax.legend()
    st.pyplot(fig)

    # Silhouette (только если k > 1 и меньше чем n_samples)
    sil_val = None
    try:
        if len(np.unique(cluster_labels)) > 1 and len(np.unique(cluster_labels)) < len(X):
            sil_val = silhouette_score(X, cluster_labels)
    except Exception:
        sil_val = None

    if sil_val is not None:
        st.metric("Silhouette score", f"{sil_val:.3f}")

    st.subheader("3) Сравнение кластеров с реальными метками")
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df_filtered, x=x_cluster, y=y_cluster, hue='species_name', palette='Set1', s=80, ax=ax2)
    ax2.set_title('Реальные виды')
    st.pyplot(fig2)

    st.subheader("4) Иерархическая кластеризация (дендрограмма)")
    if st.button("Показать дендрограмму"):
        fig, ax = plt.subplots(figsize=(12, 6))
        Z = hierarchy.linkage(X, method='ward')
        hierarchy.dendrogram(Z, ax=ax, truncate_mode='lastp', p=30)
        ax.set_title('Дендрограмма (ward)')
        st.pyplot(fig)

# ------------- Страница: Классификация -------------
elif page == "🎯 Классификация":
    st.title("🎯 Классификация видов ирисов")

    st.info("Модель логистической регрессии используется для мультиклассовой классификации (setosa / versicolor / virginica).")

    # Подготовка данных (стандартизация рекомендуема)
    X = df_filtered.iloc[:, :4].copy()
    y = df_filtered['species'].copy()

    # Стандартизация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Выбор размера теста
    test_size = st.slider("Размер тестовой выборки (%)", 10, 40, 20)
    stratify_flag = True if y.nunique() > 1 else False

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size / 100, random_state=42, stratify=y if stratify_flag else None
    )

    # Обучение модели
    model = train_logistic_model(X_train, y_train, max_iter=300)

    # Предсказания
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{acc:.3f}")
    with col2:
        st.metric("Train size", X_train.shape[0])
    with col3:
        st.metric("Test size", X_test.shape[0])

    st.subheader("Матрица ошибок (Confusion Matrix)")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['setosa', 'versicolor', 'virginica'],
                yticklabels=['setosa', 'versicolor', 'virginica'])
    ax.set_xlabel('Предсказанные метки')
    ax.set_ylabel('Истинные метки')
    st.pyplot(fig)

    st.subheader("Отчет классификации (Precision / Recall / F1)")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.subheader("Визуализация правильных/ошибочных предсказаний (по признакам лепестка)")
    # Визуализация ошибок по оригинальным (не стандартизованным) значениям
    X_test_orig = pd.DataFrame(scaler.inverse_transform(X_test), columns=X.columns)
    results_df = X_test_orig.copy()
    results_df['true_species'] = y_test.values
    results_df['predicted_species'] = y_pred
    results_df['correct'] = results_df['true_species'] == results_df['predicted_species']
    results_df['true_name'] = results_df['true_species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    results_df['pred_name'] = results_df['predicted_species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    fig, ax = plt.subplots(figsize=(10, 6))
    correct = results_df[results_df['correct']]
    wrong = results_df[~results_df['correct']]

    ax.scatter(correct['petal length (cm)'], correct['petal width (cm)'], c='green', s=80, label='Правильно', alpha=0.6)
    ax.scatter(wrong['petal length (cm)'], wrong['petal width (cm)'], c='red', s=120, marker='x', label='Ошибка', alpha=0.9)
    for idx, row in wrong.iterrows():
        ax.annotate(f"{row['true_name']}→{row['pred_name']}",
                    (row['petal length (cm)'], row['petal width (cm)']),
                    textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, color='darkred')

    ax.set_xlabel('Длина лепестка (cm)')
    ax.set_ylabel('Ширина лепестка (cm)')
    ax.set_title('Результаты классификации (зелёные = правильно, красные = ошибки)')
    ax.legend()
    st.pyplot(fig)

# ------------- Страница: Метрики и выводы -------------
elif page == "📈 Метрики / Выводы":
    st.title("📈 Метрики и ключевые выводы")

    st.subheader("Ключевые инсайты")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Виды хорошо разделяются** по признакам лепестков (особенно petal length/width).  
        - **Setosa** обычно отделяется очень чётко от остальных.  
        - Основная путаница наблюдается между **versicolor** и **virginica**.
        """)
    with col2:
        corr = df_filtered.iloc[:, :4].corr()
        st.markdown("**Корреляции (часть матрицы):**")
        st.dataframe(corr.round(3), use_container_width=True)

    st.markdown("---")
    st.subheader("Результаты моделей (сводно)")

    # Кластеризация k=3 для сводной метрики
    X_full = df_filtered.iloc[:, :4].values
    kmeans3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels3 = kmeans3.fit_predict(X_full)
    sil3 = silhouette_score(X_full, labels3)
    st.markdown(f"**Кластеризация (KMeans, k=3)** — Silhouette: `{sil3:.3f}`")

    # Классификация: тренировочный прогон на всем датасете (кросс-валидация не включена здесь,
    # но мы можем показать обученную модель на полном наборе и её важности признаков)
    st.subheader("Важность признаков (Logistic Regression)")
    # Обучаем на полном наборе (стандартизированном)
    X_all = df_filtered.iloc[:, :4]
    y_all = df_filtered['species']
    scaler_full = StandardScaler()
    X_all_scaled = scaler_full.fit_transform(X_all)
    model_full = LogisticRegression(random_state=42, max_iter=300)
    model_full.fit(X_all_scaled, y_all)
    # Для мультиклассовой логистической регрессии берём среднюю абсолютную важность по классам
    coefs = np.abs(model_full.coef_)  # shape (n_classes, n_features)
    importance_vals = coefs.mean(axis=0)
    importance_df = pd.DataFrame({
        'Признак': X_all.columns,
        'Importance': importance_vals
    }).sort_values('Importance', ascending=False)
    st.dataframe(importance_df.round(4), use_container_width=True)

    # Горизонтальный бар для важности
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(importance_df['Признак'], importance_df['Importance'])
    ax.set_xlabel('Средняя |коэффициент|')
    ax.set_title('Важность признаков (Logistic Regression)')
    st.pyplot(fig)

    st.markdown("---")
    st.success("""
    **Рекомендации:**
    1. Для классификации достаточны признаки лепестков (petal length & petal width).  
    2. KMeans с k=3 соответствует биологической интуиции и даёт хорошую сегрегацию.  
    3. Логистическая регрессия показывает высокую точность на Iris; для более надёжной оценки
       стоит добавить кросс-валидацию.
    """)

# ------------- Футер -------------
st.markdown("---")
st.caption("🌸 Iris Flower Classifier Dashboard | Курсовая работа — интерактивный дашборд")
