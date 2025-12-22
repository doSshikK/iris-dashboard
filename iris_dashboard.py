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
# ДОБАВЛЕНО: Импорт новых моделей
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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

# Загружаем данные
df = load_data()

# ------------- Sidebar -------------
st.sidebar.title("🌸 Iris Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "📌 **Выберите раздел:**",
    [" Визуализация данных", " Анализ данных", " Кластеризация", " Классификация", " Метрики / Выводы"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Дашборд для анализа датасета Iris**  
150 наблюдений, 4 признака, 3 вида ирисов.
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
if page == " Визуализация данных":
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

    # Проверка качества данных: пропуски, дубликаты и уникальность
    st.subheader("📌 Пропуски, дубликаты и уникальность")
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        missing_by_col = df_filtered.isna().sum()
        quality_df = pd.DataFrame({
            'Колонка': missing_by_col.index,
            'Пропуски': missing_by_col.values
        })
        st.dataframe(quality_df, use_container_width=True)
    
    with col_b:
        total_missing = int(df_filtered.isna().sum().sum())
        total_duplicates = int(df_filtered.duplicated().sum())
        unique_rows = df_filtered.shape[0] - total_duplicates
        
        st.metric("Всего пропусков", total_missing)
        st.metric("Всего дубликатов", total_duplicates)
        st.metric("Уникальных строк", unique_rows)

    # Описание колонок
    st.subheader("📝 Описание колонок")
    
    with st.expander("📋 Подробное описание признаков", expanded=True):
        st.markdown("""
        ### Описание всех колонок датасета Iris:
        
        **1. sepal length (cm)** - длина чашелистика в см (тип: float64)  
        *Измеряется от основания до верхушки чашелистика*
        
        **2. sepal width (cm)** - ширина чашелистика в см (тип: float64)  
        *Измеряется в самой широкой части чашелистика*
        
        **3. petal length (cm)** - длина лепестка в см (тип: float64)  
        *Измеряется от основания до верхушки лепестка*
        
        **4. petal width (cm)** - ширина лепестка в см (тип: float64)  
        *Измеряется в самой широкой части лепестка*
        
        **5. species** - вид ириса (тип: int64)  
        *Числовая кодировка вида:*  
        - **0**: setosa  
        - **1**: versicolor  
        - **2**: virginica
        
        **6. species_name** - название вида (тип: object)  
        *Текстовое название вида:*  
        - **setosa**  
        - **versicolor**  
        - **virginica**
        """)
        
        # Дополнительно в виде таблицы
        st.markdown("**Краткая сводка в таблице:**")
        columns_description = pd.DataFrame({
            'Колонка': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 
                       'petal width (cm)', 'species', 'species_name'],
            'Описание': ['Длина чашелистика (см)', 'Ширина чашелистика (см)', 
                        'Длина лепестка (см)', 'Ширина лепестка (см)',
                        'Код вида (0, 1, 2)', 'Название вида'],
            'Тип данных': ['float64', 'float64', 'float64', 'float64', 'int64', 'object'],
            'Диапазон значений': ['4.3 - 7.9 см', '2.0 - 4.4 см', '1.0 - 6.9 см', 
                                '0.1 - 2.5 см', '0-2', 'setosa, versicolor, virginica']
        })
        st.dataframe(columns_description, use_container_width=True)

    # Описательная статистика (как df.describe())
    st.subheader("📊 Описательная статистика (df.describe())")
    
    # Берем только числовые колонки
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        # Создаем таблицу как в df.describe()
        describe_df = df_filtered[numeric_cols].describe().transpose()
        describe_df = describe_df.round(3)
        
        # Переименовываем столбцы на русский
        describe_df = describe_df.rename(columns={
            'count': 'Количество',
            'mean': 'Среднее',
            'std': 'Ст. отклонение',
            'min': 'Минимум',
            '25%': '25%',
            '50%': 'Медиана',
            '75%': '75%',
            'max': 'Максимум'
        })
        
        # Показываем таблицу
        st.dataframe(describe_df, use_container_width=True)
    else:
        st.warning("Нет числовых колонок для статистики")

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
    correlation_matrix = df_filtered.iloc[:, :4].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax2, fmt='.2f')
    ax2.set_title('Корреляция между признаками')
    st.pyplot(fig2)
    
    # Выводы по корреляциям
    st.subheader("📊 Анализ корреляций")
    
    with st.expander("📈 Ключевые выводы по корреляций", expanded=True):
        st.markdown("""
        ### Сильные положительные корреляции:
        
        **1. Длина лепестка и ширина лепестка: 0.96** 
        - **Очень сильная связь** - лепестки пропорциональны: чем длиннее лепесток, тем он шире
        
        **2. Длина лепестка и длина чашелистика: 0.88** 
        - **Сильная связь** - растения с длинными лепестками обычно имеют и длинные чашелистики
        
        **3. Ширина лепестка и длина чашелистика: 0.82** 
        - **Сильная связь** - растения с широкими лепестками имеют более длинные чашелистики
        
        ### Корреляция с видом ириса:
        
        **4. Вид ириса сильно коррелирует с:**
        - **Длиной лепестка (0.88)** - главный отличительный признак
        - **Шириной лепестка (0.82)** - второй важный признак
        
        ### Самые информативные признаки для классификации:
        
        **5. Признаки-лидеры:**
        - **petal length (длина лепестка)** - самый сильный показатель
        - **petal width (ширина лепестка)** - второй по значимости
        """)

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
elif page == " Анализ данных":
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
elif page == " Кластеризация":
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

    st.info("По графику видно, что оптимальное количество кластеров является k=3, так как дальнейшее увеличение количества кластеров не приводит к значительному уменьшению инерции")

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
elif page == " Классификация":
    st.title("🎯 Классификация видов ирисов")

    st.info("Сравнение трёх моделей классификации: Логистическая регрессия, Random Forest и SVM.")

    # Проверка, что есть хотя бы 2 разных класса для классификации
    if df_filtered['species'].nunique() < 2:
        st.error("❌ Для классификации необходимо минимум 2 разных класса. Выберите больше видов ирисов в фильтре.")
    else:
        # Подготовка данных (стандартизация рекомендуема)
        X = df_filtered.iloc[:, :4].copy()
        y = df_filtered['species'].copy()

        # Стандартизация
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Выбор размера теста
        test_size = st.slider("Размер тестовой выборки (%)", 10, 40, 20)
        
        # Стратификация только если есть хотя бы 2 образца в каждом классе
        stratify_flag = all(y.value_counts() >= 2)
        
        if stratify_flag:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size / 100, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size / 100, random_state=42
            )

        # ---------- ДОБАВЛЕНО: Выбор модели ----------
        model_type = st.selectbox(
            "Выберите модель для классификации:",
            ["Логистическая регрессия", "Random Forest", "SVM (Support Vector Machine)"],
            index=0
        )
        
        # Инициализация выбранной модели
        if model_type == "Логистическая регрессия":
            model = LogisticRegression(random_state=42, max_iter=300)
            model_name = "Logistic Regression"
        elif model_type == "Random Forest":
            n_estimators = st.slider("Количество деревьев (n_estimators):", 10, 200, 100)
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model_name = f"Random Forest (n={n_estimators})"
        else:  # SVM
            kernel = st.selectbox("Ядро SVM:", ["linear", "rbf", "poly"], index=1)
            model = SVC(kernel=kernel, probability=True, random_state=42)
            model_name = f"SVM (kernel={kernel})"

        # Обучение модели
        model.fit(X_train, y_train)

        # Предсказания
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # ---------- ДОБАВЛЕНО: Сравнение моделей ----------
        st.subheader("📊 Сравнение моделей")
        
        # Создаём и обучаем все модели для сравнения
        models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=300),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM (rbf)": SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        results = {}
        for name, mdl in models.items():
            mdl.fit(X_train, y_train)
            y_pred_mdl = mdl.predict(X_test)
            results[name] = accuracy_score(y_test, y_pred_mdl)
        
        # Таблица сравнения
        comparison_df = pd.DataFrame({
            "Модель": list(results.keys()),
            "Accuracy": list(results.values())
        }).sort_values("Accuracy", ascending=False)
        
        st.dataframe(comparison_df.round(3), use_container_width=True)
        
        # Визуализация сравнения
        fig_compare, ax_compare = plt.subplots(figsize=(8, 4))
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        bars = ax_compare.bar(comparison_df["Модель"], comparison_df["Accuracy"], color=colors[:len(comparison_df)])
        ax_compare.set_ylabel("Accuracy")
        ax_compare.set_title("Сравнение точности моделей")
        ax_compare.set_ylim(0, 1.05)
        
        # Добавляем значения на столбцы
        for bar, acc in zip(bars, comparison_df["Accuracy"]):
            height = bar.get_height()
            ax_compare.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        st.pyplot(fig_compare)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{acc:.3f}", delta=f"{acc - 0.5:.3f}" if acc > 0.5 else None)
        with col2:
            st.metric("Train size", X_train.shape[0])
        with col3:
            st.metric("Test size", X_test.shape[0])

        st.subheader("Матрица ошибок (Confusion Matrix)")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Получаем имена классов
        class_names = ['setosa', 'versicolor', 'virginica'][:len(np.unique(y))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names,
                    yticklabels=class_names)
        ax.set_xlabel('Предсказанные метки')
        ax.set_ylabel('Истинные метки')
        ax.set_title(f'Матрица ошибок ({model_name})')
        st.pyplot(fig)

        st.subheader("Отчет классификации (Precision / Recall / F1)")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

        # ---------- ДОБАВЛЕНО: Важность признаков для Random Forest ----------
        if model_type == "Random Forest":
            st.subheader("🌳 Важность признаков (Random Forest)")
            feature_importance = pd.DataFrame({
                "Признак": X.columns,
                "Важность": model.feature_importances_
            }).sort_values("Важность", ascending=False)
            
            fig_importance, ax_importance = plt.subplots(figsize=(8, 4))
            ax_importance.barh(feature_importance["Признак"], feature_importance["Важность"])
            ax_importance.set_xlabel("Важность признака")
            ax_importance.set_title("Важность признаков (Random Forest)")
            st.pyplot(fig_importance)
            
            st.dataframe(feature_importance.round(4), use_container_width=True)

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
        ax.set_title(f'Результаты классификации ({model_name})')
        ax.legend()
        st.pyplot(fig)

# ------------- Страница: Метрики и выводы -------------
elif page == " Метрики / Выводы":
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

    # ---------- ДОБАВЛЕНО: Сравнение моделей классификации ----------
    st.subheader("Сравнение моделей классификации")
    
    # Подготовка данных для сравнения
    X_all = df_filtered.iloc[:, :4]
    y_all = df_filtered['species']
    scaler_full = StandardScaler()
    X_all_scaled = scaler_full.fit_transform(X_all)
    
    # Разделение на train/test для сравнения
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X_all_scaled, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    # Тестируем три модели
    models_comparison = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=300),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    comparison_results = []
    for name, model in models_comparison.items():
        model.fit(X_train_all, y_train_all)
        y_pred_all = model.predict(X_test_all)
        acc_all = accuracy_score(y_test_all, y_pred_all)
        comparison_results.append({
            "Модель": name,
            "Accuracy": round(acc_all, 3),
            "Train size": X_train_all.shape[0],
            "Test size": X_test_all.shape[0]
        })
    
    comparison_df = pd.DataFrame(comparison_results).sort_values("Accuracy", ascending=False)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Визуализация сравнения
    fig_compare, ax_compare = plt.subplots(figsize=(8, 4))
    bars = ax_compare.bar(comparison_df["Модель"], comparison_df["Accuracy"], color=['skyblue', 'lightgreen', 'lightcoral'])
    ax_compare.set_ylabel("Accuracy")
    ax_compare.set_title("Сравнение точности моделей (test_size=20%)")
    ax_compare.set_ylim(0, 1.05)
    
    # Добавляем значения на столбцы
    for bar, acc in zip(bars, comparison_df["Accuracy"]):
        height = bar.get_height()
        ax_compare.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    st.pyplot(fig_compare)

    # Классификация: тренировочный прогон на всем датасете (кросс-валидация не включена здесь,
    # но мы можем показать обученную модель на полном наборе и её важности признаков)
    if df_filtered['species'].nunique() >= 2:
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
    3. **Все три модели показывают высокую точность** (93-100%) на датасете Iris.
    4. Random Forest дополнительно предоставляет важность признаков для интерпретации.
    5. Для более надёжной оценки стоит добавить кросс-валидацию.
    """)

# ------------- Футер -------------
st.markdown("---")
st.caption("Iris Flower Classifier Dashboard | Курсовая работа — интерактивный дашборд")
