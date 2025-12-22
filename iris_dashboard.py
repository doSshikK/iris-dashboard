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
# ДОБАВЛЕНО для Random Forest
from sklearn.ensemble import RandomForestClassifier

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

# ------------- Объявление фильтра по видам -------------
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
    st.subheader("Распределение по видам")
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
    st.subheader("Корреляционная матрица")
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
    st.title("Кластеризация")

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

        # ------ Обучение двух моделей ------
        st.subheader("🤖 Сравнение моделей классификации")
        
        # Создаем две модели
        model_lr = LogisticRegression(random_state=42, max_iter=300)
        model_rf = RandomForestClassifier(random_state=42, n_estimators=100)
        
        # Обучаем модели
        model_lr.fit(X_train, y_train)
        model_rf.fit(X_train, y_train)
        
        # Предсказания
        y_pred_lr = model_lr.predict(X_test)
        y_pred_rf = model_rf.predict(X_test)
        
        # Метрики
        acc_lr = accuracy_score(y_test, y_pred_lr)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        
        # ------ Сравнение метрик ------
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        with col_metrics1:
            st.metric("📊 Logistic Regression", f"{acc_lr:.3f}", 
                     delta=f"{(acc_lr - acc_rf):+.3f}" if acc_lr != acc_rf else "0.000")
        with col_metrics2:
            st.metric("🌲 Random Forest", f"{acc_rf:.3f}",
                     delta=f"{(acc_rf - acc_lr):+.3f}" if acc_rf != acc_lr else "0.000")
        with col_metrics3:
            st.metric("📈 Разница", f"{abs(acc_lr - acc_rf):.3f}",
                     delta="Лучше LR" if acc_lr > acc_rf else "Лучше RF" if acc_rf > acc_lr else "Равны")

        # ------ Детальное сравнение в табах ------
        tab_lr, tab_rf, tab_compare = st.tabs(["Logistic Regression", "Random Forest", "Сравнение"])
        
        with tab_lr:
            st.subheader("Logistic Regression")
            
            # Важность признаков для LR
            coefs_lr = np.abs(model_lr.coef_)
            feature_importance_lr = pd.DataFrame({
                'Признак': X.columns,
                'Важность': coefs_lr.mean(axis=0)
            }).sort_values('Важность', ascending=False)
            
            col_lr1, col_lr2 = st.columns(2)
            with col_lr1:
                fig_imp_lr, ax_imp_lr = plt.subplots(figsize=(6, 4))
                ax_imp_lr.barh(feature_importance_lr['Признак'], feature_importance_lr['Важность'], 
                              color='lightgreen')
                ax_imp_lr.set_xlabel('Средняя |коэффициент|')
                ax_imp_lr.set_title('Важность признаков (LR)')
                st.pyplot(fig_imp_lr)
            
            with col_lr2:
                st.dataframe(feature_importance_lr.round(4), use_container_width=True)
            
            # Матрица ошибок для LR
            cm_lr = confusion_matrix(y_test, y_pred_lr)
            fig_cm_lr, ax_cm_lr = plt.subplots(figsize=(5, 4))
            class_names = ['setosa', 'versicolor', 'virginica'][:len(np.unique(y))]
            sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax_cm_lr,
                        xticklabels=class_names, yticklabels=class_names)
            ax_cm_lr.set_title('Матрица ошибок (LR)')
            ax_cm_lr.set_xlabel('Предсказанные метки')
            ax_cm_lr.set_ylabel('Истинные метки')
            st.pyplot(fig_cm_lr)
            
            # Отчет классификации для LR
            st.subheader("Отчет классификации")
            report_lr = classification_report(y_test, y_pred_lr, output_dict=True)
            report_df_lr = pd.DataFrame(report_lr).transpose()
            st.dataframe(report_df_lr.round(3), use_container_width=True)
        
        with tab_rf:
            st.subheader("Random Forest")
            
            # Важность признаков для RF
            feature_importance_rf = pd.DataFrame({
                'Признак': X.columns,
                'Важность': model_rf.feature_importances_
            }).sort_values('Важность', ascending=False)
            
            col_rf1, col_rf2 = st.columns(2)
            with col_rf1:
                fig_imp_rf, ax_imp_rf = plt.subplots(figsize=(6, 4))
                ax_imp_rf.barh(feature_importance_rf['Признак'], feature_importance_rf['Важность'], 
                              color='lightblue')
                ax_imp_rf.set_xlabel('Важность (Gini)')
                ax_imp_rf.set_title('Важность признаков (RF)')
                st.pyplot(fig_imp_rf)
            
            with col_rf2:
                st.dataframe(feature_importance_rf.round(4), use_container_width=True)
            
            # Матрица ошибок для RF
            cm_rf = confusion_matrix(y_test, y_pred_rf)
            fig_cm_rf, ax_cm_rf = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax_cm_rf,
                        xticklabels=class_names, yticklabels=class_names)
            ax_cm_rf.set_title('Матрица ошибок (RF)')
            ax_cm_rf.set_xlabel('Предсказанные метки')
            ax_cm_rf.set_ylabel('Истинные метки')
            st.pyplot(fig_cm_rf)
            
            # Отчет классификации для RF
            st.subheader("Отчет классификации")
            report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
            report_df_rf = pd.DataFrame(report_rf).transpose()
            st.dataframe(report_df_rf.round(3), use_container_width=True)
        
        with tab_compare:
            st.subheader("📈 Сравнение моделей")
            
            # Сравнение важности признаков
            fig_compare, ax_compare = plt.subplots(figsize=(10, 5))
            x = np.arange(len(X.columns))
            width = 0.35
            
            ax_compare.bar(x - width/2, feature_importance_lr.sort_values('Признак')['Важность'], 
                          width, label='Logistic Regression', color='lightgreen', alpha=0.8)
            ax_compare.bar(x + width/2, feature_importance_rf.sort_values('Признак')['Важность'], 
                          width, label='Random Forest', color='lightblue', alpha=0.8)
            
            ax_compare.set_xlabel('Признаки')
            ax_compare.set_ylabel('Важность')
            ax_compare.set_title('Сравнение важности признаков')
            ax_compare.set_xticks(x)
            ax_compare.set_xticklabels(X.columns, rotation=45)
            ax_compare.legend()
            st.pyplot(fig_compare)
            
            # Сравнение точности по классам
            st.subheader("Точность по классам")
            accuracy_by_class = pd.DataFrame({
                'Класс': ['setosa', 'versicolor', 'virginica'][:len(np.unique(y))],
                'Logistic Regression': [np.mean(y_pred_lr[y_test == i] == i) for i in np.unique(y)],
                'Random Forest': [np.mean(y_pred_rf[y_test == i] == i) for i in np.unique(y)]
            })
            st.dataframe(accuracy_by_class.round(3), use_container_width=True)
            
            # Визуализация ошибок обеих моделей
            st.subheader("Визуализация ошибок обеих моделей")
            
            # Подготовка данных для визуализации
            X_test_orig = pd.DataFrame(scaler.inverse_transform(X_test), columns=X.columns)
            results_df = X_test_orig.copy()
            results_df['true_species'] = y_test.values
            results_df['pred_lr'] = y_pred_lr
            results_df['pred_rf'] = y_pred_rf
            results_df['correct_lr'] = results_df['true_species'] == results_df['pred_lr']
            results_df['correct_rf'] = results_df['true_species'] == results_df['pred_rf']
            results_df['true_name'] = results_df['true_species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
            
            fig_errors, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # LR ошибки
            correct_lr = results_df[results_df['correct_lr']]
            wrong_lr = results_df[~results_df['correct_lr']]
            ax1.scatter(correct_lr['petal length (cm)'], correct_lr['petal width (cm)'], 
                       c='green', s=80, label='Правильно', alpha=0.6)
            ax1.scatter(wrong_lr['petal length (cm)'], wrong_lr['petal width (cm)'], 
                       c='red', s=120, marker='x', label='Ошибка', alpha=0.9)
            ax1.set_xlabel('Длина лепестка (cm)')
            ax1.set_ylabel('Ширина лепестка (cm)')
            ax1.set_title('Logistic Regression')
            ax1.legend()
            
            # RF ошибки
            correct_rf = results_df[results_df['correct_rf']]
            wrong_rf = results_df[~results_df['correct_rf']]
            ax2.scatter(correct_rf['petal length (cm)'], correct_rf['petal width (cm)'], 
                       c='green', s=80, label='Правильно', alpha=0.6)
            ax2.scatter(wrong_rf['petal length (cm)'], wrong_rf['petal width (cm)'], 
                       c='red', s=120, marker='x', label='Ошибка', alpha=0.9)
            ax2.set_xlabel('Длина лепестка (cm)')
            ax2.set_ylabel('Ширина лепестка (cm)')
            ax2.set_title('Random Forest')
            ax2.legend()
            
            st.pyplot(fig_errors)
            
            # Выводы
            st.info(f"""
            **Ключевые выводы:**
            - Logistic Regression: **{acc_lr:.1%}** точности
            - Random Forest: **{acc_rf:.1%}** точности
            - Разница: **{abs(acc_lr - acc_rf):.1%}**
            """)

        # ------ Интерактивный прогноз ------
        st.subheader(" Интерактивный прогноз")
        st.markdown("Введите параметры цветка для предсказания вида:")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                sepal_length = st.number_input("Длина чашелистика (см):", 
                                              min_value=0.1, max_value=10.0, 
                                              value=5.0, step=0.1)
                sepal_width = st.number_input("Ширина чашелистика (см):", 
                                             min_value=0.1, max_value=10.0, 
                                             value=3.5, step=0.1)
            
            with col2:
                petal_length = st.number_input("Длина лепестка (см):", 
                                              min_value=0.1, max_value=10.0, 
                                              value=1.5, step=0.1)
                petal_width = st.number_input("Ширина лепестка (см):", 
                                             min_value=0.1, max_value=10.0, 
                                             value=0.2, step=0.1)
            
            model_choice = st.radio("Выберите модель для прогноза:", 
                                   ["Logistic Regression", "Random Forest", "Обе модели"])
            
            submitted = st.form_submit_button("Предсказать вид")
            
            if submitted:
                # Подготовка входных данных
                input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
                input_scaled = scaler.transform(input_data)
                
                # Предсказания
                prediction_lr = model_lr.predict(input_scaled)[0]
                prediction_rf = model_rf.predict(input_scaled)[0]
                
                # Вероятности
                prob_lr = model_lr.predict_proba(input_scaled)[0] if hasattr(model_lr, 'predict_proba') else None
                prob_rf = model_rf.predict_proba(input_scaled)[0] if hasattr(model_rf, 'predict_proba') else None
                
                species_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
                
                if model_choice == "Logistic Regression" or model_choice == "Обе модели":
                    st.subheader("📊 Logistic Regression")
                    col_lr1, col_lr2 = st.columns(2)
                    with col_lr1:
                        st.success(f"**Предсказанный вид:**\n**{species_names[prediction_lr].upper()}**")
                    with col_lr2:
                        st.info(f"**Точность модели:**\n**{acc_lr:.1%}**")
                    
                    if prob_lr is not None:
                        st.subheader("Вероятности (LR):")
                        prob_df_lr = pd.DataFrame({
                            'Вид': ['setosa', 'versicolor', 'virginica'],
                            'Вероятность': prob_lr
                        }).sort_values('Вероятность', ascending=False)
                        
                        fig_prob_lr, ax_prob_lr = plt.subplots(figsize=(8, 3))
                        ax_prob_lr.bar(prob_df_lr['Вид'], prob_df_lr['Вероятность'], 
                                      color=['red' if p == max(prob_lr) else 'gray' for p in prob_lr])
                        ax_prob_lr.set_ylabel('Вероятность')
                        ax_prob_lr.set_ylim(0, 1.1)
                        for bar, prob in zip(ax_prob_lr.patches, prob_df_lr['Вероятность']):
                            ax_prob_lr.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                                          f'{prob:.1%}', ha='center', va='bottom', fontsize=10)
                        st.pyplot(fig_prob_lr)
                
                if model_choice == "Random Forest" or model_choice == "Обе модели":
                    st.subheader("🌲 Random Forest")
                    col_rf1, col_rf2 = st.columns(2)
                    with col_rf1:
                        st.success(f"**Предсказанный вид:**\n**{species_names[prediction_rf].upper()}**")
                    with col_rf2:
                        st.info(f"**Точность модели:**\n**{acc_rf:.1%}**")
                    
                    if prob_rf is not None:
                        st.subheader("Вероятности (RF):")
                        prob_df_rf = pd.DataFrame({
                            'Вид': ['setosa', 'versicolor', 'virginica'],
                            'Вероятность': prob_rf
                        }).sort_values('Вероятность', ascending=False)
                        
                        fig_prob_rf, ax_prob_rf = plt.subplots(figsize=(8, 3))
                        ax_prob_rf.bar(prob_df_rf['Вид'], prob_df_rf['Вероятность'], 
                                      color=['blue' if p == max(prob_rf) else 'gray' for p in prob_rf])
                        ax_prob_rf.set_ylabel('Вероятность')
                        ax_prob_rf.set_ylim(0, 1.1)
                        for bar, prob in zip(ax_prob_rf.patches, prob_df_rf['Вероятность']):
                            ax_prob_rf.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                                          f'{prob:.1%}', ha='center', va='bottom', fontsize=10)
                        st.pyplot(fig_prob_rf)
                
                if model_choice == "Обе модели" and prediction_lr != prediction_rf:
                    st.warning(f"⚠️ **Модели расходятся во мнениях!**\n"
                              f"- Logistic Regression: {species_names[prediction_lr]}\n"
                              f"- Random Forest: {species_names[prediction_rf]}")
                
                # Показываем введённые значения
                st.markdown("**Введённые параметры:**")
                params_df = pd.DataFrame({
                    'Признак': ['Длина чашелистика', 'Ширина чашелистика', 
                               'Длина лепестка', 'Ширина лепестка'],
                    'Значение': [sepal_length, sepal_width, petal_length, petal_width],
                    'Единица': ['см', 'см', 'см', 'см']
                })
                st.dataframe(params_df, use_container_width=True)
                
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
    3. Логистическая регрессия показывает высокую точность на Iris; для более надёжной оценки
       стоит добавить кросс-валидацию.
    4. Random Forest дополнительно предоставляет важность признаков для интерпретации.
    """)

# ------------- Футер -------------
st.markdown("---")
st.caption("Iris Flower Classifier Dashboard | Курсовая работа — интерактивный дашборд")
