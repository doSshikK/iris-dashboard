# iris_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster import hierarchy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Настройка страницы
st.set_page_config(
    page_title="Iris Flower Classifier Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка данных
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df

df = load_data()

# Сайдбар с навигацией
st.sidebar.title("🌸 Iris Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "📌 **Выберите раздел:**",
    ["📊 Визуализация данных", "🔍 Анализ данных", "🤖 Кластеризация", "🎯 Классификация", "📈 Метрики"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Дашборд для анализа датасета Iris**
- 150 наблюдений
- 4 признака
- 3 вида ирисов
""")

# ==================== СТРАНИЦА 1: Визуализация данных ====================
if page == "📊 Визуализация данных":
    st.title("📊 Визуализация исходных данных Iris")
    
    # Показываем данные
    with st.expander("📋 Показать данные", expanded=True):
        st.dataframe(df, use_container_width=True, height=300)
    
    # Основные статистики
    st.subheader("📈 Основные статистики")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего записей", df.shape[0])
    with col2:
        st.metric("Признаков", df.shape[1] - 2)
    with col3:
        st.metric("Видов ирисов", df['species_name'].nunique())
    with col4:
        st.metric("Дубликаты", df.duplicated().sum())
    
    # Гистограмма
    st.subheader("📊 Распределение признаков")
    feature = st.selectbox("Выберите признак:", df.columns[:4])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df[feature], bins=15, edgecolor='black', alpha=0.7, color='skyblue')
    ax.set_title(f'Распределение {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Частота')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Круговая диаграмма и таблица
    st.subheader("🥧 Распределение по видам")
    species_counts = df['species_name'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', 
               startangle=90, colors=colors, explode=(0.05, 0.05, 0.05))
        ax.set_title('Доля видов ирисов')
        st.pyplot(fig)
    with col2:
        st.dataframe(species_counts, use_container_width=True)
    
    # Корреляционная матрица
    st.subheader("🔥 Корреляционная матрица")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.iloc[:, :4].corr(), annot=True, cmap='coolwarm', 
                center=0, ax=ax, fmt='.2f', linewidths=0.5)
    ax.set_title('Корреляция между признаками')
    st.pyplot(fig)

# ==================== СТРАНИЦА 2: Анализ данных ====================
elif page == "🔍 Анализ данных":
    st.title("🔍 Анализ данных")
    
    # Pairplot
    st.subheader("📊 Pairplot всех признаков")
    st.info("График показывает взаимосвязи между всеми признаками, окрашенные по видам")
    
    fig = sns.pairplot(df, hue='species_name', diag_kind='hist', palette='Set2')
    fig.fig.suptitle('Pairplot по видам ирисов', y=1.02)
    st.pyplot(fig.fig)
    
    # Scatter plot с выбором осей
    st.subheader("🎯 Scatter plot с выбором признаков")
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Ось X:", df.columns[:4], index=0, key='x_scatter')
    with col2:
        y_axis = st.selectbox("Ось Y:", df.columns[:4], index=1, key='y_scatter')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
    
    for species in df['species_name'].unique():
        subset = df[df['species_name'] == species]
        ax.scatter(subset[x_axis], subset[y_axis], 
                  label=species, alpha=0.7, s=60, color=colors[species])
    
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f'{x_axis} vs {y_axis}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Boxplot
    st.subheader("📦 Boxplot по видам")
    box_feature = st.selectbox("Выберите признак для boxplot:", df.columns[:4], key='box_feature')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='species_name', y=box_feature, palette='pastel', ax=ax)
    ax.set_title(f'Распределение {box_feature} по видам')
    ax.set_xlabel('Вид ириса')
    ax.set_ylabel(box_feature)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ==================== СТРАНИЦА 3: Кластеризация ====================
elif page == "🤖 Кластеризация":
    st.title("🤖 Кластеризация")
    
    # Подготовка данных
    X = df.iloc[:, :4].values
    
    st.subheader("1. Метод локтя для определения оптимального k")
    
    # Слайдер для выбора диапазона k
    max_k = st.slider("Максимальное количество кластеров:", 2, 10, 6)
    
    # Расчет инерции
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # График метода локтя
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, max_k + 1), inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Количество кластеров (k)')
    ax.set_ylabel('Инерция')
    ax.set_title('Метод локтя для определения оптимального k')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.info("Оптимальное k там, где 'изгиб' графика (обычно k=2 или k=3)")
    
    # Кластеризация с выбранным k
    st.subheader("2. Кластеризация KMeans")
    selected_k = st.slider("Выберите количество кластеров:", 2, 5, 3)
    
    kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    
    # Визуализация кластеров
    col1, col2 = st.columns(2)
    with col1:
        x_cluster = st.selectbox("Ось X для кластеров:", df.columns[:4], index=0, key='x_cluster')
    with col2:
        y_cluster = st.selectbox("Ось Y для кластеров:", df.columns[:4], index=1, key='y_cluster')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # График 1: Кластеры KMeans
    scatter1 = ax1.scatter(df[x_cluster], df[y_cluster], c=df['cluster'], 
                          cmap='viridis', alpha=0.7, s=60)
    ax1.scatter(kmeans.cluster_centers_[:, df.columns.get_loc(x_cluster)], 
               kmeans.cluster_centers_[:, df.columns.get_loc(y_cluster)],
               c='red', s=200, marker='X', label='Центроиды')
    ax1.set_xlabel(x_cluster)
    ax1.set_ylabel(y_cluster)
    ax1.set_title(f'KMeans кластеризация (k={selected_k})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: Сравнение с реальными видами
    scatter2 = ax2.scatter(df[x_cluster], df[y_cluster], 
                          c=df['species'], cmap='Set1', alpha=0.7, s=60)
    ax2.set_xlabel(x_cluster)
    ax2.set_ylabel(y_cluster)
    ax2.set_title('Реальные виды ирисов')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Метрика качества
    silhouette = silhouette_score(X, df['cluster'])
    st.metric("Коэффициент силуэта", f"{silhouette:.3f}")
    
    # Иерархическая кластеризация
    st.subheader("3. Иерархическая кластеризация")
    
    if st.button("Показать дендрограмму"):
        fig, ax = plt.subplots(figsize=(12, 8))
        Z = hierarchy.linkage(X, method='ward')
        hierarchy.dendrogram(Z, ax=ax, truncate_mode='lastp', p=30)
        ax.set_title('Дендрограмма иерархической кластеризации')
        ax.set_xlabel('Объекты')
        ax.set_ylabel('Расстояние')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ==================== СТРАНИЦА 4: Классификация ====================
elif page == "🎯 Классификация":
    st.title("🎯 Классификация видов ирисов")
    
    st.info("Используется модель логистической регрессии для классификации 3 видов ирисов")
    
    # Подготовка данных
    X = df.iloc[:, :4]
    y = df['species']
    
    # Разделение данных
    test_size = st.slider("Размер тестовой выборки (%):", 10, 40, 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42, stratify=y
    )
    
    # Обучение модели
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Отображение результатов
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Точность (Accuracy)", f"{accuracy:.3f}")
    with col2:
        st.metric("Обучающих данных", X_train.shape[0])
    with col3:
        st.metric("Тестовых данных", X_test.shape[0])
    
    # Confusion matrix
    st.subheader("Матрица ошибок")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['setosa', 'versicolor', 'virginica'],
                yticklabels=['setosa', 'versicolor', 'virginica'])
    ax.set_xlabel('Предсказанные метки')
    ax.set_ylabel('Истинные метки')
    ax.set_title('Матрица ошибок')
    st.pyplot(fig)
    
    # Classification report
    st.subheader("Отчет классификации")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'))
    
    # Визуализация ошибок
    st.subheader("Визуализация ошибок классификации")
    
    results_df = X_test.copy()
    results_df['true_species'] = y_test.values
    results_df['predicted_species'] = y_pred
    results_df['correct'] = (y_test.values == y_pred)
    results_df['true_name'] = results_df['true_species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    results_df['pred_name'] = results_df['predicted_species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Правильные предсказания
    correct = results_df[results_df['correct']]
    ax.scatter(correct['petal length (cm)'], correct['petal width (cm)'], 
              c='green', s=100, alpha=0.6, label='Правильно', marker='o')
    
    # Ошибки
    wrong = results_df[~results_df['correct']]
    ax.scatter(wrong['petal length (cm)'], wrong['petal width (cm)'], 
              c='red', s=150, alpha=0.8, label='Ошибка', marker='X')
    
    ax.set_xlabel('Длина лепестка (см)')
    ax.set_ylabel('Ширина лепестка (см)')
    ax.set_title('Результаты классификации (зеленые = правильно, красные = ошибки)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Добавляем аннотации для ошибок
    for idx, row in wrong.iterrows():
        ax.annotate(f"{row['true_name']}→{row['pred_name']}", 
                   (row['petal length (cm)'], row['petal width (cm)']),
                   textcoords="offset points", xytext=(0,10), ha='center',
                   fontsize=9, color='darkred')
    
    st.pyplot(fig)

# ==================== СТРАНИЦА 5: Метрики ====================
elif page == "📈 Метрики":
    st.title("📈 Метрики и выводы")
    
    st.subheader("Ключевые выводы")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 Основные инсайты:
        
        1. **Виды четко разделяются** по размерам лепестков
        2. **Setosa** имеет самые маленькие лепестки
        3. **Virginica** — самые большие лепестки
        4. **Versicolor** — промежуточные значения
        
        ### 📊 Корреляции:
        - Длина и ширина лепестка сильно коррелируют (0.96)
        - Признаки чашелистика менее информативны
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Результаты моделей:
        
        **Кластеризация:**
        - Оптимальное количество кластеров: 3
        - Коэффициент силуэта: ~0.55
        
        **Классификация:**
        - Точность логистической регрессии: ~97%
        - Setosa классифицируется идеально
        - Основные ошибки между versicolor и virginica
        """)
    
    st.markdown("---")
    
    # Feature importance
    st.subheader("Важность признаков для классификации")
    
    # Обучаем модель для получения весов
    X = df.iloc[:, :4]
    y = df['species']
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    importance = pd.DataFrame({
        'Признак': X.columns,
        'Важность': np.abs(model.coef_[0])
    }).sort_values('Важность', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(importance['Признак'], importance['Важность'], color='teal')
    ax.set_xlabel('Абсолютная важность')
    ax.set_title('Важность признаков для классификации')
    
    # Добавляем значения на столбцы
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    st.pyplot(fig)
    
    st.markdown("---")
    st.success("""
    **🎯 Рекомендации:**
    1. Для классификации ирисов достаточно только признаков лепестка
    2. KMeans с k=3 хорошо соответствует реальным видам
    3. Модель показывает высокую точность (>95%)
    4. Дашборд позволяет интерактивно исследовать данные
    """)

# Футер
st.markdown("---")
st.caption("🌸 Iris Flower Classifier Dashboard | Курсовая работа по компьютерному анализу данных")
