# WalSmart - Targeted Marketing

A customer segmentation and clustering project for developing targeted marketing strategies using data mining techniques.

**Course:** Data Mining I  
**Program:** Master's in Information Management (NOVA IMS)  
**Author:** Margarida Sardinha (20221959)

---

## Project Overview

This project applies advanced data mining and machine learning techniques to segment WalSmart customers based on their purchasing behavior, value, and engagement patterns. The goal is to identify distinct customer groups to enable targeted marketing strategies and improve customer lifetime value.

### Key Features
- **Multi-perspective clustering:** Combines RFM (Recency, Frequency, Monetary) analysis with behavioral segmentation
- **Advanced clustering algorithms:** Uses both K-Means and HDBSCAN for robust segmentation
- **Comprehensive data preprocessing:** Handles missing values, outliers, and feature engineering
- **Rich visualizations:** Includes heatmaps, radar charts, and dimensionality reduction plots
- **Cluster profiling:** Detailed profiling of resulting customer segments

---

## Project Structure

```
walsmart-targeted-marketing/
├── NOVAIMS_projectData_B2C_202526.csv       # Raw customer data
├── README.md                                 # This file
│
├── Notebooks (Analysis Pipeline):
│   ├── WalSmart_DataExploration.ipynb        # 1. Exploratory data analysis
│   ├── WalSmart_Preprocessing.ipynb          # 2. Data cleaning and preprocessing
│   ├── WalSmart_Modelling.ipynb              # 3. Clustering and segmentation
│   └── WalSmart_TestModelling.ipynb          # 4. Model testing and validation
│
├── src/ (Reusable Functions):
│   ├── preproc.py                            # Data preprocessing functions
│   ├── modelling.py                          # Clustering algorithms
│   └── visualizations.py                     # Plotting utilities
│
└── outputs/ (Generated Results):
    ├── HDBSCAN_Cluster_Profiles.csv          # HDBSCAN cluster profiles
    ├── Macro_Cluster_Profiles.csv            # Consolidated macro clusters
    └── Merged_Cluster_Profiles.csv           # Merged perspective clusters
```

---

## Data

The analysis uses customer transaction and behavioral data from WalSmart (`NOVAIMS_projectData_B2C_202526.csv`), containing:

- **Customer identifiers** (ID_Client, ZIP_Code)
- **RFM metrics** (Recency, Frequency, Monetary value)
- **Product categories** (purchases across different categories)
- **Customer characteristics** (Age, Longevity, Payment methods)
- **Behavioral indicators** (Last purchase location, engagement metrics)

---

## Methodology

### 1. Data Exploration (`WalSmart_DataExploration.ipynb`)
- Assess data types, missing values, and distributions
- Analyze numeric and categorical variables
- Deep dive into product categories and customer segments
- Identify outliers and data quality issues

### 2. Data Preprocessing (`WalSmart_Preprocessing.ipynb`)
- Remove incomplete and duplicate records
- Drop meaningless/unusable columns
- Handle missing values using KNN imputation
- Engineer product category features
- Scale features using StandardScaler
- Apply noise detection using DBSCAN

### 3. Modelling (`WalSmart_Modelling.ipynb`)
- **Perspective 1: Value/RFM Clustering** - K-Means on RFM features
- **Perspective 2: Behavioral Clustering** - K-Means on product categories
- **Perspective Consolidation** - Merge perspectives using Hierarchical Clustering
- **HDBSCAN Optimization** - Apply HDBSCAN for density-based clustering with noise detection
- **Cluster visualization** - Generate radar charts and heatmaps

### 4. Additional tests (`WalSmart_TestModelling.ipynb`)
- Model testing with different models and approaches

---

## Key Functions

### Preprocessing (`src/preproc.py`)
- `walsmart_preproc()` - Main preprocessing pipeline
- `process_zip_codes()` - Process and encode ZIP code data
- `detect_outliers_dbscan()` - Identify outliers using DBSCAN
- `scale_data()` - Standardize features

### Modelling (`src/modelling.py`)
- `hc_merge_clusters()` - Merge clustering perspectives using Hierarchical Clustering
- `apply_hdbscan()` - Apply HDBSCAN for density-based clustering
- `evaluate_silhouette()` - Compute silhouette scores

### Visualizations (`src/visualizations.py`)
- `plot_heatmaps()` - Generate cluster profile heatmaps
- `plot_radar_charts()` - Create radar charts for cluster visualization
- `plot_tsne()` - Plot TSNE dimensionality reduction

---

## Outputs

The project generates three sets of cluster profiles:

1. **HDBSCAN_Cluster_Profiles.csv** - Profiles from HDBSCAN clustering
2. **Macro_Cluster_Profiles.csv** - Consolidated macro-level cluster profiles
3. **Merged_Cluster_Profiles.csv** - Results from merged perspective clustering

Each profile includes cluster statistics, feature means, and cluster sizes.

---

## Usage

### Running the Analysis

1. **Ensure dependencies are installed:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn hdbscan cartopy
   ```

2. **Run the notebooks in order:**
   - Start with `WalSmart_DataExploration.ipynb`
   - Follow with `WalSmart_Preprocessing.ipynb`
   - Execute `WalSmart_Modelling.ipynb`
   - Review results in `WalSmart_TestModelling.ipynb`

3. **Access outputs:**
   - Check the `outputs/` directory for generated cluster profiles

---

## Results Summary

The analysis identifies distinct customer segments based on:
- **Purchase value** and frequency (RFM analysis)
- **Product category preferences**
- **Customer engagement and longevity**

These segments enable:
- Targeted marketing campaigns for different customer groups
- Resource allocation optimization
- Customer lifetime value prediction
- Churn risk identification

---

## Technologies Used

- **Python 3.x**
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Machine learning (K-Means, HDBSCAN, preprocessing)
- **Matplotlib & Seaborn** - Visualization
- **Cartopy** - Geographic visualizations
- **HDBSCAN** - Density-based clustering

---

## References

- HDBSCAN: Hierarchical Density-Based Spatial Clustering of Applications with Noise
- RFM Analysis: Recency, Frequency, Monetary segmentation
- Customer Segmentation Best Practices in Retail

---

## License

This project is part of a university course assignment.

---

## Repository

Full project code and supporting materials: https://github.com/C-ph3r/walsmart-targeted-marketing
