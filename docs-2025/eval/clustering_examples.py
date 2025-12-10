# Comprehensive Clustering Evaluation Script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    rand_score,
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    mutual_info_score,
    normalized_mutual_info_score
)
from scipy.spatial.distance import pdist, squareform

# Set up random seed for reproducibility
np.random.seed(42)

# ======================================
# Custom Metrics Implementation
# ======================================

def compute_purity(y_true, y_pred):
    """
    Compute purity score for clustering evaluation.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth class labels
    y_pred : array-like
        Predicted cluster labels
        
    Returns:
    --------
    float
        Purity score (between 0 and 1)
    """
    # Compute contingency matrix
    contingency_matrix = np.zeros((np.max(y_pred) + 1, np.max(y_true) + 1))
    
    for i in range(len(y_true)):
        contingency_matrix[y_pred[i], y_true[i]] += 1
        
    # Find the maximum value for each cluster
    cluster_sizes = contingency_matrix.sum(axis=1)
    max_class_counts = contingency_matrix.max(axis=1)
    
    # Calculate purity
    purity = np.sum(max_class_counts) / np.sum(cluster_sizes)
    
    return purity

def compute_f_measure(y_true, y_pred):
    """
    Compute F-measure for clustering evaluation.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth class labels
    y_pred : array-like
        Predicted cluster labels
        
    Returns:
    --------
    float
        F-measure score (between 0 and 1)
    """
    n_samples = len(y_true)
    
    # Get unique classes and clusters
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    
    # Initialize contingency matrix
    contingency = np.zeros((len(clusters), len(classes)))
    
    # Fill contingency matrix
    for i in range(n_samples):
        contingency[y_pred[i], y_true[i]] += 1
    
    # Calculate F-measure for each class
    f_scores = []
    class_sizes = []
    
    for j, cls in enumerate(classes):
        class_size = np.sum(y_true == cls)
        class_sizes.append(class_size)
        
        # Calculate F-measure for each cluster with this class
        class_f_scores = []
        for i, cluster in enumerate(clusters):
            # Skip empty clusters
            if np.sum(contingency[i]) == 0:
                continue
                
            # Precision and recall
            precision = contingency[i, j] / np.sum(contingency[i])
            recall = contingency[i, j] / class_size
            
            # F-measure
            if precision + recall > 0:
                f_score = 2 * precision * recall / (precision + recall)
                class_f_scores.append(f_score)
            else:
                class_f_scores.append(0)
        
        # Get maximum F-score for this class
        if class_f_scores:
            f_scores.append(np.max(class_f_scores))
        else:
            f_scores.append(0)
    
    # Weighted average by class size
    weighted_f = np.sum(np.array(f_scores) * np.array(class_sizes)) / np.sum(class_sizes)
    
    return weighted_f

def compute_entropy(labels):
    """
    Compute entropy of a clustering.
    
    Parameters:
    -----------
    labels : array-like
        Cluster labels
        
    Returns:
    --------
    float
        Entropy value
    """
    n_samples = len(labels)
    
    # Count occurrences of each label
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Calculate probabilities
    probabilities = counts / n_samples
    
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add small epsilon to avoid log(0)
    
    return entropy

def compute_conditional_entropy(y_true, y_pred):
    """
    Compute conditional entropy H(true|pred).
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth class labels
    y_pred : array-like
        Predicted cluster labels
        
    Returns:
    --------
    float
        Conditional entropy value
    """
    n_samples = len(y_true)
    
    # Get unique labels
    clusters = np.unique(y_pred)
    classes = np.unique(y_true)
    
    # Initialize contingency matrix
    contingency = np.zeros((len(clusters), len(classes)))
    
    # Fill contingency matrix
    for i in range(n_samples):
        contingency[y_pred[i], y_true[i]] += 1
    
    # Calculate joint probabilities
    joint_prob = contingency / n_samples
    
    # Calculate marginal probabilities
    cluster_prob = np.sum(joint_prob, axis=1)
    
    # Calculate conditional entropy
    cond_entropy = 0
    for i, p_i in enumerate(cluster_prob):
        if p_i > 0:  # Avoid division by zero
            for j in range(len(classes)):
                p_ij = joint_prob[i, j]
                if p_ij > 0:  # Avoid log(0)
                    cond_entropy -= p_ij * np.log2(p_ij / p_i + 1e-10)  # Add small epsilon
    
    return cond_entropy

def dunn_index(X, labels):
    """
    Compute Dunn Index for clustering evaluation.
    
    Parameters:
    -----------
    X : array-like
        Data points
    labels : array-like
        Cluster labels
        
    Returns:
    --------
    float
        Dunn Index value
    """
    # Get unique clusters
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    
    if n_clusters <= 1:
        return float('nan')  # Need at least 2 clusters
    
    # Compute pairwise distances between points
    distances = squareform(pdist(X))
    
    # Initialize variables
    min_inter_cluster_dist = float('inf')
    max_intra_cluster_diam = 0
    
    # Calculate maximum intra-cluster distances (cluster diameters)
    for i in range(n_clusters):
        cluster_indices = np.where(labels == unique_clusters[i])[0]
        
        if len(cluster_indices) <= 1:
            continue  # Skip single-point clusters
            
        # Get distances within this cluster
        cluster_distances = distances[np.ix_(cluster_indices, cluster_indices)]
        
        # Update maximum intra-cluster diameter
        cluster_diam = np.max(cluster_distances)
        max_intra_cluster_diam = max(max_intra_cluster_diam, cluster_diam)
    
    # Calculate minimum inter-cluster distances
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            cluster_i_indices = np.where(labels == unique_clusters[i])[0]
            cluster_j_indices = np.where(labels == unique_clusters[j])[0]
            
            if len(cluster_i_indices) == 0 or len(cluster_j_indices) == 0:
                continue
                
            # Get distances between points in different clusters
            cross_distances = distances[np.ix_(cluster_i_indices, cluster_j_indices)]
            
            # Update minimum inter-cluster distance
            min_dist = np.min(cross_distances)
            min_inter_cluster_dist = min(min_inter_cluster_dist, min_dist)
    
    # Compute Dunn Index
    if max_intra_cluster_diam == 0:
        return float('nan')  # Avoid division by zero
        
    dunn = min_inter_cluster_dist / max_intra_cluster_diam
    
    return dunn

def calculate_wcss(X, labels):
    """
    Calculate Within-Cluster Sum of Squares (WCSS).
    
    Parameters:
    -----------
    X : array-like
        Data points
    labels : array-like
        Cluster labels
        
    Returns:
    --------
    float
        WCSS value
    """
    # Get unique clusters
    unique_clusters = np.unique(labels)
    
    # Initialize WCSS
    wcss = 0
    
    # Calculate WCSS for each cluster
    for cluster in unique_clusters:
        if cluster == -1:  # Skip noise points (for DBSCAN)
            continue
            
        # Get points in this cluster
        cluster_points = X[labels == cluster]
        
        if len(cluster_points) == 0:
            continue
            
        # Calculate centroid
        centroid = np.mean(cluster_points, axis=0)
        
        # Calculate sum of squared distances to centroid
        cluster_wcss = np.sum(np.square(cluster_points - centroid))
        wcss += cluster_wcss
    
    return wcss

def plot_radar_chart(metrics_dict, metric_names):
    """
    Create a radar chart to compare clustering algorithms.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with algorithm names as keys and lists of metric values as values
    metric_names : list
        List of metric names
    """
    # Normalize all metrics to [0,1] for comparison
    # For metrics where lower is better, we'll use 1-value
    normalized_dict = {}
    
    for alg, values in metrics_dict.items():
        normalized_values = []
        for i, value in enumerate(values):
            if metric_names[i] in ['Davies-Bouldin', 'WCSS']:
                # For these metrics, lower is better
                max_value = max([m[i] for m in metrics_dict.values() if not np.isnan(m[i])])
                min_value = min([m[i] for m in metrics_dict.values() if not np.isnan(m[i])])
                range_value = max_value - min_value
                if range_value == 0:
                    normalized_values.append(0 if np.isnan(value) else 1)
                else:
                    normalized_values.append(0 if np.isnan(value) else (1 - ((value - min_value) / range_value)))
            else:
                # For other metrics, higher is better
                normalized_values.append(0 if np.isnan(value) else value)
        normalized_dict[alg] = normalized_values
    
    # Number of metrics
    N = len(metric_names)
    
    # Compute angles for each metric
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Plot each algorithm
    colors = ['b', 'g', 'r']
    markers = ['o', 's', '^']
    
    for i, (alg, values) in enumerate(normalized_dict.items()):
        values += values[:1]  # Close the polygon
        ax.plot(angles, values, color=colors[i], linewidth=2, marker=markers[i], 
                label=alg, markersize=8)
        ax.fill(angles, values, color=colors[i], alpha=0.1)
    
    # Set ticks and labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    
    # Add legend and title
    ax.legend(loc='upper right')
    plt.title('Comparison of Clustering Algorithms', size=15)
    
    plt.tight_layout()
    plt.show()

def plot_silhouette(X, labels, algorithm_name):
    """
    Create a silhouette plot for clustering visualization.
    
    Parameters:
    -----------
    X : array-like
        Data points
    labels : array-like
        Cluster labels
    algorithm_name : str
        Name of the clustering algorithm
    """
    # Calculate silhouette scores
    try:
        silhouette_avg = silhouette_score(X, labels)
        sample_silhouette_values = silhouette_samples(X, labels)
        
        # Get unique clusters (exclude noise points if any)
        clusters = np.unique(labels)
        if -1 in clusters:
            clusters = clusters[clusters != -1]
            
        n_clusters = len(clusters)
        
        # Create silhouette plot
        fig, ax = plt.subplots(figsize=(10, 7))
        
        y_lower = 10
        
        for i, cluster in enumerate(clusters):
            # Get silhouette scores for current cluster
            ith_cluster_values = sample_silhouette_values[labels == cluster]
            ith_cluster_values.sort()
            
            # Compute size of current cluster
            size_cluster_i = ith_cluster_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            # Generate color for this cluster
            color = cm.nipy_spectral(float(i) / n_clusters)
            
            # Fill silhouette plot
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_values,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            # Label the silhouette plots with cluster numbers
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster))
            
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10
            
        # Add vertical line for average silhouette score
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        
        # Set plot styling
        ax.set_title(f"Silhouette Plot for {algorithm_name} Clustering")
        ax.set_xlabel("Silhouette Coefficient Values")
        ax.set_ylabel("Cluster")
        ax.set_yticks([])  # Clear y-axis labels
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
        plt.tight_layout()
        plt.show()
    except ValueError as e:
        print(f"Error creating silhouette plot for {algorithm_name}: {e}")

def plot_clusters_2d(X, labels, true_labels=None, algorithm_name="Clustering"):
    """
    Visualize clustering results in 2D.
    
    Parameters:
    -----------
    X : array-like
        Data points (if more than 2D, PCA will be applied)
    labels : array-like
        Cluster labels
    true_labels : array-like, optional
        True class labels for comparison
    algorithm_name : str
        Name of the clustering algorithm
    """
    # Apply PCA if X has more than 2 dimensions
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
    else:
        X_2d = X.copy()
    
    # Create subplots
    n_plots = 2 if true_labels is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]  # Make axes iterable
    
    # Plot clustering results
    scatter = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', 
                             s=50, alpha=0.8, edgecolors='w')
    axes[0].set_title(f"{algorithm_name} Results")
    
    # Add legend
    legend1 = axes[0].legend(*scatter.legend_elements(),
                            loc="best", title="Clusters")
    axes[0].add_artist(legend1)
    
    # Plot true labels if provided
    if true_labels is not None:
        scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=true_labels, cmap='viridis',
                                  s=50, alpha=0.8, edgecolors='w')
        axes[1].set_title("True Classes")
        
        # Add legend
        legend2 = axes[1].legend(*scatter2.legend_elements(),
                                loc="best", title="Classes")
        axes[1].add_artist(legend2)
    
    plt.tight_layout()
    plt.show()

def plot_elbow_method(X, max_k=10):
    """
    Plot the Elbow Method to find optimal number of clusters.
    
    Parameters:
    -----------
    X : array-like
        Data points
    max_k : int
        Maximum number of clusters to try
    """
    wcss_values = []
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss_values.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), wcss_values, marker='o', linestyle='-')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.grid(True)
    plt.show()

def compare_metrics_table(metrics_dict, metric_names):
    """
    Display a comparison table of all metrics for different algorithms.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with algorithm names as keys and lists of metric values as values
    metric_names : list
        List of metric names
    """
    # Create a DataFrame for easy display
    data = []
    for alg, values in metrics_dict.items():
        row = [alg] + values
        data.append(row)
    
    columns = ['Algorithm'] + metric_names
    df = pd.DataFrame(data, columns=columns)
    
    # Format numeric columns
    for col in df.columns[1:]:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
    
    return df

# ======================================
# Main Execution
# ======================================

def main():
    print("Starting Clustering Evaluation")
    print("=" * 50)
    
    # Load Iris dataset
    print("Loading Iris dataset...")
    X, y = datasets.load_iris(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print("=" * 50)
    
    # Apply clustering algorithms
    print("Applying clustering algorithms...")
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=3)
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.6, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    # Print cluster distribution
    print("\nCluster distribution:")
    print(f"K-means: {np.bincount(kmeans_labels)}")
    print(f"Hierarchical: {np.bincount(hierarchical_labels)}")
    print(f"DBSCAN: {np.bincount(dbscan_labels)}")
    print("=" * 50)
    
    # Visualize clustering results with PCA
    print("Visualizing clusters...")
    
    # Plot original data with true labels
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(18, 6))
    
    # Original data with true labels
    plt.subplot(141)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
    plt.title('True Classes')
    
    # K-means clustering
    plt.subplot(142)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.8)
    plt.title('K-means Clustering')
    
    # Hierarchical clustering
    plt.subplot(143)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='viridis', s=50, alpha=0.8)
    plt.title('Hierarchical Clustering')
    
    # DBSCAN clustering
    plt.subplot(144)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', s=50, alpha=0.8)
    plt.title('DBSCAN Clustering')
    
    plt.tight_layout()
    plt.show()
    
    # Plot more detailed visualizations
    plot_clusters_2d(X_scaled, kmeans_labels, y, "K-means")
    plot_clusters_2d(X_scaled, hierarchical_labels, y, "Hierarchical")
    plot_clusters_2d(X_scaled, dbscan_labels, y, "DBSCAN")
    
    print("=" * 50)
    print("Computing evaluation metrics...")
    
    # Compute External Metrics
    # Purity
    kmeans_purity = compute_purity(y, kmeans_labels)
    hierarchical_purity = compute_purity(y, hierarchical_labels)
    dbscan_purity = compute_purity(y, dbscan_labels)
    
    # Rand Index
    kmeans_ri = rand_score(y, kmeans_labels)
    hierarchical_ri = rand_score(y, hierarchical_labels)
    dbscan_ri = rand_score(y, dbscan_labels)
    
    # Adjusted Rand Index
    kmeans_ari = adjusted_rand_score(y, kmeans_labels)
    hierarchical_ari = adjusted_rand_score(y, hierarchical_labels)
    dbscan_ari = adjusted_rand_score(y, dbscan_labels)
    
    # F-measure
    kmeans_f = compute_f_measure(y, kmeans_labels)
    hierarchical_f = compute_f_measure(y, hierarchical_labels)
    dbscan_f = compute_f_measure(y, dbscan_labels)
    
    # Information-Theoretic Metrics
    # Mutual Information
    kmeans_mi = mutual_info_score(y, kmeans_labels)
    hierarchical_mi = mutual_info_score(y, hierarchical_labels)
    dbscan_mi = mutual_info_score(y, dbscan_labels)
    
    # Normalized Mutual Information
    kmeans_nmi = normalized_mutual_info_score(y, kmeans_labels)
    hierarchical_nmi = normalized_mutual_info_score(y, hierarchical_labels)
    dbscan_nmi = normalized_mutual_info_score(y, dbscan_labels)
    
    # Homogeneity
    kmeans_homogeneity = homogeneity_score(y, kmeans_labels)
    hierarchical_homogeneity = homogeneity_score(y, hierarchical_labels)
    dbscan_homogeneity = homogeneity_score(y, dbscan_labels)
    
    # Completeness
    kmeans_completeness = completeness_score(y, kmeans_labels)
    hierarchical_completeness = completeness_score(y, hierarchical_labels)
    dbscan_completeness = completeness_score(y, dbscan_labels)
    
    # V-measure
    kmeans_v = v_measure_score(y, kmeans_labels)
    hierarchical_v = v_measure_score(y, hierarchical_labels)
    dbscan_v = v_measure_score(y, dbscan_labels)
    
    # Internal Metrics
    # Silhouette Coefficient
    try:
        kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
    except Exception as e:
        kmeans_silhouette = float('nan')
        print(f"K-means Silhouette error: {e}")
    
    try:
        hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)
    except Exception as e:
        hierarchical_silhouette = float('nan')
        print(f"Hierarchical Silhouette error: {e}")
    
    try:
        # DBSCAN may have a noise cluster (-1), not valid for silhouette
        non_noise_mask = dbscan_labels != -1
        if np.sum(non_noise_mask) > 1 and len(np.unique(dbscan_labels[non_noise_mask])) > 1:
            dbscan_silhouette = silhouette_score(X_scaled[non_noise_mask], dbscan_labels[non_noise_mask])
        else:
            dbscan_silhouette = float('nan')
    except Exception as e:
        dbscan_silhouette = float('nan')
        print(f"DBSCAN Silhouette error: {e}")
    
    # Davies-Bouldin Index
    try:
        kmeans_dbi = davies_bouldin_score(X_scaled, kmeans_labels)
    except Exception as e:
        kmeans_dbi = float('nan')
        print(f"K-means Davies-Bouldin error: {e}")
    
    try:
        hierarchical_dbi = davies_bouldin_score(X_scaled, hierarchical_labels)
    except Exception as e:
        hierarchical_dbi = float('nan')
        print(f"Hierarchical Davies-Bouldin error: {e}")
    
    try:
        # DBSCAN may have a noise cluster (-1), not valid for DBI
        non_noise_mask = dbscan_labels != -1
        if np.sum(non_noise_mask) > 1 and len(np.unique(dbscan_labels[non_noise_mask])) > 1:
            dbscan_dbi = davies_bouldin_score(X_scaled[non_noise_mask], dbscan_labels[non_noise_mask])
        else:
            dbscan_dbi = float('nan')
    except Exception as e:
        dbscan_dbi = float('nan')
        print(f"DBSCAN Davies-Bouldin error: {e}")
    
    # Calinski-Harabasz Index
    try:
        kmeans_chi = calinski_harabasz_score(X_scaled, kmeans_labels)
    except Exception as e:
        kmeans_chi = float('nan')
        print(f"K-means Calinski-Harabasz error: {e}")
    
    try:
        hierarchical_chi = calinski_harabasz_score(X_scaled, hierarchical_labels)
    except Exception as e:
        hierarchical_chi = float('nan')
        print(f"Hierarchical Calinski-Harabasz error: {e}")
    
    try:
        # DBSCAN may have a noise cluster (-1), not valid for CHI
        non_noise_mask = dbscan_labels != -1
        if np.sum(non_noise_mask) > 1 and len(np.unique(dbscan_labels[non_noise_mask])) > 1:
            dbscan_chi = calinski_harabasz_score(X_scaled[non_noise_mask], dbscan_labels[non_noise_mask])
        else:
            dbscan_chi = float('nan')
    except Exception as e:
        dbscan_chi = float('nan')
        print(f"DBSCAN Calinski-Harabasz error: {e}")
    
    # Dunn Index
    kmeans_dunn = dunn_index(X_scaled, kmeans_labels)
    hierarchical_dunn = dunn_index(X_scaled, hierarchical_labels)
    
    try:
        # DBSCAN may have a noise cluster (-1)
        non_noise_mask = dbscan_labels != -1
        if np.sum(non_noise_mask) > 1 and len(np.unique(dbscan_labels[non_noise_mask])) > 1:
            dbscan_dunn = dunn_index(X_scaled[non_noise_mask], dbscan_labels[non_noise_mask])
        else:
            dbscan_dunn = float('nan')
    except Exception as e:
        dbscan_dunn = float('nan')
        print(f"DBSCAN Dunn Index error: {e}")
    
    # WCSS
    kmeans_wcss = calculate_wcss(X_scaled, kmeans_labels)
    hierarchical_wcss = calculate_wcss(X_scaled, hierarchical_labels)
    dbscan_wcss = calculate_wcss(X_scaled, dbscan_labels)
    
    print("=" * 50)
    print("Metrics Summary")
    print("=" * 50)
    
    # Organize metrics for comparison
    metric_names = [
        'Purity', 
        'Rand Index', 
        'ARI', 
        'F-measure',
        'MI', 
        'NMI', 
        'Homogeneity', 
        'Completeness', 
        'V-measure',
        'Silhouette', 
        'Davies-Bouldin', 
        'Calinski-Harabasz',
        'Dunn Index',
        'WCSS'
    ]
    
    metrics_dict = {
        'K-means': [
            kmeans_purity, kmeans_ri, kmeans_ari, kmeans_f,
            kmeans_mi, kmeans_nmi, kmeans_homogeneity, kmeans_completeness, kmeans_v,
            kmeans_silhouette, kmeans_dbi, kmeans_chi, kmeans_dunn, kmeans_wcss
        ],
        'Hierarchical': [
            hierarchical_purity, hierarchical_ri, hierarchical_ari, hierarchical_f,
            hierarchical_mi, hierarchical_nmi, hierarchical_homogeneity, hierarchical_completeness, hierarchical_v,
            hierarchical_silhouette, hierarchical_dbi, hierarchical_chi, hierarchical_dunn, hierarchical_wcss
        ],
        'DBSCAN': [
            dbscan_purity, dbscan_ri, dbscan_ari, dbscan_f,
            dbscan_mi, dbscan_nmi, dbscan_homogeneity, dbscan_completeness, dbscan_v,
            dbscan_silhouette, dbscan_dbi, dbscan_chi, dbscan_dunn, dbscan_wcss
        ]
    }
    
    # Display metrics table
    metrics_df = compare_metrics_table(metrics_dict, metric_names)
    print(metrics_df)
    
    print("\n\nFor metrics like Purity, ARI, NMI, Silhouette: Higher is better")
    print("For metrics like Davies-Bouldin, WCSS: Lower is better")
    
    # Create radar chart for selected metrics
    print("\nCreating radar chart for selected metrics...")
    selected_metrics = [
        'ARI', 'NMI', 'Homogeneity', 'Completeness', 
        'V-measure', 'Silhouette', 'Davies-Bouldin'
    ]
    
    selected_indices = [metric_names.index(m) for m in selected_metrics]
    
    selected_metrics_dict = {
        alg: [values[i] for i in selected_indices]
        for alg, values in metrics_dict.items()
    }
    
    plot_radar_chart(selected_metrics_dict, selected_metrics)
    
    # Visualize silhouette profiles
    print("\nCreating silhouette plots...")
    plot_silhouette(X_scaled, kmeans_labels, "K-means")
    plot_silhouette(X_scaled, hierarchical_labels, "Hierarchical")
    
    try:
        # Filter out noise points for DBSCAN
        non_noise_mask = dbscan_labels != -1
        if np.sum(non_noise_mask) > 1 and len(np.unique(dbscan_labels[non_noise_mask])) > 1:
            plot_silhouette(X_scaled[non_noise_mask], dbscan_labels[non_noise_mask], "DBSCAN")
        else:
            print("Cannot create silhouette plot for DBSCAN: Not enough valid clusters")
    except Exception as e:
        print(f"Cannot create silhouette plot for DBSCAN: {e}")
    
    # Create elbow method plot
    print("\nCreating elbow method plot...")
    plot_elbow_method(X_scaled, max_k=10)
    
    print("\nClustering evaluation complete!")

if __name__ == "__main__":
    main()