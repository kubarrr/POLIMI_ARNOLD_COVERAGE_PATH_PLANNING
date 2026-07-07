--config filename (mandatory!)
--method ML algorithm
	"K-Means" (default), "Mini-Batch K-Means", "GMM", "Agglomerative", "Spectral Clustering", "Fuzzy C-Means"

--clusters number_of_clusters
	3 (default)	
	
--high-quality

Example: 
	./run.sh --config paths.txt --method kmeans --k 3 --suplement nitrogen

