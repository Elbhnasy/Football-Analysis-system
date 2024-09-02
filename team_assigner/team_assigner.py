from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        """
        Initializes the TeamAssigner with empty dictionaries for team colors and player-team assignments.
        
        Attributes:
            team_colors (dict): Dictionary containing the colors of each team.
            player_team_dict (dict): Dictionary containing the team assignment for each player. 
            
        """
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image, num_clusters = 2):
        """
        Returns a KMeans clustering model trained on the given image.
        
        Args:
            image (numpy array): Image to train the clustering model on.
            num_clusters (int): Number of clusters to use for KMeans.
            
        Returns:
            KMeans: KMeans clustering model trained on the given image.
        """
        # Reshape the image to a 2D array of pixels
        pixels = image.reshape(-1, 3)
        
        # Train a KMeans clustering model
        kmeans = KMeans(n_clusters=num_clusters, init= "k-means++" , n_init= 1)
        kmeans.fit(pixels)
        
        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Returns the average color of the player in the given bounding box.
        
        Args:
            frame (numpy array): Frame containing the player.
            bbox (list): Bounding box coordinates of the player.
            
        Returns:
            tuple: Average color of the player.
        """
        # Extract player from frame
        player = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        # Get The Top half of the player
        top_half_player = player[:int(player.shape[0]/2),:]

        # Get Clustering Model
        kmeans = self.get_clustering_model(top_half_player, num_clusters = 2)

        # Get the cluster centers and labels
        labels = kmeans.labels_
        colors = kmeans.cluster_centers_

        # get cropped image dimensions
        height, width, _ = top_half_player.shape

        # reshape the labels to the shape of the image
        clustered_image = labels.reshape(height, width)

        # Get the cluster with the most pixels
        corner_cluster = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]

        # Get the most frequent cluster in the corner pixels
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)

        # Get the player cluster
        player_cluster = 1-non_player_cluster

        # Get Color of the player cluster
        player_color = colors[player_cluster]
        
        return player_color
    
    def assign_team_colors(self, frame, player_detection):
        """
        Assigns team colors to players in the given frame.
        
        Args:
            frame (numpy array): Frame to assign team colors to.
            player_detection (dict): Dictionary containing player detections in the frame.
            
        """
        player_colors = []

        for _, player_detection in player_detection.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Train a KMeans clustering model on the player colors
        kmeans = KMeans(n_clusters=2, init= "k-means++" , n_init= 10)
        kmeans.fit(player_colors)

        # Assign team colors based on the cluster centers
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Returns the team assignment for the player in the given bounding box.
        
        Args:
            frame (numpy array): Frame containing the player.
            player_bbox (list): Bounding box coordinates of the player.
            player_id (int): ID of the player.
            
        Returns:
            int: Team assignment for the player.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        # Predict the cluster of the player color
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0] + 1

        if player_id == 91:
            team_id = 1
        
        self.player_team_dict[player_id] = team_id

        return team_id