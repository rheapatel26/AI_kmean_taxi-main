import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend for non-interactive plotting
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, send_from_directory
import matplotlib.image as image

app = Flask(__name__, template_folder='template', static_folder='assets')

# Stores for taxi and passenger coordinates
taxi_locations = []

@app.route('/assets/<path:filename>')
def serve_static(filename):
    return send_from_directory('assets', filename)

@app.route('/')
def home():
    return render_template('index.html')

# Generate random passenger locations
def generate_passenger_locations(num):
    return np.array([[random.uniform(40.0, 41.0), random.uniform(-74.0, -73.0)] for _ in range(num)])

# Simplified K-Means implementation (without tol and max_iters)
def kmeans(data, centroids):
    """
    Perform K-Means clustering to assign data points to the nearest centroid.

    Parameters:
    - data: A 2D NumPy array where each row is a data point (e.g., [latitude, longitude]).
    - centroids: A 2D NumPy array where each row is a centroid (e.g., a taxi location).

    Returns:
    - labels: An array of labels indicating which cluster each data point belongs to.
    - new_centroids: The updated centroids after recomputing them based on the data points' assignments.
    """
    num_clusters = len(centroids)
    labels = np.zeros(len(data))  # Initialize an array to hold the labels for each data point

    while True:
        # Step 1: Assign each data point to the nearest centroid
        new_labels = []
        for point in data:
            distances = []  # List to store distances from the point to each centroid
            for centroid in centroids:
                distance = np.linalg.norm(point - centroid)  # Euclidean distance
                distances.append(distance)
            new_labels.append(np.argmin(distances))  # Assign the point to the nearest centroid
        new_labels = np.array(new_labels)

        # Step 2: Update centroids by averaging the points assigned to each centroid
        new_centroids = []
        for i in range(num_clusters):
            cluster_points = data[new_labels == i]  # Get the points assigned to centroid i
            if len(cluster_points) > 0:  # Prevent division by zero
                new_centroids.append(cluster_points.mean(axis=0))  # Compute the new centroid
            else:
                new_centroids.append(centroids[i])  # If no points are assigned, keep the old centroid

        new_centroids = np.array(new_centroids)

        # Step 3: Check if centroids have changed (convergence condition)
        if np.all(new_centroids == centroids):
            break  # If centroids don't change, we have converged
        centroids = new_centroids  # Update centroids for the next iteration

    return new_labels, new_centroids

@app.route('/submit_coordinates', methods=['POST'])
def submit_coordinates():
    data = request.get_json()
    latitude = data['latitude']
    longitude = data['longitude']
    
    taxi_locations.append([latitude, longitude])
    return jsonify({"message": "Coordinates received!"})

@app.route('/start_function')
def start_function():
    # Generate random passenger locations and use current taxi locations
    passengers = generate_passenger_locations(50)
    taxis = np.array(taxi_locations)

    if not taxis.any():
        return jsonify({"message": "No taxis available."}), 400

    # Perform K-Means clustering to assign passengers to taxis
    labels, _ = kmeans(passengers, taxis)

    # Create plot
    plt.figure()
    map_image = image.imread('assets/barca_map.png')
    plt.imshow(map_image, extent=[-74.0, -73.0, 40.0, 41.0])
    plt.scatter(passengers[:, 1], passengers[:, 0], c=labels, s=50, cmap='rainbow', label='Passengers')
    plt.scatter(taxis[:, 1], taxis[:, 0], s=200, c='black', marker='X', label='Taxis')

    for i, (x, y) in enumerate(taxis):
        plt.text(y, x, f'Taxi {i}', fontsize=9)

    plt.title('Passenger Assignment to Taxis')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()

    plot_path = os.path.join(app.static_folder, 'assignment_plot.png')
    plt.savefig(plot_path)
    plt.close()

    # Prepare response with assignments
    assignments = {i: np.where(labels == i)[0].tolist() for i in range(len(taxis))}

    return jsonify({
        "message": "Function executed!",
        "assignments": assignments,
        "taxi_locations": taxis.tolist(),
        "plot_url": "assignment_plot.png"
    })

if __name__ == '__main__':
    app.run(debug=True)
