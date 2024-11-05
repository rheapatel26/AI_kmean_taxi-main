import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend for non-interactive plotting
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, send_from_directory
import matplotlib.image as image

app = Flask(__name__, template_folder='template', static_folder='assets')

# Lists to hold coordinates
CorLat = []
CorLon = []
taxis = []

@app.route('/assets/<path:filename>')
def serve_static(filename):
    return send_from_directory('assets', filename)

@app.route('/')
def helloworld():
    return render_template('index.html')

def generate_passenger_locations(num_passengers):
    passengers = []
    for _ in range(num_passengers):
        lat = random.uniform(40.0, 41.0)  # Adjust for your needs
        lon = random.uniform(-74.0, -73.0)  # Adjust for your needs
        passengers.append([lat, lon])
    return np.array(passengers)

def kmeans_custom(data, initial_centroids, max_iters=100, tolerance=1e-4):
    centroids = np.array(initial_centroids)
    num_samples = data.shape[0]
    k = centroids.shape[0]
    labels = np.zeros(num_samples)

    for _ in range(max_iters):
        # Step 1: Assign clusters
        for i in range(num_samples):
            distances = np.linalg.norm(data[i] - centroids, axis=1)
            labels[i] = np.argmin(distances)

        # Step 2: Update centroids
        new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])
        
        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            break
        
        centroids = new_centroids

    return labels, centroids

@app.route('/submit_coordinates', methods=['POST'])
def submit_coordinates():
    data = request.get_json()
    latitude = data['latitude']
    longitude = data['longitude']
    clickcount = data['clickCount']

    if clickcount > len(taxis):
        taxis.append([latitude, longitude])
        CorLat.append(latitude)
        CorLon.append(longitude)
        print(f"Received coordinates - Latitude: {latitude}, Longitude: {longitude}, Click Count: {clickcount}")
        return jsonify({"message": "Coordinates received successfully!"})
    else:
        return jsonify({"message": "Coordinate submission error."}), 400

@app.route('/start_function')
def start_function():
    num_passengers = 50
    passenger_locations = generate_passenger_locations(num_passengers)
    taxi_locations = np.array(taxis)

    if len(taxi_locations) == 0:
        return jsonify({"message": "No taxis available."}), 400

    # Run custom K-Means
    labels, centroids = kmeans_custom(passenger_locations, taxi_locations)

    # Create the plot
    plt.figure()
    im = image.imread('assets/barca_map.png')  # Ensure the image path is correct
    plt.imshow(im, extent=[-74.0, -73.0, 40.0, 41.0])  # Adjust extent as needed
    plt.scatter(passenger_locations[:, 1], passenger_locations[:, 0], c=labels, s=50, cmap='rainbow', label='Passengers', zorder=2)
    plt.scatter(taxi_locations[:, 1], taxi_locations[:, 0], s=200, c='black', marker='X', label='Taxis', zorder=2)

    for i, (x, y) in enumerate(taxi_locations):
        plt.text(y, x, f'Taxi {i}', fontsize=9)

    plt.title('Passenger Assignment to Nearest Taxi Using Custom K-Means')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()

    # Save the figure
    plot_path = os.path.join(app.static_folder, 'assignment_plot.png')
    plt.savefig(plot_path)
    plt.close()

    # Prepare assignments
    assignments = {}
    for i in range(len(taxi_locations)):
        passenger_indices = np.where(labels == i)[0]
        assignments[i] = passenger_indices.tolist()  # Convert to a regular list

    # Convert taxi locations to a regular list
    taxi_locations_list = taxi_locations.tolist()

    return jsonify({
        "message": "Function executed!",
        "assignments": assignments,
        "taxi_locations": taxi_locations_list,
        "plot_url": "assignment_plot.png"
    })

if __name__ == '__main__':
    app.run(debug=True)
