#!/usr/bin/env python3
"""
Convert Turkey provinces GeoJSON to simplified SVG path data.
Outputs a JS file with province paths that can be embedded in the HTML.
"""

import json
import math

# Load GeoJSON
with open('tr-cities.json', 'r', encoding='utf-8') as f:
    geojson = json.load(f)

# Projection: simple Mercator-like for Turkey
# Turkey spans roughly 26°E-45°E longitude, 36°N-42.5°N latitude
# SVG viewBox: 0 0 800 400

def project(lon, lat):
    x = 30 + (lon - 25.5) * 38.0
    y = 20 + (42.8 - lat) * 46.0
    return (round(x, 1), round(y, 1))

def simplify_ring(coords, tolerance=0.035):
    """Douglas-Peucker simplification."""
    if len(coords) <= 2:
        return coords
    
    # Find the point with max distance from line between first and last
    dmax = 0
    index = 0
    end = len(coords) - 1
    
    for i in range(1, end):
        d = point_line_dist(coords[i], coords[0], coords[end])
        if d > dmax:
            index = i
            dmax = d
    
    if dmax > tolerance:
        left = simplify_ring(coords[:index+1], tolerance)
        right = simplify_ring(coords[index:], tolerance)
        return left[:-1] + right
    else:
        return [coords[0], coords[end]]

def point_line_dist(p, a, b):
    """Distance from point p to line segment a-b."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if dx == 0 and dy == 0:
        return math.sqrt((p[0]-a[0])**2 + (p[1]-a[1])**2)
    t = max(0, min(1, ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / (dx*dx + dy*dy)))
    proj_x = a[0] + t * dx
    proj_y = a[1] + t * dy
    return math.sqrt((p[0]-proj_x)**2 + (p[1]-proj_y)**2)

def coords_to_svg_path(multi_polygon_coords):
    """Convert MultiPolygon coordinates to SVG path string."""
    parts = []
    for polygon in multi_polygon_coords:
        for ring in polygon:
            simplified = simplify_ring(ring, tolerance=0.025)
            if len(simplified) < 3:
                continue
            projected = [project(lon, lat) for lon, lat in simplified]
            path_parts = []
            for i, (x, y) in enumerate(projected):
                if i == 0:
                    path_parts.append(f"M{x},{y}")
                else:
                    path_parts.append(f"L{x},{y}")
            path_parts.append("Z")
            parts.append("".join(path_parts))
    return " ".join(parts)

# Province name mapping for the simulation cities
CITY_ID_MAP = {
    'İstanbul': 'istanbul',
    'Ankara': 'ankara',
    'İzmir': 'izmir',
    'Bursa': 'bursa',
    'Antalya': 'antalya',
    'Adana': 'adana',
    'Konya': 'konya',
    'Gaziantep': 'gaziantep',
    'Diyarbakır': 'diyarbakir',
    'Kayseri': 'kayseri',
    'Trabzon': 'trabzon',
    'Samsun': 'samsun',
    'Erzurum': 'erzurum',
    'Van': 'van',
}

# Calculate centroids for city positioning
def centroid(multi_polygon_coords):
    """Calculate approximate centroid of multi-polygon."""
    total_x, total_y, total_n = 0, 0, 0
    for polygon in multi_polygon_coords:
        for ring in polygon:
            for lon, lat in ring:
                total_x += lon
                total_y += lat
                total_n += 1
    if total_n == 0:
        return (0, 0)
    avg_lon = total_x / total_n
    avg_lat = total_y / total_n
    return project(avg_lon, avg_lat)

# Process all provinces
provinces = []
city_positions = {}

for feature in geojson['features']:
    name = feature['properties']['name']
    geom = feature['geometry']
    coords = geom['coordinates']
    
    if geom['type'] == 'Polygon':
        coords = [coords]  # Normalize to MultiPolygon
    
    svg_path = coords_to_svg_path(coords)
    cx, cy = centroid(coords)
    
    city_id = CITY_ID_MAP.get(name, None)
    
    provinces.append({
        'name': name,
        'path': svg_path,
        'cx': cx,
        'cy': cy,
        'cityId': city_id
    })
    
    if city_id:
        city_positions[city_id] = (cx, cy)

# Sort alphabetically
provinces.sort(key=lambda p: p['name'])

# Output as JavaScript
print("// Auto-generated Turkey province SVG paths")
print("const TR_PROVINCES = [")
for p in provinces:
    city_field = f', cityId: "{p["cityId"]}"' if p['cityId'] else ''
    print(f'  {{ name: "{p["name"]}", cx: {p["cx"]}, cy: {p["cy"]}{city_field},')
    print(f'    d: "{p["path"]}" }},')
print("];")

print("\n// Updated city positions from actual province centroids")
print("const CITY_POSITIONS = {")
for city_id, (cx, cy) in sorted(city_positions.items()):
    print(f'  {city_id}: {{ x: {cx}, y: {cy} }},')
print("};")

# Also print total path data size
total_path_size = sum(len(p['path']) for p in provinces)
print(f"\n// Total SVG path data: {total_path_size} chars ({total_path_size//1024} KB)")
