import os
import time
import pandas as pd

import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.ops import unary_union


# -------------------------
# CONFIG
# -------------------------
DISTRICTS = [
    "Arcachon, Gironde, France",
    "La Teste-de-Buch, Gironde, France",
    "Gujan-Mestras, Gironde, France",
    "Le Teich, Gironde, France",
    "Biganos, Gironde, France",
    "Audenge, Gironde, France",
    "Andernos-les-Bains, Gironde, France",
    "Lège-Cap-Ferret, Gironde, France",
    "Arès, Gironde, France",
    "Marcheprime, Gironde, France",
    "Mios, Gironde, France",
    "Salles, Gironde, France",
    "Bélin-Béliet, Gironde, France",
]

OUT_DIR = "task6_outputs"
DEFAULT_SPEED_KPH = 40
MAX_STATION_SNAP_DIST_M = 3000
# -------------------------


def ensure_out_dir(path):
    os.makedirs(path, exist_ok=True)


def add_travel_time(G, default_speed_kph=40):
    for u, v, k, data in G.edges(keys=True, data=True):
        length_m = data.get("length")
        if length_m is None:
            continue

        maxspeed = data.get("maxspeed")
        speed_kph = None

        if isinstance(maxspeed, list) and len(maxspeed) > 0:
            maxspeed = maxspeed[0]

        if isinstance(maxspeed, str):
            digits = "".join(ch for ch in maxspeed if ch.isdigit())
            speed_kph = float(digits) if digits else None
        elif isinstance(maxspeed, (int, float)):
            speed_kph = float(maxspeed)

        if not speed_kph or speed_kph <= 0:
            speed_kph = float(default_speed_kph)

        speed_mps = speed_kph * 1000 / 3600
        data["travel_time"] = length_m / speed_mps


def build_union_polygon(places):
    geoms = []
    areas = []

    for place in places:
        gdf = ox.geocode_to_gdf(place)
        geom = gdf.iloc[0].geometry
        geoms.append(geom)

        area_proj = gdf.to_crs(epsg=3857)
        area_km2 = area_proj.geometry.area.iloc[0] / 1_000_000
        areas.append({"district": place, "area_km2": area_km2})

    union_poly = unary_union(geoms)
    return union_poly, pd.DataFrame(areas)


def get_fire_stations_from_polygon(polygon):
    tags = {"amenity": "fire_station"}
    gdf = ox.features_from_polygon(polygon, tags=tags)

    gdf = gdf[gdf.geometry.notnull()].copy()
    if gdf.empty:
        return gdf

    gdf_proj = gdf.to_crs(epsg=3857)
    gdf_proj["geometry"] = gdf_proj.geometry.centroid
    gdf = gdf_proj.to_crs(epsg=4326)

    if "name" not in gdf.columns:
        gdf["name"] = None

    return gdf


def station_name_from_row(row, idx):
    if "name" in row and pd.notna(row["name"]) and str(row["name"]).strip() != "":
        return str(row["name"])
    return f"station_{idx}"


def build_stations_df(G, gdf_stations):
    if gdf_stations.empty:
        return pd.DataFrame()

    stations = []
    used_nodes = set()

    for i, (_, row) in enumerate(gdf_stations.iterrows(), start=1):
        pt = row.geometry
        station_name = station_name_from_row(row, i)

        nearest_node = ox.distance.nearest_nodes(G, pt.x, pt.y)
        node_x = G.nodes[nearest_node]["x"]
        node_y = G.nodes[nearest_node]["y"]

        station_tmp = gpd.GeoSeries([pt], crs="EPSG:4326").to_crs(epsg=3857)
        node_geom = gpd.points_from_xy([node_x], [node_y], crs="EPSG:4326")[0]
        node_tmp = gpd.GeoSeries([node_geom], crs="EPSG:4326").to_crs(epsg=3857)

        snap_dist_m = station_tmp.distance(node_tmp.iloc[0]).iloc[0]
        suspicious = snap_dist_m > MAX_STATION_SNAP_DIST_M

        if nearest_node in used_nodes:
            continue
        used_nodes.add(nearest_node)

        stations.append({
            "station_number": len(stations) + 1,
            "station_name": station_name,
            "original_lon": pt.x,
            "original_lat": pt.y,
            "snapped_node": nearest_node,
            "snapped_lon": node_x,
            "snapped_lat": node_y,
            "snap_distance_m": snap_dist_m,
            "suspicious_location": suspicious,
        })

    return pd.DataFrame(stations)


def annotate_stations(ax, stations_df):
    for _, row in stations_df.iterrows():
        ax.text(
            row["snapped_lon"],
            row["snapped_lat"],
            str(int(row["station_number"])),
            fontsize=9,
            fontweight="bold",
            color="red"
        )


def main():
    ensure_out_dir(OUT_DIR)

    # -----------------------------------------
    # Area
    # -----------------------------------------
    print("Building district union polygon...")
    union_polygon, area_df = build_union_polygon(DISTRICTS)

    area_df.to_csv(os.path.join(OUT_DIR, "district_areas.csv"), index=False)
    total_area_km2 = area_df["area_km2"].sum()
    print(f"Approximate total area: {total_area_km2:.2f} km²")

    # -----------------------------------------
    # Graph
    # -----------------------------------------
    print("Downloading road network for extended area...")
    graph_t0 = time.perf_counter()

    G = ox.graph_from_polygon(union_polygon, network_type="drive", simplify=True)
    G = ox.truncate.largest_component(G, strongly=False)
    add_travel_time(G)

    graph_t1 = time.perf_counter()

    print("Graph downloaded.")
    print("Nodes:", len(G.nodes))
    print("Edges:", len(G.edges))
    print(f"Graph build time: {graph_t1 - graph_t0:.4f} seconds")

    # -----------------------------------------
    # Fire stations
    # -----------------------------------------
    print("\nExtracting fire stations...")
    gdf_stations = get_fire_stations_from_polygon(union_polygon)
    print("Stations found:", len(gdf_stations))

    stations_df = build_stations_df(G, gdf_stations)
    print("Unique stations used:", len(stations_df))

    stations_csv = os.path.join(OUT_DIR, "fire_stations_used.csv")
    stations_df.to_csv(stations_csv, index=False)

    suspicious_df = stations_df[stations_df["suspicious_location"] == True]
    suspicious_df.to_csv(os.path.join(OUT_DIR, "suspicious_stations.csv"), index=False)

    # -----------------------------------------
    # Part 1
    # -----------------------------------------
    print("\nPART 1 — First station to all nodes")

    first_station_node = int(stations_df.iloc[0]["snapped_node"])

    part1_t0 = time.perf_counter()
    times_first = nx.single_source_dijkstra_path_length(
        G,
        first_station_node,
        weight="travel_time"
    )
    part1_t1 = time.perf_counter()

    part1_time = part1_t1 - part1_t0

    print("Reachable nodes from first station:", len(times_first))
    print(f"Computation time Part 1: {part1_time:.4f} seconds")

    part1_df = pd.DataFrame({
        "node": list(times_first.keys()),
        "travel_time_sec": list(times_first.values())
    })
    part1_df["travel_time_min"] = part1_df["travel_time_sec"] / 60
    part1_df.to_csv(os.path.join(OUT_DIR, "first_station_all_nodes.csv"), index=False)

    # -----------------------------------------
    # Part 2
    # -----------------------------------------
    print("\nPART 2 — All stations to all nodes")

    all_nodes = list(G.nodes)
    matrix = pd.DataFrame(index=all_nodes)
    timing_rows = []

    total_t0 = time.perf_counter()

    for _, station in stations_df.iterrows():
        station_number = int(station["station_number"])
        station_name = station["station_name"]
        station_node = int(station["snapped_node"])

        print(f"Processing station {station_number}: {station_name}")

        st_t0 = time.perf_counter()
        station_times = nx.single_source_dijkstra_path_length(
            G,
            station_node,
            weight="travel_time"
        )
        st_t1 = time.perf_counter()

        col = f"station_{station_number}"
        matrix[col] = pd.Series(station_times)

        timing_rows.append({
            "station_number": station_number,
            "station_name": station_name,
            "reachable_nodes": len(station_times),
            "computation_time_sec": st_t1 - st_t0,
        })

    total_t1 = time.perf_counter()
    total_part2_time = total_t1 - total_t0

    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(os.path.join(OUT_DIR, "station_computation_times.csv"), index=False)

    with open(os.path.join(OUT_DIR, "global_computation_times.txt"), "w", encoding="utf-8") as f:
        f.write(f"Graph build time (time.perf_counter): {graph_t1 - graph_t0:.4f} seconds\n")
        f.write(f"Part 1 computation time (time.perf_counter): {part1_time:.4f} seconds\n")
        f.write(f"Part 2 total computation time (time.perf_counter): {total_part2_time:.4f} seconds\n")
        f.write(f"Approximate total area: {total_area_km2:.2f} km²\n")

    print(f"Part 2 total computation time: {total_part2_time:.4f} seconds")

    # convert to minutes
    matrix_min = matrix / 60
    station_only = matrix_min.copy()

    matrix_min["shortest_time_min"] = station_only.min(axis=1, skipna=True)
    matrix_min["best_station_column"] = station_only.apply(
        lambda row: row.idxmin() if row.notna().any() else None,
        axis=1
    )

    col_to_number = {
        f"station_{int(row['station_number'])}": int(row["station_number"])
        for _, row in stations_df.iterrows()
    }

    matrix_min["best_station_number"] = matrix_min["best_station_column"].map(col_to_number)

    matrix_min.to_csv(os.path.join(OUT_DIR, "full_matrix_minutes.csv"), index_label="node")

    summary_df = matrix_min[[
        "shortest_time_min",
        "best_station_column",
        "best_station_number"
    ]].copy()

    summary_df.to_csv(os.path.join(OUT_DIR, "best_response_per_node.csv"), index_label="node")

    # -----------------------------------------
    # Visualization
    # -----------------------------------------
    print("\nGenerating response-time map...")

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    nodes_gdf["best_time_min"] = nodes_gdf.index.map(
        lambda n: summary_df.loc[n, "shortest_time_min"] if n in summary_df.index else None
    )

    snapped_points = gpd.GeoDataFrame(
        stations_df.copy(),
        geometry=gpd.points_from_xy(stations_df["snapped_lon"], stations_df["snapped_lat"]),
        crs="EPSG:4326"
    )

    # Main heat map
    fig, ax = plt.subplots(figsize=(13, 13))
    edges_gdf.plot(ax=ax, linewidth=0.5, color="lightgray")
    nodes_gdf.dropna(subset=["best_time_min"]).plot(
        ax=ax,
        column="best_time_min",
        cmap="viridis",
        markersize=4,
        legend=True
    )

    # Only snapped station points, no red raw dots
    snapped_points.plot(ax=ax, color="darkblue", markersize=30)

    annotate_stations(ax, stations_df)

    plt.title("Shortest response time from nearest fire station")
    plt.axis("off")
    plt.savefig(os.path.join(OUT_DIR, "response_time_map.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Diagnostics map
    fig, ax = plt.subplots(figsize=(13, 13))
    edges_gdf.plot(ax=ax, linewidth=0.5, color="lightgray")
    snapped_points.plot(ax=ax, color="darkblue", markersize=30)
    annotate_stations(ax, stations_df)
    plt.title("Snapped fire station nodes")
    plt.axis("off")
    plt.savefig(os.path.join(OUT_DIR, "station_diagnostics_map.png"), dpi=200, bbox_inches="tight")
    plt.close()

    ox.save_graphml(G, os.path.join(OUT_DIR, "graph.graphml"))

    print("Task 6 completed successfully.")


if __name__ == "__main__":
    main()