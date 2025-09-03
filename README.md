# AMoD
# SUMO ride-hailling parking model simulation
This repository implements a comprehensive traffic simulation system focusing on roadside parking behavior and emission analysis in Beijing.
 The system integrates SUMO with advanced optimization algorithms for fleet management and ride-hailing dispatch.


+ config:Defines simulation parameters including area boundaries, vehicle densities, parking ratios, and emission factors. Supports both local and city-wide simulation modes.

+ paths: Manages file paths, SUMO installation detection, and output directory organization. Handles cross-platform compatibility for SUMO tools.

+ simulation_setup: Downloads OpenStreetMap data, converts to SUMO network format, and generates vehicle routes. Implements manual trip generation when SUMO's randomTrips.py is unavailable.

+ od_demand_generator: Implements Origin-Destination (OD) matrix-based demand generation with Beijing-specific travel patterns. Creates zone-based traffic analysis areas and generates realistic trip distributions.

+ fleet_optimizer: Implements vehicle shareability optimization based on MIT research. Constructs shareability networks and solves minimum path cover problems to determine optimal fleet sizes.

+ interfaces: Provides modular interfaces for demand generation, fleet optimization, dispatch optimization, and parking decision-making. Implements the Nature paper-based fleet optimizer and real-time dispatch algorithms.

+ dispatch_manager: Comprehensive ride-hailing dispatch system with driver-order matching algorithms. Calculates distances using Haversine formula, manages driver preferences and rejection lists.

+ ridehail_coordinator: Coordinates order generation, vehicle dispatch, and post-service parking decisions. Integrates with the main simulation loop for real-time ride-hailing operations.

+ parking_manager: Generates roadside parking density estimates based on road types and urban characteristics. Implements parking spot distribution algorithms for different road categories.

+ detector: Real-time detection and management of parking events during simulation. Implements safe parking position validation, traffic impact analysis, and emission calculations.

+ emissions:Calculates vehicle emissions based on HBEFA4 model with support for gasoline, diesel, and electric vehicles. Implements driving mode detection and CO2-equivalent calculations.

+ traci_manager:Manages SUMO TraCI connections with automatic retry mechanisms and graceful error handling.

