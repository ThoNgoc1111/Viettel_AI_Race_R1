FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# --- System dependencies for MATLAB Runtime ---
RUN apt-get update && apt-get install -y \
    wget unzip xorg \
    libxext6 libx11-6 libglu1-mesa libxi6 libxtst6 libxrender1 \
    libxrandr2 libxinerama1 libxcursor1 libxcomposite1 libxdamage1 \
    libfontconfig1 libxss1 libasound2 \
    python3.10 python3.10-distutils python3-pip \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies ---
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt -i https://pypi.org/simple

# --- MATLAB Runtime environment ---
WORKDIR /opt/mcr

# Download MATLAB Runtime installer (R2025a Linux x64)
RUN wget -q https://ssd.mathworks.com/supportfiles/downloads/R2025a/Release/0/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2025a_glnxa64.zip && \
    unzip -q MATLAB_Runtime_R2025a_glnxa64.zip -d mcr_install && \
    rm MATLAB_Runtime_R2025a_glnxa64.zip && \
    chmod +x mcr_install/install && \
    mcr_install/install -mode silent -agreeToLicense yes -destinationFolder /opt/mcr -logfile /tmp/mcr_log.txt && \
    rm -rf mcr_install

# Set library paths
ENV LD_LIBRARY_PATH=/opt/mcr/R2025a/runtime/glnxa64:/opt/mcr/R2025a/bin/glnxa64:/opt/mcr/R2025a/sys/os/glnxa64:/opt/mcr/R2025a/sys/opengl/lib/glnxa64

# --- Copy compiled MATLAB apps and shell scripts ---
WORKDIR /app
COPY runSimulationWithAnimation run_runSimulationWithAnimation.sh main_run_scenarios /app/  
COPY run_main_run_scenarios.sh /app/
COPY scenarios/ /app/scenarios/

# Pre-create MATLAB Runtime cache scenario folder so compiled binary finds scenario JSONs
# The compiled app expects scenarios under MCR_CACHE_ROOT/R2025a/<main_r*>/main_run_sce/simulation/scenarios
# Create a conservative directory (main_r0) at build time and copy scenarios into it.
RUN mkdir -p /app/mcr_cache/R2025a/main_r0/main_run_sce/simulation/scenarios && \
    cp -r /app/scenarios/* /app/mcr_cache/R2025a/main_r0/main_run_sce/simulation/scenarios/ || true

# --- Copy energy_agent Python module and make it importable ---
COPY energy_agent/ /app/energy_agent/
ENV PYTHONPATH="/app:${PYTHONPATH}"

# --- Force MATLAB Runtime cache to a fixed location ---
ENV MCR_CACHE_ROOT=/app/mcr_cache

# --- Permissions ---
RUN chmod +x runSimulationWithAnimation run_runSimulationWithAnimation.sh run_main_run_scenarios.sh main_run_scenarios || true

# Disable Docker layer caching from this point forward
ARG CACHEBUST=1

# --- Startup script ---
RUN echo '#!/bin/bash\n\
echo "MATLAB Energy Saving Simulation Container"\n\
echo "Available commands:"\n\
echo "  1. ./run_runSimulationWithAnimation.sh /opt/mcr/R2025a scenarios/indoor_hotspot.json"\n\
echo "  2. ./run_main_run_scenarios.sh /opt/mcr/R2025a"\n\
echo ""\n\
echo "For visualization, make sure to run with X11 forwarding:"\n\
echo "docker run -it --rm -e DISPLAY=\$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix <image>"\n\
echo ""\n\
/bin/bash\n\
' > /app/start.sh && chmod +x /app/start.sh

# --- X11 display environment ---
ENV DISPLAY=:0.0

# --- Default command ---
CMD ["/app/start.sh"]