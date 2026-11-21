ARG BASE_IMAGE=rpg_isaac:vbdf-student
FROM ${BASE_IMAGE}

USER root
SHELL ["/bin/bash", "-c"]
WORKDIR /workspace/isaaclab/source/rpg_isaac

# Merge the checkout into the handout. A bind mount here would hide the
# untracked Agilicious installation, Isaac assets, and simulator dependencies.
COPY source/ ./source/
COPY run/ ./run/
COPY scripts/ ./scripts/
COPY tools/ ./tools/
COPY flight_ws/src/ ./flight_ws/src/
COPY deliverable/report_manifest.json ./deliverable/report_manifest.json
COPY docker/course/environment.bash /etc/drone-course.bash
COPY docker/course/entrypoint.sh /usr/local/bin/drone-course-entrypoint
COPY docker/course/smoke.py /usr/local/lib/drone-course-smoke.py
COPY docker/course/report_smoke.py /usr/local/lib/drone-report-smoke.py

# Reinstall the editable ROS packages; the handout has copied launch files.
# Use a clean build directory, retaining the proprietary binary underlay.
RUN source /opt/ros/jazzy/setup.bash && \
    source /workspace/flight_ws/install/setup.bash && \
    cd /workspace/flight_ws && \
    /usr/bin/python3 -m colcon build --build-base /opt/course_build --install-base /opt/course_ws \
      --packages-select detection_msgs flightpilot drone_detect_deployment \
      --symlink-install --event-handlers console_direct+ && \
    echo 'source /etc/drone-course.bash' >> /root/.bashrc && \
    chmod +x /usr/local/bin/drone-course-entrypoint

# The inherited healthcheck expects Isaac Sim to render continuously.
HEALTHCHECK NONE
ENTRYPOINT ["/usr/local/bin/drone-course-entrypoint"]
CMD ["sleep", "infinity"]
