FROM apache/spark:3.5.3

USER root
RUN pip install --no-cache-dir numpy==1.24.4 \
    && mkdir -p /home/spark/.ivy2 \
    && chown -R spark:spark /home/spark
USER spark
